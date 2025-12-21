from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math
import torch

from .__about__ import __version__
from .config import (
    normalize_fuel_type,
    normalize_source_type,
    validate_combo,
    VALID_MODEL_YEARS,
    SOURCE_TYPES,
    FUEL_TYPES,
)
from .loaders import SubmodelKey, load_submodel, lookup_idling_gps, load_error_lookup


@dataclass(frozen=True)
class Inputs:
    v_ms: float          # speed [m/s]
    a_mps2: float        # acceleration [m/s^2]
    grade_pct: float     # road grade [%]
    temp_C: float        # temperature [°C]
    humid_pct: float     # relative humidity [% 0..100]


def _to_C(temp: float, unit: str) -> float:
    unit = (unit or "C").strip().upper()
    if unit == "C":
        return temp
    if unit in {"F", "°F"}:
        return (temp - 32.0) * (5.0 / 9.0)
    raise ValueError(f"Unknown temperature unit {unit!r}; use 'C' or 'F'.")


def list_available_models() -> list[SubmodelKey]:
    """
    Returns the list of available (year, source, fuel) combos
    by scanning the packaged NN_3/ weights.
    """
    # Importlib doesn't glob; instead, derive from allowed combos and check existence by try/except.
    found: list[SubmodelKey] = []
    for year in VALID_MODEL_YEARS:
        for st in sorted(set(SOURCE_TYPES.values())):
            for ft in sorted(set(FUEL_TYPES.values())):
                try:
                    key = SubmodelKey.from_user(year, st, ft)
                    # try loading, but avoid loading weights (costly); just check idling availability
                    lookup_idling_gps(key)
                    found.append(key)
                except Exception:
                    continue
    return found


def estimate_running_co2(
    v_ms: float,
    a_mps2: float,
    grade_pct: float,
    temp: float,
    humid_pct: float,
    *,
    temp_unit: str = "C",
    model_year: int,
    source_type: str,
    fuel_type: str,
    apply_idling_floor: bool = True,
    map_location: str | torch.device = "cpu",
) -> float:
    """
    Estimate per-second running-exhaust CO2 emission [g/s] using the NeuralMOVES submodel
    for the specified (model_year, source_type, fuel_type).

    Parameters
    ----------
    v_ms, a_mps2, grade_pct, temp, humid_pct : float
        Kinematics/environment. Temperature unit controlled by `temp_unit`.
    temp_unit : {'C','F'}
        Unit for `temp`. Internally converted to °C.
    model_year : int
        Cohort (maps one-to-one to vehicle age under a fixed analysis year).
    source_type : str
        e.g., 'Passenger Car', 'Motorcycles', 'Transit Bus', ...
    fuel_type : str
        'Gasoline' or 'Diesel'. (Diesel motorcycles not supported.)
    apply_idling_floor : bool
        If True, max(pred, idling_gps) is returned.
    map_location : str | torch.device
        Where to load/run the model (e.g., 'cpu').

    Returns
    -------
    float
        Running-exhaust CO2 [g/s], with idling floor if enabled.
    """
    st = normalize_source_type(source_type)
    ft = normalize_fuel_type(fuel_type)
    validate_combo(model_year, st, ft)

    temp_C = _to_C(temp, temp_unit)
    x = torch.tensor([[v_ms, a_mps2, grade_pct, temp_C, humid_pct]], dtype=torch.float32)

    key = SubmodelKey(model_year, st, ft)
    model = load_submodel(key, map_location=map_location)

    with torch.no_grad():
        pred = float(model(x).item())

    if apply_idling_floor:
        idle = lookup_idling_gps(key)  # [g/s]
        return max(pred, idle)
    return pred


def idling_rate(model_year: int, source_type: str, fuel_type: str) -> float:
    """Return the idling floor [g/s] for the specified cohort."""
    key = SubmodelKey.from_user(model_year, source_type, fuel_type)
    return lookup_idling_gps(key)


def get_expected_error(
    *,
    fuel_type: Optional[str] = None,
    source_type: Optional[str] = None,
    grade_bucket: Optional[str] = None,
) -> Optional[dict]:
    """
    Return an expected error record (MAPE/MPE/MdPE/StdPE/MAE) for a given slice, if an
    `error_lookup.csv` is packaged. The CSV lets you update error stats without changing code.

    Typical rows could include:
        scope,category,subcategory,MAPE,MPE,MdPE,StdPE,MAE_g
        overall,Overall,All Data,6.01,2.46,1.22,8.90,28.65
        category,Fuel Type,Gasoline,5.51,2.03,0.86,8.33,27.95
        combo,Gasoline × Passenger Car × Zero Grade,,4.37,-0.69,-0.99,5.75,12.53
    """
    rows = load_error_lookup()
    if rows is None:
        return None

    # Construct a simple matching heuristic
    ft = normalize_fuel_type(fuel_type) if fuel_type else None
    st = normalize_source_type(source_type) if source_type else None
    gb = (grade_bucket or "").strip() if grade_bucket else None

    # exact combo match first
    if ft and st and gb:
        target = f"{ft} × {st} × {gb}"
        for r in rows:
            if r.get("scope") == "combo" and r.get("category") == target:
                return r

    # category matches next
    if ft:
        for r in rows:
            if r.get("scope") == "category" and r.get("category") == "Fuel Type" and r.get("subcategory") == ft:
                return r
    if st:
        for r in rows:
            if r.get("scope") == "category" and r.get("category") == "Vehicle Type" and r.get("subcategory") == st:
                return r
    if gb:
        for r in rows:
            if r.get("scope") == "category" and r.get("category") == "Road Grade" and r.get("subcategory") == gb:
                return r

    # fall back to overall
    for r in rows:
        if r.get("scope") == "overall":
            return r
    return None
