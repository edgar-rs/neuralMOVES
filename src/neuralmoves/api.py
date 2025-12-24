"""Public API for NeuralMOVES emission calculations."""

from __future__ import annotations
from typing import Union
import numpy as np
import pandas as pd
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


def _to_celsius(temp: float, unit: str) -> float:
    """Convert temperature to Celsius."""
    unit = (unit or "C").strip().upper()
    if unit == "C":
        return temp
    if unit in {"F", "°F"}:
        return (temp - 32.0) * (5.0 / 9.0)
    raise ValueError(f"Unknown temperature unit {unit!r}; use 'C' or 'F'.")


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
    map_location: Union[str, torch.device] = "cpu",
) -> float:
    """
    Estimate per-second running-exhaust CO₂ emission rate.
    
    This function uses a trained neural network surrogate of EPA MOVES to predict
    emissions based on instantaneous vehicle kinematics and environmental conditions.
    
    Parameters
    ----------
    v_ms : float
        Vehicle speed in meters per second (m/s)
    a_mps2 : float
        Vehicle acceleration in meters per second squared (m/s²)
    grade_pct : float
        Road grade as percentage (100 × rise/run)
        Examples: 0 = flat, 5 = 5% uphill, -3 = 3% downhill
    temp : float
        Ambient temperature (unit specified by temp_unit)
    humid_pct : float
        Relative humidity as percentage (0-100)
    temp_unit : str, optional
        Temperature unit: 'C' for Celsius or 'F' for Fahrenheit (default: 'C')
    model_year : int
        Vehicle model year (2009-2019)
        Maps to vehicle age in the MOVES database
    source_type : str
        Vehicle category. Options:
        - 'Motorcycles' (or 'motorcycle', 'mc')
        - 'Passenger Car' (or 'passenger car', 'pc')
        - 'Passenger Truck' (or 'passenger truck', 'pt')
        - 'Light Commercial Truck' (or 'light commercial truck', 'lct')
        - 'Transit Bus' (or 'transit bus', 'bus')
    fuel_type : str
        Fuel type: 'Gasoline' (or 'gas') or 'Diesel'
        Note: Diesel motorcycles are not supported
    apply_idling_floor : bool, optional
        If True, return max(predicted_emission, idling_emission) (default: True)
        This ensures emissions don't go below the idling baseline
    map_location : str or torch.device, optional
        Device for model inference (default: 'cpu')
        
    Returns
    -------
    float
        Running-exhaust CO₂ emission rate in grams per second (g/s)
        
    Raises
    ------
    ValueError
        If invalid source_type, fuel_type, model_year, or unsupported combination
    FileNotFoundError
        If model weights or idling data not found for the specified combination
        
    Examples
    --------
    >>> import neuralmoves
    >>> # Estimate emissions for a passenger car at 15 m/s on flat road
    >>> co2_gps = neuralmoves.estimate_running_co2(
    ...     v_ms=15.0,
    ...     a_mps2=0.5,
    ...     grade_pct=0.0,
    ...     temp=25.0,
    ...     humid_pct=50.0,
    ...     model_year=2015,
    ...     source_type='Passenger Car',
    ...     fuel_type='Gasoline'
    ... )
    >>> print(f"CO2: {co2_gps:.3f} g/s")
    
    Notes
    -----
    The neural network takes inputs in this order:
    [temperature (°C), humidity (%), speed (m/s), acceleration (m/s²), grade (%)]
    
    The model predicts running emissions, and optionally applies an idling floor
    which represents the minimum emissions when the vehicle is stationary with
    engine running.
    """
    # Normalize and validate inputs
    st = normalize_source_type(source_type)
    ft = normalize_fuel_type(fuel_type)
    validate_combo(model_year, st, ft)
    
    # Convert temperature to Celsius
    temp_C = _to_celsius(temp, temp_unit)
    
    # Prepare input tensor in the order expected by the model:
    # [temperature, humidity, speed, acceleration, grade]
    x = torch.tensor(
        [[temp_C, humid_pct, v_ms, a_mps2, grade_pct]],
        dtype=torch.float32
    )
    
    # Load the appropriate submodel
    key = SubmodelKey(model_year, st, ft)
    model = load_submodel(key, map_location=map_location)
    
    # Run inference
    with torch.no_grad():
        pred = float(model(x).cpu().item())
    
    # Apply idling floor if requested
    if apply_idling_floor:
        idle_gps = lookup_idling_gps(key)
        return max(pred, idle_gps)
    
    return pred


def estimate_emissions_timeseries(
    speed_ms: np.ndarray,
    accel_mps2: np.ndarray,
    grade_pct: np.ndarray,
    temp: float,
    humid_pct: float,
    *,
    temp_unit: str = "C",
    model_year: int,
    source_type: str,
    fuel_type: str,
    apply_idling_floor: bool = True,
    map_location: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """
    Estimate CO₂ emissions for a time series of driving conditions.
    
    This is a batch version of estimate_running_co2 for processing entire
    driving cycles efficiently.
    
    Parameters
    ----------
    speed_ms : np.ndarray
        Array of vehicle speeds in m/s, shape (N,)
    accel_mps2 : np.ndarray
        Array of accelerations in m/s², shape (N,)
    grade_pct : np.ndarray
        Array of road grades in %, shape (N,)
    temp : float
        Ambient temperature (constant for the cycle)
    humid_pct : float
        Relative humidity (constant for the cycle)
    temp_unit : str, optional
        Temperature unit: 'C' or 'F' (default: 'C')
    model_year : int
        Vehicle model year (2009-2019)
    source_type : str
        Vehicle category (see estimate_running_co2 for options)
    fuel_type : str
        'Gasoline' or 'Diesel'
    apply_idling_floor : bool, optional
        Apply idling floor to all predictions (default: True)
    map_location : str or torch.device, optional
        Device for inference (default: 'cpu')
        
    Returns
    -------
    np.ndarray
        CO₂ emission rates in g/s, shape (N,)
        
    Examples
    --------
    >>> import neuralmoves
    >>> import numpy as np
    >>> # Create a simple driving cycle
    >>> speed = np.array([0, 5, 10, 15, 20, 15, 10, 5, 0])  # m/s
    >>> accel = np.diff(speed, prepend=0)  # m/s²
    >>> grade = np.zeros_like(speed)  # flat road
    >>> emissions = neuralmoves.estimate_emissions_timeseries(
    ...     speed_ms=speed,
    ...     accel_mps2=accel,
    ...     grade_pct=grade,
    ...     temp=20.0,
    ...     humid_pct=60.0,
    ...     model_year=2018,
    ...     source_type='Passenger Car',
    ...     fuel_type='Gasoline'
    ... )
    >>> print(f"Total CO2: {emissions.sum():.2f} g")
    """
    # Validate inputs
    st = normalize_source_type(source_type)
    ft = normalize_fuel_type(fuel_type)
    validate_combo(model_year, st, ft)
    
    # Convert arrays to numpy if needed
    speed_ms = np.asarray(speed_ms)
    accel_mps2 = np.asarray(accel_mps2)
    grade_pct = np.asarray(grade_pct)
    
    # Check shapes
    if not (speed_ms.shape == accel_mps2.shape == grade_pct.shape):
        raise ValueError(
            f"Shape mismatch: speed {speed_ms.shape}, "
            f"accel {accel_mps2.shape}, grade {grade_pct.shape}"
        )
    
    # Convert temperature
    temp_C = _to_celsius(temp, temp_unit)
    
    # Build input matrix: [temp, humid, speed, accel, grade]
    n = len(speed_ms)
    temp_col = np.full(n, temp_C)
    humid_col = np.full(n, humid_pct)
    
    X = np.column_stack([temp_col, humid_col, speed_ms, accel_mps2, grade_pct])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Load model
    key = SubmodelKey(model_year, st, ft)
    model = load_submodel(key, map_location=map_location)
    
    # Batch inference
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()
    
    # Apply idling floor if requested
    if apply_idling_floor:
        idle_gps = lookup_idling_gps(key)
        preds = np.maximum(preds, idle_gps)
    
    return preds


def idling_rate(model_year: int, source_type: str, fuel_type: str) -> float:
    """
    Get the idling emission rate for a specific vehicle configuration.
    
    Parameters
    ----------
    model_year : int
        Vehicle model year (2009-2019)
    source_type : str
        Vehicle category
    fuel_type : str
        'Gasoline' or 'Diesel'
        
    Returns
    -------
    float
        Idling CO₂ emission rate in g/s
        
    Examples
    --------
    >>> import neuralmoves
    >>> idle = neuralmoves.idling_rate(2015, 'Passenger Car', 'Gasoline')
    >>> print(f"Idling rate: {idle:.4f} g/s")
    """
    key = SubmodelKey.from_user(model_year, source_type, fuel_type)
    return lookup_idling_gps(key)


def list_available_models() -> list[tuple[int, str, str]]:
    """
    List all available (year, source_type, fuel_type) combinations.
    
    Returns
    -------
    list[tuple[int, str, str]]
        List of (model_year, source_type, fuel_type) tuples
        
    Examples
    --------
    >>> import neuralmoves
    >>> models = neuralmoves.list_available_models()
    >>> print(f"Found {len(models)} models")
    >>> print(models[:3])  # Show first 3
    """
    found = []
    
    for year in VALID_MODEL_YEARS:
        for st in sorted(set(SOURCE_TYPES.values())):
            for ft in sorted(set(FUEL_TYPES.values())):
                try:
                    key = SubmodelKey.from_user(year, st, ft)
                    # Just verify idling data exists (lightweight check)
                    lookup_idling_gps(key)
                    found.append((year, st, ft))
                except (ValueError, KeyError, FileNotFoundError):
                    # Skip unsupported combinations
                    continue
    
    return found


def get_expected_error(
    *,
    fuel_type: str = None,
    source_type: str = None,
    grade_bucket: str = None,
) -> dict:
    """
    Retrieve expected error metrics for model predictions.
    
    This function returns validation statistics (MAPE, MPE, etc.) from
    the optional error_lookup.csv file if available.
    
    Parameters
    ----------
    fuel_type : str, optional
        Filter by fuel type
    source_type : str, optional
        Filter by vehicle category
    grade_bucket : str, optional
        Filter by grade category (e.g., "Zero Grade", "Positive Grade")
        
    Returns
    -------
    dict or None
        Dictionary with keys: MAPE, MPE, MdPE, StdPE, MAE_g
        Returns None if error lookup is not available
        
    Notes
    -----
    The function searches for the most specific match first (combo),
    then category-level matches, and finally falls back to overall statistics.
    """
    rows = load_error_lookup()
    if rows is None:
        return None
    
    # Normalize inputs
    ft = normalize_fuel_type(fuel_type) if fuel_type else None
    st = normalize_source_type(source_type) if source_type else None
    gb = (grade_bucket or "").strip() if grade_bucket else None
    
    # Try exact combo match first
    if ft and st and gb:
        target = f"{ft} × {st} × {gb}"
        for r in rows:
            if r.get("scope") == "combo" and r.get("category") == target:
                return r
    
    # Try category matches
    if ft:
        for r in rows:
            if (r.get("scope") == "category" and 
                r.get("category") == "Fuel Type" and 
                r.get("subcategory") == ft):
                return r
    
    if st:
        for r in rows:
            if (r.get("scope") == "category" and 
                r.get("category") == "Vehicle Type" and 
                r.get("subcategory") == st):
                return r
    
    if gb:
        for r in rows:
            if (r.get("scope") == "category" and 
                r.get("category") == "Road Grade" and 
                r.get("subcategory") == gb):
                return r
    
    # Fall back to overall
    for r in rows:
        if r.get("scope") == "overall":
            return r
    
    return None


__all__ = [
    "estimate_running_co2",
    "estimate_emissions_timeseries",
    "idling_rate",
    "list_available_models",
    "get_expected_error",
]
