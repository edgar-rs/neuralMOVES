"""Layer 1: Drive cycle evaluation.

Wraps the per-second NeuralMOVES engine with standard drive cycle support
and summary metrics (g/mile, g/km, total CO2).

Example
-------
>>> from neuralmoves.cycles import DriveCycle, evaluate_cycle
>>> cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=600)
>>> result = evaluate_cycle(
...     cycle, temp=25.0, humid_pct=50.0,
...     model_year=2019, source_type="Passenger Car", fuel_type="Gasoline",
... )
>>> print(f"{result.rate_g_per_mile:.1f} g/mi")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ── Drive Cycle ───────────────────────────────────────────────────────────────

_MPH_TO_MS = 0.44704
_M_TO_MILE = 1.0 / 1609.344


@dataclass
class DriveCycle:
    """A speed trace with optional grade profile.

    Parameters
    ----------
    name : str
        Descriptive name for the cycle.
    time_s : np.ndarray
        Time stamps in seconds, shape (N,).
    speed_ms : np.ndarray
        Speed in m/s at each time step, shape (N,).
    grade_pct : np.ndarray or None
        Road grade (%) at each time step.  Defaults to 0%.
    """

    name: str
    time_s: np.ndarray
    speed_ms: np.ndarray
    grade_pct: np.ndarray = field(default=None)

    def __post_init__(self):
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.speed_ms = np.asarray(self.speed_ms, dtype=float)
        if self.grade_pct is None:
            self.grade_pct = np.zeros_like(self.speed_ms)
        else:
            self.grade_pct = np.asarray(self.grade_pct, dtype=float)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0

    @property
    def accel_mps2(self) -> np.ndarray:
        """Derive acceleration from the speed trace (forward difference)."""
        dt = np.diff(self.time_s)
        dt = np.where(dt == 0, 1.0, dt)  # avoid divide-by-zero
        dv = np.diff(self.speed_ms)
        a = np.concatenate([[0.0], dv / dt])
        return a

    @property
    def distance_miles(self) -> float:
        """Total distance traveled in miles."""
        dt = np.diff(self.time_s)
        avg_speed = (self.speed_ms[:-1] + self.speed_ms[1:]) / 2.0
        return float(np.sum(avg_speed * dt) * _M_TO_MILE)

    @property
    def avg_speed_mph(self) -> float:
        if self.duration_s == 0:
            return 0.0
        return self.distance_miles / (self.duration_s / 3600.0)

    # ── Factory methods ──────────────────────────────────────────────────

    @classmethod
    def constant_speed(
        cls, speed_mph: float, duration_s: int = 600, name: str = None,
    ) -> "DriveCycle":
        """Create a constant-speed cycle (with short ramp-up/down)."""
        if name is None:
            name = f"Constant {speed_mph:.0f} mph"
        ramp = min(10, duration_s // 4)
        t = np.arange(duration_s, dtype=float)
        v_target = speed_mph * _MPH_TO_MS
        speed = np.full(duration_s, v_target)
        # Linear ramp-up and ramp-down
        speed[:ramp] = np.linspace(0, v_target, ramp)
        speed[-ramp:] = np.linspace(v_target, 0, ramp)
        return cls(name=name, time_s=t, speed_ms=speed)

    @classmethod
    def from_csv(cls, path: Union[str, Path], name: str = None) -> "DriveCycle":
        """Load a drive cycle from a CSV file.

        Expected columns: time_s, speed_mph (or speed_ms).
        Optional columns: grade_pct.
        """
        path = Path(path)
        df = pd.read_csv(path)
        if name is None:
            name = path.stem

        time_s = df["time_s"].values if "time_s" in df.columns else np.arange(len(df))

        if "speed_ms" in df.columns:
            speed_ms = df["speed_ms"].values
        elif "speed_mph" in df.columns:
            speed_ms = df["speed_mph"].values * _MPH_TO_MS
        else:
            raise ValueError(
                f"CSV must contain 'speed_ms' or 'speed_mph' column. "
                f"Found: {list(df.columns)}"
            )

        grade = df["grade_pct"].values if "grade_pct" in df.columns else None
        return cls(name=name, time_s=time_s, speed_ms=speed_ms, grade_pct=grade)

    @classmethod
    def from_speed_array(
        cls,
        speed_mph: np.ndarray,
        dt: float = 1.0,
        grade_pct: np.ndarray = None,
        name: str = "custom",
    ) -> "DriveCycle":
        """Create a cycle from an array of speeds in mph."""
        speed_mph = np.asarray(speed_mph, dtype=float)
        time_s = np.arange(len(speed_mph)) * dt
        return cls(
            name=name,
            time_s=time_s,
            speed_ms=speed_mph * _MPH_TO_MS,
            grade_pct=grade_pct,
        )

    def __repr__(self) -> str:
        return (
            f"DriveCycle({self.name!r}, {self.duration_s:.0f}s, "
            f"{self.distance_miles:.2f}mi, avg={self.avg_speed_mph:.1f}mph)"
        )


# ── Cycle Evaluation ──────────────────────────────────────────────────────────

@dataclass
class CycleEmissionResult:
    """Results from evaluating one cycle with one vehicle configuration."""

    total_co2_g: float
    rate_g_per_mile: float
    rate_g_per_km: float
    duration_s: float
    distance_miles: float
    timeseries_g_per_s: np.ndarray
    cycle_name: str = ""
    model_year: int = 0
    source_type: str = ""
    fuel_type: str = ""

    def __repr__(self) -> str:
        return (
            f"CycleEmissionResult({self.cycle_name!r}, "
            f"{self.rate_g_per_mile:.1f} g/mi, "
            f"{self.total_co2_g:.1f} g total)"
        )


def evaluate_cycle(
    cycle: DriveCycle,
    *,
    temp: float,
    humid_pct: float,
    model_year: int,
    source_type: str,
    fuel_type: str,
    temp_unit: str = "C",
    apply_idling_floor: bool = True,
    map_location: str = "cpu",
) -> CycleEmissionResult:
    """Evaluate emissions for a drive cycle and vehicle configuration.

    Parameters
    ----------
    cycle : DriveCycle
        The drive cycle to evaluate.
    temp : float
        Ambient temperature.
    humid_pct : float
        Relative humidity (0-100).
    model_year : int
        Vehicle model year (2009-2019).
    source_type : str
        Vehicle category (e.g., 'Passenger Car').
    fuel_type : str
        'Gasoline' or 'Diesel'.
    temp_unit : str
        Temperature unit: 'C' or 'F'.
    apply_idling_floor : bool
        Apply idling emission floor.
    map_location : str
        Device for inference.

    Returns
    -------
    CycleEmissionResult
    """
    from .api import estimate_emissions_timeseries

    emissions = estimate_emissions_timeseries(
        speed_ms=cycle.speed_ms,
        accel_mps2=cycle.accel_mps2,
        grade_pct=cycle.grade_pct,
        temp=temp,
        humid_pct=humid_pct,
        temp_unit=temp_unit,
        model_year=model_year,
        source_type=source_type,
        fuel_type=fuel_type,
        apply_idling_floor=apply_idling_floor,
        map_location=map_location,
    )

    # Sum total CO2 (each value is g/s, one per second)
    dt = np.diff(cycle.time_s, prepend=cycle.time_s[0])
    dt[0] = 1.0  # first time step
    total_co2 = float(np.sum(emissions * dt))

    dist_mi = cycle.distance_miles
    rate_per_mile = total_co2 / dist_mi if dist_mi > 0 else 0.0
    rate_per_km = rate_per_mile / 1.60934

    return CycleEmissionResult(
        total_co2_g=total_co2,
        rate_g_per_mile=rate_per_mile,
        rate_g_per_km=rate_per_km,
        duration_s=cycle.duration_s,
        distance_miles=dist_mi,
        timeseries_g_per_s=emissions,
        cycle_name=cycle.name,
        model_year=model_year,
        source_type=source_type,
        fuel_type=fuel_type,
    )


def evaluate_cycle_multi(
    cycle: DriveCycle,
    *,
    temp: float,
    humid_pct: float,
    configs: list[dict],
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> pd.DataFrame:
    """Evaluate a cycle for multiple vehicle configurations.

    Parameters
    ----------
    cycle : DriveCycle
        The drive cycle.
    temp, humid_pct : float
        Environmental conditions.
    configs : list[dict]
        Each dict must have keys: model_year, source_type, fuel_type.
    temp_unit : str
        Temperature unit.
    map_location : str
        Device for inference.

    Returns
    -------
    pd.DataFrame
        One row per config with columns: model_year, source_type, fuel_type,
        total_co2_g, rate_g_per_mile, rate_g_per_km.
    """
    rows = []
    for cfg in configs:
        result = evaluate_cycle(
            cycle,
            temp=temp,
            humid_pct=humid_pct,
            model_year=cfg["model_year"],
            source_type=cfg["source_type"],
            fuel_type=cfg["fuel_type"],
            temp_unit=temp_unit,
            map_location=map_location,
        )
        rows.append({
            "model_year": cfg["model_year"],
            "source_type": cfg["source_type"],
            "fuel_type": cfg["fuel_type"],
            "total_co2_g": result.total_co2_g,
            "rate_g_per_mile": result.rate_g_per_mile,
            "rate_g_per_km": result.rate_g_per_km,
        })
    return pd.DataFrame(rows)


__all__ = [
    "DriveCycle",
    "CycleEmissionResult",
    "evaluate_cycle",
    "evaluate_cycle_multi",
]
