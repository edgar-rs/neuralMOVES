"""Layer 2: Fleet composition and fleet-average emission rates.

Builds fleet composition from MOVES defaults (age distribution × fuel mix ×
source type population) and computes fleet-average CO2 emission rates.

Example
-------
>>> from neuralmoves.fleet import FleetComposition, fleet_average_rate
>>> from neuralmoves.cycles import DriveCycle
>>> fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
>>> cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=300)
>>> result = fleet_average_rate(cycle, fleet, temp=25.0, humid_pct=50.0)
>>> print(f"Fleet avg: {result.fleet_avg_g_per_mile:.1f} g/mi")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import VALID_MODEL_YEARS, SOURCE_TYPE_ID_MAP, FUEL_TYPE_ID_MAP

# NeuralMOVES-supported source type IDs
_SUPPORTED_SOURCE_TYPES = set(SOURCE_TYPE_ID_MAP.keys())  # {11, 21, 31, 32, 42}

# Fuel type mapping for NeuralMOVES
# MOVES fuel IDs: 1=Gasoline, 2=Diesel, 3=CNG, 5=E85, 9=Electricity
_FUEL_MAP = {
    1: "Gasoline",
    2: "Diesel",
    5: "Gasoline",   # E85 → approximate as Gasoline for CO2
}
_EV_FUEL_IDS = {9}    # Electric vehicles → zero tailpipe CO2
_SKIP_FUEL_IDS = {3}  # CNG → skip (not modelable, tiny fleet share)

# Model year clamping range for NeuralMOVES NN models
_MIN_MODEL_YEAR = min(VALID_MODEL_YEARS)  # 2009
_MAX_MODEL_YEAR = max(VALID_MODEL_YEARS)  # 2019

# HPMS vehicle type → NeuralMOVES source types within that category
# Used for VMT weighting when computing population shares
_HPMS_TO_SOURCE_TYPES = {
    10: [11],              # Motorcycles
    25: [21, 31, 32],      # Passenger Cars + Light Trucks
    40: [42],              # Buses (Transit Bus only in NeuralMOVES)
}


def _clamp_model_year(model_year: int) -> int:
    """Clamp model year to NeuralMOVES supported range."""
    return max(_MIN_MODEL_YEAR, min(_MAX_MODEL_YEAR, model_year))


# ── Fleet Composition ─────────────────────────────────────────────────────────

@dataclass
class FleetComposition:
    """Fleet mix: fractions by (source_type, model_year, fuel_type).

    Attributes
    ----------
    fractions : pd.DataFrame
        Columns: source_type_id, model_year, fuel_type, nn_model_year, fraction
        fraction sums to 1.0 across all ICE vehicles.
    ev_penetration : float
        Fraction of VMT from EVs (0.0 to 1.0). EVs contribute 0 tailpipe CO2.
    calendar_year : int
        The calendar year this composition represents.
    """

    fractions: pd.DataFrame
    ev_penetration: float = 0.0
    calendar_year: int = 2019

    @classmethod
    def from_moves_defaults(
        cls,
        calendar_year: int = 2019,
        ev_penetration: Optional[float] = None,
        source_type_ids: Optional[set[int]] = None,
    ) -> "FleetComposition":
        """Build fleet composition from MOVES default tables.

        Combines:
        1. sourceTypeAgeDistribution → age fractions per source type
        2. sampleVehiclePopulation → fuel type fractions per (source type, model year)
        3. sourceTypeYear → population weights per source type

        Parameters
        ----------
        calendar_year : int
            Calendar year for the fleet (determines age distribution).
        ev_penetration : float or None
            Override EV penetration (0-1). If None, uses MOVES default EV share.
        source_type_ids : set[int] or None
            Source types to include. Default: all NeuralMOVES-supported types.
        """
        from .defaults import (
            load_age_distribution,
            load_sample_vehicle_population,
            load_source_type_population,
        )

        if source_type_ids is None:
            source_type_ids = _SUPPORTED_SOURCE_TYPES

        age_df = load_age_distribution()
        svp_df = load_sample_vehicle_population()
        pop_df = load_source_type_population()

        # Filter to supported source types and calendar year
        age_df = age_df[
            (age_df["sourceTypeID"].isin(source_type_ids))
            & (age_df["yearID"] == calendar_year)
        ].copy()

        # Derive model year from age: modelYear = calendarYear - ageID
        age_df["modelYear"] = calendar_year - age_df["ageID"]

        # Get population weights for this calendar year
        pop_year = pop_df[
            (pop_df["yearID"] == calendar_year)
            & (pop_df["sourceTypeID"].isin(source_type_ids))
        ][["sourceTypeID", "sourceTypePopulation"]].copy()

        # Normalize population to fractions
        total_pop = pop_year["sourceTypePopulation"].sum()
        if total_pop > 0:
            pop_year["popFraction"] = (
                pop_year["sourceTypePopulation"] / total_pop
            )
        else:
            pop_year["popFraction"] = 1.0 / len(pop_year)

        # Join age fractions with population weights
        merged = age_df.merge(
            pop_year[["sourceTypeID", "popFraction"]],
            on="sourceTypeID",
            how="left",
        )
        merged["popFraction"] = merged["popFraction"].fillna(0)

        # Now join with fuel type fractions from sampleVehiclePopulation
        # Filter SVP to our source types
        svp_filt = svp_df[svp_df["sourceTypeID"].isin(source_type_ids)].copy()

        # Merge on (sourceTypeID, modelYearID)
        combined = merged.merge(
            svp_filt[["sourceTypeID", "modelYearID", "fuelTypeID", "stmyFraction"]],
            left_on=["sourceTypeID", "modelYear"],
            right_on=["sourceTypeID", "modelYearID"],
            how="left",
        )

        # For model years not in SVP, default to 100% gasoline
        no_fuel = combined["fuelTypeID"].isna()
        if no_fuel.any():
            defaults = combined[no_fuel][
                ["sourceTypeID", "modelYear", "ageID", "ageFraction", "popFraction"]
            ].copy()
            defaults["fuelTypeID"] = 1.0  # Gasoline
            defaults["stmyFraction"] = 1.0
            combined = pd.concat(
                [combined[~no_fuel], defaults], ignore_index=True
            )

        combined["fuelTypeID"] = combined["fuelTypeID"].astype(int)

        # Compute fleet fraction: popFraction × ageFraction × stmyFraction
        combined["rawFraction"] = (
            combined["popFraction"]
            * combined["ageFraction"]
            * combined["stmyFraction"]
        )

        # Separate EVs from ICE vehicles
        ev_mask = combined["fuelTypeID"].isin(_EV_FUEL_IDS)
        skip_mask = combined["fuelTypeID"].isin(_SKIP_FUEL_IDS)

        default_ev_share = combined.loc[ev_mask, "rawFraction"].sum()
        ice_raw = combined[~ev_mask & ~skip_mask].copy()

        # Map fuel type IDs to NeuralMOVES fuel type names
        ice_raw["fuel_type"] = ice_raw["fuelTypeID"].map(_FUEL_MAP)
        ice_raw = ice_raw.dropna(subset=["fuel_type"])

        # Clamp model year to NN range
        ice_raw["nn_model_year"] = ice_raw["modelYear"].apply(_clamp_model_year)

        # Aggregate by (source_type_id, nn_model_year, fuel_type)
        grouped = (
            ice_raw.groupby(
                ["sourceTypeID", "nn_model_year", "fuel_type"], as_index=False
            )["rawFraction"]
            .sum()
            .rename(columns={
                "sourceTypeID": "source_type_id",
                "rawFraction": "fraction",
            })
        )

        # Determine actual EV penetration
        if ev_penetration is not None:
            actual_ev = ev_penetration
        else:
            actual_ev = default_ev_share

        # Normalize ICE fractions to sum to (1 - ev_penetration)
        ice_total = grouped["fraction"].sum()
        if ice_total > 0:
            grouped["fraction"] = grouped["fraction"] / ice_total * (1.0 - actual_ev)

        grouped["nn_model_year"] = grouped["nn_model_year"].astype(int)
        grouped["source_type_id"] = grouped["source_type_id"].astype(int)

        return cls(
            fractions=grouped.reset_index(drop=True),
            ev_penetration=actual_ev,
            calendar_year=calendar_year,
        )

    def with_ev_penetration(self, ev_fraction: float) -> "FleetComposition":
        """Return a new FleetComposition with a different EV penetration.

        ICE fractions are uniformly rescaled so that
        sum(ICE fractions) = 1 - ev_fraction.
        """
        if not 0.0 <= ev_fraction <= 1.0:
            raise ValueError(f"ev_fraction must be 0-1, got {ev_fraction}")

        new_fracs = self.fractions.copy()
        current_ice = new_fracs["fraction"].sum()
        if current_ice > 0:
            scale = (1.0 - ev_fraction) / current_ice
            new_fracs["fraction"] = new_fracs["fraction"] * scale

        return FleetComposition(
            fractions=new_fracs,
            ev_penetration=ev_fraction,
            calendar_year=self.calendar_year,
        )

    @property
    def n_vehicle_configs(self) -> int:
        return len(self.fractions)

    def summary(self) -> pd.DataFrame:
        """Summarize fleet by source type."""
        df = self.fractions.copy()
        df["source_type_name"] = df["source_type_id"].map(SOURCE_TYPE_ID_MAP)
        result = (
            df.groupby("source_type_name")["fraction"]
            .sum()
            .reset_index()
            .rename(columns={"fraction": "fleet_share"})
        )
        # Add EV row
        if self.ev_penetration > 0:
            ev_row = pd.DataFrame([{
                "source_type_name": "Electric (zero tailpipe)",
                "fleet_share": self.ev_penetration,
            }])
            result = pd.concat([result, ev_row], ignore_index=True)
        return result

    def __repr__(self) -> str:
        return (
            f"FleetComposition(year={self.calendar_year}, "
            f"configs={self.n_vehicle_configs}, "
            f"ev={self.ev_penetration:.1%})"
        )


# ── Fleet-Average Emission Rate ──────────────────────────────────────────────

@dataclass
class FleetEmissionResult:
    """Results from fleet-average emission calculation."""

    fleet_avg_g_per_mile: float
    fleet_avg_g_per_km: float
    by_config: pd.DataFrame
    ev_co2_avoided_g_per_mile: float
    fleet: FleetComposition = field(repr=False)
    cycle_name: str = ""

    def __repr__(self) -> str:
        return (
            f"FleetEmissionResult({self.cycle_name!r}, "
            f"fleet_avg={self.fleet_avg_g_per_mile:.1f} g/mi, "
            f"ev_avoided={self.ev_co2_avoided_g_per_mile:.1f} g/mi)"
        )


def fleet_average_rate(
    cycle: "DriveCycle",
    fleet: FleetComposition,
    *,
    temp: float,
    humid_pct: float,
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> FleetEmissionResult:
    """Compute fleet-average emission rate for a drive cycle.

    Evaluates the cycle for each vehicle configuration in the fleet,
    then computes a weighted average based on fleet fractions.

    Parameters
    ----------
    cycle : DriveCycle
        The drive cycle to evaluate.
    fleet : FleetComposition
        Fleet composition with fractions.
    temp : float
        Ambient temperature.
    humid_pct : float
        Relative humidity (0-100).
    temp_unit : str
        Temperature unit.
    map_location : str
        Device for inference.

    Returns
    -------
    FleetEmissionResult
    """
    from .cycles import evaluate_cycle

    rows = []
    for _, row in fleet.fractions.iterrows():
        st_id = int(row["source_type_id"])
        st_name = SOURCE_TYPE_ID_MAP.get(st_id)
        if st_name is None:
            continue

        fuel = row["fuel_type"]
        nn_year = int(row["nn_model_year"])
        frac = row["fraction"]

        if frac <= 0:
            continue

        result = evaluate_cycle(
            cycle,
            temp=temp,
            humid_pct=humid_pct,
            model_year=nn_year,
            source_type=st_name,
            fuel_type=fuel,
            temp_unit=temp_unit,
            map_location=map_location,
        )

        rows.append({
            "source_type_id": st_id,
            "source_type": st_name,
            "fuel_type": fuel,
            "nn_model_year": nn_year,
            "fraction": frac,
            "rate_g_per_mile": result.rate_g_per_mile,
            "weighted_g_per_mile": frac * result.rate_g_per_mile,
        })

    config_df = pd.DataFrame(rows)

    fleet_avg = config_df["weighted_g_per_mile"].sum()
    fleet_avg_km = fleet_avg / 1.60934

    # Estimate EV avoided emissions: if the EV share were all ICE instead
    if fleet.ev_penetration > 0:
        ice_avg = fleet_avg / (1.0 - fleet.ev_penetration) if fleet.ev_penetration < 1.0 else 0.0
        ev_avoided = ice_avg * fleet.ev_penetration
    else:
        ev_avoided = 0.0

    return FleetEmissionResult(
        fleet_avg_g_per_mile=fleet_avg,
        fleet_avg_g_per_km=fleet_avg_km,
        by_config=config_df,
        ev_co2_avoided_g_per_mile=ev_avoided,
        fleet=fleet,
        cycle_name=cycle.name,
    )


__all__ = [
    "FleetComposition",
    "FleetEmissionResult",
    "fleet_average_rate",
]
