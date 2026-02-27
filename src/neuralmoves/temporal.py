"""Layer 3: Time aggregation — scale per-cycle rates to annual/monthly totals.

Uses MOVES default VMT fractions (monthly, daily, hourly) and HPMS VMT data
to distribute emissions across time periods.

Example
-------
>>> from neuralmoves.temporal import VMTAllocation, annualize_emissions
>>> vmt = VMTAllocation.from_moves_defaults(
...     source_type_id=21, calendar_year=2019,
... )
>>> annual_co2_tons = annualize_emissions(
...     rate_g_per_mile=350.0, vmt=vmt,
... )
>>> print(f"Annual CO2: {annual_co2_tons.annual_co2_metric_tons:.1f} metric tons")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import SOURCE_TYPE_ID_MAP

# MOVES HPMS vehicle type mapping:
# sourceTypeID → HPMSVtypeID
_SOURCE_TO_HPMS = {
    11: 10,   # Motorcycles
    21: 25,   # Passenger Car → Passenger + Light Truck
    31: 25,   # Passenger Truck → Passenger + Light Truck
    32: 25,   # Light Commercial Truck → Passenger + Light Truck
    42: 40,   # Transit Bus → Buses
}

# Approximate VMT share within HPMS type 25 (from sourceTypeYear population ratios)
# These are rough national averages; users can override via annual_vmt parameter.
_HPMS25_VMT_SHARE = {
    21: 0.54,   # Passenger Cars
    31: 0.33,   # Passenger Trucks
    32: 0.13,   # Light Commercial Trucks
}

_G_PER_METRIC_TON = 1_000_000.0


@dataclass
class VMTAllocation:
    """VMT allocation for a source type across time periods.

    Attributes
    ----------
    annual_vmt : float
        Total annual VMT in miles.
    month_fractions : pd.DataFrame
        monthID (1-12), monthVMTFraction (sums to 1.0).
    source_type_id : int
        MOVES source type ID.
    calendar_year : int
        Calendar year.
    """

    annual_vmt: float
    month_fractions: pd.DataFrame
    source_type_id: int
    calendar_year: int

    @classmethod
    def from_moves_defaults(
        cls,
        source_type_id: int = 21,
        calendar_year: int = 2019,
        annual_vmt: Optional[float] = None,
    ) -> "VMTAllocation":
        """Build VMT allocation from MOVES defaults.

        Parameters
        ----------
        source_type_id : int
            MOVES source type ID (11, 21, 31, 32, or 42).
        calendar_year : int
            Calendar year for VMT data.
        annual_vmt : float or None
            Override annual VMT. If None, derives from HPMS defaults.
        """
        from .defaults import load_month_vmt_fraction, load_hpms_vtype_year

        # Load monthly VMT fractions
        month_df = load_month_vmt_fraction()
        month_frac = month_df[
            month_df["sourceTypeID"] == source_type_id
        ][["monthID", "monthVMTFraction"]].copy()

        if month_frac.empty:
            raise ValueError(
                f"No monthly VMT fractions for sourceTypeID={source_type_id}"
            )

        # Derive annual VMT from HPMS if not provided
        if annual_vmt is None:
            hpms_df = load_hpms_vtype_year()
            hpms_id = _SOURCE_TO_HPMS.get(source_type_id)

            if hpms_id is not None:
                hpms_row = hpms_df[
                    (hpms_df["HPMSVtypeID"] == hpms_id)
                    & (hpms_df["yearID"] == calendar_year)
                ]
                if not hpms_row.empty:
                    total_hpms_vmt = hpms_row["HPMSBaseYearVMT"].iloc[0]
                    # For HPMS type 25, split by source type share
                    if hpms_id == 25:
                        share = _HPMS25_VMT_SHARE.get(source_type_id, 0.33)
                        annual_vmt = total_hpms_vmt * share
                    else:
                        annual_vmt = total_hpms_vmt
                else:
                    annual_vmt = 0.0
            else:
                annual_vmt = 0.0

        return cls(
            annual_vmt=annual_vmt,
            month_fractions=month_frac.reset_index(drop=True),
            source_type_id=source_type_id,
            calendar_year=calendar_year,
        )

    def monthly_vmt(self) -> pd.DataFrame:
        """Return monthly VMT breakdown."""
        df = self.month_fractions.copy()
        df["monthly_vmt"] = df["monthVMTFraction"] * self.annual_vmt
        return df

    def __repr__(self) -> str:
        st = SOURCE_TYPE_ID_MAP.get(self.source_type_id, str(self.source_type_id))
        return (
            f"VMTAllocation({st}, year={self.calendar_year}, "
            f"annual_vmt={self.annual_vmt:,.0f})"
        )


# ── Annualization ─────────────────────────────────────────────────────────────

@dataclass
class AnnualEmissionResult:
    """Results from annualizing per-mile emission rate."""

    annual_co2_g: float
    annual_co2_metric_tons: float
    annual_vmt: float
    monthly_breakdown: pd.DataFrame
    rate_g_per_mile: float

    def __repr__(self) -> str:
        return (
            f"AnnualEmissionResult("
            f"annual_co2={self.annual_co2_metric_tons:.2f} metric tons, "
            f"vmt={self.annual_vmt:,.0f})"
        )


def annualize_emissions(
    rate_g_per_mile: float,
    vmt: VMTAllocation,
) -> AnnualEmissionResult:
    """Scale a per-mile emission rate to annual totals using VMT.

    Parameters
    ----------
    rate_g_per_mile : float
        Emission rate in g CO2 per mile.
    vmt : VMTAllocation
        VMT allocation with annual total and monthly fractions.

    Returns
    -------
    AnnualEmissionResult
    """
    annual_co2 = rate_g_per_mile * vmt.annual_vmt

    monthly = vmt.monthly_vmt()
    monthly["co2_g"] = rate_g_per_mile * monthly["monthly_vmt"]

    return AnnualEmissionResult(
        annual_co2_g=annual_co2,
        annual_co2_metric_tons=annual_co2 / _G_PER_METRIC_TON,
        annual_vmt=vmt.annual_vmt,
        monthly_breakdown=monthly,
        rate_g_per_mile=rate_g_per_mile,
    )


def annualize_with_temperature(
    cycle: "DriveCycle",
    vmt: VMTAllocation,
    *,
    monthly_temps: dict[int, float],
    monthly_humids: dict[int, float],
    model_year: int,
    source_type: str,
    fuel_type: str,
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> AnnualEmissionResult:
    """Annualize emissions accounting for monthly temperature variation.

    Evaluates the cycle at each month's average temperature, then weights
    by monthly VMT fractions. This captures seasonal effects (e.g., cold
    weather increases emissions).

    Parameters
    ----------
    cycle : DriveCycle
        Drive cycle to evaluate.
    vmt : VMTAllocation
        VMT allocation.
    monthly_temps : dict[int, float]
        Monthly average temperatures, keyed by monthID (1-12).
    monthly_humids : dict[int, float]
        Monthly average relative humidity (%), keyed by monthID (1-12).
    model_year : int
        Vehicle model year.
    source_type : str
        Vehicle type.
    fuel_type : str
        Fuel type.
    temp_unit : str
        Temperature unit for monthly_temps values.
    map_location : str
        Device for inference.

    Returns
    -------
    AnnualEmissionResult
    """
    from .cycles import evaluate_cycle

    monthly = vmt.monthly_vmt()
    co2_by_month = []

    for _, row in monthly.iterrows():
        month_id = int(row["monthID"])
        month_vmt = row["monthly_vmt"]

        temp = monthly_temps.get(month_id, 20.0)
        humid = monthly_humids.get(month_id, 50.0)

        result = evaluate_cycle(
            cycle,
            temp=temp,
            humid_pct=humid,
            model_year=model_year,
            source_type=source_type,
            fuel_type=fuel_type,
            temp_unit=temp_unit,
            map_location=map_location,
        )

        co2_by_month.append({
            "monthID": month_id,
            "rate_g_per_mile": result.rate_g_per_mile,
            "monthly_vmt": month_vmt,
            "co2_g": result.rate_g_per_mile * month_vmt,
        })

    monthly_df = pd.DataFrame(co2_by_month)
    annual_co2 = monthly_df["co2_g"].sum()
    avg_rate = annual_co2 / vmt.annual_vmt if vmt.annual_vmt > 0 else 0.0

    return AnnualEmissionResult(
        annual_co2_g=annual_co2,
        annual_co2_metric_tons=annual_co2 / _G_PER_METRIC_TON,
        annual_vmt=vmt.annual_vmt,
        monthly_breakdown=monthly_df,
        rate_g_per_mile=avg_rate,
    )


def multiyear_inventory(
    rate_g_per_mile_by_year: dict[int, float],
    vmt_by_year: dict[int, VMTAllocation],
) -> pd.DataFrame:
    """Compute multi-year emission inventory.

    Parameters
    ----------
    rate_g_per_mile_by_year : dict[int, float]
        Emission rate per mile for each year.
    vmt_by_year : dict[int, VMTAllocation]
        VMT allocation for each year.

    Returns
    -------
    pd.DataFrame
        Columns: year, annual_vmt, rate_g_per_mile,
                 annual_co2_g, annual_co2_metric_tons.
    """
    rows = []
    for year in sorted(rate_g_per_mile_by_year.keys()):
        rate = rate_g_per_mile_by_year[year]
        vmt = vmt_by_year.get(year)
        if vmt is None:
            continue
        result = annualize_emissions(rate, vmt)
        rows.append({
            "year": year,
            "annual_vmt": vmt.annual_vmt,
            "rate_g_per_mile": rate,
            "annual_co2_g": result.annual_co2_g,
            "annual_co2_metric_tons": result.annual_co2_metric_tons,
        })
    return pd.DataFrame(rows)


__all__ = [
    "VMTAllocation",
    "AnnualEmissionResult",
    "annualize_emissions",
    "annualize_with_temperature",
    "multiyear_inventory",
]
