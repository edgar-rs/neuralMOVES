"""Layer 5: Location aggregation — county/state-specific emission inventories.

Uses MOVES default meteorology and HPMS VMT data to produce location-specific
emission estimates.

Example
-------
>>> from neuralmoves.geography import LocationProfile
>>> # Get profile for a specific county (FIPS code)
>>> loc = LocationProfile.from_county_defaults(26163)  # Wayne County, MI
>>> print(f"Jan temp: {loc.avg_temp_by_month[1]:.1f}°F")
>>> print(f"Annual VMT: {loc.annual_vmt:,.0f}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LocationProfile:
    """Location-specific data for emission calculations.

    Attributes
    ----------
    location_id : str
        FIPS code or custom label.
    location_name : str
        Human-readable name.
    annual_vmt : float
        Total annual VMT for this location.
    avg_temp_by_month : dict[int, float]
        Monthly average temperature in Fahrenheit, keyed by monthID (1-12).
    avg_humid_by_month : dict[int, float]
        Monthly average relative humidity (%), keyed by monthID (1-12).
    """

    location_id: str
    location_name: str = ""
    annual_vmt: float = 0.0
    avg_temp_by_month: dict[int, float] = field(default_factory=dict)
    avg_humid_by_month: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_county_defaults(
        cls,
        fips_code: int,
        calendar_year: int = 2019,
        location_name: str = "",
    ) -> "LocationProfile":
        """Build a location profile from MOVES county meteorology defaults.

        Parameters
        ----------
        fips_code : int
            County FIPS code (e.g., 26163 for Wayne County, MI).
        calendar_year : int
            Calendar year for VMT lookup.
        location_name : str
            Optional human-readable name.
        """
        from .defaults import load_county_meteorology, load_hpms_vtype_year

        met_df = load_county_meteorology()
        county_met = met_df[met_df["countyID"] == fips_code]

        if county_met.empty:
            raise ValueError(
                f"No meteorology data for county FIPS={fips_code}. "
                f"Available counties: {sorted(met_df['countyID'].unique())[:10]}..."
            )

        temp_by_month = dict(
            zip(county_met["monthID"], county_met["temperature_F"])
        )
        humid_by_month = dict(
            zip(county_met["monthID"], county_met["relHumidity_pct"])
        )

        # Estimate VMT from HPMS (use type 25 = passenger + light truck as proxy)
        hpms_df = load_hpms_vtype_year()
        hpms_row = hpms_df[
            (hpms_df["HPMSVtypeID"] == 25)
            & (hpms_df["yearID"] == calendar_year)
        ]
        if not hpms_row.empty:
            # National VMT for type 25; scale by 1/3233 counties as rough default
            # Users should override annual_vmt for accurate county analysis
            national_vmt = hpms_row["HPMSBaseYearVMT"].iloc[0]
            n_counties = met_df["countyID"].nunique()
            annual_vmt = national_vmt / max(n_counties, 1)
        else:
            annual_vmt = 0.0

        return cls(
            location_id=str(fips_code),
            location_name=location_name or f"County {fips_code}",
            annual_vmt=annual_vmt,
            avg_temp_by_month=temp_by_month,
            avg_humid_by_month=humid_by_month,
        )

    @classmethod
    def from_custom(
        cls,
        location_id: str,
        annual_vmt: float,
        avg_temp_f: float,
        avg_humid_pct: float,
        location_name: str = "",
    ) -> "LocationProfile":
        """Create a profile with uniform temperature/humidity year-round."""
        return cls(
            location_id=location_id,
            location_name=location_name or location_id,
            annual_vmt=annual_vmt,
            avg_temp_by_month={m: avg_temp_f for m in range(1, 13)},
            avg_humid_by_month={m: avg_humid_pct for m in range(1, 13)},
        )

    def available_counties() -> list[int]:
        """Return list of available county FIPS codes."""
        from .defaults import load_county_meteorology
        return sorted(load_county_meteorology()["countyID"].unique().tolist())

    def __repr__(self) -> str:
        name = self.location_name or self.location_id
        return (
            f"LocationProfile({name!r}, vmt={self.annual_vmt:,.0f}, "
            f"months={len(self.avg_temp_by_month)})"
        )


# ── Location-level inventory ──────────────────────────────────────────────────

def location_inventory(
    location: LocationProfile,
    fleet: "FleetComposition",
    cycle: "DriveCycle",
    *,
    calendar_year: int = 2019,
    source_type_id: int = 21,
    map_location: str = "cpu",
) -> pd.DataFrame:
    """Compute monthly emission inventory for a location.

    Evaluates the cycle at each month's temperature/humidity and scales by
    monthly VMT fractions.

    Parameters
    ----------
    location : LocationProfile
        Location with meteorology and VMT data.
    fleet : FleetComposition
        Fleet composition.
    cycle : DriveCycle
        Drive cycle to evaluate.
    calendar_year : int
        Calendar year.
    source_type_id : int
        Source type for VMT fractions.
    map_location : str
        Device for inference.

    Returns
    -------
    pd.DataFrame
        Monthly breakdown: monthID, temperature_F, humidity_pct,
        fleet_rate_g_per_mile, monthly_vmt, monthly_co2_g, monthly_co2_metric_tons.
    """
    from .fleet import fleet_average_rate
    from .temporal import VMTAllocation

    vmt = VMTAllocation.from_moves_defaults(
        source_type_id=source_type_id,
        calendar_year=calendar_year,
        annual_vmt=location.annual_vmt,
    )
    monthly_vmt = vmt.monthly_vmt()

    rows = []
    for _, vmt_row in monthly_vmt.iterrows():
        month_id = int(vmt_row["monthID"])
        m_vmt = vmt_row["monthly_vmt"]

        temp_f = location.avg_temp_by_month.get(month_id, 68.0)
        humid = location.avg_humid_by_month.get(month_id, 50.0)

        result = fleet_average_rate(
            cycle, fleet,
            temp=temp_f, humid_pct=humid,
            temp_unit="F",
            map_location=map_location,
        )

        co2_g = result.fleet_avg_g_per_mile * m_vmt

        rows.append({
            "monthID": month_id,
            "temperature_F": temp_f,
            "humidity_pct": humid,
            "fleet_rate_g_per_mile": result.fleet_avg_g_per_mile,
            "monthly_vmt": m_vmt,
            "monthly_co2_g": co2_g,
            "monthly_co2_metric_tons": co2_g / 1_000_000.0,
        })

    return pd.DataFrame(rows)


def multi_location_inventory(
    locations: list[LocationProfile],
    fleet: "FleetComposition",
    cycle: "DriveCycle",
    *,
    calendar_year: int = 2019,
    source_type_id: int = 21,
    map_location: str = "cpu",
) -> pd.DataFrame:
    """Compute annual inventory across multiple locations.

    Returns
    -------
    pd.DataFrame
        One row per location: location_id, location_name, annual_vmt,
        annual_co2_metric_tons, avg_rate_g_per_mile.
    """
    rows = []
    for loc in locations:
        inv = location_inventory(
            loc, fleet, cycle,
            calendar_year=calendar_year,
            source_type_id=source_type_id,
            map_location=map_location,
        )
        annual_co2_g = inv["monthly_co2_g"].sum()
        avg_rate = annual_co2_g / loc.annual_vmt if loc.annual_vmt > 0 else 0.0

        rows.append({
            "location_id": loc.location_id,
            "location_name": loc.location_name,
            "annual_vmt": loc.annual_vmt,
            "annual_co2_g": annual_co2_g,
            "annual_co2_metric_tons": annual_co2_g / 1_000_000.0,
            "avg_rate_g_per_mile": avg_rate,
        })

    return pd.DataFrame(rows)


__all__ = [
    "LocationProfile",
    "location_inventory",
    "multi_location_inventory",
]
