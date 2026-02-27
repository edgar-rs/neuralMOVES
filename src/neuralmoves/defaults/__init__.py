"""MOVES default data loaders.

Each function loads a CSV extracted from the EPA MOVES default database
and returns a cached pandas DataFrame (loaded once per session).

Data source: EPA MOVES Model default database (movesdb20241112).
Extraction script: scripts/extract_moves_defaults.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from ._loader import _load_csv


def load_age_distribution() -> "pd.DataFrame":
    """sourceTypeAgeDistribution: ageFraction by (sourceTypeID, yearID, ageID).

    Sum rule: ageFraction sums to 1.0 per (sourceTypeID, yearID).
    """
    return _load_csv("age_distribution.csv")


def load_sample_vehicle_population() -> "pd.DataFrame":
    """sampleVehiclePopulation: fuel/engine fractions by (sourceType, modelYear).

    Columns: sourceTypeModelYearID, sourceTypeID, modelYearID, fuelTypeID,
             engTechID, regClassID, stmyFuelEngFraction, stmyFraction

    stmyFraction sums to 1.0 per (sourceTypeID, modelYearID).
    Used in place of the AVFT table (which is empty in MOVES defaults).
    """
    return _load_csv("sample_vehicle_population.csv")


def load_source_type_population() -> "pd.DataFrame":
    """sourceTypeYear: population and growth by (yearID, sourceTypeID).

    Columns: yearID, sourceTypeID, salesGrowthFactor,
             sourceTypePopulation, migrationrate
    """
    return _load_csv("source_type_population.csv")


def load_month_vmt_fraction() -> "pd.DataFrame":
    """monthVMTFraction: monthly VMT distribution by sourceTypeID.

    Sum rule: monthVMTFraction sums to 1.0 per sourceTypeID.
    """
    return _load_csv("month_vmt_fraction.csv")


def load_day_vmt_fraction() -> "pd.DataFrame":
    """dayVMTFraction: weekday/weekend VMT split.

    Columns: sourceTypeID, monthID, roadTypeID, dayID, dayVMTFraction
    Sum rule: dayVMTFraction sums to 1.0 per (sourceTypeID, monthID, roadTypeID).
    """
    return _load_csv("day_vmt_fraction.csv")


def load_hour_vmt_fraction() -> "pd.DataFrame":
    """hourVMTFraction: hourly VMT distribution.

    Columns: sourceTypeID, roadTypeID, dayID, hourID, hourVMTFraction
    Sum rule: hourVMTFraction sums to 1.0 per (sourceTypeID, roadTypeID, dayID).
    """
    return _load_csv("hour_vmt_fraction.csv")


def load_hpms_vtype_year() -> "pd.DataFrame":
    """hpmsVtypeYear: HPMS vehicle-type VMT by year.

    Columns: HPMSVtypeID, yearID, VMTGrowthFactor, HPMSBaseYearVMT
    HPMS types: 10=Motorcycles, 25=PassengerCar+LightTruck,
                40=Buses, 50=SingleUnitTruck, 60=CombinationTruck
    """
    return _load_csv("hpms_vtype_year.csv")


def load_county_meteorology() -> "pd.DataFrame":
    """County-level monthly average temperature and humidity.

    Columns: countyID, monthID, temperature_F, relHumidity_pct
    Aggregated from MOVES zoneMonthHour table (averaged across hours).
    """
    return _load_csv("county_meteorology.csv")


def load_avg_speed_distribution() -> "pd.DataFrame":
    """avgSpeedDistribution: speed bin fractions.

    Columns: sourceTypeID, roadTypeID, hourDayID, avgSpeedBinID, avgSpeedFraction
    Sum rule: avgSpeedFraction sums to 1.0 per (sourceTypeID, roadTypeID, hourDayID).
    """
    return _load_csv("avg_speed_distribution.csv")


__all__ = [
    "load_age_distribution",
    "load_sample_vehicle_population",
    "load_source_type_population",
    "load_month_vmt_fraction",
    "load_day_vmt_fraction",
    "load_hour_vmt_fraction",
    "load_hpms_vtype_year",
    "load_county_meteorology",
    "load_avg_speed_distribution",
]
