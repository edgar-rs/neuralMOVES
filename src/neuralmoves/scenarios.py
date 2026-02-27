"""Layer 6: Scenario comparison and sensitivity analysis.

Top-level module that ties all aggregation layers together to compare
baseline vs. policy scenarios.

Example
-------
>>> from neuralmoves.scenarios import ScenarioDefinition, compare_scenarios
>>> from neuralmoves.technology import TechnologyScenario
>>>
>>> baseline = ScenarioDefinition(
...     name="BAU (5% EV)",
...     technology=TechnologyScenario.constant("bau", ev_fraction=0.05),
...     years=list(range(2025, 2036)),
... )
>>> policy = ScenarioDefinition(
...     name="EV push (5→30%)",
...     technology=TechnologyScenario.linear_ev_ramp("policy", 2025, 2035, 0.05, 0.30),
...     years=list(range(2025, 2036)),
... )
>>> results = compare_scenarios([baseline, policy])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ScenarioDefinition:
    """A complete scenario definition for emission analysis.

    Parameters
    ----------
    name : str
        Descriptive name for this scenario.
    technology : TechnologyScenario
        EV adoption trajectory.
    years : list[int]
        Calendar years to evaluate.
    cycle : DriveCycle or None
        Drive cycle. If None, uses a default 30 mph constant speed cycle.
    source_type_id : int
        Primary source type for VMT allocation.
    annual_vmt : float or None
        Override annual VMT. If None, uses MOVES defaults.
    location : LocationProfile or None
        Optional location for temperature-varying analysis.
    """

    name: str
    technology: "TechnologyScenario" = None
    years: list[int] = field(default_factory=lambda: list(range(2025, 2036)))
    cycle: Optional["DriveCycle"] = None
    source_type_id: int = 21
    annual_vmt: Optional[float] = None
    location: Optional["LocationProfile"] = None


@dataclass
class ScenarioResult:
    """Results from evaluating a single scenario."""

    name: str
    yearly: pd.DataFrame  # year, ev_fraction, fleet_rate_g_per_mile, annual_vmt, annual_co2_metric_tons
    total_co2_metric_tons: float

    def __repr__(self) -> str:
        yrs = self.yearly["year"].tolist()
        return (
            f"ScenarioResult({self.name!r}, "
            f"{yrs[0]}-{yrs[-1]}, "
            f"total={self.total_co2_metric_tons:,.0f} MT)"
        )


def evaluate_scenario(
    scenario: ScenarioDefinition,
    *,
    temp: float = 20.0,
    humid_pct: float = 50.0,
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> ScenarioResult:
    """Evaluate a scenario over its year range.

    Parameters
    ----------
    scenario : ScenarioDefinition
        The scenario to evaluate.
    temp : float
        Default temperature (used if no location specified).
    humid_pct : float
        Default humidity.
    temp_unit : str
        Temperature unit.
    map_location : str
        Device for inference.

    Returns
    -------
    ScenarioResult
    """
    from .cycles import DriveCycle
    from .fleet import FleetComposition, fleet_average_rate
    from .temporal import VMTAllocation

    cycle = scenario.cycle
    if cycle is None:
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=600)

    rows = []
    for year in scenario.years:
        # Get fleet for this year
        ev_frac = scenario.technology.get_ev_fraction(year)

        # Build fleet — clamp calendar year to MOVES data range (1990-2060)
        fleet_year = max(1990, min(2060, year))
        try:
            fleet = FleetComposition.from_moves_defaults(
                calendar_year=fleet_year,
                ev_penetration=ev_frac,
            )
        except Exception:
            # If year data unavailable, use nearest available
            fleet = FleetComposition.from_moves_defaults(
                calendar_year=2019,
                ev_penetration=ev_frac,
            )

        # Get VMT
        vmt = VMTAllocation.from_moves_defaults(
            source_type_id=scenario.source_type_id,
            calendar_year=fleet_year,
            annual_vmt=scenario.annual_vmt,
        )

        # Use location-specific temperature if available
        if scenario.location is not None:
            # Average across months for a single representative rate
            temps = list(scenario.location.avg_temp_by_month.values())
            humids = list(scenario.location.avg_humid_by_month.values())
            use_temp = np.mean(temps) if temps else temp
            use_humid = np.mean(humids) if humids else humid_pct
            use_unit = "F"
        else:
            use_temp = temp
            use_humid = humid_pct
            use_unit = temp_unit

        # Compute fleet average rate
        result = fleet_average_rate(
            cycle, fleet,
            temp=use_temp, humid_pct=use_humid,
            temp_unit=use_unit,
            map_location=map_location,
        )

        annual_co2_g = result.fleet_avg_g_per_mile * vmt.annual_vmt
        rows.append({
            "year": year,
            "ev_fraction": ev_frac,
            "fleet_rate_g_per_mile": result.fleet_avg_g_per_mile,
            "annual_vmt": vmt.annual_vmt,
            "annual_co2_g": annual_co2_g,
            "annual_co2_metric_tons": annual_co2_g / 1_000_000.0,
        })

    yearly_df = pd.DataFrame(rows)
    total = yearly_df["annual_co2_metric_tons"].sum()

    return ScenarioResult(
        name=scenario.name,
        yearly=yearly_df,
        total_co2_metric_tons=total,
    )


def compare_scenarios(
    scenarios: list[ScenarioDefinition],
    *,
    temp: float = 20.0,
    humid_pct: float = 50.0,
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> pd.DataFrame:
    """Compare multiple scenarios side by side.

    Returns
    -------
    pd.DataFrame
        Wide format with year rows and one column per scenario for
        annual CO2 (metric tons).
    """
    results = {}
    for scenario in scenarios:
        result = evaluate_scenario(
            scenario,
            temp=temp, humid_pct=humid_pct,
            temp_unit=temp_unit, map_location=map_location,
        )
        results[scenario.name] = result

    # Build comparison DataFrame
    all_years = set()
    for r in results.values():
        all_years.update(r.yearly["year"].tolist())

    rows = []
    for year in sorted(all_years):
        row = {"year": year}
        for name, result in results.items():
            yr_data = result.yearly[result.yearly["year"] == year]
            if not yr_data.empty:
                row[f"{name}_co2_MT"] = yr_data["annual_co2_metric_tons"].iloc[0]
                row[f"{name}_ev_pct"] = yr_data["ev_fraction"].iloc[0]
                row[f"{name}_g_per_mi"] = yr_data["fleet_rate_g_per_mile"].iloc[0]
            else:
                row[f"{name}_co2_MT"] = np.nan
                row[f"{name}_ev_pct"] = np.nan
                row[f"{name}_g_per_mi"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def ev_impact_analysis(
    ev_fractions: list[float],
    *,
    calendar_year: int = 2019,
    cycle: Optional["DriveCycle"] = None,
    temp: float = 20.0,
    humid_pct: float = 50.0,
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> pd.DataFrame:
    """Analyze emission impact across a range of EV penetration levels.

    Parameters
    ----------
    ev_fractions : list[float]
        EV penetration levels to test (0-1).
    calendar_year : int
        Calendar year for fleet composition.
    cycle : DriveCycle or None
        Drive cycle. Default: 30 mph constant.
    temp, humid_pct : float
        Environmental conditions.

    Returns
    -------
    pd.DataFrame
        Columns: ev_fraction, fleet_rate_g_per_mile, reduction_pct
    """
    from .cycles import DriveCycle
    from .fleet import FleetComposition, fleet_average_rate

    if cycle is None:
        cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=600)

    base_fleet = FleetComposition.from_moves_defaults(
        calendar_year=calendar_year, ev_penetration=0.0,
    )

    # Get baseline rate (0% EV)
    base_result = fleet_average_rate(
        cycle, base_fleet,
        temp=temp, humid_pct=humid_pct,
        temp_unit=temp_unit, map_location=map_location,
    )
    base_rate = base_result.fleet_avg_g_per_mile

    rows = []
    for ev_frac in ev_fractions:
        fleet = base_fleet.with_ev_penetration(ev_frac)
        result = fleet_average_rate(
            cycle, fleet,
            temp=temp, humid_pct=humid_pct,
            temp_unit=temp_unit, map_location=map_location,
        )
        reduction = (
            (1.0 - result.fleet_avg_g_per_mile / base_rate) * 100.0
            if base_rate > 0 else 0.0
        )
        rows.append({
            "ev_fraction": ev_frac,
            "fleet_rate_g_per_mile": result.fleet_avg_g_per_mile,
            "reduction_pct": reduction,
        })

    return pd.DataFrame(rows)


def sensitivity_analysis(
    base_scenario: ScenarioDefinition,
    parameter: str,
    values: list,
    *,
    temp: float = 20.0,
    humid_pct: float = 50.0,
    temp_unit: str = "C",
    map_location: str = "cpu",
) -> pd.DataFrame:
    """Run sensitivity analysis on a single parameter.

    Parameters
    ----------
    base_scenario : ScenarioDefinition
        Base scenario definition.
    parameter : str
        Parameter to vary. Options: 'ev_penetration', 'annual_vmt', 'temp'.
    values : list
        Values to test for the parameter.

    Returns
    -------
    pd.DataFrame
        Columns: parameter_value, total_co2_metric_tons, pct_change
    """
    from .technology import TechnologyScenario

    base_result = evaluate_scenario(
        base_scenario, temp=temp, humid_pct=humid_pct,
        temp_unit=temp_unit, map_location=map_location,
    )
    base_total = base_result.total_co2_metric_tons

    rows = []
    for val in values:
        modified = ScenarioDefinition(
            name=f"{base_scenario.name}_{parameter}={val}",
            technology=base_scenario.technology,
            years=base_scenario.years,
            cycle=base_scenario.cycle,
            source_type_id=base_scenario.source_type_id,
            annual_vmt=base_scenario.annual_vmt,
            location=base_scenario.location,
        )

        use_temp = temp
        use_humid = humid_pct

        if parameter == "ev_penetration":
            modified.technology = TechnologyScenario.constant(
                f"ev_{val}", ev_fraction=val, years=base_scenario.years,
            )
        elif parameter == "annual_vmt":
            modified.annual_vmt = val
        elif parameter == "temp":
            use_temp = val
        else:
            raise ValueError(f"Unknown parameter: {parameter!r}")

        result = evaluate_scenario(
            modified, temp=use_temp, humid_pct=use_humid,
            temp_unit=temp_unit, map_location=map_location,
        )

        pct_change = (
            (result.total_co2_metric_tons - base_total) / base_total * 100.0
            if base_total > 0 else 0.0
        )
        rows.append({
            "parameter_value": val,
            "total_co2_metric_tons": result.total_co2_metric_tons,
            "pct_change_from_base": pct_change,
        })

    return pd.DataFrame(rows)


__all__ = [
    "ScenarioDefinition",
    "ScenarioResult",
    "evaluate_scenario",
    "compare_scenarios",
    "ev_impact_analysis",
    "sensitivity_analysis",
]
