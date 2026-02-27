"""
NeuralMOVES: Lightweight microscopic surrogate of EPA MOVES for vehicle CO₂ emissions.

This package provides fast, accurate emission predictions using neural network
surrogates trained on EPA MOVES output.

Modules
-------
api : Per-second, per-vehicle CO₂ estimates (core engine).
cycles : Layer 1 — drive cycle evaluation and summary metrics.
fleet : Layer 2 — fleet composition and fleet-average rates.
temporal : Layer 3 — time aggregation (annual/monthly totals via VMT).
technology : Layer 4 — technology scenario modeling (EV adoption curves).
geography : Layer 5 — location-specific inventories (county meteorology).
scenarios : Layer 6 — scenario comparison and sensitivity analysis.
defaults : MOVES default data loaders (bundled CSV tables).
"""

from .__about__ import __version__
from .api import (
    estimate_running_co2,
    estimate_emissions_timeseries,
    idling_rate,
    list_available_models,
    get_expected_error,
)
from .config import VALID_MODEL_YEARS
from .cycles import DriveCycle, CycleEmissionResult, evaluate_cycle
from .fleet import FleetComposition, FleetEmissionResult, fleet_average_rate
from .temporal import VMTAllocation, annualize_emissions
from .technology import TechnologyScenario
from .geography import LocationProfile
from .scenarios import (
    ScenarioDefinition,
    evaluate_scenario,
    compare_scenarios,
    ev_impact_analysis,
)

__all__ = [
    # Metadata
    "__version__",
    "VALID_MODEL_YEARS",
    # Core API (per-second engine)
    "estimate_running_co2",
    "estimate_emissions_timeseries",
    "idling_rate",
    "list_available_models",
    "get_expected_error",
    # Layer 1: Drive cycles
    "DriveCycle",
    "CycleEmissionResult",
    "evaluate_cycle",
    # Layer 2: Fleet
    "FleetComposition",
    "FleetEmissionResult",
    "fleet_average_rate",
    # Layer 3: Temporal
    "VMTAllocation",
    "annualize_emissions",
    # Layer 4: Technology
    "TechnologyScenario",
    # Layer 5: Geography
    "LocationProfile",
    # Layer 6: Scenarios
    "ScenarioDefinition",
    "evaluate_scenario",
    "compare_scenarios",
    "ev_impact_analysis",
]
