"""
NeuralMOVES: Lightweight microscopic surrogate of EPA MOVES for vehicle CO₂ emissions.

This package provides fast, accurate emission predictions using neural network
surrogates trained on EPA MOVES output.
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

__all__ = [
    "__version__",
    "estimate_running_co2",
    "estimate_emissions_timeseries",
    "idling_rate",
    "list_available_models",
    "get_expected_error",
    "VALID_MODEL_YEARS",
]
