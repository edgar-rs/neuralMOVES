"""Configuration constants and validation for NeuralMOVES."""

from __future__ import annotations

# Canonical string names (as appear in filenames/user-facing API)
SOURCE_TYPES = {
    "motorcycle": "Motorcycles",
    "mc": "Motorcycles",
    "motorcycles": "Motorcycles",
    "passenger car": "Passenger Car",
    "pc": "Passenger Car",
    "passenger truck": "Passenger Truck",
    "pt": "Passenger Truck",
    "light commercial truck": "Light Commercial Truck",
    "lct": "Light Commercial Truck",
    "transit bus": "Transit Bus",
    "bus": "Transit Bus",
}

FUEL_TYPES = {
    "gas": "Gasoline",
    "gasoline": "Gasoline",
    "diesel": "Diesel",
}

# Mapping of MOVES numeric IDs to canonical names (for CSV and model filenames)
SOURCE_TYPE_ID_MAP = {
    11: "Motorcycles",
    21: "Passenger Car",
    31: "Passenger Truck",
    32: "Light Commercial Truck",
    42: "Transit Bus",
}

FUEL_TYPE_ID_MAP = {
    1: "Gasoline",
    2: "Diesel",
}

# Reverse mappings for filename generation
SOURCE_TYPE_TO_ID = {v: k for k, v in SOURCE_TYPE_ID_MAP.items()}
FUEL_TYPE_TO_ID = {v: k for k, v in FUEL_TYPE_ID_MAP.items()}

# Supported model years (cohorts)
VALID_MODEL_YEARS = tuple(range(2009, 2020))

# Combinations not available (e.g., diesel motorcycles)
INVALID_COMBOS = {("Motorcycles", "Diesel")}


def normalize_source_type(x: str) -> str:
    """Normalize user input to canonical source type name."""
    key = (x or "").strip().lower()
    if key not in SOURCE_TYPES:
        valid = sorted(set(SOURCE_TYPES.values()))
        raise ValueError(
            f"Unknown source type: {x!r}. Valid options: {valid}"
        )
    return SOURCE_TYPES[key]


def normalize_fuel_type(x: str) -> str:
    """Normalize user input to canonical fuel type name."""
    key = (x or "").strip().lower()
    if key not in FUEL_TYPES:
        valid = sorted(set(FUEL_TYPES.values()))
        raise ValueError(
            f"Unknown fuel type: {x!r}. Valid options: {valid}"
        )
    return FUEL_TYPES[key]


def validate_combo(model_year: int, source_type: str, fuel_type: str) -> None:
    """Validate that the requested combination is supported."""
    if model_year not in VALID_MODEL_YEARS:
        raise ValueError(
            f"model_year={model_year} not in VALID_MODEL_YEARS {VALID_MODEL_YEARS}"
        )
    if (source_type, fuel_type) in INVALID_COMBOS:
        raise ValueError(
            f"Unsupported combination: {source_type} × {fuel_type}"
        )
