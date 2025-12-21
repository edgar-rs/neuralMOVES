from __future__ import annotations

# Canonical tokens used in filenames and tables
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

# 11 cohorts (example: 2009–2019)
VALID_MODEL_YEARS = tuple(range(2009, 2020))

# Combinations not present in MOVES or not trained (e.g., diesel motorcycles)
INVALID_COMBOS = {("Motorcycles", "Diesel")}


def normalize_source_type(x: str) -> str:
    key = (x or "").strip().lower()
    if key not in SOURCE_TYPES:
        raise ValueError(f"Unknown source type: {x!r}. Valid: {sorted(set(SOURCE_TYPES.values()))}")
    return SOURCE_TYPES[key]


def normalize_fuel_type(x: str) -> str:
    key = (x or "").strip().lower()
    if key not in FUEL_TYPES:
        raise ValueError(f"Unknown fuel type: {x!r}. Valid: {sorted(set(FUEL_TYPES.values()))}")
    return FUEL_TYPES[key]


def validate_combo(model_year: int, source_type: str, fuel_type: str) -> None:
    if model_year not in VALID_MODEL_YEARS:
        raise ValueError(f"model_year={model_year} not in VALID_MODEL_YEARS {VALID_MODEL_YEARS}")
    if (source_type, fuel_type) in INVALID_COMBOS:
        raise ValueError(f"Unsupported combination: {source_type} × {fuel_type}")
