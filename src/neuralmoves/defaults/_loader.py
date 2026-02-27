"""Internal CSV loading utilities for MOVES default data."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files, as_file

import pandas as pd


def _defaults_path(filename: str):
    """Return the importlib resource for a file in the defaults/ package."""
    return files(__package__) / filename


@lru_cache(maxsize=None)
def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the defaults package and cache the result."""
    res = _defaults_path(filename)
    with as_file(res) as path:
        return pd.read_csv(path)
