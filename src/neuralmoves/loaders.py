from __future__ import annotations
import csv
from dataclasses import dataclass
from importlib.resources import files, as_file
from pathlib import Path
from typing import Optional

import torch

from .config import (
    normalize_fuel_type,
    normalize_source_type,
    validate_combo,
    SOURCE_TYPE_TO_ID,
    FUEL_TYPE_TO_ID,
)
from .model import Net


@dataclass(frozen=True)
class SubmodelKey:
    model_year: int
    source_type: str  # canonical name
    fuel_type: str    # canonical name

    @classmethod
    def from_user(cls, model_year: int, source_type: str, fuel_type: str) -> "SubmodelKey":
        st = normalize_source_type(source_type)
        ft = normalize_fuel_type(fuel_type)
        validate_combo(model_year, st, ft)
        return cls(model_year, st, ft)

    @property
    def filename(self) -> str:
        # Match your on-disk naming convention with numeric IDs:
        # NN_3/NN_model_{model_year}_{source_id}_{fuel_id}.pt
        source_id = SOURCE_TYPE_TO_ID[self.source_type]
        fuel_id = FUEL_TYPE_TO_ID[self.fuel_type]
        return f"NN_model_{self.model_year}_{source_id}_{fuel_id}.pt"


def _resource_path(rel: str) -> Path:
    """
    Return a filesystem path for a package resource inside neuralmoves/.
    """
    res = files(__package__) / rel
    return res


def load_submodel(key: SubmodelKey, map_location: str | torch.device = "cpu") -> Net:
    """
    Loads and returns a torch.nn.Module for the requested (year, source, fuel) combo.
    """
    rel = Path("NN_3") / key.filename
    res = _resource_path(str(rel))
    if not res.exists():
        # Some environments require a real path; as_file handles zips too.
        with as_file(res) as tmp:
            if not Path(tmp).exists():
                raise FileNotFoundError(f"Model weights not found at packaged path: {rel}")
    # Instantiate architecture and load state dict
    model = Net()
    with as_file(res) as model_path:
        state = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model


def load_idling_table() -> list[dict]:
    """
    Load idling_emissions.csv from the package. Expected columns:
    fuelTypeID, modelYear, sourceTypeID, TH, emission_per_second_MOVES
    
    Note: TH column is ignored; we average over all TH values for a given vehicle spec.
    """
    from .config import SOURCE_TYPE_ID_MAP, FUEL_TYPE_ID_MAP
    
    res = _resource_path("idling_emissions.csv")
    with as_file(res) as csv_path:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # Convert numeric IDs to canonical names
                fuel_id = int(row["fuelTypeID"])
                source_id = int(row["sourceTypeID"])
                
                rows.append({
                    "model_year": int(row["modelYear"]),
                    "source_type": SOURCE_TYPE_ID_MAP.get(source_id, f"Unknown_{source_id}"),
                    "fuel_type": FUEL_TYPE_ID_MAP.get(fuel_id, f"Unknown_{fuel_id}"),
                    "idling_gps": float(row["emission_per_second_MOVES"])
                })
        return rows


def lookup_idling_gps(key: SubmodelKey) -> float:
    table = load_idling_table()
    for row in table:
        if (
            row["model_year"] == key.model_year
            and row["source_type"] == key.source_type
            and row["fuel_type"] == key.fuel_type
        ):
            return row["idling_gps"]
    raise KeyError(
        f"No idling value for (year={key.model_year}, source={key.source_type}, fuel={key.fuel_type}). "
        "Make sure idling_emissions.csv contains this cohort."
    )


def load_error_lookup() -> Optional[list[dict]]:
    """
    Optionally load error_lookup.csv with columns like:
    scope,category,subcategory,MAPE,MPE,MdPE,StdPE,MAE_g
    Returns None if the file isn't packaged yet.
    """
    res = _resource_path("error_lookup.csv")
    try:
        with as_file(res) as csv_path:
            rows = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append(row)
            return rows
    except FileNotFoundError:
        return None
