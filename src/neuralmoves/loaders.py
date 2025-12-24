"""Resource loaders for NeuralMOVES model weights and data files."""

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
    SOURCE_TYPE_ID_MAP,
    FUEL_TYPE_ID_MAP,
)
from .model import Net


@dataclass(frozen=True)
class SubmodelKey:
    """Identifier for a specific (year, source, fuel) submodel."""
    
    model_year: int
    source_type: str  # canonical name (e.g., "Passenger Car")
    fuel_type: str    # canonical name (e.g., "Gasoline")

    @classmethod
    def from_user(cls, model_year: int, source_type: str, fuel_type: str) -> SubmodelKey:
        """Create key from user-friendly inputs with validation."""
        st = normalize_source_type(source_type)
        ft = normalize_fuel_type(fuel_type)
        validate_combo(model_year, st, ft)
        return cls(model_year, st, ft)

    @property
    def source_type_id(self) -> int:
        """Get MOVES numeric ID for source type."""
        return SOURCE_TYPE_TO_ID[self.source_type]
    
    @property
    def fuel_type_id(self) -> int:
        """Get MOVES numeric ID for fuel type."""
        return FUEL_TYPE_TO_ID[self.fuel_type]

    @property
    def filename(self) -> str:
        """
        Generate model filename matching the NN_3/ naming convention.
        Format: NN_model_{year}_{sourceID}_{fuelID}.pt
        """
        return f"NN_model_{self.model_year}_{self.source_type_id}_{self.fuel_type_id}.pt"


def _resource_path(rel: str):
    """
    Return a resource reference for a file inside the neuralmoves package.
    Uses importlib.resources for proper installed-package support.
    """
    return files(__package__) / rel


def load_submodel(
    key: SubmodelKey,
    map_location: str | torch.device = "cpu"
) -> Net:
    """
    Load a pre-trained neural network model for the specified submodel.
    
    Parameters
    ----------
    key : SubmodelKey
        Identifier for the model to load
    map_location : str or torch.device
        Device to load the model onto (default: "cpu")
        
    Returns
    -------
    Net
        Loaded model in evaluation mode
        
    Raises
    ------
    FileNotFoundError
        If the model weights file doesn't exist
    """
    rel_path = Path("NN_3") / key.filename
    res = _resource_path(str(rel_path))
    
    # Use as_file to handle both filesystem and zip-based installations
    with as_file(res) as model_path:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model weights not found: {key.filename}\n"
                f"Expected at: {rel_path}\n"
                f"Combination: year={key.model_year}, source={key.source_type}, fuel={key.fuel_type}"
            )
        
        # Instantiate model with correct architecture
        model = Net(input_dim=5, hidden_dim=64, output_dim=1)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model


def load_idling_table() -> list[dict]:
    """
    Load the idling emissions lookup table from the packaged CSV.
    
    Returns
    -------
    list[dict]
        List of records with keys: model_year, source_type, fuel_type, idling_gps
        
    Notes
    -----
    The CSV uses numeric IDs (sourceTypeID, fuelTypeID) which are converted
    to canonical string names for consistency with the API.
    """
    res = _resource_path("idling_emissions.csv")
    
    with as_file(res) as csv_path:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric IDs to canonical names
                source_id = int(row["sourceTypeID"])
                fuel_id = int(row["fuelTypeID"])
                
                if source_id not in SOURCE_TYPE_ID_MAP:
                    continue  # Skip unknown source types
                if fuel_id not in FUEL_TYPE_ID_MAP:
                    continue  # Skip unknown fuel types
                
                rows.append({
                    "model_year": int(row["modelYear"]),
                    "source_type": SOURCE_TYPE_ID_MAP[source_id],
                    "fuel_type": FUEL_TYPE_ID_MAP[fuel_id],
                    "idling_gps": float(row["emission_per_second_MOVES"]),
                })
        
        return rows


def lookup_idling_gps(key: SubmodelKey) -> float:
    """
    Look up the idling emission rate for a specific submodel.
    
    Parameters
    ----------
    key : SubmodelKey
        Identifier for the submodel
        
    Returns
    -------
    float
        Idling emission rate in g/s
        
    Raises
    ------
    KeyError
        If no idling value exists for the specified combination
    """
    table = load_idling_table()
    
    for row in table:
        if (
            row["model_year"] == key.model_year
            and row["source_type"] == key.source_type
            and row["fuel_type"] == key.fuel_type
        ):
            return row["idling_gps"]
    
    raise KeyError(
        f"No idling value found for:\n"
        f"  year={key.model_year}\n"
        f"  source={key.source_type}\n"
        f"  fuel={key.fuel_type}\n"
        f"Ensure idling_emissions.csv contains this combination."
    )


def load_error_lookup() -> Optional[list[dict]]:
    """
    Optionally load error statistics lookup table.
    
    Returns
    -------
    list[dict] or None
        Error statistics if available, otherwise None
        
    Notes
    -----
    This is an optional file for providing validation error metrics.
    Expected columns: scope, category, subcategory, MAPE, MPE, MdPE, StdPE, MAE_g
    """
    res = _resource_path("error_lookup.csv")
    
    try:
        with as_file(res) as csv_path:
            if not Path(csv_path).exists():
                return None
                
            rows = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            return rows
    except (FileNotFoundError, Exception):
        return None
