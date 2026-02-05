[![Paper](https://img.shields.io/badge/paper-TR--C%202026-blue)](https://www.sciencedirect.com/science/article/pii/S0968090X26000185) 

# NeuralMOVES

-----

## Table of Contents

- [Overview](#overview)
- [Paper and Citation](#paper-and-citation)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Future developments](#future-developments)
- [License](#license)
- [Contact](#contact)


## Overview

NeuralMOVES is an open-source Python package that provides surrogate models for diverse vehicle emission calculations. It offers a fast, accurate, and lightweight alternative to the EPA's Motor Vehicle Emission Simulator (MOVES).

## Paper and Citation
NeuralMOVES is an academic effort. If you use NeuralMOVES, please cite the paper: 

**Paper:** NeuralMOVES — Transportation Research Part C (2026): https://authors.elsevier.com/a/1mXcD,M0mRcnKv

**Citation:** (BibTeX: [`CITATION.bib`](./CITATION.bib)) **OR** (Use GitHub’s **Cite this repository** button to export a citation.)

## Key Features

- **Microscopic emission modeling** - Second-by-second CO₂ estimation
- **Diverse scenario parameters** - Vehicle types, ages, fuel types, temperatures, humidity, road grades
- **Real-time computation** - Millisecond-scale evaluation suitable for optimization and control
- **High accuracy** - 6% mean absolute percentage error compared to EPA MOVES
- **Lightweight** - 99.98% compression (2.4 MB vs dozens of GB)
- **Transparent error reporting** - Built-in expected error statistics for academic credibility
- **Production-ready API** - Clean, documented interface with comprehensive examples

## Installation

NeuralMOVES can be installed using pip:

```bash
python -m pip install "git+https://github.com/edgar-rs/neuralMOVES.git@main"
```

## Usage

### Quick Start

Here's a basic example of estimating CO₂ emissions for a single driving condition:

```python
import neuralmoves

# Single-second emission estimation
emission = neuralmoves.estimate_running_co2(
    v_ms=15.0,              # Speed: 15 m/s (≈54 km/h)
    a_mps2=0.5,             # Acceleration: 0.5 m/s²
    grade_pct=0.0,          # Road grade: flat (0%)
    temp=25,                # Temperature: 25°C
    temp_unit='C',          
    humid_pct=50,           # Relative humidity: 50%
    model_year=2019,        # Vehicle model year
    source_type='Passenger Car',
    fuel_type='Gasoline'
)

print(f"CO₂ emission: {emission:.2f} g/s")
```

### Time-Series Analysis

Process a complete driving cycle:

```python
import pandas as pd
import neuralmoves

# Define a driving cycle
driving_cycle = pd.DataFrame({
    'speed_ms': [0, 5, 10, 15, 20, 20, 15, 10, 5, 0],  # m/s
    'acceleration_mps2': [0, 1, 1, 1, 0, 0, -1, -1, -1, 0],  # m/s²
    'grade_pct': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # flat road
})

# Calculate emissions for each second
emissions = []
for _, row in driving_cycle.iterrows():
    em = neuralmoves.estimate_running_co2(
        v_ms=row['speed_ms'],
        a_mps2=row['acceleration_mps2'],
        grade_pct=row['grade_pct'],
        temp=25, temp_unit='C', humid_pct=50,
        model_year=2019,
        source_type='Passenger Car',
        fuel_type='Gasoline'
    )
    emissions.append(em)

driving_cycle['co2_gs'] = emissions
total_co2 = driving_cycle['co2_gs'].sum()
print(f"Total CO₂ over cycle: {total_co2:.2f} g")
```

### Error Reporting and Transparency

NeuralMOVES provides expected error statistics for transparent academic reporting:

```python
import neuralmoves

# Get overall model performance
overall_error = neuralmoves.get_expected_error()
print(f"MAPE: {overall_error['MAPE']}%")  # Mean Absolute Percentage Error
print(f"MPE: {overall_error['MPE']}%")    # Mean Percentage Error

# Get error for specific vehicle configuration
error = neuralmoves.get_expected_error(
    fuel_type='Gasoline',
    source_type='Passenger Car'
)
print(f"Gasoline Passenger Car MAPE: {error['MAPE']}%")

# Report emissions with confidence bounds
emission = neuralmoves.estimate_running_co2(
    v_ms=15.0, a_mps2=0.5, grade_pct=0.0,
    temp=25, temp_unit='C', humid_pct=50,
    model_year=2019, source_type='Passenger Car', fuel_type='Gasoline'
)

mape = float(error['MAPE'])
margin = emission * mape / 100
print(f"Emission: {emission:.2f} ± {margin:.2f} g/s (MAPE: {mape:.1f}%)")
```

### Additional Utilities

```python
import neuralmoves

# Get idling emission rate
idling = neuralmoves.idling_rate(
    model_year=2019,
    source_type='Passenger Car',
    fuel_type='Gasoline'
)
print(f"Idling rate: {idling:.3f} g/s")

# List all available model configurations
available_models = neuralmoves.list_available_models()
print(f"Available models: {len(available_models)}")
```

### Comprehensive Examples

For more detailed examples including:
- Multi-vehicle comparisons
- Eco-driving analysis
- Integration with traffic simulation (SUMO, Vissim, etc.)
- Real-world scenario analysis

See the [comprehensive usage guide](examples/comprehensive_usage_guide.ipynb) in the `examples/` directory.
## Future Developments 

There are several natural avenues for extending NeuralMOVES (community contributions welcome), including:

- **Traffic simulator integration** - Native support for SUMO, Vissim, and other major platforms
- **Additional pollutants** - Extending beyond CO₂ to NOx, PM, and other emissions
- **Expanded vehicle coverage** - CNG, hybrid, and emerging vehicle technologies
- **High-resolution emission fields** - Tools for generating spatiotemporal emission inventories
- **Real-time data pipelines** - Integration with GPS/AVL data sources

For feature requests or collaboration inquiries, please open an issue on GitHub or contact the team. 


## License

`neuralmoves` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Contact

For questions and support, please contact Edgar Ramirez-Sanchez at edgarrs@mit.edu
