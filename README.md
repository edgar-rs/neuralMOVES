[![Paper](https://img.shields.io/badge/paper-TR--C%202026-blue)](https://www.sciencedirect.com/science/article/pii/S0968090X26000185) 

# NeuralMOVES

-----

## Table of Contents

- [Overview](#overview)
- [Paper and Citation](#paper-and-citation)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Fleet-Level Analysis (v0.4.0)](#fleet-level-analysis-v040)
- [Architecture](#architecture)
- [Future Developments](#future-developments)
- [License](#license)
- [Contact](#contact)


## Overview

NeuralMOVES is an open-source Python package that provides surrogate models for diverse vehicle emission calculations. It offers a fast, accurate, and lightweight alternative to the EPA's Motor Vehicle Emission Simulator (MOVES).

## Paper and Citation
NeuralMOVES is an academic effort. If you use NeuralMOVES, please cite the paper: 

**Paper:** NeuralMOVES — Transportation Research Part C (2026): https://authors.elsevier.com/a/1mXcD,M0mRcnKv

**Citation:** (BibTeX: [`CITATION.bib`](./CITATION.bib)) **OR** (Use GitHub’s **Cite this repository** button to export a citation.)

## Key Features

- **Microscopic emission modeling** - Second-by-second CO₂ estimation for individual vehicles
- **Fleet-level analysis** - Fleet-average emission rates using MOVES default fleet compositions
- **Policy scenario modeling** - EV adoption ramps, scenario comparison, and sensitivity analysis
- **3,200+ US counties** - Bundled county-level meteorology for location-specific estimates
- **Diverse scenario parameters** - Vehicle types, ages, fuel types, temperatures, humidity, road grades
- **Real-time computation** - Millisecond-scale evaluation suitable for optimization and control
- **High accuracy** - 6% mean absolute percentage error compared to EPA MOVES
- **Lightweight** - No MOVES installation or database required
- **Transparent error reporting** - Built-in expected error statistics for academic credibility

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

## Fleet-Level Analysis (v0.4.0)

NeuralMOVES v0.4.0 adds six aggregation layers that scale per-second estimates to fleet averages, annual totals, and policy comparisons — using the same default data as EPA MOVES but without requiring a MOVES installation or database.

For the full tutorial with visualizations, see the [fleet analysis guide](examples/fleet_analysis_guide.ipynb). For a detailed explanation of how the layers work, see the [architecture documentation](docs/architecture.md).

### Drive Cycle Evaluation

Evaluate total CO₂ over a standard drive cycle instead of one second at a time:

```python
from neuralmoves import DriveCycle, evaluate_cycle

cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=600)
result = evaluate_cycle(
    cycle, temp=25, humid_pct=50,
    model_year=2019, source_type='Passenger Car', fuel_type='Gasoline',
)
print(f"{result.rate_g_per_mile:.1f} g/mi over {result.distance_miles:.1f} miles")
```

### Fleet-Average Emission Rates

Compute the fleet-weighted average across all vehicle types, model years, and fuel types for a given calendar year:

```python
from neuralmoves import DriveCycle, FleetComposition, fleet_average_rate

cycle = DriveCycle.constant_speed(speed_mph=30, duration_s=300)
fleet = FleetComposition.from_moves_defaults(calendar_year=2019)
result = fleet_average_rate(cycle, fleet, temp=25, humid_pct=50)
print(f"Fleet average: {result.fleet_avg_g_per_mile:.1f} g/mi ({fleet.n_vehicle_configs} configs)")
```

### Annual Emissions from VMT

Scale per-mile rates to annual totals using MOVES VMT data:

```python
from neuralmoves import VMTAllocation, annualize_emissions

vmt = VMTAllocation.from_moves_defaults(source_type_id=21, calendar_year=2019)
annual = annualize_emissions(rate_g_per_mile=300.0, vmt=vmt)
print(f"Annual CO₂: {annual.annual_co2_metric_tons:,.0f} metric tons")
```

### EV Policy Scenarios

Model how increasing EV adoption reduces fleet CO₂ over a decade:

```python
from neuralmoves import TechnologyScenario, ScenarioDefinition, compare_scenarios

baseline = ScenarioDefinition(
    name="5% EV constant",
    technology=TechnologyScenario.constant("bau", ev_fraction=0.05),
    years=list(range(2025, 2036)),
)
policy = ScenarioDefinition(
    name="5-30% EV ramp",
    technology=TechnologyScenario.linear_ev_ramp("policy", 2025, 2035, 0.05, 0.30),
    years=list(range(2025, 2036)),
)
results = compare_scenarios([baseline, policy])
print(results[["year", "5% EV constant_co2_MT", "5-30% EV ramp_co2_MT"]])
```

### County-Level Analysis

Use bundled meteorology for any of 3,200+ US counties:

```python
from neuralmoves import LocationProfile

wayne_county = LocationProfile.from_county_defaults(26163)  # Wayne County, MI
print(f"Jan temp: {wayne_county.avg_temp_by_month[1]:.1f} F")
print(f"Jul temp: {wayne_county.avg_temp_by_month[7]:.1f} F")
```

## Architecture

NeuralMOVES is organized as a layered architecture. Each layer builds on the ones below but can be used independently — enter at whatever level matches your analysis needs:

```
Layer 6: Scenarios    compare_scenarios(), sensitivity_analysis()
Layer 5: Geography    LocationProfile, county meteorology
Layer 4: Technology   TechnologyScenario, EV adoption curves
Layer 3: Temporal     VMTAllocation, annual/monthly totals
Layer 2: Fleet        FleetComposition, fleet-average rates
Layer 1: Cycles       DriveCycle, evaluate_cycle()
---------------------------------------------------------
Core Engine           estimate_running_co2() (per-second NN inference)
```

See [docs/architecture.md](docs/architecture.md) for a detailed description of each layer, how they compose, and the MOVES default data used.

## Future Developments

There are several natural avenues for extending NeuralMOVES (community contributions welcome), including:

- **Traffic simulator integration** - Native support for SUMO, Vissim, and other major platforms
- **Additional pollutants** - Extending beyond CO₂ to NOx, PM, and other emissions
- **Expanded vehicle coverage** - CNG, hybrid, and emerging vehicle technologies
- **Standard drive cycles** - Bundle official EPA cycles (FTP-75, HWFET, US06)
- **Real-time data pipelines** - Integration with GPS/AVL data sources

For feature requests or collaboration inquiries, please open an issue on GitHub or contact the team.


## License

`neuralmoves` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Contact

For questions and support, please contact Edgar Ramirez-Sanchez at edgarrs@mit.edu
