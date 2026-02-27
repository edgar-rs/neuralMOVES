# NeuralMOVES Architecture

This document describes the internal architecture of NeuralMOVES, with a focus on how the aggregation layers build on the core per-second engine to support fleet-level and policy analysis.

## Overview

NeuralMOVES operates at two levels:

1. **Microscopic mode** — Per-second, per-vehicle CO₂ estimation using 99 lightweight neural network surrogates of EPA MOVES. This is the core engine described in the [paper](https://www.sciencedirect.com/science/article/pii/S0968090X26000185). It takes instantaneous driving conditions (speed, acceleration, grade, temperature, humidity) and a vehicle configuration (source type, model year, fuel type) and returns grams of CO₂ per second.

2. **Fleet/policy mode** — Six aggregation layers that scale microscopic results up to fleet averages, annual inventories, and multi-year scenario comparisons. These layers use the same default data as EPA MOVES (bundled as CSV files, no MOVES installation needed).

Users can enter at any level depending on their analysis needs. A traffic control researcher might only use the core engine, while a policy analyst might jump straight to scenario comparison.

---

## The Core Engine

NeuralMOVES contains **99 neural network submodels**, one for each combination of:

| Parameter | Values | Count |
|-----------|--------|-------|
| Source type | Motorcycle (11), Passenger Car (21), Passenger Truck (31), Light Commercial Truck (32), Transit Bus (42) | 5 |
| Model year | 2009, 2010, ..., 2019 | 11 |
| Fuel type | Gasoline, Diesel | 2 |

Minus one invalid combination (Motorcycle + Diesel) = **99 models**.

Each model is a 2-layer neural network with 5 hidden units and tanh activation. Total package size: 2.4 MB.

**Inputs** (per second):
- Speed (m/s)
- Acceleration (m/s²)
- Road grade (%)
- Temperature (°C or °F)
- Relative humidity (%)

**Output**: CO₂ emission rate (g/s)

**Accuracy**: 6.013% mean absolute percentage error (MAPE) compared to EPA MOVES across all operating conditions.

### Key functions

```python
# Single-second estimate
neuralmoves.estimate_running_co2(v_ms, a_mps2, grade_pct, temp, humid_pct, ...)

# Full time-series (vectorized, much faster than looping)
neuralmoves.estimate_emissions_timeseries(speed_ms, accel_mps2, grade_pct, temp, humid_pct, ...)
```

---

## Aggregation Layer Architecture

The six aggregation layers form a stack. Each layer builds on the ones below but can be used independently:

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Layer 6: Scenarios                                                 │
 │  compare_scenarios(), ev_impact_analysis(), sensitivity_analysis()  │
 │  Compare baseline vs. policy scenarios over multi-year horizons     │
 ├──────────────────────────────────────────────────────────────────────┤
 │  Layer 5: Geography                                                 │
 │  LocationProfile, location_inventory()                              │
 │  County-specific meteorology and VMT for 3,200+ US counties        │
 ├──────────────────────────────────────────────────────────────────────┤
 │  Layer 4: Technology                                                │
 │  TechnologyScenario (linear ramp, S-curve, constant)                │
 │  Model EV adoption trajectories and fleet transitions over time     │
 ├──────────────────────────────────────────────────────────────────────┤
 │  Layer 3: Temporal                                                  │
 │  VMTAllocation, annualize_emissions()                               │
 │  Scale per-mile rates to annual/monthly totals using MOVES VMT data │
 ├──────────────────────────────────────────────────────────────────────┤
 │  Layer 2: Fleet                                                     │
 │  FleetComposition, fleet_average_rate()                             │
 │  Weighted average across vehicle types, model years, and fuels      │
 ├──────────────────────────────────────────────────────────────────────┤
 │  Layer 1: Drive Cycles                                              │
 │  DriveCycle, evaluate_cycle()                                       │
 │  Summarize per-second emissions over a complete speed trace         │
 ├──────────────────────────────────────────────────────────────────────┤
 │  Core Engine                                                        │
 │  estimate_running_co2(), estimate_emissions_timeseries()            │
 │  Per-second NN inference (99 submodels)                             │
 └──────────────────────────────────────────────────────────────────────┘
```

### How to choose your entry point

| You need... | Enter at | Key function |
|---|---|---|
| Per-second CO₂ for one vehicle | Core Engine | `estimate_running_co2()` |
| Total CO₂ over a drive cycle | Layer 1 | `evaluate_cycle()` |
| Fleet-average g/mile for a calendar year | Layer 2 | `fleet_average_rate()` |
| Annual CO₂ in metric tons | Layer 3 | `annualize_emissions()` |
| Year-by-year fleet under EV adoption | Layer 4 | `TechnologyScenario.get_fleet()` |
| County-specific monthly inventory | Layer 5 | `location_inventory()` |
| Baseline vs. policy comparison | Layer 6 | `compare_scenarios()` |

---

## Layer Details

### Layer 1: Drive Cycles (`cycles.py`)

A `DriveCycle` is a speed trace (speed in m/s at each second) with optional road grade. The module wraps `estimate_emissions_timeseries()` and computes summary metrics.

**What it does**: Takes a speed profile and a single vehicle config, runs the NN for every second, and returns total CO₂ (g), emission rate (g/mile, g/km), and the full per-second timeseries.

**Factory methods** for creating cycles:
- `DriveCycle.constant_speed(speed_mph, duration_s)` — Constant speed with ramp up/down
- `DriveCycle.from_speed_array(speed_mph, dt)` — From a speed array (e.g., from a traffic simulator)
- `DriveCycle.from_csv(path)` — From a CSV file with `time_s` and `speed_mph` columns

### Layer 2: Fleet Composition (`fleet.py`)

A `FleetComposition` represents the distribution of vehicle types, model years, and fuel types on the road in a given calendar year.

**What it does**: Builds the fleet mix from three MOVES default tables, then evaluates the drive cycle for every vehicle configuration in the fleet and computes a weighted average emission rate.

**How the fleet is built** (mirroring MOVES):
1. `sourceTypeAgeDistribution` — What fraction of each source type is age 0, 1, 2, ...? (age → model year)
2. `sampleVehiclePopulation` — For each (source type, model year), what fraction is gasoline vs. diesel vs. electric?
3. `sourceTypePopulation` — How many passenger cars vs. trucks vs. buses are on the road?

These three are cross-multiplied to produce a fleet fraction for each (source type, model year, fuel type) combination. The result is typically ~99 unique vehicle configurations, each evaluated once per cycle.

**EV handling**: EVs contribute zero tailpipe CO₂. The `ev_penetration` parameter scales down ICE fractions so that ICE share + EV share = 100%.

### Layer 3: Temporal Aggregation (`temporal.py`)

`VMTAllocation` represents how many miles vehicles travel per year, broken down by month.

**What it does**: Multiplies a per-mile emission rate by annual VMT to get annual CO₂ in grams (or metric tons). Monthly VMT fractions from MOVES allow breaking this down by month.

**Data source**: Annual VMT comes from the HPMS (Highway Performance Monitoring System) data in the MOVES database. Monthly fractions come from the `monthVMTFraction` table.

**Temperature-varying mode**: `annualize_with_temperature()` evaluates the drive cycle at each month's average temperature separately, capturing seasonal effects (cold weather increases CO₂).

### Layer 4: Technology Scenarios (`technology.py`)

`TechnologyScenario` defines how EV adoption changes over time.

**What it does**: Maps calendar years to EV penetration fractions. For any year, `get_fleet(year)` returns a `FleetComposition` with the appropriate EV share.

**Adoption curves**:
- `constant(ev_fraction)` — Fixed EV share (e.g., 5% in all years)
- `linear_ev_ramp(start_year, end_year, start_pct, end_pct)` — Linear growth
- `s_curve_ev_adoption(start_year, end_year, start_pct, end_pct, steepness)` — Logistic S-curve (more realistic for technology adoption)

### Layer 5: Geography (`geography.py`)

`LocationProfile` stores county-specific meteorology (monthly temperature and humidity) and VMT.

**What it does**: Enables location-specific emission inventories by evaluating cycles at local temperature/humidity conditions for each month.

**Data**: Monthly average temperature and humidity for 3,232 US counties, from the MOVES `zoneMonthHour` table.

**Key functions**:
- `LocationProfile.from_county_defaults(fips_code)` — Load a county by FIPS code
- `location_inventory(location, fleet, cycle)` — Monthly emission breakdown for one county
- `multi_location_inventory(locations, fleet, cycle)` — Compare across counties

### Layer 6: Scenario Comparison (`scenarios.py`)

`ScenarioDefinition` bundles a technology trajectory, drive cycle, VMT, and optional location into a complete analysis scenario.

**What it does**: Evaluates one or more scenarios over a range of years and produces comparison tables.

**Key functions**:
- `evaluate_scenario(scenario)` — Year-by-year CO₂ for one scenario
- `compare_scenarios([baseline, policy])` — Side-by-side comparison table
- `ev_impact_analysis(ev_fractions)` — CO₂ reduction curve across EV penetration levels
- `sensitivity_analysis(base, parameter, values)` — Vary one parameter (EV fraction, VMT, temperature)

---

## MOVES Default Data

NeuralMOVES bundles 9 CSV tables extracted from the official EPA MOVES database (`movesdb20241112`). These are stored in `src/neuralmoves/defaults/` and loaded automatically — no MOVES installation or MariaDB database is needed.

| Table | Rows | Description |
|---|---|---|
| `age_distribution.csv` | 33,579 | Fleet age distribution by source type and calendar year |
| `sample_vehicle_population.csv` | 16,428 | Fuel type fractions by source type and model year |
| `source_type_population.csv` | 819 | Vehicle population counts by source type and year |
| `month_vmt_fraction.csv` | 156 | Monthly VMT distribution (sums to 1.0 per source type) |
| `day_vmt_fraction.csv` | 1,248 | Daily VMT fractions by month and road type |
| `hour_vmt_fraction.csv` | 2,496 | Hourly VMT fractions by day type |
| `hpms_vtype_year.csv` | 315 | National VMT totals from HPMS (1990-2060) |
| `county_meteorology.csv` | 38,784 | Monthly avg temperature/humidity for 3,232 counties |
| `avg_speed_distribution.csv` | 39,936 | Speed bin distributions by source type and road type |

**Extraction**: These CSVs were extracted from the EPA MOVES SQL dump using `scripts/extract_moves_defaults.py`, which parses the SQL INSERT statements directly (no database setup required). The extraction script is included for reproducibility but is not needed for normal use.

---

## Key Design Decisions

### Model year clamping
NeuralMOVES has NN models for model years 2009-2019 only. For fleet vehicles with model year < 2009 (older cars still on the road), the 2009 model is used. This is reasonable for CO₂ because MOVES does not apply age-based emission degradation to CO₂ — unlike criteria pollutants (THC, CO, NOx), CO₂ is tied directly to fuel consumption via carbon balance and does not change with vehicle age.

### EV modeling
Electric vehicles are modeled as a fleet fraction parameter, not as a distinct vehicle type with its own NN model. The `ev_penetration` parameter (0.0 to 1.0) uniformly reduces ICE vehicle fractions. EVs contribute 0 tailpipe CO₂. This is a simplification — it does not model upstream electricity generation emissions.

### Fuel type mapping
MOVES defines several fuel types. NeuralMOVES maps them as follows:
- Fuel 1 (Gasoline) → Gasoline NN model
- Fuel 2 (Diesel) → Diesel NN model
- Fuel 5 (E85/Ethanol) → Gasoline NN model (approximation — similar CO₂ characteristics)
- Fuel 9 (Electricity) → Zero tailpipe CO₂
- Fuel 3 (CNG) → Skipped (tiny fleet share, no NN model)

### Fleet composition data
The MOVES `avft` (Alternative Vehicle Fuels and Technologies) table is empty in the default database (it is user-populated). NeuralMOVES uses the `sampleVehiclePopulation` table instead, which provides pre-computed fuel type fractions (`stmyFraction`) that sum to 1.0 per (source type, model year).

### Performance
Fleet evaluation (all ~99 vehicle configs over a 300-second cycle) completes in roughly 0.2 seconds on CPU. The bottleneck is NN inference, which is vectorized per-vehicle using PyTorch.

---

## Module Map

```
src/neuralmoves/
    __init__.py              # Public API (all layers)
    api.py                   # Core engine: per-second CO₂
    config.py                # Constants: source types, fuel types, model years
    model.py                 # Net class (NN architecture)
    loaders.py               # Model weight loading
    cycles.py                # Layer 1: Drive cycle evaluation
    fleet.py                 # Layer 2: Fleet composition and averaging
    temporal.py              # Layer 3: Time aggregation (VMT)
    technology.py            # Layer 4: Technology scenarios (EV ramps)
    geography.py             # Layer 5: Location-specific inventories
    scenarios.py             # Layer 6: Scenario comparison
    defaults/                # Bundled MOVES default data (CSVs)
    NN_3/                    # 99 NN model weight files
```
