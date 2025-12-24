# NeuralMOVES

**Lightweight microscopic surrogate of EPA MOVES for vehicle CO₂ emissions**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

NeuralMOVES is an open-source Python package that provides fast, accurate vehicle emission calculations using neural network surrogates trained on EPA's Motor Vehicle Emission Simulator (MOVES). It offers real-time microscopic emission modeling with minimal computational overhead.

## 🌟 Key Features

- **⚡ Fast**: 99.98% compression rate - condensing dozens of gigabytes into a 2.4 MB package
- **🎯 Accurate**: 6% mean average percentage error compared to MOVES
- **🔬 Microscopic**: Second-by-second emission predictions
- **🚗 Comprehensive**: Supports multiple vehicle types, model years, and fuel types
- **📦 Lightweight**: Simple pip installation with minimal dependencies
- **🧪 Well-tested**: Validated against EPA MOVES output

## 📊 Supported Configurations

- **Vehicle Types**: Motorcycles, Passenger Cars, Passenger Trucks, Light Commercial Trucks, Transit Buses
- **Fuel Types**: Gasoline, Diesel
- **Model Years**: 2009-2019 (maps to vehicle age cohorts)
- **Environmental Conditions**: Temperature and humidity variations
- **Road Conditions**: Variable speed, acceleration, and grade

## 📦 Installation

### From GitHub (Recommended)

```bash
pip install "git+https://github.com/edgar-rs/neuralMOVES.git@main"
```

### For Development

```bash
git clone https://github.com/edgar-rs/neuralMOVES.git
cd neuralMOVES
pip install -e .
```

## 🚀 Quick Start

### Single-Second Emission Estimate

```python
import neuralmoves

# Estimate CO₂ for a passenger car at 15 m/s (≈34 mph)
co2_gps = neuralmoves.estimate_running_co2(
    v_ms=15.0,           # Speed: 15 m/s
    a_mps2=0.5,          # Acceleration: 0.5 m/s²
    grade_pct=0.0,       # Road grade: 0% (flat)
    temp=25.0,           # Temperature: 25°C
    humid_pct=50.0,      # Humidity: 50%
    model_year=2015,     # Vehicle model year
    source_type='Passenger Car',
    fuel_type='Gasoline'
)

print(f"CO₂ emission rate: {co2_gps:.3f} g/s")
```

### Driving Cycle Analysis

```python
import neuralmoves
import numpy as np

# Define a simple driving cycle (10 seconds)
time = np.arange(10)  # seconds
speed = np.array([0, 5, 10, 15, 20, 20, 15, 10, 5, 0])  # m/s
accel = np.diff(speed, prepend=0)  # m/s²
grade = np.zeros_like(speed)  # flat road

# Calculate emissions for the entire cycle
emissions = neuralmoves.estimate_emissions_timeseries(
    speed_ms=speed,
    accel_mps2=accel,
    grade_pct=grade,
    temp=20.0,
    humid_pct=60.0,
    model_year=2018,
    source_type='Passenger Car',
    fuel_type='Gasoline'
)

# Analyze results
total_co2_g = emissions.sum()
total_co2_kg = total_co2_g / 1000
distance_m = (speed * 1).sum()  # 1 second intervals
distance_km = distance_m / 1000

print(f"Total CO₂: {total_co2_kg:.3f} kg")
print(f"Distance: {distance_km:.3f} km")
print(f"Average rate: {total_co2_g/10:.3f} g/s")
if distance_km > 0:
    print(f"Emission intensity: {total_co2_kg/distance_km:.1f} kg/km")
```

### Using Pandas DataFrames

```python
import neuralmoves
import pandas as pd

# Load or create driving cycle data
df = pd.DataFrame({
    'time_s': [0, 1, 2, 3, 4],
    'speed_ms': [0, 5, 10, 15, 20],
    'accel_mps2': [0, 5, 5, 5, 5],
    'grade_pct': [0, 0, 1, 2, 1]
})

# Calculate emissions
df['co2_gps'] = neuralmoves.estimate_emissions_timeseries(
    speed_ms=df['speed_ms'].values,
    accel_mps2=df['accel_mps2'].values,
    grade_pct=df['grade_pct'].values,
    temp=25.0,
    humid_pct=50.0,
    model_year=2016,
    source_type='Passenger Car',
    fuel_type='Gasoline'
)

print(df)
```

### Temperature Units

```python
import neuralmoves

# Using Fahrenheit
co2_F = neuralmoves.estimate_running_co2(
    v_ms=20.0,
    a_mps2=0.0,
    grade_pct=0.0,
    temp=77.0,           # 77°F
    temp_unit='F',       # Specify Fahrenheit
    humid_pct=50.0,
    model_year=2015,
    source_type='Passenger Car',
    fuel_type='Gasoline'
)

# Using Celsius (default)
co2_C = neuralmoves.estimate_running_co2(
    v_ms=20.0,
    a_mps2=0.0,
    grade_pct=0.0,
    temp=25.0,           # 25°C
    temp_unit='C',       # Optional, 'C' is default
    humid_pct=50.0,
    model_year=2015,
    source_type='Passenger Car',
    fuel_type='Gasoline'
)

assert abs(co2_F - co2_C) < 0.001  # Should be nearly identical
```

## 📚 API Reference

### Main Functions

#### `estimate_running_co2()`

Estimate per-second CO₂ emission rate for instantaneous conditions.

**Parameters:**
- `v_ms` (float): Vehicle speed [m/s]
- `a_mps2` (float): Acceleration [m/s²]
- `grade_pct` (float): Road grade [%] (100 × rise/run)
- `temp` (float): Temperature (unit set by `temp_unit`)
- `humid_pct` (float): Relative humidity [%]
- `temp_unit` (str): 'C' (default) or 'F'
- `model_year` (int): Vehicle model year (2009-2019)
- `source_type` (str): Vehicle category
- `fuel_type` (str): 'Gasoline' or 'Diesel'
- `apply_idling_floor` (bool): Apply minimum idling rate (default: True)
- `map_location` (str): Device for inference (default: 'cpu')

**Returns:** float - CO₂ emission rate [g/s]

#### `estimate_emissions_timeseries()`

Batch version for processing entire driving cycles efficiently.

**Parameters:**
- `speed_ms` (np.ndarray): Speed values [m/s]
- `accel_mps2` (np.ndarray): Acceleration values [m/s²]
- `grade_pct` (np.ndarray): Grade values [%]
- `temp` (float): Constant temperature for cycle
- `humid_pct` (float): Constant humidity for cycle
- Other parameters same as `estimate_running_co2()`

**Returns:** np.ndarray - CO₂ emission rates [g/s]

#### `idling_rate()`

Get the idling emission rate for a specific vehicle configuration.

**Parameters:**
- `model_year` (int)
- `source_type` (str)
- `fuel_type` (str)

**Returns:** float - Idling CO₂ rate [g/s]

#### `list_available_models()`

List all available (year, source_type, fuel_type) combinations.

**Returns:** list[tuple[int, str, str]]

### Vehicle Type Options

Use any of these names (case-insensitive):

- **Motorcycles**: 'Motorcycles', 'motorcycle', 'mc'
- **Passenger Car**: 'Passenger Car', 'passenger car', 'pc'
- **Passenger Truck**: 'Passenger Truck', 'passenger truck', 'pt'
- **Light Commercial Truck**: 'Light Commercial Truck', 'light commercial truck', 'lct'
- **Transit Bus**: 'Transit Bus', 'transit bus', 'bus'

### Fuel Type Options

- **Gasoline**: 'Gasoline', 'gasoline', 'gas'
- **Diesel**: 'Diesel', 'diesel'

⚠️ **Note**: Diesel motorcycles are not supported (not available in EPA MOVES).

## 🔬 Model Details

### Neural Network Architecture

- **Input features** (5): Temperature (°C), Humidity (%), Speed (m/s), Acceleration (m/s²), Grade (%)
- **Hidden layers**: Two layers with 64 neurons each
- **Activation**: Hyperbolic tangent (tanh)
- **Output**: Single value - running-exhaust CO₂ rate [g/s]

### Idling Floor

The model applies a minimum emission rate based on vehicle idling:

```
final_emission = max(predicted_running, idling_baseline)
```

This ensures emissions never drop below the baseline for a stationary vehicle with engine running.

## 🧪 Validation

NeuralMOVES has been validated against EPA MOVES with the following performance:

- **Mean Average Percentage Error (MAPE)**: ~6%
- **Coverage**: Validated across diverse driving cycles, vehicle types, and environmental conditions
- **Compression**: 99.98% size reduction while maintaining accuracy

For detailed validation results, see the [research paper](https://github.com/edgar-rs/neuralMOVES).

## 💡 Use Cases

- **Transportation planning**: Assess emission impacts of traffic management strategies
- **Research**: Analyze vehicle emissions at microscopic level
- **Traffic simulation**: Integrate with SUMO, VISSIM, or custom simulators
- **Real-time monitoring**: Estimate emissions from GPS trajectory data
- **Policy analysis**: Evaluate fleet composition and technology scenarios

## 🛠️ Troubleshooting

### Import Errors

```python
# ❌ Don't do this
from neuralmoves import NeuralMOVES  # No class needed!

# ✅ Do this
import neuralmoves
co2 = neuralmoves.estimate_running_co2(...)
```

### Model Not Found Errors

```python
# Check available models
models = neuralmoves.list_available_models()
print(f"Available: {len(models)} models")
print(models[:5])  # Show first 5

# Verify specific combination
try:
    idle = neuralmoves.idling_rate(2015, 'Passenger Car', 'Gasoline')
    print(f"Model exists! Idling rate: {idle:.4f} g/s")
except (KeyError, FileNotFoundError) as e:
    print(f"Model not available: {e}")
```

### Unsupported Combinations

```python
import neuralmoves

# ❌ This will raise ValueError (diesel motorcycles not supported)
try:
    co2 = neuralmoves.estimate_running_co2(
        v_ms=10.0, a_mps2=0.0, grade_pct=0.0,
        temp=25.0, humid_pct=50.0,
        model_year=2015,
        source_type='Motorcycles',
        fuel_type='Diesel'  # Not supported!
    )
except ValueError as e:
    print(f"Error: {e}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Edgar Ramírez Sánchez**  
Email: edgarrs@mit.edu  
GitHub: [@edgar-rs](https://github.com/edgar-rs)

## 🙏 Acknowledgments

- EPA for developing the MOVES model
- MIT research community
- All contributors to this project

## 📖 Citation

If you use NeuralMOVES in your research, please cite:

```bibtex
@software{neuralmoves2024,
  author = {Ramírez Sánchez, Edgar},
  title = {NeuralMOVES: Lightweight Microscopic Surrogate of EPA MOVES},
  year = {2024},
  url = {https://github.com/edgar-rs/neuralMOVES}
}
```

## 🔮 Future Development

The NeuralMOVES team is currently working on:

- Direct integration with popular traffic simulators (SUMO, VISSIM)
- Extended support for additional pollutants (NOx, PM, CO)
- Newer vehicle model years and electric vehicles
- Web API for cloud-based emission calculations
- Enhanced validation datasets and error metrics

---

**Made with ❤️ for sustainable transportation research**
