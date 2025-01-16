# NeuralMOVES

-----

## Table of Contents

- [Overview](#overview)
- [Key Features](#key_features)
- [Installation](#installation)
- [Usage](#usage)
- [Future develpments](#future)
- [License](#license)
- [Contact](#contact)


## Overview

NeuralMOVES is an open-source Python package that provides surrogate models for diverse vehicle emission calculations. It offers a fast, accurate, and lightweight alternative to the EPA's Motor Vehicle Emission Simulator (MOVES).


## Key Features

- Microscopic emission modeling
- Diverse scenario parameters (vehicle types, ages, fuel types, regions, road grade, etc.)
- Real-time computation
- High accuracy compared to MOVES (6% mean average percentage error)
- Lightweight (99.98% compression rate, condensing dozens of gigabytes into a 2.4 MB representation)

## Installation

NeuralMOVES can be installed using pip:

```console
python -m pip install git+https://github.mit.edu/edgarrs/neuralmoves.git@main
```

## Usage

Here's a basic example of how to use NeuralMOVES:
```console
import neuralmoves
import pandas as pd

# Example driving cycle data
df_speed_acceleration_grade = pd.DataFrame({
    'speed': [0, 5, 10, 15, 20],
    'acceleration': [0, 1, 0, -1, 0],
    'grade': [0, 0, 1, 1, 0]
})

# Calculate emissions
emissions = neuralmoves.get_emissions(
    source_type=21,  # Passenger Car
    model_year=2020,
    fuel_type=1,     # Gasoline
    temperature=20,  # Celsius
    humidity=50,     # Percent
    df_speed_acceleration_grade=df_speed_acceleration_grade
)

print(emissions)
```
## Future developments 
The NeuralMOVES team is currently working towards incorporating NeuralMOVES into some of the most used traffic simulators like SUMO to further increase the accesibility and ease of use of MOVES. 


## License

`neuralmoves` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Contact

For questions and support, please contact Edgar Ramirez-Sanchez at edgarrs@mit.edu
