# NeuralMOVES

-----

## Table of Contents

- [Overview](#overview)
- [Publications and impact](#publications_and_impact)
- [Key Features](#key_features)
- [Installation](#installation)
- [Usage](#usage)
- [Future develpments](#future)
- [License](#license)
- [Contact](#contact)


## Overview

NeuralMOVES is an open-source Python package that provides surrogate models for diverse vehicle emission calculations. It offers a fast, accurate, and lightweight alternative to the EPA's Motor Vehicle Emission Simulator (MOVES).

## Publications and impact

The development of Neuralmoves has been documented and shared at: 

- Finished project presented at the Conference in Emerging Technologies in Transportation Systems (TRC-30)
September 02-03, 2024, Crete, Greece: https://easychair.org/smart-program/TRC-30/2024-09-04.html#talk:261921
Methodology and results are summarized in the TRC-30 extended abstract available as a pdf file in the github repository. 
A full journal paper on the development of NeuralMOVES will be linked here soon. 
- (Previous) Early develpments presented at the Advances in Neural Information Processing Systems (NeurIPS) Conference, 2022. Workshop on Tackling Climate Change with Machine Learning. Paper, slides, and recorded talk available at: https://www.climatechange.ai/papers/neurips2022/90

*Notable applications*: 
NeuralMOVES has been a key enabler in the large-scale impact assessment in eco-driving for signalized intersections study, which shows that intersection emissions can be cut by 11-22%, which could lead to a reduction in US carbon emissions by up to around 123 million tonnes. Featured in New Scientist: https://www.newscientist.com/article/2445202-a-simple-driving-trick-could-make-a-big-dent-in-cars-carbon-emissions/ 
Full project information and publication: Mitigating Metropolitan Carbon Emissions with Semi-autonomous Vehicles using Deep Reinforcement Learning. https://vindulamj.github.io/eco-drive/

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
