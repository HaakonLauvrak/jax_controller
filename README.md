# Control System Simulation

This project is a simulation of a control system. It includes different types of plants and controllers, and allows for the tuning of controller parameters to optimize the performance of the system.

## Key Components

- `Plants`: These are the systems being controlled. The project includes simulations of a bathtub, a Cournot duopoly, and a chemical reaction. More details can be found in the [plants documentation](plants/README.md).

- `Controllers`: These are the algorithms that adjust the inputs to the plant to achieve a desired output. The project includes a classic PID controller and an artificial neural network (ANN) controller. More details can be found in the [controllers documentation](controllers/README.md).

- `CONSYS`: This class represents the control system that includes the plant and the controller. It includes methods to run the system for a specified number of epochs, update the controller parameters, and calculate the error. More details can be found in the [consys documentation](consys/README.md).

- `config.py`: This file contains the configuration parameters for the control system simulation. More details can be found in the [config documentation](config/README.md).

## Usage

To run the simulation, adjust the configuration parameters in `config.py` as needed, and then run `main.py`. The simulation will run for the specified number of epochs, and the average mean squared error for each epoch will be printed to the console. The error will be plotted and presented as a graph when the simulation is complete. 
