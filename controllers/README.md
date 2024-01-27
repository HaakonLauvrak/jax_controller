# CONTROLLER Class

This class is an abstract base class (ABC) that represents a generic controller. It provides a method signature for computing a control signal.

## Methods

- `compute_control_signal(params, state)`: This is an abstract method that must be implemented by any class that inherits from `CONTROLLER`. It takes two parameters:
    - `params`: The parameters of the controller. The exact nature of these parameters will depend on the specific type of controller.
    - `state`: The current state of the system that the controller is controlling.

The `compute_control_signal` method is expected to compute and return a control signal based on the provided parameters and state.

# ANNcontroller Class

This class is used to create an Artificial Neural Network (ANN) controller. It provides methods to execute different activation functions and generate parameters for the network.

## Attributes

- `activation_function`: A string that specifies the activation function to be used in the network. It can be "sigmoid", "tanh", or "relu".
- `weight_range`: A float that specifies the range within which the weights and biases of the network will be initialized.

## Methods

- `sigmoid(x)`: A static method that takes a number `x` and returns the sigmoid activation function applied to `x`.
- `tanh(x)`: A static method that takes a number `x` and returns the hyperbolic tangent activation function applied to `x`.
- `relu(x)`: A static method that takes a number `x` and returns the rectified linear unit (ReLU) activation function applied to `x`.
- `execute_activation_func(signal)`: This method takes a signal and applies the specified activation function to it.
- `gen_jaxnet_params()`: This method generates the weights and biases for the network layers within the specified weight range.
- `compute_control_signal(params, state)`: This method computes the control signal for the ANN controller. It takes two parameters: `params`, which are the weights and biases of the network, and `state`, which is the current state of the system. The method computes the control signal by passing the current state through the network, applying the specified activation function at each layer, and returning the output of the final layer as the control signal.

Note: The class uses `jnp` (JAX numpy) for mathematical operations and `np` (numpy) for generating random weights and biases.    



# PIDCONTROLLER Class

This class inherits from the `CONTROLLER` class and implements a Proportional-Integral-Derivative (PID) controller.

## Methods

- `compute_control_signal(params, state)`: This method computes the control signal for the PID controller. It takes two parameters:
    - `params`: A list of three elements representing the proportional, integral, and derivative gains, respectively.
    - `state`: A dictionary containing the current state of the system. It should have a key `"error_history"` that maps to a list of past error values. The most recent error is at the end of the list.

The `compute_control_signal` method computes the control signal based on the PID formula: `P*error + I*integral(error) + D*derivative(error)`, where `P`, `I`, and `D` are the proportional, integral, and derivative gains, respectively.

**Note:** This documentation is AI-generated based on the code we have written ourselves. We have read through it to ensure it is accurate.  