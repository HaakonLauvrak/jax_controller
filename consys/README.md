# CONSYS Class

This class represents the control system that includes the plant and the controller.

## Attributes

- `num_epochs`: The number of training epochs.
- `num_timesteps`: The number of simulation timesteps of the CONSYS per epoch.
- `learning_rate`: The learning rate for tuning the controller parameters.
- `noise_range`: The range of acceptable values for noise/disturbance (D).
- `param_history`: A list to store the history of parameters.
- `params`: The parameters of the controller.
- `plant`: An instance of the `PLANT` class or its subclasses.
- `controller`: An instance of the `CONTROLLER` class or its subclasses.

## Methods

- `__init__`: Initializes the control system. It sets the plant and controller based on the configuration.

- `run_system`: Runs the control system for a specified number of epochs. It initializes the parameters, runs one epoch at a time, updates the parameters based on the gradients, and stores the parameters and the average error for each epoch.

- `run_one_epoch`: Runs the control system for one epoch. It initializes the state, runs the control system for a specified number of timesteps, and calculates the mean squared error over all timesteps.

- `update_params`: Updates the parameters of the controller based on the gradients. It checks the type of controller and calls the appropriate method to update the parameters.

- `update_params_classic`: Updates the parameters of a classic controller. It takes the current parameters and the gradients, and returns the updated parameters. The direction of the update depends on whether the goal is to maximize or minimize the error.

- `update_params_ann`: Updates the parameters of an ANN controller. It takes the current parameters and the gradients, and returns the updated parameters. The weights and biases of the ANN are updated separately.

**Note:** This documentation is AI-generated based on the code we have written ourselves. We have read through it to ensure it is accurate.  