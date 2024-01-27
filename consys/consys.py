import random
import numpy as np
import jax
import jax.numpy as jnp
from controllers.ANNcontroller import ANNCONTROLLER
from controllers.PIDcontroller import PIDCONTROLLER
from plants.bathtub_plant import BATHTUB_PLANT
import config.config as config
from plants.cournot_plant import COURNOT_PLANT
from plants.chemical_reaction_plant import CHEMICAL_REACTION_PLANT


class CONSYS:
    """
    The CONSYS class represents a control system. It initializes the control system with the specified plant and controller configurations,
    and provides methods to run the system and update the parameters.
    """

    def __init__(self):
        """
        Initializes a new instance of the CONSYS class.
        """
        self.num_epochs = config.num_epochs
        self.num_timesteps = config.num_timesteps
        self.learning_rate = config.learning_rate
        self.noise_range = config.noise_range
        self.param_history = []
        self.params = []

        if config.plant == "bathtub":
            self.plant = BATHTUB_PLANT(config.A, config.C, config.H0)
        elif config.plant == "cournot":
            self.plant = COURNOT_PLANT(
                config.pmax, config.cm, config.cournot_target, config.q1, config.q2)
        elif config.plant == "chemical_reaction":
            self.plant = CHEMICAL_REACTION_PLANT(
                config.k, config.chemical_plant_target, config.initial_concentration)
        else:
            raise AttributeError(f"{config.plant} not supported")

        if config.controller == "classic":
            self.controller = PIDCONTROLLER()
        elif config.controller == "ann":
            self.controller = ANNCONTROLLER(config.layers, config.activation_function, config.weight_range)
        else:
            raise AttributeError(f"{config.controller} not supported")

    def run_system(self):
        """
        Runs the control system for the specified number of epochs.

        Returns:
            epoch_error_history (list): A list of average error values for each epoch.
            param_history (list): A list of parameter values for each epoch.
        """
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0)

        # Initialize params
        if config.controller == "classic":
            self.params = jnp.array([config.kp, config.ki, config.kd])
        elif config.controller == "ann":
            self.params = self.controller.gen_jaxnet_params()
        else:
            raise AttributeError(f"{config.controller} not supported")

        epoch_error_history = []
        counter = 0
        for _ in range(self.num_epochs):
            counter += 1
            print(f"Epoch {counter}")
            avg_error, gradients = gradfunc(self.params)
            epoch_error_history.append(avg_error)
            self.params = self.update_params(self.params, gradients)
            self.param_history.append(self.params)
            print(f"Average error: {avg_error}")

        return epoch_error_history, self.param_history

    def run_one_epoch(self, params):
        """
        Runs one epoch of the control system with the given parameters.

        Args:
            params (list): The parameters of the controller.

        Returns:
            mean_squared_error (float): The mean squared error of the control system for the epoch.
        """
        state = {
            "error_history": [0, 0],
            "disturbance": 0,
            "control_signal": 0,
            "value": self.plant.get_initial_value(),
            "additional": self.plant.get_additional_values()
        }

        for _ in range(self.num_timesteps):
            state["control_signal"] = self.controller.compute_control_signal(
                params, state)
            state["disturbance"] = random.uniform(
                -self.noise_range, self.noise_range)
            state["value"] = self.plant.execute_timestep(state)
            state["error_history"].append(self.plant.calculate_error(state))
        error_history_jax = jnp.array(state["error_history"])
        mean_squared_error = jnp.mean(jnp.square(error_history_jax))
        return mean_squared_error

    def update_params(self, params, gradients):
        """
        Updates the parameters of the controller based on the gradients.

        Args:
            params (list): The current parameters of the controller.
            gradients (list): The gradients of the parameters.

        Returns:
            new_params (list): The updated parameters of the controller.
        """
        if config.controller == "classic":
            return self.update_params_classic(params, gradients)
        elif config.controller == "ann":
            return self.update_params_ann(params, gradients)
        else:
            raise AttributeError(f"{config.controller} not supported")
        
    def update_params_classic(self, params, gradients):  
        """
        Updates the parameters of the classic controller.

        Args:
            params (list): The current parameters of the controller.
            gradients (list): The gradients of the parameters.

        Returns:
            new_params (list): The updated parameters of the controller.
        """
        if config.max_or_min == "max":
            q = 1
        else:
            q = -1
        new_params = []
        for i in range(len(params)):
            new_params.append(
                params[i] + q * self.learning_rate * gradients[i])
        return new_params 
    
    def update_params_ann(self, params, gradients):
        """
        Updates the parameters of the ANN controller.

        Args:
            params (list): The current parameters of the controller.
            gradients (list): The gradients of the parameters.

        Returns:
            new_params (list): The updated parameters of the controller.
        """
        return [(w - self.learning_rate * dw, b - self.learning_rate * db) 
                for (w, b), (dw, db) in zip(params, gradients)]
