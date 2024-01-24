import random
import numpy as np
import jax
import jax.numpy as jnp
from controllers.ANNcontroller import ANNCONTROLLER
from controllers.PIDcontroller import PIDCONTROLLER
from plants.bathtub_plant import BATHTUB_PLANT
import config
from plants.cournot_plant import COURNOT_PLANT


class CONSYS:

    def __init__(self):

        self.num_epochs = config.num_epochs
        self.num_timesteps = config.num_timesteps
        self.learning_rate = config.learning_rate
        self.noise_range = config.noise_range
        self.param_history = []

        if config.plant == "bathtub":
            self.plant = BATHTUB_PLANT(config.A, config.C, config.H0)
        elif config.plant == "cournot":
            self.plant = COURNOT_PLANT(config.pmax, config.cm, config.target)
        else:
            AttributeError(f"{config.plant} not supported")

        if config.controller == "classic":
            self.controller = PIDCONTROLLER()
        elif config.controller == "ann":
            self.controller = ANNCONTROLLER(
                config.num_layers, config.activation, config.weight_range)
        else:
            AttributeError(f"{config.controller} not supported")


    def run_system(self):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0)

        # Initialize params
        if config.controller == "classic":
            self.params = jnp.array([config.kp, config.ki, config.kd])
        elif config.controller == "ann":
            for _ in range(config.num_layers):
                self.params.append(
                    np.random.uniform(-config.weight_range, config.weight_range))
        else:
            AttributeError(f"{config.controller} not supported")


        epoch_error_history = []
        counter = 0
        for _ in range(self.num_epochs):
            counter += 1
            print(f"Epoch {counter}")
            avg_error, gradients = gradfunc(self.params)
            epoch_error_history.append(avg_error)
            self.params = self.update_params(self.params, gradients)
            self.param_history.append(self.params)
        
        return epoch_error_history, self.param_history

    def run_one_epoch(self, params):
        state = {
            "error_history": [0, 0],
            "disturbance": 0,
            "control_signal": 0,
            "value": self.plant.get_inital_value(),
        }

        for _ in range(self.num_timesteps):
            state["control_signal"] = self.controller.compute_control_signal(params, state)
            state["disturbance"] = random.uniform(-self.noise_range, self.noise_range)
            state["value"] = self.plant.execute_timestep(state)
            state["error_history"].append(self.plant.calculate_error(state))
        error_history_jax = jnp.array(state["error_history"])
        mean_squared_error = jnp.mean(jnp.square(error_history_jax))
        return mean_squared_error


    def update_params(self, params, gradients):
        if config.max_or_min == "max":
            q = 1
        else:
            q = -1
        new_params = []
        for i in range(len(params)):
            new_params.append(params[i] + q * self.learning_rate * gradients[i])
        return new_params
