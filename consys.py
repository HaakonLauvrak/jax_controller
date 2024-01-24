import numpy as np
import jax
import jax.numpy as jnp
from plants.bathtub_plant import BATHTUB_PLANT
import config
from plants.cournot_plant import COURNOT_PLANT

class CONSYS: 

    def __init__(self):

        self.num_epochs = config.num_epochs
        self.num_timesteps = config.num_timesteps
        self.learning_rate = config.learning_rate
        self.noise_range = config.noise_range

        if  config.plant == "bathtub":
            self.plant = BATHTUB_PLANT()
        elif config.plant == "cournot":
            self.plant = COURNOT_PLANT(config.pmax, config.cm)
        else:
            AttributeError(f"{config.plant} not supported")
        
        if config.controller == "classic":
            self.controller = PIDCONTROLLER()
        elif config.controller == "neural":
            self.controller = NEURALCONTROLLER(config.num_layers, config.activation, config.weight_range)
        else:
            AttributeError(f"{config.controller} not supported")
    
        
            
    def run_system(num_epochs):
        gradfunc = jax.value_and_grad(run_one_epoch, argnums=0)
        # .. init params and state
        for _ in range(num_epochs):
            avg_error, gradients = gradfunc(params,state)
            # .. execute run_one_epoch via gradfunc
            update_params(params, gradients) # Use gradients to update controller params

    def run_one_epoch(self, params, state):
        for _ in range(self.num_timesteps):
            self.plant.execute_timestep(state)
            # .. get error at this timestep
            # .. update state

        # .. state gets updated at each timestep

        return avg_of_all_timestep_errors
