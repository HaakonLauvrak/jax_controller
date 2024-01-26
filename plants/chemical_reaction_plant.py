from plants.plant import PLANT
import jax.numpy as jnp

class CHEMICAL_REACTION_PLANT(PLANT):

    def __init__(self, k, target, initial_concentration):
        super(CHEMICAL_REACTION_PLANT, self).__init__(target)
        self.k = k
        self.initial_concentration = initial_concentration

    def execute_timestep(self, state):
        control_signal = state["control_signal"]
        disturbance = state["disturbance"]
        current_concentration = state["value"]

        # Update concentration based on reaction rate and control signal
        new_concentration = current_concentration - self.k * current_concentration + control_signal + disturbance
        return new_concentration

    def get_initial_value(self):
        return self.initial_concentration

    def calculate_error(self, state):
        error = self.target - state["value"]
        return error
