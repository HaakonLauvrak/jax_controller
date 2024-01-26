from plants.plant import PLANT, abstractmethod
import jax.numpy as jnp


class BATHTUB_PLANT(PLANT):
    
  def __init__(self, A, C, target):
    super(BATHTUB_PLANT, self).__init__(target)
    self.A = A
    self.C = C


  def execute_timestep(self, state):
    disturbance = state["disturbance"]
    control_signal = state["control_signal"]
    volume = state["value"] * self.A
    drain = self.C * jnp.sqrt(2 * 9.8 * state["value"])
    return max((volume + disturbance + control_signal - drain) / self.A, 0)

  def get_initial_value(self):
    return self.target
    
  def calculate_error(self, state):
    error = self.target - state["value"]
    return error
  