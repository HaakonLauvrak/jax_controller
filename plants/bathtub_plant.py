from plants.plant import PLANT, abstractmethod
import jax.numpy as jnp


class BATHTUB_PLANT(PLANT):
    
  def __init__(self, A, C, H0):
    self.A = A
    self.C = C
    self.H0 = H0

  def get_inital_value(self):
    return self.H0

  def execute_timestep(self, state):
    disturbance = state["disturbance"]
    control_signal = state["control_signal"]
    volume = state["value"] * self.A
    drain = self.C * jnp.sqrt(2 * 9.8 * state["value"])
    return (volume + disturbance + control_signal - drain) / self.A

    
  def calculate_error(self, state):
    error = self.H0 - state["value"]
    return error

    
    