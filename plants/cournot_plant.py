from plants.plant import PLANT
import jax.numpy as jnp


class COURNOT_PLANT(PLANT):

  def __init__(self, pmax, cm, target, q1, q2):
    super(COURNOT_PLANT, self).__init__(target)
    self.pmax = pmax
    self.cm = cm
    self.q1_0 = q1
    self.q2_0 = q2
    
  # def execute_timestep(self, state):
  #   disturbance = state["disturbance"]
  #   values = state["value"]
  #   control_signal = state["control_signal"]
  #   new_values = {}

  #   new_values["q1"] = jnp.clip(values["q1"] + control_signal, 0, 1)
  #   new_values["q2"] = jnp.clip(values["q2"] + disturbance, 0, 1)

  #   q = new_values["q1"] + new_values["q2"] # q = q1 + q2
  #   p = self.pmax - q #price = pmax - q
  #   new_values["profit"] = new_values["q1"] * (p - self.cm) #profit = q1 * (price - marginal cost)
  #   return new_values
  
  def execute_timestep(self, state):
        disturbance = state["disturbance"]
        control_signal = state["control_signal"]
        q1 = state["additional"][0]
        q2 = state["additional"][1]

        # Update q1 and q2 within the constraints
        new_q1 = jnp.clip(q1 + control_signal, 0, 1)
        new_q2 = jnp.clip(q2 + disturbance, 0, 1)

        # Calculate total production and price
        q = new_q1 + new_q2
        p = self.pmax - q

        return new_q1 * (p - self.cm)
  

  def get_initial_value(self):
        return 0
  
  def calculate_error(self, state):
    error = self.target - state["value"]
    return error

  def get_additional_values(self):
      return [self.q1_0, self.q2_0]