from plants.plant import PLANT, abstractmethod



class COURNOT_PLANT(PLANT):

  def __init__(self, pmax, cm, target, q1, q2):
    super(self, target)
    self.pmax = pmax
    self.cm = cm
    self.q1 = q1
    self.q2 = q2
    
  def execute_timestep(self, state):
    disturbance = state["disturbance"]
    profit = state["value"]
    q1 = state["control_signal"]

    p = self.pmax -
    return p
