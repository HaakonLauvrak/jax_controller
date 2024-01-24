from plants.plant import PLANT, abstractmethod



class COURNOT_PLANT(PLANT):

  def __init__(self, pmax, cm):
    self.pmax = pmax
    self.cm = cm
    
    
  def execute_timestep(args):
    p = self.pmax - q