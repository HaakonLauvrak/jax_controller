from abc import ABC, abstractmethod

class PLANT: 

    def __init__(self, target):
        self.target = target

    @abstractmethod
    def execute_timestep(args):
        pass

    def get_initial_value(self):
        return self.target

    def calculate_error(self, state):
        error = self.target - state["value"]
        return error
    
