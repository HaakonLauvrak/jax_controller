from abc import ABC, abstractmethod

class PLANT: 

    def __init__(self, target):
        self.target = target

    @abstractmethod
    def execute_timestep(args):
        pass

    @abstractmethod
    def get_initial_value(self):
        pass

    @abstractmethod
    def calculate_error(self, state):
        pass

    @abstractmethod
    def get_additional_values(self):
        return None
    
