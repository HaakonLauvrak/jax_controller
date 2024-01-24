from abc import ABC, abstractmethod

class PLANT: 

    @abstractmethod
    def execute_timestep(args):
        pass

    @abstractmethod 
    def get_initial_value():
        pass