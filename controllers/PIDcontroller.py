from controllers.controller import CONTROLLER, abstractmethod
import jax

class PIDCONTROLLER(CONTROLLER):


    def compute_control_signal(self, params, state):
        error = state["error_history"][-1]  
        error_derivative = state["error_history"][-2] - error
        error_integral = sum(state["error_history"])
        return params[0] * error + params[1] * error_derivative + params[2] * error_integral
