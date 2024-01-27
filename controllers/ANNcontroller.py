import numpy as np
from controllers.controller import CONTROLLER, abstractmethod
import jax.numpy as jnp
import numpy as np
import random

class ANNCONTROLLER(CONTROLLER):

    def __init__(self, layers, activation_function, weight_range):
        self.layers = layers
        self.activation_function = activation_function
        self.weight_range = weight_range

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
    
    @staticmethod
    def tanh(x):
        return jnp.tanh(x)
    
    @staticmethod
    def relu(x):
        return jnp.maximum(0, x)
    
    
    def execute_activation_func(self, signal):
        if self.activation_function == "sigmoid":
            return self.sigmoid(signal)
        elif self.activation_function == "tanh":
            return self.tanh(signal)
        elif self.activation_function == "relu":
            return self.relu(signal)

    def gen_jaxnet_params(self):
        sender = 3
        self.layers.append(1)
        params = []
        for receiver in self.layers:
            weights = np.random.uniform(-self.weight_range, self.weight_range,(receiver, sender))
            biases = np.random.uniform(-self.weight_range, self.weight_range,(1,receiver))
            sender = receiver
            params.append([weights, biases])
        return params

    def compute_control_signal(self, params, state):
        current_input = jnp.array([state["error_history"][-1], 
                   state["error_history"][-2] - state["value"], 
                   sum(state["error_history"])])
        for i in range(len(params)):
            weights = params[i][0]
            biases = params[i][1][0]
            activations = []
            for j in range(len(weights)):
                dot_product = jnp.dot(current_input,weights[j]) + biases[j]
                activations.append(self.execute_activation_func(dot_product))
            current_input = jnp.array(activations)
        return current_input[0]

    

