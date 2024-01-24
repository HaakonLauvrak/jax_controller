# 1. The plant to simulate: bathtub, Cournot competition, your additional model, etc.
plant = "bathtub"
# 2. The controller to use: classic or AI-based
controller = "classic"
# 3. Number of layers and number of neurons in each layer of the neural network. Your system should
# handle anywhere between 0 and 5 hidden layers.
num_layers = 1
# 4. Activation function used for each layer of the neural network. Your system must include at least
# Sigmoid, Tanh and RELU.
activation = "sigmoid"
# 5. Range of acceptable initial values for each weight and bias in the neural network.
weight_range = 1
# 6. Number of training epochs
num_epochs = 5
# 7. Number of simulation timesteps of the CONSYS per epoch
num_timesteps = 50
# 8. Learning rate for tuning the controller parameters, whether classic PID or neural-net-based.
learning_rate = 0.1
# 9. Range of acceptable values for noise / disturbance (D).
noise_range = 0.01
# 10. Cross-sectional area (A) of the bathtub
A = 1
# 11. Cross-sectional area (C) of the bathtub’s drain.
C = 0.05
# 12. Initial height (H0) of the bathtub water.
H0 = 1
# 13. The maximum price (pmax) for Cournot competition
pmax = 1
# 14. The marginal cost (cm) for Cournot competition.
cm = 0.1
# Target for the price of the Cournot competition
target = 0.5
# q1 initial amount produced
q1 = 0.5
# q2 initial amount produced
q2 = 0.5
# 15. At least two parameters for your third plant.

# Initial values for pid parameters
kp, ki, kd = 0.1, 0.1, 0.1

max_or_min = "min"

