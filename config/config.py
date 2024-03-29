# 1. The plant to simulate: "bathtub", "cournot", "chemical_reaction"
plant = "bathtub"
# 2. The controller to use: "classic" or "ann"
controller = "ann"
# 3. Number of layers and number of neurons in each layer of the neural network.
layers = [5, 10, 5]
# 4. Activation function used for each layer of the neural network. Possible values: "tanh", "sigmoid", "RELU".
activation_function = "relu"
# 5. Range of acceptable initial values for each weight and bias in the neural network.
# For bathtub, cournot, weight range = 0.001 works good.
weight_range = 0.1
# 6. Number of training epochs
num_epochs = 20
# 7. Number of simulation timesteps of the CONSYS per epoch
# For ann chemical_reaction: num_timesteps = 10 works good. Struggles with higher timesteps for some reason.
num_timesteps = 50
# 8. Learning rate for tuning the controller parameters, whether classic PID or neural-net-based.
#For classic chemical_reaction: learning_rate = 0.05 works good. 
# For classic bathtub and cournot: learning_rate = 0.3 works good.
# For ann sigmoid bathtub: learning_rate = 0.01 works good.
# For ann cournot sigmoid: learning_rate = 0.3 works good.
learning_rate = 0.01
# 9. Range of acceptable values for noise / disturbance (D).
# For classic chemical_reaction: noise_range = 0.05 works good.
# For classic bathtub and cournot: noise_range = 0.01 works good.
# For ann bathtub: noise_range = 0.1 works good.
# For ann cournot: noise_range = 0.01 works good.
noise_range = 0.01
# Initial values for pid parameters
#For classic chemical_reaction: kp = 0.3, ki = 0.3, kd = 0.3 works good.
#For classic bathtub and cournot: kp = 0.1, ki = 0.1, kd = 0.1 works good.
kp, ki, kd = 0.1, 0.1, 0.1

max_or_min = "min"

# Bathtub
# 10. Cross-sectional area (A) of the bathtub
A = 1
# 11. Cross-sectional area (C) of the bathtub’s drain.
C = 0.05
# 12. Initial height (H0) of the bathtub water.
H0 = 1

# Cournot competition
# 13. The maximum price (pmax) for Cournot competition
pmax = 2
# 14. The marginal cost (cm) for Cournot competition.
cm = 0.50
# Target for the price of the Cournot competition
cournot_target = 0.2
# q1 initial amount produced
q1 = 0.5
# q2 initial amount produced
q2 = 0.5
# 15. At least two parameters for your third plant.

# Chemical reaction plant
# 16. The rate constant (k) for the chemical reaction
# For ann: k = 0.01 works good. Struggles with high k, as control signal is limited to [-1, 1] with sign
k = 0.01
# 17. The target concentration for the chemical reaction
chemical_plant_target = 10
# 18. The initial concentration for the chemical reaction
initial_concentration = 10
