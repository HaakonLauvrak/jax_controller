# PLANT Class

This class is an abstract base class (ABC) that represents a generic plant in a control system. It provides method signatures for executing a timestep, getting the initial value, calculating the error, and getting additional values.

## Attributes

- `target`: The target value that the plant is trying to achieve.

## Methods

- `execute_timestep(args)`: This abstract method is responsible for advancing the plant's operation by one timestep. The `args` parameter provides any necessary inputs for this operation.

- `get_initial_value()`: This abstract method provides the starting point for the plant's operation. It returns the initial value that the plant should have at the beginning of its operation.

- `calculate_error(state)`: This abstract method assesses the plant's performance. It takes the current `state` of the plant and calculates the deviation from the desired target, returning this as an error value.

- `get_additional_values()`: This abstract method is a placeholder for any extra information that specific plant implementations might need to provide. By default, it doesn't return anything (`None`), but subclasses can override it to return relevant additional data.

# BATHTUB_PLANT Class

This class inherits from the `PLANT` class and represents a bathtub in a control system. The goal is to maintain the water level in the bathtub at a target value.

## Attributes

- `A`: The area of the bathtub.
- `C`: The drain coefficient of the bathtub.
- `target`: The target water level that the bathtub is trying to maintain.

## Methods

- `execute_timestep(state)`: This method advances the bathtub's operation by one timestep. It takes the current `state` of the bathtub, which includes the disturbance (e.g., additional water from a faucet or water removed by a user), the control signal (e.g., opening or closing the drain), and the current water volume. It calculates the new water level based on these inputs and returns it.

- `get_initial_value()`: This method returns the initial water level of the bathtub, which is the target water level.

- `calculate_error(state)`: This method calculates the deviation of the current water level from the target water level. It takes the current `state` of the bathtub and returns this error value.

# COURNOT_PLANT Class

This class inherits from the `PLANT` class and represents a Cournot competition model in a control system. The goal is to maximize the profit of a firm in a duopoly.

## Attributes

- `pmax`: The maximum price that can be charged for the product.
- `cm`: The marginal cost of production.
- `target`: The target profit that the firm is trying to achieve.
- `q1_0`, `q2_0`: The initial quantities of the products produced by the two firms.

## Methods

- `execute_timestep(state)`: This method advances the operation of the Cournot model by one timestep. It takes the current `state` of the model, which includes the disturbance (e.g., changes in the market or the actions of the other firm), the control signal (e.g., changes in the quantity produced by the firm), and the current quantities produced by the two firms. It calculates the new quantities, the price, and the profit based on these inputs and returns them.

- `get_initial_value()`: This method returns the initial quantities of the products produced by the two firms.

- `calculate_error(state)`: This method calculates the deviation of the current profit from the target profit. It takes the current `state` of the model and returns this error value.

- `get_additional_values(state)`: This method returns the current quantities, price, and profit. It takes the current `state` of the model and returns these additional values.

# CHEMICAL_REACTION_PLANT Class

This class inherits from the `PLANT` class and represents a chemical reaction in a control system. The goal is to maintain the concentration of a chemical at a target value.

## Attributes

- `k`: The reaction rate constant.
- `target`: The target concentration that the reaction is trying to maintain.
- `initial_concentration`: The initial concentration of the chemical.

## Methods

- `execute_timestep(state)`: This method advances the reaction by one timestep. It takes the current `state` of the reaction, which includes the disturbance (e.g., additional chemicals added or removed), the control signal (e.g., adding a catalyst or a reactant), and the current concentration. It calculates the new concentration based on these inputs and the reaction rate, and returns it.

- `get_initial_value()`: This method returns the initial concentration of the chemical.

- `calculate_error(state)`: This method calculates the deviation of the current concentration from the target concentration. It takes the current `state` of the reaction and returns this error value.