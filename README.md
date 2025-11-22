This repository implements loopy belief propagation as an initializing step in solving grid-world reinforcement learning problems, testing if the algorithm can speed up training by reducing the iterations of RL needed to converge to an optimal policy. Loopy BP implementation is adapted from https://github.com/krashkov/Belief-Propagation.

# Enviroment Setup
environment.yml contains the conda environment with all packages needed to run the code. An environment can be created with ```conda env create -f environment.yml```

# Config.py and grid world definition
Before executing ```rlbp/main.py``` to run belief propagation and policy iteration, you can create a grid and set hyperparameters for the algorithms in a config.json file. The rlbp/grids folder contains four grids that were tested on. Grids are square. The top left cell is numbered 0 and the bottom right is ```grid_size ** 2 - 1```. There are a few special types of cells:
Empty: The agent can move freely through these
Obstacle states: the agent cannot move into these cells. If it attempts to move into an obstacle cell, it will stay in its current cell. If it is initialized in an obstacle, it cannot leave.
Terminal states: When an agent enters a terminal state, the trajectory ends.
Reward states: Any cell can be assigned a reward for entering. In my grid worlds, I assign one destination cell with a reward of +1, and several death cells with rewards of -1. Also in my grid worlds, the reward states are terminal.

A random grid can be created from specified dimensions and set number of obstacles and "lightnings": (terminal states with reward -1) using ```notebooks/random_grid.ipynb```.

Hyperparameters:
The config file also contains many hyperparameters for BP and RL:
- Evidence: A dictionary of cell numbers and their set distributions to propagate with LBP. The distribution is a geometric RV with four entries: up, down, left, and right.
- alpha and beta: These parameters control the initialization of conditional distributions for LBP. Alpha is generally greater than beta, which means that neighbors in the graph generally have the same optimal policy. The strength of this ratio controls how quickly that relationship between neighbors decays.
- num_actions: 4, for up, down, left, right. Additional actions would require modifications to other parts of the code as well.
- run_bp: A boolean to compare pure RL training vs LBP initialization.
- gamma: Discount factor for RL
- pi_iters: Max iterations of policy iteration
- eposides: Number of monte carlo trajectories used to estimate the return of a policy
- p_correct_action: Change that the agent takes its desired action. It is common to model an agent that has a small chance of going right, for example, when it meant to go up. This probablity is assigned to the desired action, with the other actions evenly dividing the remainder.

# Running the code
Everything required to run belief propagation and policy iteration on a grid world is contained in ```rlbp/main.py```. A config file can be specified from the command line.
