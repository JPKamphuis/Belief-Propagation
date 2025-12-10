This repository implements loopy belief propagation as an initializing step in solving grid-world reinforcement learning problems, testing if the algorithm can speed up training by reducing the iterations of RL needed to converge to an optimal policy. Loopy BP implementation is adapted from https://github.com/krashkov/Belief-Propagation.

# Idea
Many reinforcement learning algorithms work by iteratively improving a policy, performing several small updates that are each garunteed to increase the expected value (measured with the state value function V or state-action value function Q) of the policy. Intuively, if we could initialize the policy to be closer to an optimum, then fewer of these steps would be required to reach that optimum. Belief propagation, JTA, etc. seem like natural choices to use some small amount of information that we could expect a human to have about a particular reinforcement learning environment in order to enact sweeping changes to a policy all at once. Obviously the results of running belief propagation with a small amount of information will not reasonably create an optimal policy, but should hopefully provide an initiazation that will require fewer RL iterations.

# Methodology
The grid-world environment is a classic benchmark for RL algorithms. In this setting, an agent has to learn to avoid obstacles and reach some positive reward as quickly as possible. It is a *very simplifed* simluation of real RL tasks, like a roomba navigating a house or a self-driving car finding parking in a lot. This setting is also easily translated to a graphical model; each cell in the grid-world becomes a node in the graph, with edges between itself and its neighbors. Using code from the repository mentioned above, a grid world is translated into a factor graph where evidence can be injected and belief propagation can run. Obviously, this graph contains many cycles, so we use loopy belief propagation, in which normal BP is run for many iterations and an approximate solution is reached. 

The evidence in this case is some knowledge that the human has about the optimal policy. We might reasonably expect to know, for example, that a roomba's best action when it is in a corner is to turn 180 degrees. Similarly, we know in a grid world that if the positive reward ('treasure cell') is directly to the right of the agent, it should move right. When we propagate this evidence, we make a simple assumption that the optimal action in a given cell is similar to the optimal action in neighboring cells, so the conditional distributions in the graph model this assumption. After belief propagation, cells that had evidence in them will have their initial policies set exactly to the evidence, cells close to them will be influenced toward the evidence, and cells far away from evidence will be unaffected. 

Looking at  ```rlbp/main.py```, this is implemented in a few function calls. We create a grid from a config file that tells us how large the grid is, where obstacles are, what our evidence is, etc. and then translate it to a graph. The code from krashkov is used to run loopy belief propagation and create an initial policy that is hopefully closer to an optimal policy than a uniform initialization. After the policy is initialized, we run a policy iteration algorithm (SARSA) to find an optimal policy. SARSA continously update the state-action value function (Q function) as the agent explores the environment. Each time the agent moves, the state, action and reward from the move as well as the next state and action (hence SARSA: state action reward state action) are used to update the Q function according to
`Q(s,a) ← Q(s,a) + α [ r + γ Q(s', a') − Q(s,a) ]`, where alpha is a learning rate and gamma is a discount factor so that future rewards are less valuable (encouraging the agent to finish the task quickly). The Q function is easily translated to a policy: when we are in a state, we choose the action that has the highest Q(s,a) value.

I track the progress of SARSA training by sampling trajectories from the policy intermittently and calculating their reward. From this, we can see whether the BP initialized policy reach high reward faster than uniform initialization.

# Results/Contributions
The BP initialized policies converged quicker than uniform initialized policy in simple grids, and in complicated grids that had very strong evidence (lots of evidence along an optimal trajectory). In more complicated settings with larger grids, more obstacles, etc., the two methods performed equally. The mostly likely cause of this underperformance is the assumption made that optimal policies in cells are similar to their neighbors. In reality, the best policy in one cell might be to move right to head toward the treasure, but immediately below that cell, moving right takes the agent to a danger cell with negative reward. Future work with this initialization needs to more carefully construct the conditional distributions between neighbors in the graphical model.



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
