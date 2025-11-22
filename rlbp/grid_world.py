import numpy as np
from config import GridWorldConfig
from factor import factor
from graph import factor_graph
from belief_propagation import loopy_belief_propagation

def move(state, action, grid_size):
        r, c = divmod(state, grid_size)

        if action == "up" or action == 0:
            if r == 0: 
                return state
            return (r - 1) * grid_size + c

        elif action == "down" or action == 1:
            if r == grid_size - 1:
                return state
            return (r + 1) * grid_size + c

        elif action == "left" or action == 2:
            if c == 0:
                return state
            return r * grid_size + (c - 1)

        elif action == "right" or action == 3:
            if c == grid_size - 1:
                return state
            return r * grid_size + (c + 1)

def get_transition_matrix(cfg):
    """
    Creates stochastic transition matrices for each action.
    
    p_intended: probability of going in the intended direction.
                 The remaining probability is split among other actions.
    """
    grid_size = cfg.grid_size
    n_states = grid_size * grid_size
    actions = ["up", "down", "left", "right"]
    n_actions = len(actions)

    P_actions = []

    for action_idx, action in enumerate(actions):
        P = np.zeros((n_states, n_states))

        for state in range(n_states):
            # Absorbing states (terminal or obstacle)
            if state in cfg.terminal_states or state in cfg.obstacle_states:
                P[state, state] = 1.0
                continue

            p_correct_action = cfg.p_correct_action
            p_error = (1 - p_correct_action) / (n_actions - 1)
            probs = {a: p_error for a in actions}
            probs[action] = p_correct_action

            for a, p in probs.items():
                next_state = move(state, a, cfg.grid_size)
                if next_state in cfg.obstacle_states: # Moved to a mountain
                    next_state = state
                P[state, next_state] += p
        
        P_actions.append(P)

    return P_actions

def get_reward_matrix(cfg):
    grid_size = cfg.grid_size
    n_states = grid_size * grid_size
    R = np.zeros(n_states)
    for s, r in cfg.reward_states.items():
        R[s] = r
    return R

def make_bp_graph(cfg):
    rows = cols = cfg.grid_size
    fg = factor_graph()

    # create variables
    for s in range(rows * cols):
        fg.add_variable_node(f"V{s}")
        fg.get_graph().vs.find(name=f"V{s}")["rank"] = cfg.num_actions

    # Potts factor
    pairwise = cfg.beta * np.ones((cfg.num_actions, cfg.num_actions))
    np.fill_diagonal(pairwise, cfg.alpha)

    # add edges
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c

            # right neighbor
            if c + 1 < cols:
                s2 = s + 1
                fg.add_factor_node(f"F{s}_{s2}", factor([f"V{s}", f"V{s2}"], pairwise))

            # down neighbor
            if r + 1 < rows:
                s2 = s + cols
                fg.add_factor_node(f"F{s}_{s2}", factor([f"V{s}", f"V{s2}"], pairwise))

    return fg

def graph_to_array(bp: loopy_belief_propagation, rows, cols):
    # Convert all BP marginals directly into a flat (rows*cols, 4)
    # stochastic policy array.

    n_states = rows * cols
    policy_flat = np.zeros((n_states, 4))

    # Number of iterations BP has already run
    num_iter = bp.get_t()

    for node in range (rows * cols):
            vname = f"V{node}"

            # Get marginal belief (messages already updated globally)
            belief = bp.belief(vname, num_iter)
            dist = belief.get_distribution()

            policy_flat[node] = dist

    return policy_flat

def softmax(x, tau=1.0):
    x = x / tau
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def policy_iteration(P_actions, R, gamma, policy, cfg,
                     alpha=0.01, tau=0.1, mix=1,
                     iterations=5000, max_steps=100):
    """
    TD(0)-style policy iteration with policy mixing to preserve BP priors.
    
    mix: how much of the improved policy to blend in each update.
    """
    n_states = len(R)
    n_actions = len(P_actions)
    V = np.zeros(n_states)

    for it in range(iterations):

        # random start state
        s = np.random.randint(n_states)

        for step in range(max_steps):

            # sample action using current policy
            a = np.random.choice(n_actions, p=policy[s])

            # sample next state
            s_prime = np.random.choice(n_states, p=P_actions[a][s])

            # TD(0) update
            V[s] += alpha * (R[s] + gamma * V[s_prime] - V[s])

            # compute local Q-values
            Q = np.zeros(n_actions)
            for a_i in range(n_actions):
                Q[a_i] = R[s] + gamma * (P_actions[a_i][s] @ V)

            # softmax improvement
            improved = np.exp(Q / tau)
            improved /= improved.sum()

            # *** POLICY MIXING ***
            # policy[s] = (1-mix)*old + mix*new
            policy[s] = (1 - mix) * policy[s] + mix * improved
            policy[s] /= policy[s].sum()

            # move to next state
            s = s_prime

        # progress logging, comment out for accurate runtime
        if it % 100 == 0:
            print(f"Expected return from start state {cfg.start_state} "
                  f"after {it} iterations: {monte_carlo_return(policy, R, cfg)}")

    return V, policy


def monte_carlo_return(policy, R, cfg):
    returns = []
    n_states = cfg.grid_size * cfg.grid_size
    for _ in range(cfg.episodes):
        s = cfg.start_state
        G = 0
        t = 0
        while s not in cfg.terminal_states:
            a = np.random.choice(4, p=policy[s])
            s = move(s, a, cfg.grid_size) #TODO: Move doesnt use the probability of correct move
            G += (cfg.gamma**t) * R[s]
            t += 1
        returns.append(G)
    return np.mean(returns)

def print_graph_as_grid(fg, grid_size):
    """
    Reads the factor_graph object directly.
    Prints only variable nodes V#, arranged into a (rows x cols) grid
    based on the integer index in their name.

    Example:
        V0 V1 V2
        V3 V4 V5
        V6 V7 V8
    """

    g = fg.get_graph()

    # Extract variable nodes and parse their indices
    variables = []
    for v in g.vs:
        if not v["is_factor"]:
            name = v["name"]
            if name.startswith("V"):
                try:
                    idx = int(name[1:])
                    variables.append((idx, name))
                except:
                    pass

    # Sort by index to ensure correct ordering
    variables.sort(key=lambda x: x[0])

    # Build a 2D grid
    grid = [["??" for _ in range(grid_size)] for _ in range(grid_size)]

    for idx, name in variables:
        r = idx // grid_size
        c = idx % grid_size
        grid[r][c] = name

    # Print
    print("\n=== BP Graph Variable Layout ===")
    for r in range(grid_size):
        print("  ".join(f"{grid[r][c]:3s}" for c in range(grid_size)))
    print("")

def print_pretty_grid(cfg):
    """
    Pretty-print the grid as a uniform boxed layout with centered symbols.

    Legend:
      (space) = empty
      M = obstacle
      L = lightning (negative terminal)
      T = treasure
      S = start state
    """

    size = cfg.grid_size

    treasure_cells = {int(s) for s, r in cfg.reward_states.items() if r > 0}
    lightning_cells = {int(s) for s, r in cfg.reward_states.items() if r < 0}
    obstacle_cells = set(cfg.obstacle_states)
    start = cfg.start_state

    def symbol(idx):
        if idx == start: return "S"
        if idx in treasure_cells: return "T"
        if idx in lightning_cells: return "L"
        if idx in obstacle_cells: return "M"
        return " "

    print("\n=== GRID WORLD ===\n")

    # For each row
    for r in range(size):
        # Upper border of cells
        print(("+" + "---") * size + "+")

        # Middle line with centered symbols
        row_symbols = []
        for c in range(size):
            idx = r * size + c
            row_symbols.append(f"| {symbol(idx)} ")
        print("".join(row_symbols) + "|")

        # Lower border of cells
        print(("+" + "---") * size + "+")

    print()

def print_policy(policy, cfg):
    """
    Print the stochastic policy for each cell along with what the cell contains:
      T = treasure
      L = lightning (negative terminal)
      M = obstacle
      S = start
      . = normal empty cell
    """
    treasure_cells = {int(s) for s, r in cfg.reward_states.items() if r > 0}
    lightning_cells = {int(s) for s, r in cfg.reward_states.items() if r < 0}
    obstacle_cells = set(cfg.obstacle_states)
    start_state = cfg.start_state

    def cell_type(idx):
        if idx == start_state:
            return "S"
        if idx in treasure_cells:
            return "T"
        if idx in lightning_cells:
            return "L"
        if idx in obstacle_cells:
            return "M"
        return "."

    for i in range(policy.shape[0]):
        ctype = cell_type(i)
        p = policy[i]
        print("Cell {:3d} [{}] policy: [{:.3f}, {:.3f}, {:.3f}, {:.3f}]"
              .format(i, ctype, p[0], p[1], p[2], p[3]))

