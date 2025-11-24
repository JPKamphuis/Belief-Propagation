import json

class GridWorldConfig:
    def __init__(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self.grid_size = data["grid_size"]
        self.reward_states = {int(k): v for k, v in data["reward_states"].items()}
        self.obstacle_states = set(data["obstacle_states"])
        self.terminal_states = set(data["terminal_states"])
        self.alpha = data["alpha"]
        self.beta = data["beta"]
        self.num_actions = data.get("num_actions", 4)
        self.start_state = data["start_state"]
        self.evidence = {int(k): v for k, v in data.get("evidence", {}).items()}
        self.bp_iters = data["bp_iters"]
        self.run_bp = data["run_bp"]
        self.gamma = data["gamma"]
        self.pi_iters = data["pi_iters"]
        self.episodes = data["episodes"]
        self.p_correct_action = data["p_correct_action"]
        self.print_every = data["print_every"]

    def __repr__(self):
        return f"<GridWorldConfig {self.__dict__}>"
