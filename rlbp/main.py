from config import GridWorldConfig
from grid_world import *
from belief_propagation import loopy_belief_propagation
import argparse
import time

def main():
  parser = argparse.ArgumentParser(description= 'BP init for grid world')
  parser.add_argument('-c', '--config', default='grids/config_10_grid.json',
                      type=str, help="name of config file")
  args = parser.parse_args()
    
  cfg = GridWorldConfig(args.config)
  print_pretty_grid(cfg)

  # Build MDP components
  P = get_transition_matrix(cfg)
  R = get_reward_matrix(cfg)

  # Build BP factor graph
  fg = make_bp_graph(cfg)

  # Run loopy belief propagation
  bp_start = time.time()
  if (cfg.run_bp):
    fg.apply_evidence(cfg.evidence)
    bp = loopy_belief_propagation(fg)
    bp.belief("V0", cfg.bp_iters)
  else: # Pure RL, just initialize bp for graph_to_array()
    bp = loopy_belief_propagation(fg)
  bp_end = time.time()

  # Convert to policy array
  policy = graph_to_array(bp, cfg.grid_size, cfg.grid_size)
  # print_policy(policy, cfg)

  # Run value iteration algorithm
  mix = 1
  if(cfg.run_bp): mix = 0.3

  rl_start = time.time()
  V, policy_opt = policy_iteration(P, R, cfg.gamma, policy, cfg, iterations=cfg.pi_iters, mix=mix)
  rl_end = time.time()

  # print_policy(policy_opt, cfg)
  expected_return = monte_carlo_return(policy_opt, R, cfg)
  print(f"Expected return from start state {cfg.start_state} in optimal policy: {expected_return}")
  print(f"Belief propagation took {bp_end - bp_start} seconds for {cfg.bp_iters} iters.")
  print(f"Policy iteration took {rl_end - rl_start} seconds for {cfg.pi_iters} iters.")

  
if __name__ == '__main__':
  main()