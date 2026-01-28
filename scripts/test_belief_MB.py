import numpy as np
from envs import SocialGridworld, Lab_env_HRI, Human, Lab_env_HRI_LatentObs
from agents import Epsilon_greedy_MB, Epsilon_BeliefMB, Rmax_BeliefMB
from play_function import play
import constants as const

np.random.seed(42)
human = Human(**const.basic_human_param)
nav_env = SocialGridworld(size=120)

env_standard = Lab_env_HRI(nav_env, human)

env_latent = Lab_env_HRI_LatentObs(nav_env, human)

print("Testing e_greedy_MB (standard env)...")
agent_mb = Epsilon_greedy_MB(env_standard, **const.e_greedy_MB_param)
rewards_mb = play(env_standard, agent_mb, trials=50000, max_step=20)
print(f"Average reward: {np.mean(rewards_mb)}")

print("\nTesting e_greedy_belief_MB (latent obs env)...")
agent_belief = Epsilon_BeliefMB(env_latent, **const.e_greedy_belief_MB_param)
rewards_belief = play(env_latent, agent_belief, trials=50000, max_step=20)
print(f"Average reward: {np.mean(rewards_belief)}")

print("\nTesting Rmax_belief_MB (latent obs env)...")
agent_rmax = Rmax_BeliefMB(env_latent, **const.Rmax_belief_MB_param)
rewards_rmax = play(env_latent, agent_rmax, trials=50000, max_step=20)
print(f"Average reward: {np.mean(rewards_rmax)}")