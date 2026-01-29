from envs import Human, Lab_env_HRI, SocialGridworld
from agents import Epsilon_greedy_MB, MFLearnerOnMB, Epsilon_greedy_MF
from play_function import play
import constants as const
import variables as var
import numpy as np

from graphics import plot_curves, extracting_results



name_human = 'basic_human'

# ---------------------------------------------------------------------------- #
# Generating the initial Q-maps for the MF agent
# ---------------------------------------------------------------------------- #


play_parameters = {'trials': 25000, 'max_step': 20}
nb_tests = 8

def get_q_table_MF(nb_tests, play_params):
    for i in range(nb_tests):

        human = Human(**var.human_parameters[name_human])

        nav_env = SocialGridworld(size=120)

        environment = Lab_env_HRI(nav_env, human)
        MB_agent = Epsilon_greedy_MB(environment,
                                     **const.e_greedy_MB_param)
        agent = MFLearnerOnMB(MB_agent, 
                              **const.e_greed_MF_no_decision_param)
        play(environment, agent, **play_params)
        np.save('../data/MF_tables/agent_Q_MF_'+str(i)+'_.npy', agent.Q_MF)


get_q_table_MF(nb_tests, play_parameters)



# ---------------------------------------------------------------------------- #
# Generating the rewards with learned initial Q-values
# ---------------------------------------------------------------------------- #



play_parameters = {'trials': 75000, 'max_step': 20}
nb_tests = 8

def get_MF_performance_with_q_tables(nb_tests, play_parameters):
    rewards = {}
    for i in range(nb_tests):

        human = Human(**var.human_parameters[name_human])

        nav_env = SocialGridworld(size=120)
        environment = Lab_env_HRI(nav_env, human)
        init_Q = np.load('../data/MF_tables/agent_Q_MF_'+str(i)+'_.npy')
        agent = Epsilon_greedy_MF(environment, **const.e_greedy_MF_param)
        setattr(agent, 'Q', init_Q)
        reward = play(environment, agent, **play_parameters)
        trial_name = ('social_basic','MF_on_MB', i)
        rewards[(trial_name)] = reward
    np.save('../data/MF_tables/reward_MF_on_MB.npy', rewards)


get_MF_performance_with_q_tables(nb_tests, play_parameters)


# ---------------------------------------------------------------------------- #
# Getting the rewards to plot
# ---------------------------------------------------------------------------- #

rewards = np.load('../data/MF_tables/rewards_many.npy',
                  allow_pickle=True).item()
# rewards = {(str(k[0]), str(k[1]), k[2]): v for k, v in rewards.items()}

rewards = {key: value for key, value in rewards.items()
            if key not in [('social_basic', 'Rmax_MB_soc', i)
                         for i in range(10)]}

MF_MB_rewards = np.load('../data/MF_tables/reward_MF_on_MB.npy',
                         allow_pickle=True).item()
# print(MF_MB_rewards.keys())
# MF_MB_rewards = {(str(k[0]), str(k[1]), k[2]): v for k, v in MF_MB_rewards.items()}


MF_MB_rewards = {('social_basic','MF_on_MB', key[2]): value for key, value 
                 in MF_MB_rewards.items() for i in range(10)}

print(MF_MB_rewards.keys())

rewards.update(MF_MB_rewards)

np.save('../data/MF_tables/all_rewards_MB_MF.npy', rewards)

print(rewards.keys())

# ---------------------------------------------------------------------------- #
# Plotting everything together
# ---------------------------------------------------------------------------- #

all_rewards = np.load('../data/MF_tables/all_rewards_MB_MF.npy',
                      allow_pickle=True).item()
# all_rewards = {(str(k[0]), str(k[1]), k[2]): v for k, v in all_rewards.items()}

# agent_to_test = ['MF_on_MB', 'e_greedy_MB', 'e_greedy_MF']
agent_to_test = ['MF_on_MB', 'Epsilon_greedy_MB', 'Epsilon_greedy_MF']
play_parameters = {'trials': 100000, 'max_step': 20}
environment_to_test = ['social_basic']
nb_tests = 8
total_steps = {agent_name: play_parameters['trials'] *
               play_parameters['max_step'] for agent_name in agent_to_test}

total_steps['MF_on_MB'] = 75000*20


mean, std = extracting_results(all_rewards, batches=100)


plot_curves(mean, std, total_steps)

