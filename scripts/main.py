import numpy as np
from play_function import main_function
from graphics import extracting_results, plot_curves


def launch_and_plot(agent_to_test,
                    env_to_test,
                    nb_tests,
                    play_parameters,
                    starting_seed,
                    nb_processes=5,
                    condition='agent'):

    if condition == 'agent':
        total_steps = {agent_name: play_parameters['trials'] *
                       play_parameters['max_step'] for agent_name in agent_to_test}
    elif condition == 'social':
        total_steps = {name_cond: play_parameters['trials'] *
                       play_parameters['max_step'] for name_cond in env_to_test}
    else:
        raise ValueError('Wrong condition')

    rewards = main_function(agent_to_test,
                            env_to_test,
                            nb_tests,
                            play_parameters,
                            starting_seed,
                            nb_processes)

    mean, std = extracting_results(rewards, 100, condition)

    plot_curves(mean, std, total_steps)


def only_plot(rewards,
              agent_to_test,
              env_to_test,
              play_parameters,
              condition='agent'):

    if condition == 'agent':
        total_steps = {agent_name: play_parameters['trials'] *
                       play_parameters['max_step'] for agent_name in agent_to_test}
    elif condition == 'social':
        total_steps = {name_cond: play_parameters['trials'] *
                       play_parameters['max_step'] for name_cond in env_to_test}
    else:
        raise ValueError('Wrong condition')

    mean, std = extracting_results(rewards, 100, condition)

    plot_curves(mean, std, total_steps)


if __name__ == '__main__':
    # ---------------------------------------------------------------------------- #
    # Navigation
    # ---------------------------------------------------------------------------- #

    agent = ['Rmax_MB_nav', 'e_greedy_MB', 'e_greedy_MF']
    env = ['gridworld']
    nb_iters = 10
    play_params = {'trials': 25000, 'max_step': 20}
    start_seed = 1
    proc = 5
    cond = 'agent'

    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)

    # ---------------------------------------------------------------------------- #
    # Social
    # ---------------------------------------------------------------------------- #
    agent = ['Rmax_MB_soc', 'e_greedy_MB', 'e_greedy_MF']
    agent = ['e_greedy_MF']
    env = ['social_basic']
    nb_iters = 10
    play_params = {'trials': 100000, 'max_step': 20}
    start_seed = 2
    proc = 8
    cond = 'agent'

    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)

    # ---------------------------------------------------------------------------- #
    # Go to Human Vision
    # ---------------------------------------------------------------------------- #
    agent = ['Rmax_MB_soc', 'e_greedy_MB', 'e_greedy_MF']
    agent = ['e_greedy_MF']
    env = ['go_to_h']
    nb_iters = 10
    play_params = {'trials': 10000, 'max_step': 20}
    start_seed = 3
    proc = 10
    cond = 'agent'

    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)


    # ---------------------------------------------------------------------------- #
    # Three humans comparison - Social Task
    # ---------------------------------------------------------------------------- #

    agent = ['e_greedy_MB']
    env= ['social_basic', 'social_fast', 'social_hard']
    nb_iters = 10
    play_params = {'trials': 25000, 'max_step': 20}
    start_seed = 4
    proc = 8
    cond = 'social'


    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)

    # ---------------------------------------------------------------------------- #
    # Three humans comparison 2 - Social Task
    # ---------------------------------------------------------------------------- #

    agent = ['e_greedy_MB']
    env = ['social_basic', 'social_basic_speed_2', 'social_basic_speed_random' 
        'social_basic_speed_3']
    nb_iters = 10
    play_params = {'trials': 25000, 'max_step': 20}
    start_seed = 4
    proc = 10
    cond = 'social'


    launch_and_plot(agent,
                    env,
                    nb_iters,
                    play_params,
                    start_seed,
                    proc,
                    cond)
        

