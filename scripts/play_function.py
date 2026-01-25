
import numpy as np
from visualization import Visualization
from multiprocessing import Pool
import constants as const
from variables import envs, env_names, agents, agent_names
from variables import env_subparams, env_subclasses, agent_params
import time


def play(environment,
         agent,
         trials=100,
         max_step=30,
         visual={'render': False,
                 'trial': [],
                 'task_type': "HRI"}):
    """Simulate a reinforcement learning agent on an environment in a 
    sequential task.

    Parameters
    ----------
    environment : _type_
        Any class from the envs.py file.
    agent : _type_
        Any agent from the agents.py file.
    trials : int, optional
        Number of trials to run.
    max_step : int, optional
        Maximal steps per trial.
    visual : dict, optional
        Permits to plot a 2D visualization of the agent on specific trials. 
        'render' is a boolean, if True, a plot is generated.
        'trial' is a list of the trials to plot.
        'task_type' is the type of the task_type, "navigation", "social" or 
        "HRI".

    Returns
    -------
    list
        Accumulated reward over all trials.
    """
    if visual['render']:
        # visualization_mode is "navigation", "social" or "HRI"
        visualization = Visualization(environment, visual['task_type'])
        visualization.clean_folder()  # remove the files from "tmp-img" folder
    reward_per_episode = []
    for trial in range(trials):
        cumulative_reward, step, game_over = 0, 0, False
        while not game_over:
            old_state = environment.agent_state
            action = agent.choose_action(old_state)
            reward, new_state, observation = environment.make_step(action)
            agent.learn(old_state, reward, new_state, action, observation)
            cumulative_reward += reward
            step += 1
            if visual['render']:
                if trial in visual['trial']:
                    background = visualization.get_background()
                    visualization.update_scenario(background,
                                                  environment,
                                                  action,
                                                  trial,
                                                  step)
            if step == max_step:
                game_over = True
                environment.new_episode()
        reward_per_episode.append(cumulative_reward)
    return reward_per_episode


def get_simulation_to_do(agent_to_test,
                         env_to_test,
                         nb_tests,
                         play_parameters,
                         starting_seed):
    simulation_to_do = []
    seed = starting_seed
    for agent_name in agent_to_test:
        for env_name in env_to_test:
            for iteration in range(nb_tests):
                trial_name = (env_name, agent_name, iteration)
                simulation_to_do.append({'trial_name': trial_name,
                                         'env_name': env_name,
                                         'agent_name': agent_name,
                                         'seed': seed,
                                         'play_parameters': play_parameters})

                seed += 1
    return simulation_to_do


def one_parameter_play_function(all_params_one_trial):

    seed = all_params_one_trial['seed']
    play_parameters = all_params_one_trial['play_parameters']
    env_name = all_params_one_trial['env_name']
    agent_name = all_params_one_trial['agent_name']
    trial_name = all_params_one_trial['trial_name']

    np.random.seed(seed)

    # Generate new subclasses for the environment definition
    sub_class_env = env_subclasses[env_name]
    sub_param_env = env_subparams[env_name]
    environment_parameters = {key: sub_class_env[key](**sub_param_env[key])
                              for key in sub_class_env.keys()}

    environment = envs[env_name](**environment_parameters)
    agent = agents[agent_name](environment, **agent_params[agent_name])

    return trial_name, play(environment, agent, **play_parameters)


def main_function(agent_to_test,
                  env_to_test,
                  nb_tests,
                  play_parameters,
                  starting_seed,
                  nb_processes=5):

    before = time.time()
    every_simulation = get_simulation_to_do(agent_to_test,
                                            env_to_test,
                                            nb_tests,
                                            play_parameters,
                                            starting_seed)
    pool = Pool(processes=nb_processes)
    results = pool.map(one_parameter_play_function, every_simulation)
    pool.close()
    pool.join()
    rewards = {}
    for result in results:
        rewards[result[0]] = result[1]
    time_after = time.time()
    print('Computation time: '+str(time_after - before))
    np.save('../data/all_rewards/'+str(before)+' .npy', rewards)
    return rewards
