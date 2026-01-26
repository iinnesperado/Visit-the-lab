import numpy as np
import time
import os
import copy

from environment_generation import Lab_structure
from envs import Lab_env, Gridworld, GoToHumanVision
from envs import Lab_env_HRI, Human, Lab_HRI_evaluation
from agents import Epsilon_greedy_MF, Rmax, Epsilon_greedy_MB, MFLearnerOnMB
from nav_interaction import NavigationInteraction
from graphics import get_max_Q_values_and_policy, plot_with_std
from graphics import plot_2D, plot_1D, basic_plot, plot_vs_Q, plot_vs_distance

# from visualization import save_gif, save_mp4

from play_function import play

from tqdm import tqdm
from visualization import Visualization


import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import itertools

# Find a good Epsilon for Epsilon MF

# navigation
"""
nb_trials_navigation = 50000
nb_steps_navigation = 20
navigation_environment = Gridworld()
navigation_agent = Epsilon_greedy_MF(navigation_environment, gamma=0.9, epsilon=0.05, alpha=0.1)

navigation_rewards = play(navigation_environment, navigation_agent,
                          nb_trials_navigation, nb_steps_navigation)

best_q_values_navigation, best_actions_navigation = get_max_Q_values_and_policy(navigation_agent.Q)
np.save("some results/Epsilon_MF/navigation/V_navigation_2.npy", best_q_values_navigation)
np.save("some results/Epsilon_MF/navigation/action_navigation_2.npy", best_actions_navigation)
np.save("some results/Epsilon_MF/navigation/navigation_rewards.npy", navigation_rewards)

for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment,
            "some results/Epsilon_MF/navigation/navigation_"+str(i)+".pdf")


plot_1D(navigation_rewards, navigation_agent, task="Navigation Epsilon MF",
        path="some results/Epsilon_MF/navigation/reward_navigation.pdf")

"""
"""
# social

nb_trials_social = 400000
nb_steps_social = 20

navigation_environment = Gridworld()

human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.05,
              orientation_change_rate=0.2, random_movement=0.1)

social_environment = Lab_environment_HRI(navigation_environment, human)


social_agent = Epsilon_greedy_MF(social_environment, gamma=0.9, epsilon=0.05, alpha=0.1)
social_rewards = play(social_environment, social_agent, nb_trials_social, nb_steps_social)

best_q_values_social, best_actions_social = get_max_Q_values_and_policy(social_agent.Q)
np.save("some results/Epsilon_MF/social/V_social_0_01.npy", best_q_values_social)
np.save("some results/Epsilon_MF/social/reward_social_0_01.npy", social_rewards)
np.save("some results/Epsilon_MF/social/best_actions_0_01.npy", best_actions_social)

plot_1D(social_rewards, social_agent, task="Social",
        path="some results/Epsilon_MF/social/reward_social_0_01.pdf")

"""

# Get V*(s) for both tasks

"""
# navigation
nb_trials_navigation = 30000
nb_steps_navigation = 20
navigation_environment = Gridworld()
navigation_agent = Rmax(navigation_environment, gamma=0.9,
                        m=1, Rmax=1, max_iterations=1, step_update=1000)

navigation_rewards = play(navigation_environment, navigation_agent,
                          nb_trials_navigation, nb_steps_navigation)

best_q_values_navigation, best_actions_navigation = get_max_Q_values_and_policy(navigation_agent.Q)
np.save("some results/opt_q_values/navigation/opt_V_navigation_2.npy", best_q_values_navigation)
np.save("some results/opt_q_values/navigation/opt_action_navigation_2.npy", best_actions_navigation)


for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment,
            "some results/opt_q_values/navigation/opt_V_navigation_2_"+str(i)+".pdf")


plot_1D(navigation_rewards, navigation_agent, task="Navigation")



# social

nb_trials_social = 200000
nb_steps_social = 20

navigation_environment = Gridworld()

human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.05,
              orientation_change_rate=0.2, random_movement=0.1)

social_environment = Lab_environment_HRI(navigation_environment, human)


social_agent = Epsilon_greedy_MB(social_environment, gamma=0.9,
                                 epsilon=0.1, max_iterations=1, step_update=1000)
social_rewards = play(social_environment, social_agent, nb_trials_social, nb_steps_social)

best_q_values_social, best_actions_social = get_max_Q_values_and_policy(social_agent.Q)
np.save("some results/opt_q_values/social/opt_V_social_2.npy", best_q_values_social)
np.save("some results/opt_q_values/social/opt_action_social_2.npy", best_actions_social)

plot_1D(social_rewards, social_agent, task="Social")


# MF learner on MB / navigation



# What to save

navigation_environment = Gridworld()
MB_agent = Rmax(navigation_environment, gamma=0.9, Rmax=1,
                m=1, max_iterations=1, step_update=1000)
navigation_agent_1 = MFLearnerOnMB(MB_agent, gamma=0.9, alpha=0.1)
navigation_agent_2 = Epsilon_greedy_MF(navigation_environment, gamma=0.9, alpha=0.1, epsilon=0.05)

number_of_trials = 10000
number_of_steps = 20
pictures = []

navigation_rewards_1_1 = play(navigation_environment, navigation_agent_1,
                              number_of_trials, number_of_steps, pictures, "navigation")

navigation_rewards_2_1 = play(navigation_environment, navigation_agent_2,
                              number_of_trials, number_of_steps, pictures, "navigation")

for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent_1.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment)


for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent_1.Q_MF[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment)


for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent_2.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment)


np.save("starting_Q_table_nav.npy", navigation_agent_1.Q_MF)
starting_Q = np.load("starting_Q_table_nav.npy")

navigation_agent_3 = Epsilon_greedy_MF(navigation_environment, gamma=0.9,
                                       alpha=0.2, epsilon=0.05, initial_Q=starting_Q)

number_of_trials = 10000
number_of_steps = 20
pictures = []

navigation_rewards_1_2 = play(navigation_environment, navigation_agent_1,
                              number_of_trials, number_of_steps, pictures)


navigation_rewards_2_2 = play(navigation_environment, navigation_agent_2,
                              number_of_trials, number_of_steps, pictures)

navigation_rewards_3_2 = play(navigation_environment, navigation_agent_3,
                              number_of_trials, number_of_steps, pictures)

navigation_rewards_3_1 = [0]*len(navigation_rewards_3_2)

navigation_rewards_1 = navigation_rewards_1_1 + navigation_rewards_1_2
navigation_rewards_2 = navigation_rewards_2_1 + navigation_rewards_2_2
navigation_rewards_3 = navigation_rewards_3_1 + navigation_rewards_3_2


# plot_rewards_agents(rewards, rewards2, navigation_agent, agent2, task = 'Navigation')

plot_1D(navigation_rewards_1, navigation_agent_1, task='Navigation')
plot_1D(navigation_rewards_2, navigation_agent_2, task='Navigation')
plot_1D(navigation_rewards_3, navigation_agent_3, task='Navigation')

plot_rewards_agents(navigation_rewards_1,
                    navigation_rewards_2,
                    navigation_rewards_3)


for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent_1.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment)


for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent_3.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment)


for i in range(int(navigation_environment.max_label)+1):
    q_values, best_actions = get_max_Q_values_and_policy(
        navigation_agent_2.Q[i*navigation_environment.number_navigation_states:(i+1)*navigation_environment.number_navigation_states])
    plot_2D(q_values, navigation_environment)
"""

# Compare the agents

nb_iters = 8
starting_seed = 1

# agents = ['Rmax', 'Epsilon_greedy_MB', 'Epsilon_greedy_MF']
agents = ['Epsilon_greedy_MB', 'Epsilon_greedy_MF']

# agents = ['VI_softmax']
agents_with_MF = ['Rmax', 'Epsilon_greedy_MB']
passive_MF_agent_parameters = {'alpha': 0.5, 'gamma': 0.9}
# environments = ["navigation", "social"]
environments = ["navigation"]
# environments = ["social"]

navigation_environment = Gridworld()
human = Human(speeds=[0.5, 0.5, 0], failing_rate=0.05, pointing_need=0.5, losing_attention=0.05,
              orientation_change_rate=0.2, random_movement=0.1)
social_environment = Lab_env_HRI(navigation_environment, human)

environments_name = {"navigation": navigation_environment, "social": social_environment}
play_parameters = {"navigation": {'trials': 20000, 'nb_steps': 20},
                   "social": {'trials': 40000, 'nb_steps': 20}}

agent_names = {'Rmax': Rmax,
               'Epsilon_greedy_MB': Epsilon_greedy_MB,
               'Epsilon_greedy_MF': Epsilon_greedy_MF}


agent_parameters = {
    'Epsilon_greedy_MB': {'gamma': 0.9, 'epsilon': 0.05, 'max_iterations': 1, 'step_update': 1000},
    'Rmax': {'gamma': 0.9, 'm': 1, 'Rmax': 1, 'max_iterations': 1, 'step_update': 1000},
    'VI_softmax': {'gamma': 0.9, 'tau': 0.1, 'max_iterations': 1, 'step_update': 1000},
    'Epsilon_greedy_MF': {'gamma': 0.9, 'epsilon': 0.05, 'alpha': 0.5}}


def get_max_Q_values_and_policy(table):
    best_values = np.max(table, axis=1)
    best_actions = np.argmax(table + 1e-5 * np.random.random(table.shape), axis=1)
    return best_values, best_actions


def play_multiprocess(environment, agent, name_agent, name_environment, trials=100, nb_steps=30):
    start_time = time.time()
    reward_per_episode = []
    for trial in range(trials):
        cumulative_reward, step, game_over = 0, 0, False
        while not game_over:
            if trial == trials - 1 and step == 0:
                mid_time = time.time()
                directory_name = generate_directory_name(
                    "data_MFMB/"+name_agent+'_'+name_environment)
                max_q_values, best_actions = get_max_Q_values_and_policy(agent.Q)
                np.save(directory_name+"/max_q_values_halfway.npy", max_q_values)
                if agent.__class__.__name__ == "MFLearnerOnMB":
                    np.save(directory_name+"/MF_qvalues_halfway.npy", agent.Q_MF)
                end_mid_time = time.time()
            old_state = environment.agent_state
            action = agent.choose_action(old_state)
            reward, new_state = environment.make_step(action)
            agent.learn(old_state, reward, new_state, action)
            cumulative_reward += reward
            step += 1
            if step == nb_steps:
                game_over = True
                environment.new_episode()
        reward_per_episode.append(cumulative_reward)
    stop_time = time.time()
    total_time = stop_time - end_mid_time + mid_time - start_time
    max_q_values, best_actions = get_max_Q_values_and_policy(agent.Q)
    np.save(directory_name+"/rewards.npy", reward_per_episode)
    np.save(directory_name+"/max_q_values.npy", max_q_values)
    np.save(directory_name+"/time.npy", total_time)
    np.save(directory_name+"/best_actions.npy", best_actions)
    if agent.__class__.__name__ == "MFLearneronMB":
        np.save(directory_name+"MF_qvalues.npy", agent.Q_MF)

    return reward_per_episode, stop_time - start_time


def generate_directory_name(name, x=1):
    dir_name = (name + ('_'+str(x) if x != 0 else '')).strip()
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        return dir_name
    else:
        return generate_directory_name(name, x + 1)


def getting_simulations_to_do(*args):
    return [i for i in itertools.product(*args)]


def play_with_params(play_parameters, seed, agent_parameters, simulation_to_do):
    np.random.seed(seed)
    name_environment, name_agent, iteration = simulation_to_do[:3]
    environment = environments_name[name_environment]
    agent = agent_names[name_agent](environment, **agent_parameters[name_agent])
    if name_agent in agents_with_MF:
        agent = MFLearnerOnMB(agent, **passive_MF_agent_parameters)
    return simulation_to_do, play_multiprocess(environment, agent, name_agent, name_environment,
                                               **play_parameters[name_environment])


def evaluate_agents(environments, agents, nb_iters, play_params, agent_parameters, starting_seed):
    """Launch the experiment, extract the results and plot them."""
    every_simulation = getting_simulations_to_do(environments, agents, range(nb_iters))
    all_seeds = [starting_seed+i for i in range(len(every_simulation))]
    rewards, times = main_function(all_seeds, every_simulation, play_params, agent_parameters)
    return rewards, times
    # rewards = extracting_results(
    # rewards, environments, agents, nb_iters)
    # save_and_plot(optimal, real, rewards, agents, environments,
    # play_params, environments, agent_parameters)


def one_parameter_play_function(all_params):
    return play_with_params(all_params[0], all_params[1], all_params[2], all_params[3])


def get_mean_and_SEM(dictionary, name_env, agents_tested, nb_iters):
    """Compute the mean and the standard error of the mean of a dictionnary of results."""
    mean = {name_agent: np.average([dictionary[env, name_agent, i]
                                    for i in range(nb_iters) for env in name_env], axis=0)
            for name_agent in agents_tested}
    SEM = {name_agent: (np.std([dictionary[env, name_agent, i]
                               for env in name_env for i in range(nb_iters)], axis=0)
           / np.sqrt(nb_iters*len(name_env))) for name_agent in agents_tested}

    return (mean, SEM)


def get_agent_parameters(name_agent, basic_parameters, list_of_new_parameters):
    """Generate all the agent parameters to test during parameter fitting."""
    agent_parameters = []
    for dic in list_of_new_parameters:
        d = copy.deepcopy(basic_parameters)
        for key, value in dic.items():
            d[name_agent][key] = value
        agent_parameters.append(d)
    return agent_parameters


def main_function(all_seeds, every_simulation, play_params, agent_parameters):
    """Run many play functions in parallel using multiprocessing"""
    before = time.time()
    if type(agent_parameters) == dict:
        all_parameters = [[play_params, all_seeds[index_seed], agent_parameters,
                           every_simulation[index_seed]] for index_seed in range(len(all_seeds))]
    else:
        all_parameters = [[play_params, all_seeds[index_seed], agent_parameters[index_seed],
                           every_simulation[index_seed]] for index_seed in range(len(all_seeds))]
    # pool = Pool(cpu_count()-1)
    pool = Pool(4)
    results = pool.map(one_parameter_play_function, all_parameters)
    pool.close()
    pool.join()
    rewards, times = {}, {}
    for result in results:
        rewards[result[0]] = result[1][0]
        times[result[0]] = result[1][1]
    time_after = time.time()
    print('Computation time: '+str(time_after - before))
    return rewards, times

if __name__ == '__main__':
    rewards, times = evaluate_agents(environments, agents, nb_iters, play_parameters,
                                    agent_parameters, starting_seed)
    np.save('data/MF_tables/rewards_many.npy', rewards)
    


# Compute the values on the new MF agents


def generate_MF_tables(agents_with_MF, environments, play_parameters):
    for i in range(11, 21):
        for agent_name in agents_with_MF:
            for env_name in environments:
                path_to_save = 'data_MFMB/'+agent_name + '_' + env_name + '_' + str(i)
                starting_Q = np.load(path_to_save+'/MF_qvalues_halfway.npy')
                environment = environments_name[env_name]
                agent = Epsilon_greedy_MF(environment, gamma=0.9,
                                          alpha=0.5, epsilon=0.05, initial_Q=starting_Q)
                rewards = play(environment, agent,
                               **play_parameters[env_name])
                np.save(path_to_save+'/rewards_MBMF_agent.npy', rewards)
                np.save(path_to_save+'/final_Q_MBMF_agent.npy', agent.Q)


def plot_all_rewards_curves(agents_with_MF, environments):
    for agent_name in agents_with_MF:
        for env_name in environments:
            all_rewards_MB, all_rewards_MF, all_rewards_MBMF = [], [], []
            for i in range(1, 11):
                path_to_save = 'data_MFMB/'+agent_name + '_' + env_name + '_' + str(i)
                path_MF = 'data_MFMB/Epsilon_greedy_MF'+'_' + env_name + '_' + str(i)
                rewards_MB = np.load(path_to_save+'/rewards.npy')
                rewards_MF = np.load(path_MF+'/rewards.npy')
                rewards_MBMF = np.load(path_to_save+'/rewards_MBMF_agent.npy')
                all_rewards_MB.append(rewards_MB)
                all_rewards_MF.append(rewards_MF)
                all_rewards_MBMF.append(rewards_MBMF)
                # plot_rewards_agents(rewards_MB, rewards_MF, rewards_MBMF,
                #                     path=path_to_save+'/rewards_comparison.pdf',
                #                     title='MF performance with '+agent_name+' in the '+env_name + ' task')
            plot_with_std(all_rewards_MB, all_rewards_MF, all_rewards_MBMF,
                          path='data_MFMB/'+agent_name + '_' + env_name+'_rewards_comparison.png',
                          title='')


def plot_Q_curves(agents_with_MF):
    for agent_name in agents_with_MF:
        env_name = "navigation"
        big_MF, big_MB, big_MBMF = [], [], []
        big_best, big_MF_half, big_MB_half, big_MBMF_half = [], [], [], []
        for i in range(1, 11):
            path_to_save = 'data_MFMB/'+agent_name + '_' + env_name + '_' + str(i)
            path_MF = 'data_MFMB/Epsilon_greedy_MF'+'_' + env_name + '_' + str(i)
            q_table_MF = np.load(path_MF+'/max_q_values.npy')
            q_table_MF_halfway = np.load(path_MF+'/max_q_values_halfway.npy')
            q_table_MB = np.load(path_to_save+'/max_q_values.npy')
            q_table_MB_halfway = np.load(path_to_save+'/max_q_values_halfway.npy')

            q_table_MBMF_halfway, _ = get_max_Q_values_and_policy(
                np.load(path_to_save+'/MF_qvalues_halfway.npy'))
            q_table_MBMF, _ = get_max_Q_values_and_policy(
                np.load(path_to_save+'/final_Q_MBMF_agent.npy'))
            best_q_per_task = np.load('some results/opt_q_values/navigation/opt_V_navigation.npy')

            big_MF += list(q_table_MF)
            big_MB += list(q_table_MB)
            big_MBMF += list(q_table_MBMF)
            big_MF_half += list(q_table_MF_halfway)
            big_MB_half += list(q_table_MB_halfway)
            big_MBMF_half += list(q_table_MBMF_halfway)
            big_best += list(best_q_per_task)

        big_MF = np.array(big_MF)
        big_MB = np.array(big_MB)
        big_MBMF = np.array(big_MBMF)
        big_MF_half = np.array(big_MF_half)
        big_MB_half = np.array(big_MB_half)
        big_MBMF_half = np.array(big_MBMF_half)
        big_best = np.array(big_best)

        plot_vs_Q(big_MB, big_MF, big_MBMF, env_name, big_best, name_agent=agent_name,
                  path='data_MFMB/'+agent_name+'_q_comparison.png')
        plot_vs_Q(big_MB_half, big_MF_half, big_MBMF_half, env_name, big_best, name_agent=agent_name,
                  path='data_MFMB/'+agent_name+'_q_comparison_halfway.png')


def plot_2D_maps(agents_with_MF):
    for agent_name in agents_with_MF:
        env_name = "navigation"
        for i in range(1, 2):
            path_to_save = 'data_MFMB/'+agent_name + '_' + env_name + '_' + str(i)
            q_table_MB = np.load(path_to_save+'/max_q_values.npy')
            q_table_MB_halfway = np.load(path_to_save+'/max_q_values_halfway.npy')
            q_table_MBMF_halfway, _ = get_max_Q_values_and_policy(
                np.load(path_to_save+'/MF_qvalues_halfway.npy'))
            q_table_MBMF, _ = get_max_Q_values_and_policy(
                np.load(path_to_save+'/final_Q_MBMF_agent.npy'))
            step = len(q_table_MB)//9

            path_MF = 'data_MFMB/Epsilon_greedy_MF'+'_' + env_name + '_' + str(i)
            q_table_MF = np.load(path_MF+'/max_q_values.npy')
            q_table_MF_halfway = np.load(path_MF+'/max_q_values_halfway.npy')
            name_to_table = {"q_table_MB": q_table_MB,
                             "q_table_MB_halfway": q_table_MB_halfway,
                             "q_table_MBMF_halfway": q_table_MBMF_halfway,
                             "q_table_MBMF": q_table_MBMF,
                             "q_table_MF": q_table_MF,
                             "q_table_MF_halfway": q_table_MF_halfway}
            for i in range(9):
                all_paths = {"q_table_MB": path_to_save + '/tmp1/2D_plot_'+str(i)+'.png',
                             "q_table_MB_halfway": path_to_save + '/tmp1/2D_plot_halfway_'+str(i)+'.png',
                             "q_table_MBMF_halfway": path_to_save + '/tmp1/2D_plot_MBMF_halfway_'+str(i)+'.png',
                             "q_table_MBMF": path_to_save + '/tmp1/2D_plot_MBMF_'+str(i)+'.png',
                             "q_table_MF": path_MF+'/tmp1/2D_plot_MF_'+str(i)+'.png',
                             "q_table_MF_halfway": path_MF+'/tmp1/2D_plot_MF_halfway'+str(i)+'.png'}
                for table_name in ["q_table_MB", "q_table_MB_halfway", "q_table_MBMF_halfway", "q_table_MBMF", "q_table_MF", "q_table_MF_halfway"]:
                    q_values = name_to_table[table_name][i*step:(i+1)*step]
                    plot_2D(q_values, navigation_environment, path=all_paths[table_name])


def cluster_per_distance():
    distance_qs = np.round(np.max(np.load("data/q_table_navigation_one_distance.npy"), axis=1), 2)
    get_all_distances = np.unique(distance_qs)
    get_ordered_distances = {get_all_distances[i]: np.argsort(
        -get_all_distances)[i] for i in range(len(get_all_distances))}
    all_distances = np.vectorize(get_ordered_distances.get)(distance_qs)+1
    return all_distances


def plot_distances(agents_with_MF):
    for agent_name in agents_with_MF:
        env_name = "navigation"
        big_MF, big_MB, big_MBMF = [], [], []
        big_best, big_MF_half, big_MB_half, big_MBMF_half = [], [], [], []
        for i in range(11, 21):
            path_to_save = 'data_MFMB/'+agent_name + '_' + env_name + '_' + str(i)
            path_MF = 'data_MFMB/Epsilon_greedy_MF'+'_' + env_name + '_' + str(i)
            # q_table_MF = np.load(path_MF+'/max_q_values.npy')
            q_table_MF_halfway = np.load(path_MF+'/max_q_values_halfway.npy')
            # q_table_MB = np.load(path_to_save+'/max_q_values.npy')
            q_table_MB_halfway = np.load(path_to_save+'/max_q_values_halfway.npy')

            q_table_MBMF_halfway, _ = get_max_Q_values_and_policy(
                np.load(path_to_save+'/MF_qvalues_halfway.npy'))
            # q_table_MBMF, _ = get_max_Q_values_and_policy(
            #     np.load(path_to_save+'/final_Q_MBMF_agent.npy'))
            best_q_per_task = np.load('some results/opt_q_values/navigation/opt_V_navigation.npy')

            # big_MF += list(q_table_MF)
            # big_MB += list(q_table_MB)
            # big_MBMF += list(q_table_MBMF)
            big_MF_half += list(q_table_MF_halfway)
            big_MB_half += list(q_table_MB_halfway)
            big_MBMF_half += list(q_table_MBMF_halfway)
            big_best += list(best_q_per_task)

        big_MF = np.array(big_MF)
        big_MB = np.array(big_MB)
        big_MBMF = np.array(big_MBMF)
        big_MF_half = np.array(big_MF_half)
        big_MB_half = np.array(big_MB_half)
        big_MBMF_half = np.array(big_MBMF_half)
        big_best = np.array(big_best)
        all_distances = cluster_per_distance()
        # plot_vs_distance(big_MB, big_MF, big_MBMF, env_name, big_best, all_distances,
        #                  agent_name,
        #                  path='data_MFMB/article/'+agent_name+'_distance_comparison.pdf')
        plot_vs_distance(big_MB_half, big_MF_half, big_MBMF_half, env_name, big_best, all_distances, name_agent=agent_name,
                         path='data_MFMB/article/'+agent_name+'_distance_comparison_halfway.pdf')
