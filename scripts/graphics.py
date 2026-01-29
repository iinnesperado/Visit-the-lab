import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_mean_and_std(dictionary, condition='agent'):
    """Compute the mean and the standard error of the mean of 
    a dictionnary of results."""
    keys = list(dictionary.keys())
    nb_iters = max(map(lambda x: x[2], keys)) + \
        1  # gets the max of the iterations
    # gets all the environments
    all_env = list(np.unique(list(map(lambda x: x[0], keys))))
    # gets all the agents
    all_agents = list(np.unique(list(map(lambda x: x[1], keys))))

    if condition == 'agent':
        mean = {name_agent: np.mean([dictionary[env, name_agent, i]
                                     for i in range(nb_iters) for env in all_env], axis=0)
                for name_agent in all_agents}

        std = {name_agent: np.std([dictionary[env, name_agent, i]
                                   for i in range(nb_iters) for env in all_env], axis=0)
               for name_agent in all_agents}
    else:
        mean = {name_env: np.mean([dictionary[name_env, name_agent, i]
                                   for i in range(nb_iters) for name_agent in all_agents], axis=0)
                for name_env in all_env}
        std = {name_env: np.std([dictionary[name_env, name_agent, i]
                                 for i in range(nb_iters) for name_agent in all_agents], axis=0)
               for name_env in all_env}
    return (mean, std)


def reducing_with_batches(array, batch_numbers):
    if len(array) % batch_numbers != 0:
        print("Batches are not be equal.")
    batch_size = len(array) // batch_numbers
    return np.mean(array[:batch_size*batch_numbers].reshape(-1, batch_size), axis=1)


def get_moving_average(rewards, avg=100):
    index_avg = [i * len(rewards) // avg for i in range(avg)]

    moving_average_rewards = [np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg)-1)]
    moving_average_rewards.append(np.mean(rewards[index_avg[-1]:]))

    shift_index = (index_avg[1]-index_avg[0])/2
    index_avg = [i+shift_index for i in index_avg]
    return index_avg, moving_average_rewards


def extracting_results(dic_of_rewards, batches=100, condition='agent'):
    mean, std = get_mean_and_std(dic_of_rewards, condition=condition)
    reduced_mean = {name: reducing_with_batches(
        table, batch_numbers=batches) for name, table in mean.items()}
    reduced_std = {name: reducing_with_batches(
        table, batch_numbers=batches) for name, table in std.items()}

    return reduced_mean, reduced_std


def plot_curves(means, stds, all_agent_steps, condition='agent', title=''):
    """Plot the results"""

    colors = {'e_greedy_MB': 'tab:blue',
              'e_greedy_MF': 'tab:green',
              'Epsilon_greedy_MB': 'tab:blue',
              'Epsilon_greedy_MF': 'tab:green',
              'Rmax_MB_nav': 'tab:orange',
              'Rmax_MB_soc': 'tab:orange',
              'MF_on_MB': 'tab:gray',
              'social_basic': 'tab:blue',
              'social_fast': 'tab:orange',
              'social_hard': 'tab:green',
              'social_slow': 'tab:blue',
              'social_no_pointing': 'tab:gray',
              'social_basic_speed_2':'black',
              'social_basic_speed_3':'tab:red',
              'social_basic_speed_random':'tab:green'}

    fig = plt.figure(dpi=300)
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.grid(linestyle='--')

    condition_tested = list(means.keys())
    # length_trial = len(means[agents_tested[0]])
    # x_range = [i * total_steps/length_trial for i in range(length_trial)]
    for idx_cond, name_cond in enumerate(condition_tested):

        length_trial = len(means[name_cond])
        x_range = [(i+1/2) * all_agent_steps[name_cond] /
                   length_trial for i in range(length_trial)]
        if name_cond == 'MF_on_MB':
            shift = 25000*20
            x_range = np.array(x_range)+shift
            plt.axvline(shift, color=colors[name_cond], linestyle='--')
        yerr0 = means[name_cond] - stds[name_cond]
        yerr1 = means[name_cond] + stds[name_cond]

        plt.fill_between(x_range, yerr0, yerr1,
                         color=colors[name_cond], alpha=0.25)

        plt.plot(x_range, means[name_cond],
                 color=colors[name_cond], label=name_cond)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    plt.legend()
    if title != '':
        plt.title(title)
    plt.savefig('../all_imgs/1D-plots/' + str(time.time()) +
                '.pdf', bbox_inches='tight')


def get_max_Q_values_and_policy(table):
    best_values = np.max(table, axis=1)
    random_noise = 1e-5 * np.random.random(table.shape)
    best_actions = np.argmax(table + random_noise, axis=1)
    return best_values, best_actions


def plot_2D(table, environment, path=''):
    plt.figure(dpi=300)
    table = np.reshape(table, (environment.height, environment.width))
    sns.heatmap(table, cmap='crest', cbar=False, annot=table, fmt='.1f',
                annot_kws={"size": 35 / (np.sqrt(len(table)) + 2.5)})
    plt.xticks([])
    plt.yticks([])
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()


def get_mean_and_std_list(rewards):
    return np.mean(rewards, axis=0), np.std(rewards, axis=0)


def plot_vs_Q(q_table_MB,
              q_table_MF,
              q_table_MFMB,
              task_name,
              best_q_per_task,
              name_agent,
              title='',
              path=''):

    best_q = best_q_per_task
    normalized_q_MF = q_table_MF / best_q
    normalized_q_MFMB = q_table_MFMB / best_q
    normalized_q_MB = q_table_MB / best_q
    print(best_q)
    idx = np.argsort(best_q)

    list_splits = np.round(list(set(list(best_q))), 2)
    print(list_splits)

    best_q = np.round(best_q, 2)
    packs_by_value = [np.where(best_q == list_splits[i])[0]
                      for i in range(len(list_splits))]

    norm_sort_MF = [np.array(normalized_q_MF)[idx_value]
                    for idx_value in packs_by_value]
    norm_sort_MB = [np.array(normalized_q_MB)[idx_value]
                    for idx_value in packs_by_value]
    norm_sort_MFMB = [np.array(normalized_q_MFMB)[idx_value]
                      for idx_value in packs_by_value]

    plt.figure(figsize=(18, 6), dpi=300)
    titles = {0: name_agent, 1: "MF", 2: "MF on " + name_agent}
    data = {0: norm_sort_MB, 1: norm_sort_MF, 2: norm_sort_MFMB}
    for i in range(3):
        plt.subplot(1, 3, i+1)
        sns.set(style="whitegrid")

        sns.boxplot(data=data[i])

        plt.title(titles[i])
        plt.xticks(range(len(list_splits)), list_splits)
        plt.xlabel('V*(s)', fontsize=12)
        plt.ylabel('V(s)/V*(s)', fontsize=12)
        plt.ylim(-0.05, 1.05)
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_vs_distance(q_table_MB,
                     q_table_MF,
                     q_table_MFMB,
                     task_name,
                     best_q_per_task,
                     distances,
                     name_agent,
                     title='',
                     path=''):

    best_q = best_q_per_task
    normalized_q_MF = q_table_MF / best_q
    normalized_q_MFMB = q_table_MFMB / best_q
    normalized_q_MB = q_table_MB / best_q

    max_distance = np.max(distances)

    list_splits = np.arange(1, max_distance+1)
    print(list_splits)

    best_q = np.round(best_q, 2)
    packs_by_value = [np.where(distances == list_splits[i])[0]
                      for i in range(len(list_splits))]

    norm_sort_MF = [np.array(normalized_q_MF)[idx_value]
                    for idx_value in packs_by_value]
    norm_sort_MB = [np.array(normalized_q_MB)[idx_value]
                    for idx_value in packs_by_value]
    norm_sort_MFMB = [np.array(normalized_q_MFMB)[idx_value]
                      for idx_value in packs_by_value]

    plt.figure(figsize=(18, 6), dpi=300)
    titles = {0: name_agent, 1: "MF", 2: "MF on " + name_agent}
    data = {0: norm_sort_MB, 1: norm_sort_MF, 2: norm_sort_MFMB}
    for i in range(3):
        plt.subplot(1, 3, i+1)
        sns.set(style="whitegrid")

        sns.boxplot(data=data[i])

        plt.title(titles[i])
        plt.xticks(range(len(list_splits)), list_splits)
        plt.xlabel('Number of 1-step action to get a reward', fontsize=12)
        plt.ylabel('V(s)/V*(s)', fontsize=12)
        plt.ylim(-0.05, 1.05)
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_with_std(list_rewards_MB,
                  list_rewards_MF,
                  list_rewards_MBMF,
                  path='',
                  title=''):

    mean_MB, std_MB = get_mean_and_std_list(list_rewards_MB)
    mean_MF, std_MF = get_mean_and_std_list(list_rewards_MF)
    mean_MBMF, std_MBMF = get_mean_and_std_list(list_rewards_MBMF)
    print(len(mean_MB))
    index_MB, mean_moving_MB = get_moving_average(mean_MB)
    _, std_moving_MB = get_moving_average(std_MB)

    index_MF, mean_moving_MF = get_moving_average(mean_MF)
    _, std_moving_MF = get_moving_average(std_MF)

    index_MBMF, mean_moving_MBMF = get_moving_average(mean_MBMF)
    _, std_moving_MBMF = get_moving_average(std_MBMF)

    plt.figure(dpi=300)
    plt.rc('axes', axisbelow=True)
    plt.grid()

    colors = {0: '#1E88E5', 1: '#FFC107', 2: '#D81B60'}

    plt.plot(index_MB[: -1], mean_moving_MB,
             color=colors[0], linewidth=3, label="MB")
    plt.plot(index_MF[:-1], mean_moving_MF,
             color=colors[1], linewidth=3, label="MF")
    plt.plot(np.array(index_MBMF[:-1])+max(index_MBMF),
             mean_moving_MBMF,
             color=colors[2], linewidth=3, label="MF on MB")

    plt.axvline(max(index_MBMF), color='red', linestyle='--')

    plt.fill_between(index_MB[:-1],
                     np.array(mean_moving_MB) - np.array(std_moving_MB),
                     np.array(mean_moving_MB) + np.array(std_moving_MB),
                     color=colors[0],
                     alpha=0.2)
    plt.fill_between(index_MF[:-1],
                     np.array(mean_moving_MF) - np.array(std_moving_MF),
                     np.array(mean_moving_MF) + np.array(std_moving_MF),
                     color=colors[1],
                     alpha=0.2)

    plt.fill_between(np.array(index_MBMF[:-1])+max(index_MBMF),
                     np.array(mean_moving_MBMF) - np.array(std_moving_MBMF),
                     np.array(mean_moving_MBMF) + np.array(std_moving_MBMF),
                     color=colors[2],
                     alpha=0.2)

    plt.ylabel('Mean Reward', fontsize=12)
    plt.xlabel('Trial (x20 for steps)', fontsize=12)
    plt.legend(loc='upper left')
    if title != '':
        plt.title(title)
    if path != '':
        plt.savefig(path, bbox_inches='tight')
    plt.show()


def plot_1D(rewards, path=''):
    # plt.figure(dpi=300)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    index_avg = [i * len(rewards) // 20 for i in range(20)]
    moving_median = np.array([np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg) - 1)])
    quartile = np.array([0.6745 * np.std(rewards[index_avg[index]:index_avg[index + 1]])
                         for index in range(len(index_avg) - 1)])
    plt.fill_between(index_avg[:-1], moving_median - quartile,
                     moving_median + quartile, color='tab:blue', alpha=0.2)
    plt.plot(index_avg[:-1], moving_median, 'tab:blue',
             linewidth=3)
    # plt.scatter(np.arange(len(rewards)), rewards, color='tab:blue', marker='+', s=4, alpha=0.4)
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Trial', fontsize=12)
    # plt.legend(loc='lower right')
    if path != '':
        plt.savefig(path)
    plt.show()


def basic_plot(rewards):
    index_avg = [i * len(rewards) // 20 for i in range(20)]
    moving_average_rewards = [np.mean(rewards[index_avg[index]:index_avg[index + 1]])
                              for index in range(len(index_avg) - 1)]
    plt.grid()
    plt.plot(index_avg[: -1], moving_average_rewards,
             color='tab:blue', linewidth=3, label="Easy Human")
    plt.ylabel('Reward', fontsize=12)
    plt.xlabel('Trial', fontsize=12)
    plt.show()


def plot_different_humans(all_rewards, title=''):

    plt.figure(dpi=300)
    colors = {'e_greedy_MB': 'tab:blue',
              'e_greedy_MF': 'tab:green',
              'Rmax_MB_nav': 'tab:orange',
              'Rmax_MB_soc': 'tab:orange',
              'MF_on_MB': 'tab:gray',
              'social_basic': 'tab:blue',
              'social_fast': 'tab:orange',
              'social_hard': 'tab:green',
              'social_slow': 'tab:blue',
              'social_no_pointing': 'tab:green',
              'social_basic_speed_2':'black',
              'social_basic_speed_3':'tab:red',
              'fast_human': 'tab:orange',
              'basic_human': 'tab:blue',
              'pointing_need_human': 'tab:grey',
              'hard_human': 'tab:red',
              'basic_human_speed_2': 'black',
              'basic_human_speed_3': 'tab:red',
              'social_basic_speed_random':'tab:green',
              'basic_human_speed_random':'tab:green'}
    
    for name, reward in all_rewards.items():
        mean_reward, std_reward = np.mean(
            reward, axis=0), np.std(reward, axis=0)
        index_avg, avg_rewards = get_moving_average(mean_reward, avg=50)
        _, avg_std = get_moving_average(std_reward, avg=50)

        plt.plot(index_avg, avg_rewards,
                 color=colors[name],
                 label=name)

        plt.fill_between(index_avg,
                         np.array(avg_rewards) - np.array(avg_std),
                         np.array(avg_rewards) + np.array(avg_std),
                         color=colors[name],
                         alpha=0.2)
    plt.ylabel('#Goals reached')
    plt.xlabel('#Trials (Steps x100)')
    plt.legend(loc='upper left')
    plt.grid(linestyle='--')
    if title != '':
        plt.title(title)
    plt.savefig("../all_imgs/1D-plots/three_humans"+str(time.time())+".pdf")
