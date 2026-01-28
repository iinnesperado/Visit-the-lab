from agents import Epsilon_greedy_MF, Rmax, Epsilon_greedy_MB, Rmax_BeliefMB, Epsilon_BeliefMB
from envs import Gridworld, Lab_env, Lab_env_HRI, GoToHumanVision, Human, Lab_env_HRI_LatentObs
from envs import Lab_HRI_evaluation, SocialGridworld
from environment_generation import Lab_structure
import constants as const


# ---------------------------------------------------------------------------- #
# Humans definition
# ---------------------------------------------------------------------------- #



human_parameters = {
    'basic_human': const.basic_human_param,
    'fast_human': const.fast_human_param,
    'hard_human': const.hard_human_param,
    'pointing_need_human': const.pointing_need_human_param,
    'basic_human_speed_2': const.basic_human_speed_2_param,
    'basic_human_speed_3': const.basic_human_speed_3_param,
    'basic_human_speed_random': const.basic_human_speed_random_param}

human_names = list(human_parameters.keys())

humans = {human_name: Human for human_name in human_names}


# ---------------------------------------------------------------------------- #
# Agents definition
# ---------------------------------------------------------------------------- #


agents = {'e_greedy_MB': Epsilon_greedy_MB,
          'e_greedy_MF': Epsilon_greedy_MF,
          'Rmax_MB_nav': Rmax,
          'Rmax_MB_soc': Rmax,
          'e_greedy_MB_no_explo': Epsilon_greedy_MB,
          'e_greedy_belief_MB' : Epsilon_BeliefMB,
          'Rmax_belief_MB' : Rmax_BeliefMB
          }

agent_names = list(agents.keys())

agent_params = {'e_greedy_MB': const.e_greedy_MB_param,
                'e_greedy_MF': const.e_greedy_MF_param,
                'Rmax_MB_nav': const.Rmax_MB_nav,
                'Rmax_MB_soc': const.Rmax_MB_soc,
                'e_greedy_MB_no_explo': const.e_greedy_MB_no_explo_param,
                'e_greedy_belief_MB': const.e_greedy_belief_MB_param,
                'Rmax_belief_MB': const.Rmax_belief_MB_param
                }

# ---------------------------------------------------------------------------- #
# Environments definition
# ---------------------------------------------------------------------------- #


envs = {'gridworld': Gridworld,
        'lab_nav': Lab_env,
        'social_basic': Lab_env_HRI,
        'social_hard': Lab_env_HRI,
        'social_latent': Lab_env_HRI_LatentObs,
        'social_fast': Lab_env_HRI,
        'social_basic_speed_2': Lab_env_HRI,
        'social_basic_speed_3': Lab_env_HRI,
        'social_basic_speed_random': Lab_env_HRI,
        'HRI_basic': Lab_HRI_evaluation,
        'HRI_hard': Lab_HRI_evaluation,
        'HRI_fast': Lab_HRI_evaluation,
        'go_to_h': GoToHumanVision,
        }


env_subparams = {
    'gridworld': {},
    'lab_nav': {'structure': {}},
    'social_basic': {'nav_env': {},
                     'human': const.basic_human_param},
    'social_latent': {'nav_env': {},
                      'human': const.basic_human_param},
    'social_basic_speed_2': {'nav_env': {},
                             'human': const.basic_human_speed_2_param},
    'social_basic_speed_3': {'nav_env': {},
                             'human': const.basic_human_speed_3_param},
    'social_basic_speed_random': {'nav_env': {},
                                  'human': const.basic_human_speed_random_param},
    'social_hard': {'nav_env': {},
                    'human': const.hard_human_param},
    'social_fast': {'nav_env': {},
                    'human': const.fast_human_param},
    'HRI_basic': {'nav_env': {},
                  'human': const.basic_human_param},
    'HRI_hard': {'nav_env': {},
                 'human': const.hard_human_param},
    'HRI_fast': {'nav_env': {},
                 'human': const.fast_human_param},
    'go_to_h': {'nav_env': {}}
}

env_subclasses = {
    'gridworld': {},
    'lab_nav': {'structure': Lab_structure},
    'social_basic': {'nav_env': SocialGridworld, 'human': Human},
    'social_latent': {'nav_env': SocialGridworld, 'human': Human},
    'social_hard': {'nav_env': SocialGridworld, 'human': Human},
    'social_fast': {'nav_env': SocialGridworld, 'human': Human},
    'social_basic_speed_2': {'nav_env': SocialGridworld, 'human': Human},
    'social_basic_speed_3': {'nav_env': SocialGridworld, 'human': Human},
    'social_basic_speed_random': {'nav_env': SocialGridworld, 'human': Human},
    'HRI_basic': {'nav_env': Gridworld, 'human': Human},
    'HRI_hard': {'nav_env': Gridworld, 'human': Human},
    'HRI_fast': {'nav_env': Gridworld, 'human': Human},
    'go_to_h': {'nav_env': Gridworld}
}


env_names = list(envs.keys())
