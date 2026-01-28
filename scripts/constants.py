# HUMANS
# argparse
# scripts bash
#


# -----------------------------------------------------------------------------#
# Parameters Human
# -----------------------------------------------------------------------------#

# Speeds

SPEED_1 = [1, 0, 0]
SPEED_2 = [0, 1, 0]
SPEED_3 = [0, 0, 1]

SPEED_FAST = [0, 0.5, 0.5]
SPEED_RANDOM = [0.33, 0.33, 0.34]

# Hello Failing Rate

NO_FAIL_HELLO = 0
SMALL_FAIL_HELLO = 0.05
HALF_FAIL_HELLO = 0.5

# Pointing need

NO_POINTING = 0
MEDIUM_POINTING = 0.5
POINTING = 1

# Losing attention

NO_ATTENTION_LOSS = 0
SMALL_ATTENTION_LOSS = 0.05
HIGH_ATTENTION_LOSS = 0.1

# Orientation change rate

SMALL_CHANGE_VISION = 0.1
MEDIUM_CHANGE_VISION = 0.15
HIGH_CHANGE_VISION = 0.3

# Random movement

SMALL_RANDOM_MVT = 0.05
MEDIUM_RANDOM_MVT = 0.1
HIGH_RANDOM_MVT = 0.2


basic_human_param = {'speeds': SPEED_1,
                     'failing_rate': NO_FAIL_HELLO,
                     'pointing_need': NO_POINTING,
                     'losing_attention': NO_ATTENTION_LOSS,
                     'orientation_change_rate': SMALL_CHANGE_VISION,
                     'random_movement': MEDIUM_RANDOM_MVT}

hard_human_param = {'speeds': SPEED_RANDOM,
                    'failing_rate': HALF_FAIL_HELLO,
                    'pointing_need': MEDIUM_POINTING,
                    'losing_attention': HIGH_ATTENTION_LOSS,
                    'orientation_change_rate': HIGH_CHANGE_VISION,
                    'random_movement': HIGH_RANDOM_MVT}


fast_human_param = {'speeds': SPEED_FAST,
                    'failing_rate': SMALL_FAIL_HELLO,
                    'pointing_need': NO_POINTING,
                    'losing_attention': SMALL_ATTENTION_LOSS,
                    'orientation_change_rate': MEDIUM_CHANGE_VISION,
                    'random_movement': SMALL_RANDOM_MVT}

pointing_need_human_param = fast_human_param.copy()
pointing_need_human_param['pointing_need'] = POINTING

basic_human_speed_2_param = basic_human_param.copy()
basic_human_speed_2_param['speeds'] = SPEED_2

basic_human_speed_3_param = basic_human_param.copy()
basic_human_speed_3_param['speeds'] = SPEED_3

basic_human_speed_random_param = basic_human_param.copy()
basic_human_speed_3_param['speeds'] = SPEED_RANDOM

# -----------------------------------------------------------------------------#
# Parameters Agents
# -----------------------------------------------------------------------------#

GAMMA = 0.9
EPSILON = 0.05
MAX_ITERATIONS = 1
STEP_UPDATE = 1000
RMAX = 1
M_NAV = 1
M_SOC = 5
ALPHA = 0.5


e_greedy_MB_param = {'gamma': GAMMA,
                     'epsilon': EPSILON,
                     'max_iterations': MAX_ITERATIONS,
                     'step_update': STEP_UPDATE}

e_greedy_MF_param = {'gamma': GAMMA,
                     'epsilon': EPSILON,
                     'alpha': ALPHA}

Rmax_MB_nav = {'gamma': GAMMA,
               'Rmax': RMAX,
               'm': M_NAV,
               'max_iterations': MAX_ITERATIONS,
               'step_update': STEP_UPDATE}

Rmax_MB_soc = Rmax_MB_nav.copy()
Rmax_MB_soc['m'] = M_SOC

e_greedy_MB_no_explo_param = e_greedy_MB_param.copy()
e_greedy_MB_no_explo_param['epsilon'] = 0

e_greed_MF_no_decision_param = {'gamma': GAMMA,
                                'alpha': ALPHA}

e_greedy_belief_MB_param = {'gamma': GAMMA,
                             'epsilon': EPSILON,
                             'num_latent_states': 8,
                             'max_iterations': MAX_ITERATIONS,
                             'step_update': STEP_UPDATE}

Rmax_belief_MB_param = {'gamma': GAMMA,
                         'Rmax': RMAX,
                         'm': M_SOC,
                         'num_latent_states': 8,
                         'max_iterations': MAX_ITERATIONS,
                         'step_update': STEP_UPDATE}
