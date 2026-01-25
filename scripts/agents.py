
import numpy as np

# ---------Model-free agents--------------


class Epsilon_greedy_MF:

    def __init__(
            self,
            environment,
            gamma=0.95,
            alpha=0.7,
            epsilon=0.05):

        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.Q = np.zeros((self.size_environment, self.size_actions))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions)
        else:
            q_values = self.Q[state]
            max_indexes = np.flatnonzero(q_values == q_values.max())
            action = np.random.choice(max_indexes)
        return action

    def learn(self, old_state, reward, new_state, action):
        max_q = np.max(self.Q[new_state])
        delta = reward + self.gamma * max_q - self.Q[old_state][action]
        self.Q[old_state][action] = self.Q[old_state][action]+self.alpha*delta

# --------- End of Model-free agents--------------

# --------Model-based agents-------


class Basic_MB:

    def __init__(self,
                 environment,
                 gamma=0.95,
                 max_iterations=10000,
                 step_update=1):

        self.environment = environment
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)

        self.gamma = gamma
        self.max_iterations = max_iterations
        self.step_update = step_update

        self.shape_SA = (self.size_environment, self.size_actions)
        self.shape_SAS = (self.size_environment,
                          self.size_actions, self.size_environment)

        self.R = np.zeros(self.shape_SA)
        self.Rsum = np.zeros(self.shape_SA)
        self.R_VI = np.zeros(self.shape_SA)  # Reward for value iteration

        self.nSA = np.zeros(self.shape_SA)
        self.nSAS = np.zeros(self.shape_SAS)

        self.tSAS = np.ones(self.shape_SAS) / self.size_environment
        self.Q = np.zeros(self.shape_SA)

        self.counter = 0

    def choose_action(self, state):
        q_values = self.Q[state]
        max_indexes = np.flatnonzero(q_values == q_values.max())
        return np.random.choice(max_indexes)

    def learn_the_model(self, old_state, reward, new_state, action):
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward
        new_reward = self.Rsum[old_state][action] / self.nSA[old_state][action]
        self.R[old_state][action] = new_reward

    def learn(self, old_state, reward, new_state, action):

        self.learn_the_model(old_state, reward, new_state, action)
        self.compute_reward_VI(old_state, action)
        self.compute_transitions(old_state, action)
        self.counter += 1
        self.value_iteration()

    def compute_transitions(self, old_state, action):
        transitions = self.nSAS[old_state][action] / \
            self.nSA[old_state][action]
        self.tSAS[old_state][action] = transitions

    def compute_reward_VI(self, old_state, action):
        self.R_VI[old_state][action] = self.R[old_state][action]

    def value_iteration(self):
        if self.counter % self.step_update == 0:
            threshold = 1e-3
            converged = False
            nb_iters = 0
            while (not converged and nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

                diff = np.abs(self.Q - new_Q)
                self.Q = new_Q
                if np.max(diff) < threshold:
                    converged = True


class Epsilon_greedy_MB(Basic_MB):

    def __init__(self,
                 environment,
                 gamma,
                 epsilon,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, max_iterations, step_update)
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions)
        else:
            q_values = self.Q[state]
            action = np.random.choice(
                np.flatnonzero(q_values == q_values.max()))
        return action


class Rmax(Basic_MB):

    def __init__(self,
                 environment,
                 gamma,
                 Rmax,
                 m,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, max_iterations, step_update)

        self.Rmax = Rmax
        self.m = m
        self.R_VI = np.ones(self.shape_SA) * self.Rmax
        self.Q = np.ones(self.shape_SA) * self.Rmax / (1 - self.gamma)
        self.known_states = np.zeros(self.shape_SA)
        self.update_Q = False

    def compute_reward_VI(self, old_state, action):
        '''Use frequentist reward after m passages'''
        if self.nSA[old_state][action] == self.m:
            self.R_VI[old_state][action] = self.R[old_state][action]

    def compute_transitions(self, old_state, action):
        if self.nSA[old_state][action] == self.m:
            self.tSAS[old_state][action] = self.nSAS[old_state][action] / \
                self.nSA[old_state][action]
            self.known_states[old_state][action] = 1
            self.update_Q = True

    def value_iteration(self):
        if self.update_Q and self.counter % self.step_update == 0:
            threshold = 1e-3
            converged = False
            nb_iters = 0
            while not converged and (nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

                diff = np.abs(self.Q[self.known_states > 0] -
                              new_Q[self.known_states > 0])
                self.Q[self.known_states > 0] = new_Q[self.known_states > 0]
                if np.max(diff) < threshold:
                    converged = True
            self.Q_value_changed = False

# -------------End of MB agents-----------

# -------------Start of Belief-based MB agents-----------


class BeliefMB(Basic_MB):
    '''
    Model-based agent with belief over latent interaction stated.
    Belief is updated using Bayesian Inference based on observation.
    '''

    def __init__(self,
                 environment,
                 gamma=0.95,
                 num_latent_states=8,
                 max_iterations=10000,
                 step_update=1):
        
        super().__init__(environment,gamma, max_iterations, step_update)

        self.num_latent_states = num_latent_states
        self.belief = np.ones(num_latent_states)/num_latent_states

        self.latent_transition_counts = np.zeros((num_latent_states,
                                                  self.size_actions,
                                                  num_latent_states))
        self.latent_transition_model = np.ones((num_latent_states,
                                                self.size_actions,
                                                num_latent_states)) / num_latent_states
        
        self.prev_observation = None
        self.prev_latent_estimate = None

    def reset_belief(self):
        self.belief = np.ones(self.num_latent_states)/self.num_latent_states
    
    def compute_observation_likelihood(self, obs_human_moved, obs_human_rotated,
                                       obs_distance_changed, latent_state):
        '''
        Compute P(observation | latent_state, action)

        :param obs_human_moved: int (0 or 1)
        :param obs_human_rotated: int
        :param obs_distance_changed: int
        :param latent_state: int (0-7)
            The interaction_state value
        '''
        if latent_state <= 4:  # Vision states (not engaged)
            # Low probability of movement/rotation when not engaged
            p_moved = 0.1 if obs_human_moved == 1 else 0.9
            p_rotated = 0.2 if obs_human_rotated == 1 else 0.8
            p_dist_changed = 0.15 if obs_distance_changed == 1 else 0.85
        else:  # Engaged states (human_state 1-3)
            # Higher probability of movement/rotation when engaged
            engagement_level = latent_state - 4  # 1, 2, or 3
            p_moved = 0.2 + 0.25 * engagement_level if obs_human_moved == 1 else 0.8 - 0.25 * engagement_level
            p_rotated = 0.3 + 0.2 * engagement_level if obs_human_rotated == 1 else 0.7 - 0.2 * engagement_level
            p_dist_changed = 0.25 + 0.2 * engagement_level if obs_distance_changed == 1 else 0.75 - 0.2 * engagement_level
        
        # Independence assumption: P(o1, o2, o3 | s) = P(o1|s) * P(o2|s) * P(o3|s)
        return p_moved * p_rotated * p_dist_changed
    
    def get_latent_transition_prob(self, next_latent_state, current_latent_state, action):
        '''Get P(next_latent_state | current_latent_state, action) from learned model.'''
        return self.latent_transition_model[current_latent_state, action, next_latent_state]

    def learn_latent_transitions(self, prev_latent, action, next_latent):
        '''Update latent transition model based on observed state transitions.'''
        self.latent_transition_counts[prev_latent, action, next_latent] += 1

        total_count = np.sum(self.latent_transition_counts[prev_latent, action, next_latent])
        if total_count > 0:
            self.latent_transition_counts[prev_latent, action] = \
                self.latent_transition_counts[prev_latent,action] / total_count
            
    def update_belief(self, action, observation):
        '''
        Bayesian belief update: b_next[s'] ∝ O(o | s', a) * sum_s T(s' | s, a) * b[s]
        
        :param action: int
            The action taken
        :param observation: tuple (human_moved, human_rotated, distance_changed)
        '''
        human_moved, human_rotated, distance_changed = observation

        new_belief = np.zeros(self.num_latent_states)

        for next_state in range(self.num_latent_states):
            obs_linelihood = self.compute_observation_likelihood(
                human_moved,
                human_rotated,
                distance_changed,
                next_state
            )
        
            transition_sum = 0.0
            for current_state in range(self.num_latent_states):
                transition_prob = self.get_latent_transition_prob(
                    next_state,
                    current_state,
                    action
                )
                transition_sum += transition_prob * self.belief[current_state]

            new_belief[next_state] = obs_linelihood * transition_sum

        # Normalize
        belief_sum = np.sum(new_belief)
        if belief_sum > 0 :
            self.belief = new_belief / belief_sum
        else :
            self.belief = np.ones(self.num_latent_states) / self.num_latent_states

    def get_belief_state_estimate(self):
        '''Get the most likely latent state based on current belief.'''
        return np.argmax(self.belief)
    
    def learn(self, old_state, reward, new_state, action, observation=None):
        '''Extended learning that includes belief update and latent transition learning.'''

        super().learn(old_state, reward, new_state, action)

        # Update belief if observation provided
        if observation is not None :
            # prev_latent_estimate = self.get_belief_state_estimate()

            self.update_belief(action, observation)

            new_latent_estimate = self.get_belief_state_estimate()

            if self.prev_latent_estimate is not None :
                self.learn_latent_transitions(
                    self.prev_latent_estimate,
                    action, 
                    new_latent_estimate
                )

                self.prev_latent_estimate = new_latent_estimate
                self.prev_observation = observation

class Epsilon_BeliefMB(BeliefMB):
    """
    Epsilon-greedy version of Belief-based MB agent.
    """

    def __init__(self,
                 environment,
                 gamma=0.95,
                 epsilon=0.05,
                 num_latent_states=8,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, num_latent_states, 
                        max_iterations, step_update)
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions)
        else:
            q_values = self.Q[state]
            max_indexes = np.flatnonzero(q_values == q_values.max())
            action = np.random.choice(max_indexes)
        return action

class Rmax_BeliefMB(BeliefMB):
    """
    R-max version of Belief-based MB agent.
    """

    def __init__(self,
                 environment,
                 gamma=0.95,
                 Rmax=1.0,
                 m=5,
                 num_latent_states=8,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, num_latent_states,
                        max_iterations, step_update)

        self.Rmax = Rmax
        self.m = m
        self.R_VI = np.ones(self.shape_SA) * self.Rmax
        self.Q = np.ones(self.shape_SA) * self.Rmax / (1 - self.gamma)
        self.known_states = np.zeros(self.shape_SA)
        self.update_Q = False

    def compute_reward_VI(self, old_state, action):
        '''Use frequentist reward after m passages'''
        if self.nSA[old_state][action] == self.m:
            self.R_VI[old_state][action] = self.R[old_state][action]

    def compute_transitions(self, old_state, action):
        if self.nSA[old_state][action] == self.m:
            self.tSAS[old_state][action] = self.nSAS[old_state][action] / \
                self.nSA[old_state][action]
            self.known_states[old_state][action] = 1
            self.update_Q = True

    def value_iteration(self):
        if self.update_Q and self.counter % self.step_update == 0:
            threshold = 1e-3
            converged = False
            nb_iters = 0
            while not converged and (nb_iters < self.max_iterations):
                nb_iters += 1
                max_Q = np.max(self.Q, axis=1)
                new_Q = self.R_VI + self.gamma * np.dot(self.tSAS, max_Q)

                diff = np.abs(self.Q[self.known_states > 0] -
                              new_Q[self.known_states > 0])
                self.Q[self.known_states > 0] = new_Q[self.known_states > 0]
                if np.max(diff) < threshold:
                    converged = True
            self.update_Q = False

# -------------End of Belief-based MB agents-----------

# ------------MF on MB agents----------


class MFLearnerOnMB:
    def __init__(self, MB_agent, gamma=0.95, alpha=0.3):
        self.environment = MB_agent.environment
        self.MB_agent = MB_agent
        self.gamma = gamma
        self.alpha = alpha
        self.size_environment = len(self.environment.states)
        self.size_actions = len(self.environment.actions)
        self.shape_SA = (self.size_environment, self.size_actions)
        self.Q_MF = np.zeros(self.shape_SA) / (1 - self.gamma)
        self.Q = self.MB_agent.Q

    def learn(self, old_state, reward, new_state, action):
        '''Learning of the MB and of the MF agent'''
        self.MB_agent.learn(old_state, reward, new_state, action)

        self.learn_MF(old_state, reward, new_state, action)

        self.Q = self.MB_agent.Q

    def learn_MF(self, old_state, reward, new_state, action):
        '''Model-free update of one Q-value of the MF agent'''
        max_q = np.max(self.Q_MF[new_state])
        delta = reward + self.gamma * max_q - self.Q_MF[old_state][action]
        self.Q_MF[old_state][action] = self.Q_MF[old_state][action] + \
            self.alpha*delta

    def choose_action(self, state):
        '''The action is chosen by the MB agent'''
        return self.MB_agent.choose_action(state)

# ------------ End of MF on MB agents ----------


# ------------ Finite Horizon MB agents ----------

class FiniteHorizonMB(Basic_MB):

    def __init__(self,
                 environment,
                 gamma=0.95,
                 horizon=10,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, gamma, max_iterations, step_update)

        self.horizon = horizon

        # horizon tables
        self.shape_SAH = (self.size_environment,
                          self.size_actions, self.horizon)
        self.R_horizon = np.zeros(self.shape_SAH)
        self.nSA_horizon = np.zeros(self.shape_SAH, dtype='int')

    def learn_the_model(self, old_state, reward, new_state, action):
        '''Learns the model with respect to the horizon'''
        self.nSA[old_state][action] += 1
        self.nSAS[old_state][action][new_state] += 1
        self.Rsum[old_state][action] += reward

        index_update = int(self.nSA[old_state][action] % self.horizon)

        if self.nSA[old_state][action] > self.horizon:
            forget_state = self.nSA_horizon[old_state][action][index_update]
            forget_reward = self.R_horizon[old_state][action][index_update]

            self.Rsum[old_state][action] -= forget_reward
            self.nSAS[old_state][action][forget_state] -= 1

        self.nSA_horizon[old_state][action][index_update] = new_state
        self.R_horizon[old_state][action][index_update] = reward

        self.norm_factor = min(self.nSA[old_state][action], self.horizon)

        new_reward = self.Rsum[old_state][action] / self.norm_factor
        self.R[old_state][action] = new_reward

    def compute_transitions(self, old_state, action):
        transitions = self.nSAS[old_state][action] / self.norm_factor
        self.tSAS[old_state][action] = transitions


class Epsilon_MB_horizon(FiniteHorizonMB):
    def __init__(self,
                 environment,
                 gamma,
                 horizon,
                 epsilon,
                 max_iterations=10000,
                 step_update=1):

        super().__init__(environment, 
                         gamma, 
                         horizon, 
                         max_iterations, 
                         step_update)

        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.randint(self.size_actions)
        else:
            q_values = self.Q[state]
            max_indexes = np.flatnonzero(q_values == q_values.max())
            action = np.random.choice(max_indexes)
        return action


# ------------ End of Finite Horizon MB agents ----------
