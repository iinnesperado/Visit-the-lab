import numpy as np


# action_names = {0: "Up",
#                 1: "Down",
#                 2: "Left",
#                 3: "Right",
#                 4: "Up x2",
#                 5: "Down x2",
#                 6: "Left x2",
#                 7: "Right x2",
#                 8: "Up x3",
#                 9: "Down x3",
#                 10: "Left x3",
#                 11: "Right x3",
#                 12: "Up-Left ",
#                 13: "Up-Right",
#                 14: "Down-Left",
#                 15: "Down-Right",
#                 16: "Up-Left x2",
#                 17: "Up-Right x2",
#                 18: "Down-Left x2",
#                 19: "Down-Right x2",
#                 20: "Up-Left x3",
#                 21: "Up-Right x3",
#                 22: "Down-Left x3",
#                 23: "Down-Right x3",
#                 24: "Stay",
#                 25: "Look Human",
#                 26: "Hello!",
#                 27: "Come!",
#                 28: "Pointing",
#                 29: "Go to Human"}


class Gridworld():
    """
    Basic nav environment without walls, can be used as a parent 
    class for other environments.
    """

    def __init__(self, size=24):
        """_summary_

        Parameters
        ----------
        size : int, optional
            Gridworld is a squared environment of size * size states.
        """
        self.height = size
        self.width = size

        self.number_nav_states = self.height * self.width
        self.number_actions = 25
        self.nav_states = np.arange(self.number_nav_states)
        self.actions = np.arange(self.number_actions)

        self.walls = np.zeros((self.height, self.width, 4))
        self.walls[0, :, 0] = 1
        self.walls[:, 0, 2] = 1
        self.walls[:, self.width - 1, 3] = 1
        self.walls[self.height - 1, :, 1] = 1
        self.walls = self.walls.reshape(self.height * self.width, 4)

        self.labels_2D = np.zeros((self.height, self.width))
        for h in range(self.height):
            for w in range(self.width):
                h_3 = h//(self.height//3)
                w_3 = w//(self.width//3)
                self.labels_2D[h, w] = (h_3)*3+w_3
        self.labels = self.labels_2D.flatten()
        self.max_label = int(np.max(self.labels))

        self.compute_transitions()

        self.number_states = self.number_nav_states * (self.max_label+1)
        self.states = np.arange(self.number_states)

        self.new_episode()

    def compute_transitions(self):
        self.transitions = np.zeros((self.number_nav_states,
                                     self.number_actions,
                                     self.number_nav_states))

        self.effect_of_each_action = {0: -self.width,
                                      1: self.width,
                                      2: -1,
                                      3: 1}

        for action in range(12):
            self.transitions[:, action, :] = self.basic_action(action)

        for action in range(12, 24):
            self.transitions[:, action, :] = self.diagonal_action(action)

        self.transitions[:, 24, :] = np.identity(self.number_nav_states)

    def basic_action(self, number_action):
        """Take a basic action and return the associated transition array.
        """
        nb_iters = number_action // 4 + 1
        modif_state = self.effect_of_each_action[number_action % 4]
        action_array = np.identity(self.number_nav_states)
        for i in range(nb_iters):
            for x in range(self.number_nav_states):
                arrival_state = np.argmax(action_array[x])
                if self.walls[arrival_state, number_action % 4] != 1:
                    action_array[x, :] = np.zeros(self.number_nav_states)
                    action_array[x, (arrival_state + modif_state)] = 1
        return action_array

    def diagonal_action(self, action):
        """Take a diagonal action and return the associated transition array.
        Diagonal actions are numbered between 12 and 23 included, with four
        actions for each speed, 1, 2 and 3. The direction is given by % 4
        and the speed by // 4. We make sure that there is no wall on the 
        horizontal and vertical pathways, whether the agent starts to move
        vertically first and then horizontally or vice versa.
        """
        action_array = np.identity(self.number_nav_states)
        nb_iters = action // 4 - 2  # speed associated with the action
        # associate each diagonal action to two basic actions , e.g up-right
        diagonal_to_basic_action = {0: (0, 2),
                                    1: (0, 3),
                                    2: (1, 2),
                                    3: (1, 3)}
        vert, horiz = diagonal_to_basic_action[action % 4]
        vert_modif = self.effect_of_each_action[vert]
        horiz_modif = self.effect_of_each_action[horiz]
        for i in range(nb_iters):
            for x in range(self.number_nav_states):
                arrival_state = np.argmax(action_array[x])
                cond_vert = (self.walls[arrival_state, vert] != 1)
                cond_horiz = (self.walls[arrival_state, horiz] != 1)
                if cond_vert and cond_horiz:
                    vert_state = arrival_state + vert_modif
                    horiz_state = arrival_state + horiz_modif
                    cond_vert_horiz = (self.walls[vert_state, horiz] != 1)
                    cond_horiz_vert = (self.walls[horiz_state, vert] != 1)
                    if cond_vert_horiz and cond_horiz_vert:
                        arrival_state += vert_modif+horiz_modif
                        action_array[x, :] = np.zeros(self.number_nav_states)
                        action_array[x, arrival_state] = 1
        return action_array

    def new_episode(self):
        self.pos = np.random.randint(
            self.number_nav_states)
        self.rewarded_room = np.random.randint(self.max_label+1)
        self.agent_state = self.nav_state_to_state(
            self.pos, self.rewarded_room)
        self.rewards = (self.labels == self.rewarded_room).astype(float)

    def make_step(self, action):
        transition_probas = self.transitions[self.pos][action]
        self.pos = np.random.choice(
            self.nav_states, p=transition_probas)
        self.agent_state = self.nav_state_to_state(
            self.pos, self.rewarded_room)
        return self.rewards[self.pos], self.agent_state

    def nav_state_to_state(self, state, rewarded_room):
        return state + rewarded_room*self.number_nav_states


class Lab_env(Gridworld):
    """
    nav environment with different labelled rooms and walls,
    meant to represent a laboratory. Uses the structure randomly generated in 
    environment_generation.py.
    """

    def __init__(self, structure):
        self.height = structure.total_height
        self.width = structure.total_width

        self.number_nav_states = self.height*self.width
        self.number_actions = 25
        self.nav_states = np.arange(self.number_nav_states)
        self.actions = np.arange(self.number_actions)

        self.labels = structure.labels_1D
        self.walls = structure.walls_1D

        self.max_label = max(self.labels)
        self.number_states = self.number_nav_states * (self.max_label+1)
        self.states = np.arange(self.number_states)

        self.compute_transitions()
        self.new_episode()


class Gridworld_one_distance(Gridworld):

    def __init__(self):
        super().__init__()
        self.number_actions = 8
        self.actions = np.arange(self.number_actions)
        self.transitions = np.concatenate((
            self.transitions[:, :4, :], self.transitions[:, 12:16, :]), axis=1)


class SocialGridworld(Gridworld):
    def __init__(self, size=140):
        super().__init__(size)

    def compute_transitions(self):

        self.transitions = np.zeros((self.number_nav_states,
                                     self.number_actions), dtype = int)

        self.effect_of_each_action = {0: -self.width,
                                      1: self.width,
                                      2: -1,
                                      3: 1}

        for action in range(12):
            tmp_states = super().basic_action(action)
            self.transitions[:, action] = np.argmax(tmp_states, axis = 1)

        for action in range(12, 24):
            tmp_states = super().diagonal_action(action)
            self.transitions[:, action] = np.argmax(tmp_states, axis = 1)

        self.transitions[:, 24] = np.arange(self.number_nav_states)

    def new_episode(self):
        self.pos = np.random.randint(self.number_nav_states)
        self.rewarded_room = np.random.randint(self.max_label+1)
        self.agent_state = self.nav_state_to_state(self.pos, self.rewarded_room)
        self.rewards = (self.labels == self.rewarded_room).astype(float)

    def make_step(self, action):
        self.pos = self.transitions[self.pos][action]
        self.agent_state = self.pos + self.rewarded_room*self.number_nav_states
        return self.rewards[self.pos], self.agent_state


# ------Social environment-------


class SocialStructure():
    '''This class is not to be used alone, it is a base class for all the 
    classes which use social interaction.'''

    def __init__(self, nav_env):

        # Navigation
        self.nav_env = nav_env

        self.height = nav_env.height
        self.width = nav_env.width

        self.number_nav_states = nav_env.number_nav_states
        self.number_nav_actions = nav_env.number_actions
        self.nav_states = np.arange(self.number_nav_states)
        self.nav_actions = np.arange(self.number_nav_actions)
        self.nav_rewards = nav_env.rewards

        self.walls = nav_env.walls
        self.transitions = nav_env.transitions
        self.labels = nav_env.labels

        self.direction_to_action = np.array(
            [3, 13, 0, 12, 2, 14, 1, 15])
        self.action_to_direction = np.array(
            [2, 6, 4, 0] * 3 + [3, 1, 5, 7] * 3)

        # Social
        self.min_distance = 1
        self.danger_distance_min = self.min_distance + 1
        self.max_distance = 5
        self.danger_distance_max = self.max_distance - 1

        self.direction_to_action = np.array(
            [3, 13, 0, 12, 2, 14, 1, 15])
        self.action_to_direction = np.array(
            [2, 6, 4, 0] * 3 + [3, 1, 5, 7] * 3)

        # Setting up a new episode for an episodic task
        # self.new_episode()

    def multiD_to_1D(self, cardinalities, multiD_state):
        state_1D, multiply = 0, 1
        for index_state in range(len(multiD_state)):
            state_1D += multiD_state[index_state] * multiply
            multiply *= cardinalities[index_state]
        return state_1D

    def oneD_to_multiD(self, cardinalities, state_1D):
        multiD_state = []
        total_cardinal = np.prod(cardinalities)
        for cardinal in reversed(cardinalities):
            total_cardinal //= cardinal
            multiD_state.append(state_1D // total_cardinal)
            state_1D %= total_cardinal
        multiD_state.reverse()
        return np.array(multiD_state)

    def one_to_two_D(self, position):
        return (position // self.width, position % self.width)

    def two_to_one_D(self, position_2D):
        return position_2D[0] * self.width + position_2D[1]

    def new_episode(self):

        self.pos = np.random.randint(self.number_nav_states)
        self.human_pos = np.random.randint(self.number_nav_states)
        self.orientation = np.random.randint(8)
        self.human_orientation = np.random.randint(8)

        # self.update_position()
        # self.update_vision()

    def update_position(self):

        self.robot_pos_2D = self.one_to_two_D(self.pos)
        self.human_pos_2D = self.one_to_two_D(self.human_pos)

        # Change of referential : x <- y , y <- - x
        self.relative_pos = (-self.robot_pos_2D[1] + self.human_pos_2D[1],
                             +self.robot_pos_2D[0] - self.human_pos_2D[0])

        # Computing human-robot distance
        dist_vert = abs(self.relative_pos[0])
        dist_horiz = abs(self.relative_pos[1])
        self.distance = max(dist_vert, dist_horiz)
        self.all_distances = np.array([self.min_distance,
                                       self.danger_distance_min,
                                       self.danger_distance_max,
                                       self.max_distance])
        dist_array = self.distance > self.all_distances
        self.distance_for_the_robot = np.sum(dist_array)
        greater_min = self.distance > self.min_distance
        smaller_max = self.distance <= self.max_distance
        self.appropriate_distance = greater_min and smaller_max

        # Computing human-robot position angle
        pos_angle_radian = np.arctan2(self.relative_pos[1],
                                      self.relative_pos[0])
        tmp_angle = (pos_angle_radian / (np.pi / 4)) % 8
        self.relative_pos_angle = np.round(tmp_angle, 2)
        # A bit weird - to debug
        tmp_angle_2 = np.round(self.relative_pos_angle, 0) % 8
        self.position_angle = (tmp_angle_2).astype(int)

    def update_vision(self):
        tmp_angle = self.human_orientation-self.relative_pos_angle
        human_vision_angle = (tmp_angle-4) % 8

        self.H_sees_R = human_vision_angle <= 2 or human_vision_angle >= 6
        self.H_looks_at_R = human_vision_angle <= 1 or human_vision_angle >= 7

        robot_vision_angle = (self.orientation - self.relative_pos_angle) % 8

        self.R_looks_at_H = robot_vision_angle <= 1 or robot_vision_angle >= 7

        self.seeing_each_other = self.H_sees_R and self.R_looks_at_H
        self.looking_at_each_other = self.H_looks_at_R and self.R_looks_at_H

# ----------------changements fait pour obs latente de interation_state----------------------
class Lab_env_HRI(SocialStructure):

    def __init__(self, 
                 nav_env, 
                 human, 
                 random_human_pos=False, 
                 deterministic = True):

        super().__init__(nav_env)

        # INTERACTION
        self.human = human
        self.random_human_pos = random_human_pos
        self.deterministic = deterministic
        # Observations: [human_moved, human_rotated, distance_changed, distance, required_direction, position_angle]
        self.cardinalities = np.array([2, 2, 2, 5, 8, 8])

        self.number_states = np.prod(self.cardinalities)

        self.number_actions = 30
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)

        # GENERAL
        self.state_counter = np.zeros(self.number_states)
        self.step = 0

        # Vars track previous state
        self.prev_human_pos = None
        self.prev_human_orientation = None
        self.prev_distance = None

        # Setting up a new episode for an episodic task
        self.new_episode()

    def new_episode(self):

        self.human_state = 0

        self.first_orientation = np.random.randint(8)
        self.first_orientation_human = np.random.randint(8)
        self.orientation = self.first_orientation
        self.human_orientation = self.first_orientation_human

        self.pos = np.random.randint(self.number_nav_states)
        if self.random_human_pos:
            self.human_pos = np.random.randint(self.number_nav_states)
        else:
            self.human_pos = self.height // 2 * self.width + self.width // 2

        random_direction = np.random.randint(8)
        self.new_required_direction(random_direction)

        self.update_position()
        self.update_vision()

        # Init prev state
        self.prev_human_pos = self.human_pos
        self.prev_human_orientation = self.human_orientation
        self.prev_distance = self.distance_for_the_robot

        self.update_agent_state()

    def new_required_direction(self, direction: int):

        self.required_direction = direction
        basic_action = self.direction_to_action[self.required_direction]

        self.human_required_actions = [basic_action,
                                       basic_action + 4,
                                       basic_action + 8]
        

    def update_agent_state(self):
        if self.looking_at_each_other:
            self.vision_state = 4
        elif self.seeing_each_other:
            self.vision_state = 3
        elif self.H_looks_at_R:
            self.vision_state = 2
        elif self.H_sees_R:
            self.vision_state = 1
        else:
            self.vision_state = 0

        self.interaction_state = self.vision_state
        if self.human_state != 0:
            self.interaction_state = self.human_state + 4

        # Obs changes in human behaviour
        human_moved = int(self.human_pos != self.prev_human_pos)
        human_rotated = int(self.human_orientation != self.prev_human_orientation)
        distance_changed = int(self.distance_for_the_robot != self.prev_distance)

        multiD_agent_state = np.array([human_moved,
                                       human_rotated,
                                       distance_changed,
                                       self.distance_for_the_robot,
                                       self.required_direction,
                                       self.position_angle])
        self.agent_state = self.multiD_to_1D(self.cardinalities,
                                             multiD_agent_state)

        self.state_counter[self.agent_state] += 1

    def make_step(self, action):
        # Store previous state before making changes
        self.prev_human_pos = self.human_pos
        self.prev_human_orientation = self.human_orientation
        self.prev_distance = self.distance_for_the_robot

        # nav action (change in position and in orientation)
        if action < 24:

            if self.deterministic :
                self.pos = self.transitions[self.pos][action]
            else : 
                probas = self.transitions[self.pos][action]
                self.pos = np.random.choice(self.nav_states,
                                        p=probas)
                
            self.orientation = self.action_to_direction[action]

        # Looking at human (change in orientation)
        if action == 25:
            self.orientation = self.position_angle
        # Teleport the robot to the human visual field, with random orientation.
        # The position is the one reached doing a two step movement from the
        # human position in the direction vector of the human visual field
        if action == 29:
            # Only if the human is not engaged
            if self.human_state == 0:

                two_step = self.direction_to_action[self.human_orientation] + 4

                if self.deterministic : 
                    self.pos = self.transitions[self.human_pos][two_step]
                else : 
                    probabilities = self.transitions[self.human_pos][two_step]
                    goal_state = np.random.choice(self.nav_states, p=probabilities)

                    # Teleporting the robot to the desired state
                    self.pos = goal_state
                self.orientation = np.random.randint(8)

        self.human_state = self.human.update_state(
            action,
            self.H_sees_R,
            self.H_looks_at_R,
            self.R_looks_at_H,
            self.appropriate_distance,
            self.human_orientation)

        # the human_action is updated depending on the robot action
        self.human_action = self.human.update_action(self.position_angle,
                                                     action)

        self.human_orientation = self.human.update_vision(self.position_angle)


        if self.deterministic : 
            self.human_pos = self.transitions[self.human_pos][self.human_action]
        else : 
            probas_pos_human = self.transitions[self.human_pos][self.human_action]
            self.human_pos = np.random.choice(self.nav_states, p=probas_pos_human)
        self.update_position()
        self.update_vision()
        self.update_agent_state()

        reward = self.get_reward()

        observation = (self.human_pos, self.human_orientation, self.distance_for_the_robot)

        return reward, self.agent_state, observation

    def get_reward(self):
        cond_human_action = self.human_action in self.human_required_actions
        cond_human_state = self.human_state == 3
        if cond_human_action and cond_human_state:
            reward = 1
            # if np.random.random() < 1/5 :
            #     random_direction = np.random.randint(8)
            #     self.new_required_direction(random_direction)
            #     multiD_agent_state = np.array([self.interaction_state,
            #                                self.distance_for_the_robot,
            #                                self.required_direction,
            #                                self.position_angle])
            #     self.agent_state = self.multiD_to_1D(self.cardinalities,
            #                                      multiD_agent_state)
        else:
            reward = 0
        if np.random.random() < 1/5 :
            random_direction = np.random.randint(8)
            self.new_required_direction(random_direction)

            human_moved = int(self.human_pos != self.prev_human_pos)
            human_rotated = int(self.human_orientation != self.prev_human_orientation)
            distance_changed = int(self.distance_for_the_robot != self.prev_distance)

            multiD_agent_state = np.array([human_moved,
                                       human_rotated,
                                       distance_changed,
                                       self.distance_for_the_robot,
                                       self.required_direction,
                                       self.position_angle])
            self.agent_state = self.multiD_to_1D(self.cardinalities,
                                             multiD_agent_state)
        return reward




class Lab_HRI_evaluation(Lab_env_HRI):

    def __init__(self, nav_env, human):
        super().__init__(nav_env, human, True, False)
        self.labels = self.nav_env.labels
        self.max_label = self.nav_env.max_label
        self.rewards = self.nav_rewards
        self.rewarded_room = None
        self.new_goal()

    def new_goal(self):
        rewarded_room_before = self.rewarded_room
        while self.rewarded_room == rewarded_room_before:
            self.rewarded_room = np.random.randint(self.max_label)

        self.rewards = (self.labels == self.rewarded_room).astype(float)
        self.label_state_human = self.nav_state_to_state(self.human_pos,
                                                         self.rewarded_room)
        self.label_state = self.nav_state_to_state(self.pos,
                                                   self.rewarded_room)

    def get_reward(self):
        # The agent is rewarded if the human reaches a rewarded position
        reward = self.rewards[self.human_pos]
        if reward > 0:
            self.new_goal()

        return reward

    def nav_state_to_state(self, state, label):
        return state + label*self.number_nav_states


# ----- Go to human vision class-------

class GoToHumanVision(SocialStructure):

    def __init__(self, nav_env):
        super().__init__(nav_env)

        # INTERACTION

        self.cardinalities = np.array([8, 8, 5])

        self.number_states = np.prod(self.cardinalities)
        self.number_actions = 25
        self.states = np.arange(self.number_states)
        self.actions = np.arange(self.number_actions)

        self.rewards = np.zeros(self.number_nav_states)

        # GENERAL
        self.state_counter = np.zeros(self.number_states)
        self.step = 0

        # Setting up a new episode for an episodic task
        self.new_episode()

    def new_episode(self):

        self.human_state = 0

        self.required_direction = np.random.randint(8)
        self.orientation = np.random.randint(8)
        self.human_orientation = np.random.randint(8)

        self.pos = np.random.randint(self.number_nav_states)
        self.human_pos = np.random.randint(self.number_nav_states)
        # self.human_pos = self.height // 2 * self.width + self.width // 2
        self.update_position()
        self.update_vision()
        self.update_agent_state()

    def make_step(self, action):
        if action < 24:
            self.pos = np.random.choice(
                self.nav_states, p=self.transitions[self.pos][action])
            self.orientation = self.action_to_direction[action]

        self.update_position()
        self.update_vision()

        self.update_agent_state()

        reward = self.get_reward()

        return reward, self.agent_state

    def get_reward(self):
        appropriate_distance = self.distance_for_the_robot in [1, 2, 3]

        if self.H_looks_at_R and appropriate_distance:
            reward = 1
        else:
            reward = 0
        return reward

    def update_agent_state(self):

        multiD_agent_state = np.array([self.position_angle,
                                       self.human_orientation,
                                       self.distance_for_the_robot])
        self.agent_state = self.multiD_to_1D(self.cardinalities,
                                             multiD_agent_state)

# --------End of Go to human vision--------

# ----Human Class used for social environments------


class Human:

    def __init__(self,
                 speeds=[1, 0, 0],
                 failing_rate=0.1,
                 pointing_need=0.5,
                 losing_attention=0.05,
                 orientation_change_rate=0.1,
                 random_movement=0.05):

        self.speeds = speeds
        self.failing_rate = failing_rate
        self.pointing_need = pointing_need
        self.losing_attention = losing_attention
        self.orientation_change_rate = orientation_change_rate
        self.random_movement = random_movement

        self.direction_to_action = np.array(
            [3, 13, 0, 12, 2, 14, 1, 15])
        self.action_to_direction = np.array(
            [2, 6, 4, 0] * 3 + [3, 1, 5, 7] * 3)

        self.human_state = 0

    def going_in_a_direction(self, direction):

        speed = np.random.choice(np.arange(3), p=self.speeds)
        direction_action_human = self.direction_to_action[direction]
        action_of_the_human = (direction_action_human + speed * 4).astype(int)
        return action_of_the_human

    def update_state(self,
                     robot_action,
                     H_sees_R,
                     H_looks_R,
                     R_looks_H,
                     appropriate_distance,
                     human_orientation):

        # Loss of attention from the human
        attention_loss = np.random.random() < self.losing_attention
        if not appropriate_distance or attention_loss:
            self.human_state = 0

        # Hello action
        appropriate_vision = H_sees_R and R_looks_H
        if robot_action == 26 and appropriate_distance and appropriate_vision:
            successful_hello = np.random.random() > self.failing_rate
            if H_looks_R or successful_hello:
                self.human_state = max(1, self.human_state)

        # Come action
        elif robot_action == 27 and self.human_state == 1:
            pointing_success = int(np.random.random() > self.pointing_need)
            self.human_state = 2 + pointing_success

        # Pointing action
        elif robot_action == 28 and self.human_state >= 2:
            self.human_state = 3

        self.orientation = human_orientation

        return self.human_state

    def update_vision(self, position_angle):

        # random orientation change when the human is not engaged
        orientation_change = np.random.random() < self.orientation_change_rate
        if orientation_change and self.human_state == 0:
            self.orientation = np.random.randint(8)

        # The human's gaze follows the robot after the hello action
        if self.human_state >= 1:
            self.orientation = (position_angle + 4) % 8

        return self.orientation

    def update_action(self, position_angle, robot_action):

        # The human does not move in general
        self.action = 24

        # If not engaged, it can move randomly
        noise_movement = np.random.random() < self.random_movement
        if noise_movement and self.human_state == 0:
            self.action = self.going_in_a_direction(np.random.randint(8))
            self.orientation = self.action_to_direction[self.action]

        # Hello, come and pointing can make the human stop moving
        # else it moves towards the robot when engaged
        stop_actions = [26, 27, 28]
        if self.human_state == 3 and robot_action not in stop_actions:
            self.action = self.going_in_a_direction((position_angle + 4) % 8)

        return self.action

# ---End of human class------
