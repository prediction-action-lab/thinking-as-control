import random
import numpy as np
import gym


class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 5
        self.max_steps = 20
        self.letter_goals = {
            5: (self.grid_size, self.grid_size),  # 'A'
            6: (1, 1),  # 'B'
            # 7: (5, 3),  # 'C'
        }
        self.letters = list(self.letter_goals.keys())
        self.action_meanings = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C"]
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.observation_space = gym.spaces.Dict(
            {
                "letter": gym.spaces.Discrete(len(self.letters)),  # 0 = A, 1 = B, 2 = C
                "position": gym.spaces.Box(
                    low=1, high=self.grid_size, shape=(2,), dtype=np.int32
                ),
            }
        )

    def reset(self):
        self.agent_pos = np.array(
            [np.random.randint(1, self.grid_size), np.random.randint(1, self.grid_size)]
        )
        # self.agent_pos = random.choice([np.array([4, 3]), np.array([1,2])])
        # self.agent_pos = np.array([1, 2])
        self.letter = random.choice(self.letters)
        self.goal = self.letter_goals[self.letter]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return {"letter": self.letter, "position": self.agent_pos.copy()}

    def step(self, action):
        self.steps += 1
        if action < 5:  # Directional action
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
            new_pos = self.agent_pos + np.array(delta)
            if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                self.agent_pos = new_pos

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def generate_expert_action(self):
        goal = np.array(self.goal)
        act = 5
        direction = goal - self.agent_pos
        if direction[0] < 0:
            act = 1
        elif direction[0] > 0:
            act = 2
        elif direction[1] < 0:
            act = 3
        elif direction[1] > 0:
            act = 4
        return act


class TwoStageGridWorldEnv(GridWorldEnv):
    def __init__(self):
        super(TwoStageGridWorldEnv, self).__init__()
        self.max_steps = 20
        self.letter_goals = {
            5: [(self.grid_size, self.grid_size)],  # 'A'
            6: [(1, 1)],  # 'B'
            # 7: None,  # 'C'
        }
        self.letter_goals[7] = [self.letter_goals[5][0], self.letter_goals[6][0]]
        self.letters = list(self.letter_goals.keys())
        self.action_meanings = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "C"]
        self.action_space = gym.spaces.Discrete(len(self.action_meanings))
        self.observation_space = gym.spaces.Dict(
            {
                "letter": gym.spaces.Discrete(len(self.letters)),
                "position": gym.spaces.Box(
                    low=1, high=self.grid_size, shape=(2,), dtype=np.int32
                ),
            }
        )

    def reset(self):
        self.agent_pos = np.array(
            [np.random.randint(1, self.grid_size), np.random.randint(1, self.grid_size)]
        )
        # self.agent_pos = random.choice([np.array([4, 3]), np.array([1,2])])
        # self.agent_pos = np.array([1, 2])
        self.letter = 7
        self.goal = self.letter_goals[self.letter]
        self.steps = 0
        self.goal_ind = 0
        return self._get_obs()

    def _get_obs(self):
        return {"letter": self.letter, "position": self.agent_pos.copy()}

    def step(self, action):
        self.steps += 1
        if action < 5:  # Directional action
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)][action - 1]
            new_pos = self.agent_pos + np.array(delta)
            if np.all((1 <= new_pos) & (new_pos <= self.grid_size)):
                self.agent_pos = new_pos

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal[self.goal_ind]:
            self.goal_ind += 1
            if len(self.goal) == self.goal_ind:
                reward = 1.0
                done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def generate_expert_action(self):
        goal = np.array(self.goal[self.goal_ind])
        act = 5
        # print(goal, self.agent_pos)
        direction = goal - self.agent_pos
        if direction[0] < 0:
            act = 1
        elif direction[0] > 0:
            act = 2
        elif direction[1] < 0:
            act = 3
        elif direction[1] > 0:
            act = 4
        return act
