import random
import numpy as np
import gym


# BASE SETTING
# GRID_SIZE = 5
# MAX_STEPS = 20

GRID_SIZE = 5
MAX_STEPS = (GRID_SIZE // 2) * 2 + 2 * GRID_SIZE + 6


class GridWorldEnv(gym.Env):
    def __init__(self, n_goals=2):
        super(GridWorldEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS
        self.deterministic_start = False
        self.letter_goals = {
            5: (self.grid_size, self.grid_size),  # 'A'
        }
        if n_goals > 1:
            self.letter_goals[6] = (1, 1)  # 'C'
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
        if self.deterministic_start:
            self.agent_pos = [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
        else:
            self.agent_pos = np.array(
                [
                    np.random.randint(1, self.grid_size),
                    np.random.randint(1, self.grid_size),
                ]
            )
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
        self.grid_size = GRID_SIZE
        self.max_steps = MAX_STEPS
        self.deterministic_start = True
        self.letter_goals = {
            5: [(self.grid_size, self.grid_size)],  # 'A'
            6: [(1, 1)],  # 'B'
        }
        # Goal for "C" (7) is to do "A" and then "B"
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
        if self.deterministic_start:
            self.agent_pos = np.array(
                [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
            )
        else:
            self.agent_pos = np.array(
                [
                    np.random.randint(1, self.grid_size),
                    np.random.randint(1, self.grid_size),
                ]
            )
        self.letter = 7
        self.goal = self.letter_goals[self.letter]
        self.steps = 0
        self.goal_ind = 0
        return self._get_obs()

    def _get_obs(self):
        return {"letter": self.letter, "position": self.agent_pos.copy()}

    def step(self, action):
        self.steps += 1
        if action < 5 and action > 0:  # Directional action, action 0 is pad value
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


class PlayGridWorldEnv(gym.Env):
    def __init__(self):
        super(PlayGridWorldEnv, self).__init__()
        self.grid_size = GRID_SIZE  # 3
        self.max_steps = MAX_STEPS  # 10
        self.deterministic_start = False
        self.letter_goals = {
            5: (self.grid_size, self.grid_size),  # 'A'
            6: (1, 1),  # 'B'
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
        if self.deterministic_start:
            self.agent_pos = [self.grid_size // 2 + 1, self.grid_size // 2 + 1]
        else:
            self.agent_pos = np.array(
                [
                    np.random.randint(1, self.grid_size),
                    np.random.randint(1, self.grid_size),
                ]
            )
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

        if action >= 5:
            self.letter = action
            self.goal = self.letter_goals[self.letter]

        reward = 0.0
        done = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            # done = True

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
        else:
            act = np.random.choice([5, 6])
        # Re-sample goal with probability 0.25
        act = np.random.choice([act, 5, 6], p=[0.75, 0.125, 0.125])
        return act


class DebugEnv(GridWorldEnv):
    def __init__(self):
        self.letter = 5
        self.agent_pos = [1, 1]

    def reset(self):
        return self._get_obs()

    def step(self, action):
        if action == 5:
            reward = 1
        else:
            reward = 0

        return self._get_obs(), reward, True, {}
