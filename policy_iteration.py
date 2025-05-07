import numpy as np
import matplotlib.pyplot as plt


class GridworldMDP:
    def __init__(self, grid_size, terminal_states, rewards, gamma=0.9):
        self.grid_size = grid_size  # (rows, cols)
        self.terminal_states = terminal_states  # {(i, j): reward}
        self.rewards = rewards  # Default reward for each state
        self.gamma = gamma  # Discount factor
        self._actions = ["U", "D", "L", "R"]  # Up, Down, Left, Right
        self.values = np.zeros(self.grid_size)  # Value function

        # Policy initialization
        # self.policy = np.random.choice(self._actions, size=self.grid_size)  # random
        # self.policy = np.full(self.grid_size, 'L', dtype=str)  # sub-optimal
        # self.policy = np.full(self.grid_size, 'R', dtype=str)  # Optimal
        thought_policy = np.full((self.grid_size[0] - 1, self.grid_size[1]), 'R', dtype=str)
        no_thought_policy = np.full((1, self.grid_size[1]), 'L', dtype=str)
        self.policy = np.vstack([thought_policy, no_thought_policy])

    def actions(self):
        return self._actions

    def is_terminal(self, state):
        return state in self.terminal_states

    def get_next_state(self, state, action):
        if self.is_terminal(state):
            return state
        i, j = state
        if action == "U":
            i = max(i - 1, 0)
        elif action == "D":
            i = min(i + 1, self.grid_size[0] - 1)
        elif action == "L":
            j = max(j - 1, 0)
        elif action == "R":
            j = min(j + 1, self.grid_size[1] - 1)
        return (i, j)

    def step(self, state, action):
        action_letter = self._actions[action]
        nstate = self.get_next_state(state, action_letter)
        reward = self.rewards.get(state, 0)
        done = self.is_terminal(state)
        return nstate, reward, done

    def init_Q(self):
        return np.zeros((self.grid_size[0], self.grid_size[1], 4))

    def init_policy_logits(self, init='uniform'):
        if init == 'uniform':
            return np.zeros((self.grid_size[0], self.grid_size[1], 4))
        return np.zeros((self.grid_size[0], self.grid_size[1], 4))

    def initial_state(self):
        return (self.grid_size[0] - 1, 0)

    def policy_evaluation(self, theta=1e-4):
        while True:
            delta = 0
            new_values = np.copy(self.values)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    state = (i, j)
                    action = self.policy[i, j]
                    next_state = self.get_next_state(state, action)
                    reward = self.rewards.get(state, 0)
                    if self.is_terminal(state):
                        target = reward
                    else:
                        target = reward + self.gamma * self.values[next_state]
                    new_values[i, j] = target
                    delta = max(delta, abs(new_values[i, j] - self.values[i, j]))
            self.values = new_values
            if delta < theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state = (i, j)
                if self.is_terminal(state):
                    continue
                old_action = self.policy[i, j]
                action_values = {}
                for action in self._actions:
                    next_state = self.get_next_state(state, action)
                    reward = self.rewards.get(state, 0)
                    action_values[action] = (
                        reward + self.gamma * self.values[next_state]
                    )
                best_action = max(action_values, key=action_values.get)
                self.policy[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def step_policy_iteration(self):
        self.policy_improvement()
        self.policy_evaluation()

    def display_policy(self):
        for row in self.policy:
            print(" ".join(row))

    def display_values(self):
        print(np.round(self.values, 2))

    def plot_policy_and_values(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(13.5, 7.5)
        ax.imshow(self.values, cmap='coolwarm', interpolation='nearest')
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                action = self.policy[i, j]
                if action == 'U':
                    ax.arrow(j, i + 0.15, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
                elif action == 'D':
                    ax.arrow(j, i - 0.15, 0, 0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
                elif action == 'L':
                    ax.arrow(j + 0.15, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
                elif action == 'R':
                    ax.arrow(j - 0.15, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
                # ax.text(j, i, round(self.values[i, j], 2), ha='center', va='center', color='white')
        plt.xticks(range(self.grid_size[1]))
        plt.yticks(range(self.grid_size[0]))
        plt.gca().invert_yaxis()
        plt.colorbar(ax.imshow(self.values, cmap='coolwarm', interpolation='nearest'))
        plt.title("State Values")
        plt.show()


if __name__ == "__main__":
    n_rows, n_cols = 2, 10
    grid_size = (n_rows, n_cols)
    terminal_states, rewards = {}, {}
    for row in range(n_rows):
        terminal_states[(row, n_cols - 1)] = 1
        rewards[(row, n_cols - 1)] = 1
        rewards[(row, 1)] = 0.0

    mdp = GridworldMDP(grid_size, terminal_states, rewards)
    mdp.policy_evaluation()
    for itr in range(10):
        print("Iteration %d" % itr)
        mdp.step_policy_iteration()
        print("Optimal Policy:")
        mdp.display_policy()
        print("State Values:")
        mdp.display_values()
        print()
    mdp.plot_policy_and_values()
