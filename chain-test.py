import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the Chain MDP class
class ChainMDP:
    def __init__(self, num_states=10, num_actions=2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state = num_states // 2  # Default start at state 5 each episode

    def reset(self):
        self.state = self.num_states // 2
        return self.state

    def step(self, action):
        if action == 0 and self.state > 0:
            self.state -= 1  # Move left
        elif action == 1 and self.state < self.num_states - 1:
            self.state += 1  # Move right

        reward = 1 if self.state in [1, self.num_states // 2 + 2] else 0
        done = False  # The episode can continue indefinitely
        return self.state, reward, done


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, num_actions)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.num_states = num_states

    def forward(self, state):
        x = torch.eye(self.num_states)[state].float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.output_layer(x))


# Define the value network
class ValueNetwork(nn.Module):
    def __init__(self, num_states):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output_layer = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.num_states = num_states

    def forward(self, state):
        x = torch.eye(self.num_states)[state].float()
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output_layer(x).squeeze()


# Policy Gradient agent
class PolicyGradientAgent:
    def __init__(self, num_states=10, num_actions=2, lr=0.01, gamma=0.99):
        self.num_actions = num_actions
        self.gamma = gamma
        self.policy = PolicyNetwork(num_states, num_actions)
        self.value = ValueNetwork(num_states)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []

    def select_action(self, state):
        state_tensor = torch.tensor([state], dtype=torch.long)
        action_probs = self.policy(state_tensor).detach().numpy()[0]
        return np.random.choice(self.num_actions, p=action_probs)

    def store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def train(self):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.episode_rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        state_tensor = torch.tensor(self.episode_states, dtype=torch.long)
        action_tensor = torch.tensor(self.episode_actions, dtype=torch.long)

        # Compute value baseline
        state_values = self.value(state_tensor)
        advantages = discounted_rewards - state_values.detach()

        # Update value network
        value_loss = torch.mean((state_values - discounted_rewards) ** 2)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute policy loss
        action_probs = self.policy(state_tensor)
        selected_action_probs = action_probs.gather(1, action_tensor.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(torch.log(selected_action_probs) * advantages)

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []


def train_agent():
    # Train the agent
    n_states = 20
    env = ChainMDP(num_states=n_states)
    agent = PolicyGradientAgent(num_states=n_states)
    n_episodes = 1000
    reward_history = []
    state_counts = np.zeros(env.num_states)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(50):  # Limit steps per episode
            state_counts[state] += 1
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            total_reward += reward
        agent.train()
        reward_history.append(total_reward)

        # if episode % 100 == 0:
        #     print(f"Episode {episode}, Total Reward: {total_reward}")
    print(f"\tState counts: {state_counts}")
    return np.argmax(state_counts)

count_equal_one = 0

n_trials = 100
for _ in range(n_trials):
    max_state = train_agent()
    if max_state == 1:
        count_equal_one += 1

print(count_equal_one/n_trials)

# state = env.reset()
# for _ in range(50):
#     action = agent.select_action(state)
#     next_state, reward, done = env.step(action)
#     print(state, action, reward, next_state)
#     state = next_state

# plt.plot(reward_history)
# plt.xlabel('Episodes')
# plt.ylabel('Total Reward')
# plt.title('Policy Gradient in Chain MDP')
# plt.show()
