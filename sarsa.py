import numpy as np
from policy_iteration import GridworldMDP


def epsilon_greedy_policy(Q, state, epsilon, actions):
    if np.random.rand() < epsilon:
        ind = np.random.choice(len(actions))
        return ind
    else:
        return np.argmax(Q[state])


def sarsa(env, episodes=100, alpha=0.1, gamma=0.9, epsilon=0.1):

    Q = env.init_Q()
    policy = env.init_policy_logits()
    explore_eps_greedy = False
    alpha_pi = 0.1

    returns = []

    for episode in range(episodes):
        state = env.initial_state()
        pi_probs = np.exp(policy[state])
        pi_probs /= np.sum(pi_probs, axis=-1)

        if explore_eps_greedy:
            action = epsilon_greedy_policy(Q, state, epsilon, env.actions())
        else:
            action = np.random.choice(len(env.actions()), p=pi_probs)

        done = False
        G, t = 0, 0
        while not done:
            next_state, reward, done = env.step(state, action)
            G += gamma ** t * reward
            t += 1

            if explore_eps_greedy:
                next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.actions())
            else:
                pi_probs = np.exp(policy[next_state])
                pi_probs /= np.sum(pi_probs, axis=-1)
                next_action = np.random.choice(len(env.actions()), p=pi_probs)

            # SARSA update rule
            delta = reward + gamma * Q[next_state][next_action] * (not done) - Q[state][action]
            Q[state][action] += alpha * delta

            # Updates policy
            grad_log_probs = -pi_probs
            grad_log_probs[action] += 1
            policy[state] += alpha_pi * delta * grad_log_probs

            state, action = next_state, next_action

        returns.append(G)
        if episode % 100 == 0:
            print(f"Episode {episode}: Completed")
            print(np.mean(returns))
            returns = []

    return Q


if __name__ == "__main__":
    n_rows, n_cols = 2, 10
    grid_size = (n_rows, n_cols)
    terminal_states, rewards = {}, {}
    for row in range(n_rows):
        terminal_states[(row, n_cols - 1)] = 1
        rewards[(row, n_cols - 1)] = 1
        rewards[(row, 1)] = 0.0
    env = GridworldMDP(grid_size, terminal_states, rewards)
    Q = sarsa(env, episodes=2000, epsilon=0.3)
    print("Learned Q-Table:")
    print(Q)
    print(np.argmax(Q, axis=2))
