import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

from data_utils import (
    RolloutBuffer,
    sample_mini_batches,
    NextTokenDataset,
    collate_fn,
    rl_collate_fn,
    RLDataset,
    compute_returns_and_advantages,
)
from envs import GridWorldEnv, TwoStageGridWorldEnv, DebugEnv
from policies import TransformerPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run a pretrained PyTorch model with specified options.")

    parser.add_argument(
        '--model_path', type=str, required=True, default='pretrained_5x5.pth',
        help='Path to the pretrained PyTorch model file (.pt or .pth)'
    )

    parser.add_argument(
        '--mask_thinking', action='store_true', default=False,
        help='If set, mask out "thinking" actions during evaluation'
    )

    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output_file', type=str, required=True,
        help='Path to write evaluation results (e.g., results.json or results.csv)'
    )

    return parser.parse_args()


def train_rl_v2(output_file_base, seed, model_path, use_action_mask=False):
    env = TwoStageGridWorldEnv()
    # env = GridWorldEnv()
    # env = DebugEnv()
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")

    # from_scratch = False
    # use_mask = False

    output_file = f"{output_file_base}_{seed}"

    if model_path is not None:
        policy = torch.load(model_path, weights_only=False)
    else:
        policy = TransformerPolicy().to(device)

    np.random.seed(seed)
    torch.manual_seed(np.random.randint(1e5))
    random.seed(np.random.randint(1e5))

    action_mask = None
    if use_action_mask:
        action_mask = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0])
        # filename += 'use-mask_'
    # filename += f"{seed}"

    vf_and_policy_optimizer = optim.Adam(policy.parameters(), lr=1e-5, weight_decay=0.0)
    vf_optimizer = optim.Adam(policy.parameters(), lr=1e-5, weight_decay=0.0)

    num_episodes = 200
    num_iterations = 100
    vf_burn_in_iters = 1
    rewards = np.zeros(num_iterations)
    frac_thinking_actions = np.zeros(num_iterations)

    for itr in range(num_iterations):

        episodes = []
        total_reward = 0.0
        action_counts = np.zeros(8)

        # 1. Collect data
        for episode in range(num_episodes):
            obs = env.reset()
            done = False

            state_seq = [torch.tensor(obs["position"], device=device)]
            action_seq = [torch.tensor(obs["letter"], device=device)]
            rew_seq = []
            log_probs = []
            values = []
            timestep = 0

            while not done:

                sseq = torch.stack(state_seq).unsqueeze(0)
                aseq = torch.stack(action_seq).unsqueeze(0)

                with torch.no_grad():
                    logits, value = policy(sseq, aseq, action_mask=action_mask)
                    probs = F.softmax(logits, dim=-1)[0][-1]
                    dist = Categorical(probs)
                    action = dist.sample()
                    log_probs.append(dist.log_prob(action).item())
                    values.append(value[0][-1].item())
                    action_counts[action.item()] += 1

                # if timestep == 0:
                #     action_counts[action] += 1

                next_obs, reward, done, _ = env.step(action)

                total_reward += reward
                rew_seq.append(reward)
                action_seq.append(action)
                obs = next_obs
                timestep += 1
                state_seq.append(torch.tensor(obs["position"], device=device))

            sseq = torch.stack(state_seq)
            aseq = torch.stack(action_seq)
            log_prob_seq = torch.tensor(log_probs)
            returns, advs = compute_returns_and_advantages(rew_seq, values)
            episodes.append((sseq, aseq, returns, advs, log_prob_seq))

        frac_thinking_actions[itr] = action_counts[5:].sum() / action_counts.sum()
        rewards[itr] = total_reward / num_episodes
        print(f'Iter {itr}: Avg Return = {rewards[itr]}, Frac Thinking = {frac_thinking_actions[itr]}')
        # print(action_counts)
        # if rewards[itr] == 0:
        #     continue

        dataset = RLDataset(episodes)

        # Policy Optimization
        loader = DataLoader(dataset, batch_size=num_episodes, shuffle=True, collate_fn=rl_collate_fn)
        # loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad tokens

        if vf_burn_in_iters > 0 and itr == 0:
            num_epochs = vf_burn_in_iters
        else:
            num_epochs = 1
        USE_PPO = False

        for epoch in range(num_epochs):

            total_mse = 0.0
            adv_mean = 0.0
            value_mean = 0.0

            for states, acts, targets, returns, advs, old_logprobs, mask in loader:

                logits, values = policy(states, acts, mask, action_mask=action_mask)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(targets)

                advs = returns
                binary_mask = mask.to(dtype=torch.uint8)  # Or torch.int, torch.long, etc.

                if USE_PPO:
                    ratio = torch.exp(logprobs - old_logprobs)
                    surr1 = ratio * advs
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * advs
                    surr = torch.min(surr1, surr2) * binary_mask
                else:
                    surr = advs * logprobs * binary_mask
                masked_squared_error = (returns - values) ** 2 * binary_mask
                sum_masked_squared_error = torch.sum(masked_squared_error)
                num_non_masked_elements = binary_mask.sum()

                if num_non_masked_elements == 0:
                    mse_loss = torch.tensor(0.0)
                else:
                    mse_loss = sum_masked_squared_error / num_non_masked_elements
                # print(surr, masked_squared_error)
                if vf_burn_in_iters > 0 and itr == 0:
                    loss = mse_loss
                    optimizer = vf_optimizer
                    print('Value Loss', loss.item())
                else:
                    loss = -(surr.sum() / num_non_masked_elements) + mse_loss
                    optimizer = vf_and_policy_optimizer

                total_mse += mse_loss.item()
                value_mean += values.mean().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        np.save(f"{output_file}.npy", rewards)
        np.save(f"{output_file}-thinkactions.npy", frac_thinking_actions)
    plt.plot(rewards)
    plt.show()

    return env, policy


def evaluate_agent(env, agent):

    num_episodes = 100
    total_reward = 0.0
    debug = False

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        letter = torch.tensor([obs["letter"]])
        state_seq = [torch.tensor(obs["position"])]
        action_seq = [torch.tensor(obs["letter"])]

        while not done:
            sseq = torch.stack(state_seq).unsqueeze(0)
            if debug:
                print(action_seq)
            aseq = torch.stack(action_seq).unsqueeze(0)

            with torch.no_grad():
                logits, _ = agent(sseq, aseq)
                probs = F.softmax(logits, dim=-1)[0][-1]
                if debug:
                    print(sseq, aseq, probs)
                dist = Categorical(probs)
                action = dist.sample()
                if debug:
                    print(action)
                # action = env.generate_expert_action()

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            action_seq.append(action)
            obs = next_obs
            state_seq.append(torch.tensor(obs["position"]))

    print(total_reward / num_episodes)


if __name__ == "__main__":

    args = parse_args()
    seed = args.seed
    model_path = args.model_path
    results_file = args.output_file
    use_action_mask = args.mask_thinking

    # env, agent = train_sl()
    # evaluate_agent(env, agent)
    # torch.save(agent, 'initial_model.pth')
    # agent = torch.load("initial_model.pth", weights_only=False)

    env2, agent = train_rl_v2(results_file, seed, model_path, use_action_mask)
    evaluate_agent(env2, agent)
