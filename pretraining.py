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
from envs import PlayGridWorldEnv, GridWorldEnv, TwoStageGridWorldEnv
from policies import TransformerPolicy


def train_sl():
    env = PlayGridWorldEnv()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    policy = TransformerPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    episodes = []
    num_episodes = 200

    for episode in range(num_episodes):
        obs = env.reset()
        done = False

        state_seq = [torch.tensor(obs["position"], device=device)]
        action_seq = [torch.tensor(obs["letter"], device=device)]

        while not done:

            action = env.generate_expert_action()

            next_obs, reward, done, _ = env.step(action)

            action_seq.append(torch.tensor(action, dtype=torch.long, device=device))
            obs = next_obs
            state_seq.append(torch.tensor(obs["position"], device=device))

        sseq = torch.stack(state_seq)
        aseq = torch.stack(action_seq)
        episodes.append((sseq, aseq))

    dataset = NextTokenDataset(episodes, block_size=1)

    # Optimization
    loader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad tokens

    NUM_EPOCHS = 500
    for epoch in range(NUM_EPOCHS):

        for states, acts, targets, mask in loader:

            logits, _ = policy(states, acts, mask)

            B, T, V = logits.size()
            # print(B, T, V, logits.size())
            loss = loss_fn(logits.reshape(B * T, V), targets.view(B * T))
            # exit()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")

    return env, policy


def evaluate_agent(agent):

    env = GridWorldEnv()
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

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            action_seq.append(action)
            obs = next_obs
            state_seq.append(torch.tensor(obs["position"]))

    print(total_reward / num_episodes)


def evaluate_agent_in_two_stage(agent):

    env = TwoStageGridWorldEnv()
    num_episodes = 500
    total_reward = 0.0
    debug = False
    action_counts = np.zeros(8)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        timestep = 0
        past_checkpoint = False

        letter = torch.tensor([obs["letter"]])
        state_seq = [torch.tensor(obs["position"])]
        action_seq = [torch.tensor(obs["letter"])]

        while not done:
            sseq = torch.stack(state_seq).unsqueeze(0)
            if debug:
                print(action_seq)
            aseq = torch.stack(action_seq).unsqueeze(0)

            action_mask = None
            action_mask = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0])

            with torch.no_grad():
                logits, _ = agent(sseq, aseq, action_mask=action_mask)
                probs = F.softmax(logits, dim=-1)[0][-1]
                if debug:
                    print(sseq, aseq, probs)
                dist = Categorical(probs)
                action = dist.sample()
                if debug:
                    # print(action)
                    print(probs)
                # if timestep == 0:
                #     action = torch.tensor(5)
                # elif (env.agent_pos == env.letter_goals[5][0]).all() and not past_checkpoint:
                #     if debug:
                #         print("Goal reached")
                #     action = torch.tensor(6)
                #     past_checkpoint = True
                # elif action >= 5:
                #     p = probs[1:5]
                #     action = np.argmax(p) + 1
                action_counts[action] += 1

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            timestep += 1

            action_seq.append(action)
            obs = next_obs
            state_seq.append(torch.tensor(obs["position"]))

    print(total_reward / num_episodes)
    print(action_counts)


if __name__ == "__main__":

    # env, agent = train_sl()
    agent = torch.load("pretrained_3x3.pth", weights_only=False)
    evaluate_agent(agent)
    evaluate_agent_in_two_stage(agent)
    torch.save(agent, 'pretrained_3x3.pth')
