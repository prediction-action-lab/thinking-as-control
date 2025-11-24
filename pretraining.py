"""Code to produce thinking-primed pre-trained model."""
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
    # RolloutBuffer,
    sample_mini_batches,
    NextTokenDataset,
    collate_fn,
    rl_collate_fn,
    RLDataset,
    compute_returns_and_advantages,
)
from envs import PlayGridWorldEnv, GridWorldEnv, TwoStageGridWorldEnv
from policies import TransformerPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-train a transformer model in the "PlayingGridworldEnv."'
    )

    parser.add_argument(
        "--model_load_path",
        type=str,
        default=None,
        help="Path to a pretrained PyTorch model file to load (.pt or .pth)",
    )

    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Path to save the trained model file (.pt or .pth)",
    )

    parser.add_argument(
        "--mask_thinking",
        action="store_true",
        default=False,
        help='If set, mask out "thinking" actions during evaluation',
    )

    parser.add_argument(
        "--prompt_thinking",
        action="store_true",
        default=False,
        help='If set, force agent to "think" at the right time in evaluation.',
    )

    return parser.parse_args()


def train_sl():
    env = PlayGridWorldEnv(wrong_goals=True)
    device = torch.device("cpu")
    policy = TransformerPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    episodes = []
    num_episodes = 500

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
    loader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=collate_fn)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad tokens

    NUM_EPOCHS = 1000
    for epoch in range(NUM_EPOCHS):

        for states, acts, targets, mask in loader:

            logits, _ = policy(states, acts, mask)

            B, T, V = logits.size()
            loss = loss_fn(logits.reshape(B * T, V), targets.view(B * T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")

    return env, policy


def evaluate_pretrained_agent(
    agent, prompt_thinking=False, mask_thinking_actions=False
):

    env = TwoStageGridWorldEnv()
    num_episodes = 200
    total_reward = 0.0
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
            aseq = torch.stack(action_seq).unsqueeze(0)

            action_mask = None
            if mask_thinking_actions:
                action_mask = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0])

            with torch.no_grad():
                logits, _ = agent(sseq, aseq, action_mask=action_mask)
                probs = F.softmax(logits, dim=-1)[0][-1]
                dist = Categorical(probs)
                action = dist.sample()

                if prompt_thinking:
                    if timestep == 0:
                        action = torch.tensor(5)
                    elif (
                        env.agent_pos == env.letter_goals[5][0]
                    ).all() and not past_checkpoint:
                        action = torch.tensor(6)
                        past_checkpoint = True
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

    print("Success rate:", total_reward / num_episodes)
    print(action_counts)


if __name__ == "__main__":

    args = parse_args()

    if args.model_load_path is None:
        env, agent = train_sl()
    else:
        agent = torch.load(args.model_load_path)

    evaluate_pretrained_agent(agent, args.prompt_thinking, args.mask_thinking)

    if args.model_save_path is not None:
        torch.save(agent, args.model_save_path)
