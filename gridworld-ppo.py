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

from data_utils import (
    RolloutBuffer,
    sample_mini_batches,
    NextTokenDataset,
    collate_fn,
    rl_collate_fn,
    RLDataset,
    compute_returns_and_advantages,
)
from envs import GridWorldEnv, TwoStageGridWorldEnv


class TransformerPolicy(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        vocab_size = 7 + 1
        self.letter_embedding = nn.Embedding(3, d_model)
        self.state_embedding = nn.Linear(2, d_model)
        self.action_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * d_model, nhead=nhead, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(2 * d_model, vocab_size)
        self.value_head = nn.Linear(2 * d_model, 1)

    def forward(self, state_seq, action_seq, mask=None, debug=False):

        batch_size = state_seq.shape[0]
        seq_len = state_seq.shape[1]

        if mask is None:
            mask = torch.tensor(
                [[True for _ in range(seq_len)] for _ in range(batch_size)]
            )

        state_embed = self.state_embedding(state_seq.float())  # [B, seq, D]
        action_embed = self.action_embedding(action_seq)

        interleaved = torch.zeros(
            batch_size, seq_len, 2 * self.d_model, device=state_embed.device
        )
        interleaved[:, :, : self.d_model] = action_embed
        interleaved[:, :, self.d_model :] = state_embed
        input_seq = interleaved
        if debug:
            print(input_seq.shape)
            print(mask.shape)
            print(mask.transpose(0, 1).shape)
            print(interleaved)

        # x = self.transformer(input_seq.permute(1, 0, 2))
        src_key_padding = ~(mask.transpose(0, 1))
        x = self.transformer(input_seq, src_key_padding_mask=src_key_padding)
        if debug:
            print("x")
            print(x)
            print(x.shape)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


def train_sl():
    env = GridWorldEnv()
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
            loss = loss_fn(logits.view(B * T, V), targets.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")

    return env, policy


def train_rl_v2(policy):
    # env = TwoStageGridWorldEnv()
    env = GridWorldEnv()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    # policy = TransformerPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-6)

    num_episodes = 100
    num_iterations = 5

    for itr in range(num_iterations):

        episodes = []
        total_reward = 0.0

        # 1. Collect data
        for episode in range(num_episodes):
            obs = env.reset()
            done = False

            state_seq = [torch.tensor(obs["position"], device=device)]
            action_seq = [torch.tensor(obs["letter"], device=device)]
            rew_seq = []
            log_probs = []
            values = []

            while not done:

                sseq = torch.stack(state_seq).unsqueeze(0)
                aseq = torch.stack(action_seq).unsqueeze(0)

                with torch.no_grad():
                    logits, value = policy(sseq, aseq)
                    probs = F.softmax(logits, dim=-1)[0][-1]
                    dist = Categorical(probs)
                    action = dist.sample()
                    # action = torch.tensor(env.generate_expert_action())
                    log_probs.append(dist.log_prob(action).item())
                    values.append(value[0][-1].item())

                next_obs, reward, done, _ = env.step(action)

                total_reward += reward
                rew_seq.append(reward)
                action_seq.append(action)
                obs = next_obs
                state_seq.append(torch.tensor(obs["position"], device=device))

            sseq = torch.stack(state_seq)
            aseq = torch.stack(action_seq)
            log_prob_seq = torch.tensor(log_probs)
            returns, advs = compute_returns_and_advantages(rew_seq, values)
            episodes.append((sseq, aseq, returns, advs, log_prob_seq))

        print(f'Avg Reward for Iter {itr}: {total_reward / num_episodes}')

        dataset = RLDataset(episodes)

        # Policy Optimization
        loader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=rl_collate_fn)
        # loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad tokens

        NUM_EPOCHS = 1
        for epoch in range(NUM_EPOCHS):

            for states, acts, targets, returns, advs, old_logprobs, mask in loader:

                logits, values = policy(states, acts, mask)

                # B, T, V = logits.size()
                # loss = loss_fn(logits.view(B*T, V), targets.view(B*T))
                # print(logits.size())
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(targets.squeeze())
                ratio = torch.exp(logprobs - old_logprobs)
                print(advs[0])
                binary_mask = mask.to(dtype=torch.uint8)  # Or torch.int, torch.long, etc.
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advs
                surr = torch.min(surr1, surr2) * binary_mask
                masked_squared_error = (returns - values) ** 2 * binary_mask
                sum_masked_squared_error = torch.sum(masked_squared_error)
                num_non_masked_elements = binary_mask.sum()

                if num_non_masked_elements == 0:
                    mse_loss = torch.tensor(0.0)
                else:
                    mse_loss = sum_masked_squared_error / num_non_masked_elements
                print(surr, masked_squared_error)
                loss = -(surr.sum() / num_non_masked_elements) + mse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

    # env, agent = train_sl()
    # evaluate_agent(env, agent)
    # torch.save(agent, 'initial_model.pth')
    agent = torch.load("initial_model.pth")
    env2, agent = train_rl_v2(agent)
    evaluate_agent(env2, agent)


def train_rl():
    env = GridWorldEnv()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    policy = TransformerPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    buffer = RolloutBuffer()

    for episode in range(1000):
        obs = env.reset()
        done = False

        letter = torch.tensor([obs["letter"]], device=device)
        state_seq = [torch.tensor(obs["position"], device=device)]
        action_seq = []

        while not done:
            if action_seq:
                sseq = torch.stack(state_seq).unsqueeze(0)
                aseq = torch.stack(action_seq).unsqueeze(0)
            else:
                sseq = torch.stack(state_seq).unsqueeze(0)
                aseq = torch.zeros((1, 0), dtype=torch.long, device=device)

            with torch.no_grad():
                logits, value = policy(letter, sseq, aseq)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()

            next_obs, reward, done, _ = env.step(action.item())

            buffer.add(
                letter.item(),
                obs["position"],
                action.item(),
                reward,
                done,
                dist.log_prob(action).item(),
                value.item(),
            )
            action_seq.append(action)
            obs = next_obs
            state_seq.append(torch.tensor(obs["position"], device=device))

        # Optimization
        MAX_BUFFER_SIZE = 100
        if len(buffer.rewards) >= MAX_BUFFER_SIZE:
            # print('Training')
            # returns, advs = buffer.compute_returns_and_advantages()

            states = torch.tensor(
                buffer.states, dtype=torch.float32, device=device
            ).unsqueeze(1)
            actions = torch.tensor(
                buffer.actions, dtype=torch.long, device=device
            ).unsqueeze(1)
            letters = torch.tensor(buffer.letters, dtype=torch.long, device=device)
            logprobs_old = torch.tensor(buffer.logprobs, device=device)
            advs = (advs.to(device) - advs.mean()) / (advs.std() + 1e-8)
            returns = returns.to(device)
            print(f"Mean return: {np.mean(returns.numpy())}")

            NUM_EPOCHS = 1
            for _ in range(NUM_EPOCHS):
                for batch_idx in sample_mini_batches(buffer, batch_size=100):
                    b_states = states[batch_idx]
                    b_actions = actions[batch_idx]
                    b_letters = letters[batch_idx]
                    b_returns = returns[batch_idx]
                    b_advs = advs[batch_idx]
                    b_logprobs_old = logprobs_old[batch_idx]

                    logits, values = policy(b_letters, b_states, b_actions)
                    dist = Categorical(logits=logits)
                    logprobs = dist.log_prob(b_actions.squeeze())
                    ratio = torch.exp(logprobs - b_logprobs_old)

                    surr1 = ratio * b_advs
                    surr2 = torch.clamp(ratio, 0.8, 1.2) * b_advs
                    loss = -torch.min(surr1, surr2).mean() + F.mse_loss(
                        values, b_returns
                    )  # loss =

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # if episode % 10 == 0:
            #     print(f"Episode {episode}, reward mean: {np.mean(buffer.rewards):.3f}")

            buffer.clear()