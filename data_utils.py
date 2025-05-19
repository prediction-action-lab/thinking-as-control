"""Utility functions and classes for data during training."""
import random
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class NextTokenDataset:
    def __init__(self, episodes, block_size):
        self.block_size = block_size
        self.samples = []
        for episode in episodes:
            state_seq, action_seq = episode
            if len(action_seq) < 1:
                continue
            self.samples.append((state_seq[:-1], action_seq[:-1], action_seq[1:]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def clear():
        self.samples = []


class RLDataset:
    def __init__(self, episodes):
        self.samples = []
        for episode in episodes:
            state_seq, action_seq, returns, advs, log_prob_seq = episode
            if len(action_seq) < 1:
                continue
            self.samples.append(
                (
                    state_seq[:-1],
                    action_seq[:-1],
                    action_seq[1:],
                    returns,
                    advs,
                    log_prob_seq,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def clear():
        self.samples = []


def sample_mini_batches(buffer, batch_size=64):
    dataset_size = len(buffer.rewards)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield batch_idx


def collate_fn(batch):
    states, actions, ys = zip(*batch)
    states_pad = pad_sequence(states, batch_first=True, padding_value=0)
    actions_pad = pad_sequence(actions, batch_first=True, padding_value=0)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    # state_mask = (states_pad != 0)  # True for tokens, False for pad
    mask = actions_pad != 0
    return states_pad, actions_pad, ys_pad, mask


def rl_collate_fn(batch):
    states, actions, ys, returns, advs, log_probs = zip(*batch)
    states_pad = pad_sequence(states, batch_first=True, padding_value=0)
    actions_pad = pad_sequence(actions, batch_first=True, padding_value=0)
    returns_pad = pad_sequence(returns, batch_first=True, padding_value=-1)
    advs_pad = pad_sequence(advs, batch_first=True, padding_value=-1)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=0)
    log_probs_pad = pad_sequence(log_probs, batch_first=True, padding_value=1)
    mask = advs_pad != -1
    return states_pad, actions_pad, ys_pad, returns_pad, advs_pad, log_probs_pad, mask


def compute_returns_and_advantages(rewards, values, gamma=0.99, lam=1):
    returns = np.zeros(len(rewards))
    advs = np.zeros_like(returns)
    gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advs[t] = gae
        next_value = values[t]
        returns[t] = gae + values[t]
    return torch.tensor(returns), torch.tensor(advs)
