import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        print(pe[:, 0])
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


def generate_causal_mask(seq_len, device='cpu'):
    # Shape: [seq_len, seq_len]
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask.to(device)


class TransformerPolicy(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, max_len=50):
        super().__init__()
        self.d_model = d_model
        vocab_size = 7 + 1
        # self.letter_embedding = nn.Embedding(3, d_model)
        self.state_embedding = nn.Linear(2, d_model)
        self.action_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(2 * d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * d_model, nhead=nhead, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(2 * d_model, vocab_size)
        self.value_head = nn.Linear(2 * d_model, 1)
        self.temperature = torch.tensor(1.0)
        self.use_position_encoding = True
        self.max_len = max_len

    def forward(self, state_seq, action_seq, input_mask=None, debug=False, action_mask=None):

        batch_size = state_seq.shape[0]
        seq_len = state_seq.shape[1]
        causal_mask = generate_causal_mask(seq_len)

        if input_mask is None:
            input_mask = torch.tensor(
                [[True for _ in range(seq_len)] for _ in range(batch_size)]
            )

        state_embed = self.state_embedding(state_seq.float())  # [B, seq, D]
        action_embed = self.action_embedding(action_seq)

        interleaved = torch.zeros(
            batch_size, seq_len, 2 * self.d_model, device=state_embed.device
        )
        interleaved[:, :, :self.d_model] = action_embed
        interleaved[:, :, self.d_model:] = state_embed
        input_seq = interleaved
        if debug:
            print(input_seq.shape)
            print(input_mask.shape)
            print(input_mask.transpose(0, 1).shape)
            print(interleaved)

        if self.use_position_encoding:
            input_seq *= math.sqrt(self.d_model)
            input_seq = self.pos_encoder(input_seq)

        src_key_padding = ~(input_mask.transpose(0, 1))

        input_seq = input_seq.transpose(0, 1)
        src_key_padding = src_key_padding.transpose(0, 1)

        x = self.transformer(input_seq, mask=causal_mask, src_key_padding_mask=src_key_padding)
        if debug:
            print("x")
            print(x)
            print(x.shape)
        logits = self.policy_head(x)
        logits = logits * self.temperature
        logits = logits.transpose(0, 1)

        if action_mask is not None:
            # Set invalid action logits to -1e10
            logits = logits.masked_fill(~action_mask.bool(), -1e10)

        value = self.value_head(x).transpose(0, 1).squeeze(-1)
        return logits, value