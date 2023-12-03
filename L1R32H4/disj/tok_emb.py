import torch
import torch.nn as nn


class ParityEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.block_size
        self.n_emb = config.n_embd
        self.embedding_layer = nn.Embedding(
            self.seq_len,
            self.n_emb
        )

    def forward(self, x):
        x = self.embedding_layer(x)

        return x