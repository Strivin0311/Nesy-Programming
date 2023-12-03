import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DisjDataset(Dataset):
    def __init__(self, root="../data/disj", seq_len=20, is_train=True, n_train=9000):
        if seq_len not in [20, 40, 60, 80, 100, 200]:
            raise NotImplementedError(f"The dataset of seq_len {seq_len} is not implemented")
        self.root = os.path.join(root, str(seq_len))
        self.X_path = os.path.join(self.root, "features.pt")
        self.y_path = os.path.join(self.root, "labels.pt")
        self.n_train = n_train

        with open(self.X_path, 'rb') as f:
            self.X = torch.load(f).long()
        with open(self.y_path, 'rb') as f:
            self.y = torch.load(f).long()

        self.X = self.X[:n_train] if is_train else self.X[n_train:]
        self.y = self.y[:n_train] if is_train else self.y[n_train:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]