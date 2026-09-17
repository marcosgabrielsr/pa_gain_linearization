import numpy as np
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, target_dim=1):
        self.window_size = window_size
        self.target_dim = target_dim

        # Converte os sinais para tensores uma única vez; __getitem__ só fatia (sem cópia)
        # inputs: (N, 4) com colunas x.real, y.real, x.imag, y.imag | targets: (N, 2) com y.real, y.imag
        self.inputs = torch.tensor(np.stack([data.x.real, data.y.real, data.x.imag, data.y.imag], axis=1))
        self.targets = torch.tensor(np.stack([data.y.real, data.y.imag], axis=1))

    def __len__(self):
        return (len(self.inputs) - self.window_size - self.target_dim + 1)

    def __getitem__(self, index):
        start_index = index
        end_index = start_index + self.window_size

        x = self.inputs[start_index:end_index]
        y = self.targets[end_index:end_index+self.target_dim].squeeze(0)

        return x,y
