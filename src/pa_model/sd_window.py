import torch
import numpy as np
from torch.utils.data import Dataset

class SWDataset(Dataset):
    def __init__(self, data, window_size, target_dim=1):
        self.window_size = window_size
        self.target_dim = target_dim

        # Converting the input on a tensor
        self.x = torch.tensor(np.stack([data.x.real, data.x.imag], axis=1), dtype=torch.float32)  # (N, 2)

        # Converting the outupts(targets) on a tensor
        self.tragets = torch.tensor(np.stack([data.y.real, data.y.imag], axis=1), dtype=torch.float32)

    def __len__(self):
        return (len(self.x) - self.window_size - self.target_dim + 2)

    def __getitem__(self, start_index):
        end_index = start_index + self.window_size

        x = self.x[start_index:end_index] # (W, 2) -> x[t-W+1 ... t]
        y = self.targets[end_index-1:end_index-1+self.target_dim].squeeze(0)

        return x, y
