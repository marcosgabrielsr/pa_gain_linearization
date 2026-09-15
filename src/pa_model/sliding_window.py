import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, target_dim=1):
        self.data = data
        self.window_size = window_size
        self.target_dim = target_dim

    def __len__(self):
        return (len(self.data) - self.window_size - self.target_dim + 1)

    def __getitem__(self, index):
        start_index = index
        end_index = start_index + self.window_size

        x = torch.stack([
            torch.tensor(self.data.x.real[start_index:end_index]),
            torch.tensor(self.data.y.real[start_index:end_index]),
            torch.tensor(self.data.x.imag[start_index:end_index]),
            torch.tensor(self.data.y.imag[start_index:end_index])
        ], dim=1)

        y = torch.stack([
            torch.tensor(self.data.y.real[end_index:end_index+self.target_dim]),
            torch.tensor(self.data.y.imag[end_index:end_index+self.target_dim])
        ], dim=1).squeeze(0)

        return x,y