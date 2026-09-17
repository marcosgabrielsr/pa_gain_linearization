import numpy as np
import torch
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, target_dim=1):
        self.window_size = window_size
        self.target_dim = target_dim

        # Sinais separados para poder fatiar cada um com seu próprio deslocamento
        self.x = torch.tensor(np.stack([data.x.real, data.x.imag], axis=1))  # (N, 2)
        self.y = torch.tensor(np.stack([data.y.real, data.y.imag], axis=1))  # (N, 2)

        # Realimentacao atrasada de uma amostra: y_prev[n] = y[n-1], com y[-1] = 0.
        # E' esse atraso que faz a mesma fatia terminar em x[t] e em y[t-1].
        self.y_prev = torch.cat([torch.zeros(1, 2, dtype=self.y.dtype), self.y[:-1]])
        self.targets = self.y

    def __len__(self):
        return (len(self.x) - self.window_size - self.target_dim + 2)

    def __getitem__(self, start_index):
        end_index = start_index + self.window_size

        x_win = self.x[start_index:end_index]        # x[t-W+1 … t]   <- x[t] e' o ultimo
        y_win = self.y_prev[start_index:end_index]   # y[t-W … t-1]   <- y[t] fica de fora

        x = torch.cat([x_win, y_win], dim=1)         # (W, 4)
        y = self.targets[end_index-1:end_index-1+self.target_dim].squeeze(0)

        return x, y
