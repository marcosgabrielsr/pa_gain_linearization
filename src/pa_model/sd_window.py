import math
import torch
import numpy as np
from torch.utils.data import Dataset

class SWDataset(Dataset):
    """Janelas deslizantes: entrada x[i .. i+W-1] -> alvo y[i+W-1] (target_dim = 1).

    Todas as janelas são montadas de uma vez com unfold (uma view: não copia os dados).
    Com device="cuda" a série inteira já fica na GPU, e um lote é uma única indexação.
    """
    def __init__(self, data, window_size, target_dim=1, device="cpu"):
        self.window_size = window_size
        self.target_dim = target_dim

        # Converting the input and the outputs (targets) on tensors: (N, 2) = [I, Q]
        self.x = torch.tensor(np.stack([data.x.real, data.x.imag], axis=1), dtype=torch.float32, device=device)
        self.targets = torch.tensor(np.stack([data.y.real, data.y.imag], axis=1), dtype=torch.float32, device=device)

        n = len(self.x) - window_size - target_dim + 2     # número de janelas

        # (N-W+1, 2, W) -> (N-W+1, W, 2): (batch, tempo, features), como a LSTM espera
        self.windows = self.x.unfold(0, window_size, 1).transpose(1, 2)[:n]

        # Alvo da janela i começa em y[i+W-1] (último instante da janela)
        start = window_size - 1
        if target_dim == 1:
            self.window_targets = self.targets[start:start + n]                                  # (n, 2)
        else:
            self.window_targets = self.targets.unfold(0, target_dim, 1).transpose(1, 2)[start:start + n]  # (n, T, 2)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """idx pode ser um inteiro (uma janela) ou um tensor de índices (um lote inteiro)."""
        return self.windows[idx], self.window_targets[idx]


class WindowLoader:
    """Substitui o DataLoader para um SWDataset: cada lote é uma única indexação do tensor,
    sem chamadas Python por amostra nem collate.

    Com shuffle=True e o mesmo generator, produz os lotes NA MESMA ORDEM que
    DataLoader(dataset, batch_size, shuffle=True, generator=generator).
    """
    def __init__(self, dataset, batch_size, shuffle=False, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            # O DataLoader sorteia uma semente base no início de cada época (para os workers),
            # consumindo um número do generator antes do randperm do RandomSampler.
            # Repetimos esse sorteio só para manter a mesma sequência de lotes.
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            idx = torch.randperm(n, generator=self.generator)
        else:
            idx = torch.arange(n)

        idx = idx.to(self.dataset.windows.device)
        for batch_idx in idx.split(self.batch_size):
            yield self.dataset[batch_idx]
