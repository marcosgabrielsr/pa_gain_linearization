import torch.nn as nn
from lstm import PaLSTM


class DpdLSTM(nn.Module):
    # Residual DPD with memory: u[n] = x[n] + f(x[n-W+1 .. n]), with f a PaLSTM (window -> I/Q correction)
    def __init__(self, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.f = PaLSTM(input_size=2, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        # Last layer starts at zero: before training the DPD is the identity (u = x)
        nn.init.zeros_(self.f.fc.weight)
        nn.init.zeros_(self.f.fc.bias)

    def forward(self, windows):
        # windows: (B, W, 2) -> u: (B, 2), the pre-distorted sample at the last instant of each window
        return windows[:, -1, :] + self.f(windows)
