import torch
import torch.nn as nn

class PaLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.25, output_size=2, batch_first=True):
        super(PaLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=batch_first
        )
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out
