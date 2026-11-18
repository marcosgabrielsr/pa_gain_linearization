import json
import numpy as np
import torch

from lstm import PaLSTM, ResultsStorage


def load_pa_twin(run_dir, device):
    # Loads the best PA LSTM of a grid search folder as a frozen digital twin.
    # Returns (pa, ws, scaler_x, scaler_y)
    storage = ResultsStorage(run_dir)
    params = json.load(open(storage.best_params_path))["params"]

    pa = PaLSTM(2, params["hs"], params["nl"], params["dr"]).to(device)
    pa.load_state_dict(torch.load(storage.best_model_path, map_location=device, weights_only=True))

    # Frozen: the DPD training never changes the PA weights (the gradient still flows to the input)
    for p in pa.parameters():
        p.requires_grad_(False)

    # On the GPU the cuDNN LSTM only allows backward in train() mode. Dropout is zeroed so that
    # train() does not make the twin random (dropout only matters while training the PA itself)
    pa.lstm.dropout = 0.0
    pa.train()

    scaler_x, scaler_y = storage.load_scalers()
    return pa, int(params["ws"]), scaler_x, scaler_y


def cascade(dpd, pa, x_windows, wd, wp):
    # x_windows: (B, wd+wp-1, 2) -> (y_hat: (B, 2), u: (B, wp, 2))
    # Each window is split into wp sub-windows of size wd; the DPD turns them into the wp samples
    # of u that the PA LSTM needs (its own window) to produce y_hat at the last instant
    B = x_windows.size(0)
    sub = x_windows.unfold(1, wd, 1).permute(0, 1, 3, 2)      # (B, wp, wd, 2)
    u = dpd(sub.reshape(B * wp, wd, 2)).reshape(B, wp, 2)
    return pa(u), u


def _to_tensor(z, scaler, device):
    iq = np.stack([z.real, z.imag], axis=1)
    return torch.tensor(scaler.transform(iq), dtype=torch.float32, device=device)


def _to_complex(t, scaler):
    iq = scaler.inverse_transform(t.detach().cpu().numpy().astype(np.float64))
    return iq[:, 0] + 1j * iq[:, 1]


@torch.no_grad()
def run_pa(pa, x, wp, scaler_x, scaler_y, device):
    # PA LSTM alone over a whole signal (complex, physical units).
    # Output k corresponds to x[k + wp - 1]
    xs = _to_tensor(x, scaler_x, device)
    return _to_complex(pa(xs.unfold(0, wp, 1).transpose(1, 2)), scaler_y)


@torch.no_grad()
def run_dpd(dpd, pa, x, wd, wp, scaler_x, scaler_y, device):
    # DPD + PA LSTM over a whole signal (complex, physical units). Returns (u, y_hat):
    #   u[k]     corresponds to x[k + wd - 1]
    #   y_hat[k] corresponds to x[k + wd + wp - 2]
    dpd.eval()
    xs = _to_tensor(x, scaler_x, device)
    u = dpd(xs.unfold(0, wd, 1).transpose(1, 2))                       # (N-wd+1, 2), normalized
    y = pa(u.unfold(0, wp, 1).transpose(1, 2))                         # (N-wd-wp+2, 2)
    return _to_complex(u, scaler_x), _to_complex(y, scaler_y)
