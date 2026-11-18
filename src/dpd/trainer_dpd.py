import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lstm import EarlyStopping
from metrics.metrics import NMSE, EVM
from .chain import cascade

KEYS = ("avg_loss", "nmse", "evm", "max_u")


class TrainerDPD():
    # Trains the DPD through the frozen PA LSTM (direct learning): loss = MSE(PA(DPD(x)), G·x).
    # Same structure as TrainerLSTM; the metrics compare the cascade output with the target G·x
    def __init__(self, dpd, pa, wd, wp, lr, scaler_x, scaler_y, n_epochs=50, device=torch.device("cpu"),
                 early_stopping=False, patience=5, delta=0, verbose=False):
        self.dpd = dpd
        self.pa = pa
        self.wd, self.wp = wd, wp
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(dpd.parameters(), lr=lr)     # only the DPD weights are trained
        self.scaler_x, self.scaler_y = scaler_x, scaler_y
        self.n_epochs = n_epochs
        self.device = device
        self.nmse, self.evm = NMSE(), EVM()
        self.verbose = verbose
        self.early_stopping = EarlyStopping(patience, delta, verbose) if early_stopping else None

    def fit(self, train_loader, val_loader, key_metric="nmse"):
        history = {f"{split}_{k}": [] for split in ("train", "val") for k in KEYS}

        for epoch in range(self.n_epochs):
            train_results = dict(zip(KEYS, self._fit(train_loader)))
            val_results = dict(zip(KEYS, self.evaluate(val_loader)))

            for k in KEYS:
                history[f"train_{k}"].append(train_results[k])
                history[f"val_{k}"].append(val_results[k])

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch: {epoch+1}/{self.n_epochs}")
                for name, d in (("train", train_results), ("val", val_results)):
                    print(f"{name}: " + ", ".join(f"{k} = {v:.4g}" for k, v in d.items()))
                print()

            if self.early_stopping is not None:
                self.early_stopping.check_early_stop(val_results[key_metric], self.dpd)
                if self.early_stopping.stop_training:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if self.early_stopping is not None:
            return self.early_stopping.best_metric, self.early_stopping.best_model, history
        return val_results[key_metric], self.dpd, history

    def _fit(self, train_loader):
        self.dpd.train()
        return self._run_epoch(train_loader, train=True)

    def evaluate(self, val_loader):
        self.dpd.eval()
        with torch.no_grad():
            return self._run_epoch(val_loader, train=False)

    def _run_epoch(self, loader, train):
        # Loss, outputs and max|u| are accumulated on the device; one GPU -> CPU sync per epoch
        y_pred, y_true = [], []
        total_loss, total = 0.0, 0
        max_u = torch.zeros((), device=self.device)

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if train:
                self.optimizer.zero_grad()
            outputs, u = cascade(self.dpd, self.pa, inputs, self.wd, self.wp)
            loss = self.criterion(outputs, targets)
            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.detach().double() * inputs.size(0)
            y_pred.append(outputs.detach())
            y_true.append(targets)
            max_u = torch.maximum(max_u, self._abs_physical(u.detach()).max())
            total += inputs.size(0)

        avg_loss = (total_loss / total).item()
        y_pred = self.scaler_y.inverse_transform(torch.cat(y_pred).cpu().numpy().astype(np.float64))
        y_true = self.scaler_y.inverse_transform(torch.cat(y_true).cpu().numpy().astype(np.float64))
        return avg_loss, self.nmse.compute(y_true, y_pred), self.evm.compute(y_true, y_pred), max_u.item()

    def _abs_physical(self, u):
        # |u| in physical units (undoing the StandardScaler of the PA input): shows how far the DPD extrapolates
        mean = torch.as_tensor(self.scaler_x.mean_, dtype=u.dtype, device=u.device)
        scale = torch.as_tensor(self.scaler_x.scale_, dtype=u.dtype, device=u.device)
        iq = u * scale + mean
        return torch.sqrt(iq[..., 0]**2 + iq[..., 1]**2)
