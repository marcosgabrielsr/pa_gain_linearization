import torch
import torch.optim as optim
import numpy as np
from metrics.metrics import RMSE, EVM, RSQR, AIC, BIC
from .earlystopping import EarlyStopping

KEYS = ("avg_loss", "rmse", "evm", "rsqr", "aic", "bic")

class TrainerLSTM():
    """ Train the LSTM Model (by default uses Adam as optimizer) """
    def __init__(self, model, criterion, lr, output_scaler, n_epochs=50, device=torch.device('cpu'), early_stopping=False, patience=5, delta=0, verbose=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = device
        self.scaler = output_scaler
        self.n_epochs = n_epochs
        self.rmse = RMSE()
        self.evm = EVM()
        self.rsqr = RSQR()
        # AIC and BIC penalize the number of trainable parameters of the model
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.aic = AIC(n_params)
        self.bic = BIC(n_params)
        self.verbose = verbose
        self.early_stopping = EarlyStopping(patience, delta, verbose) if early_stopping else None

    def fit(self, train_loader, val_loader, key_metric="evm"):
        history = {f"{split}_{k}": [] for split in ("train", "val") for k in KEYS}

        for epoch in range(self.n_epochs):
            train_results = dict(zip(KEYS, self._fit(train_loader)))
            val_results = dict(zip(KEYS, self.evaluate(val_loader)))

            for k in KEYS:
                history[f"train_{k}"].append(train_results[k])
                history[f"val_{k}"].append(val_results[k])

            if self.verbose and (epoch+1)%5==0:
                print(f"Epoch: {epoch+1}/{self.n_epochs}")
                for (name,d) in (("train",train_results),("val",val_results)):
                    print(f"{name}:")
                    for k,v in d.items():
                        print(f"\t{k}: {v}")
                    print()

            if self.early_stopping is not None:
                self.early_stopping.check_early_stop(val_results[key_metric], self.model)

                if self.early_stopping.stop_training:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if self.early_stopping is not None:
            return self.early_stopping.best_metric, self.early_stopping.best_model, history

        return val_results[key_metric], self.model, history

    def _fit(self, train_loader):
        """ Treina o modelo LSTM por uma época """
        self.model.train()
        return self._run_epoch(train_loader, train=True)

    def evaluate(self, val_loader):
        """Avalia o modelo LSTM no conjunto de validação."""
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(val_loader, train=False)

    def _run_epoch(self, loader, train):
        """ Percorre o loader uma vez. Loss e previsões ficam acumuladas no device:
            a GPU só é sincronizada com a CPU uma vez, no fim da época. """
        y_pred, y_true = [], []
        total_loss, total = 0.0, 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)   # no-op se já estiverem no device

            if train:
                self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.detach().double() * inputs.size(0)    # tensor no device (sem .item())
            y_pred.append(outputs.detach())
            y_true.append(targets)
            total += outputs.size(0)

        avg_loss = (total_loss / total).item()
        y_pred = torch.cat(y_pred).cpu().numpy().astype(np.float64)
        y_true = torch.cat(y_true).cpu().numpy().astype(np.float64)
        return (avg_loss, *self._metrics(y_true, y_pred))

    def _metrics(self, y_true, y_pred):
        """ Métricas no domínio original (desnormalizado) """
        y_pred, y_true = self.scaler.inverse_transform(y_pred), self.scaler.inverse_transform(y_true)
        return (
            self.rmse.compute(y_true, y_pred),
            self.evm.compute(y_true, y_pred),
            self.rsqr.compute(y_true, y_pred),
            self.aic.compute(y_true, y_pred),
            self.bic.compute(y_true, y_pred),
        )
