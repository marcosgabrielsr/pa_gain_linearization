import copy
import torch
import torch.optim as optim
import numpy as np
from src.metrics.metrics import RMSE, EVM
from .earlystopping import EarlyStopping

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
        self.verbose = verbose
        self.early_stopping = EarlyStopping(patience, delta, verbose) if early_stopping else None

    def fit(self, train_loader, val_loader, key_metric="evm"):
        keys = ("avg_loss", "rmse", "evm")
        history = {
            "train_avg_loss": [], "train_rmse": [], "train_evm": [],
            "val_avg_loss": [], "val_rmse": [], "val_evm": []
        }

        for epoch in range(self.n_epochs):
            train_results = dict(zip(keys, self._fit(train_loader)))
            val_results = dict(zip(keys, self.evaluate(val_loader)))

            history["train_avg_loss"].append(train_results["avg_loss"])
            history["train_rmse"].append(train_results["rmse"])
            history["train_evm"].append(train_results["evm"])
            history["val_avg_loss"].append(val_results["avg_loss"])
            history["val_rmse"].append(val_results["rmse"])
            history["val_evm"].append(val_results["evm"])

            if self.verbose and (epoch+1)%5==0:
                print(f"Epoch: {epoch+1}/50")
                for (name,d) in (("train",train_results),("val",val_results)):
                    print(f"{name}:")
                    for k,v in d.items():
                        print(f"\t{k}: {v}")
                    print()

            if self.early_stopping is not None:
                self.early_stopping.check_early_stop(val_results[key_metric],copy.deepcopy(self.model))

                if self.early_stopping.stop_training:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.early_stopping.best_metric, self.early_stopping.best_model, history


    def _fit(self, train_loader):
        """ Treina o modelo LSTM por uma época """
        self.model.train()
        y_pred, y_true = [], []
        total_loss, total = 0.0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            y_pred.append(outputs.detach().cpu())
            y_true.append(targets.cpu())
            total += outputs.size(0)

        y_pred, y_true = np.concatenate(y_pred,axis=0), np.concatenate(y_true,axis=0)
        y_pred, y_true = self.scaler.inverse_transform(y_pred), self.scaler.inverse_transform(y_true)
        avg_loss = total_loss / total
        rmse = self.rmse.compute(y_true, y_pred)
        evm = self.evm.compute(y_true, y_pred)
        return avg_loss, rmse, evm

    def evaluate(self, val_loader):
        """Avalia o modelo LSTM no conjunto de validação."""
        self.model.eval()
        y_pred, y_true = [], []
        total_loss, total = 0.0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                y_pred.append(outputs.detach().cpu())
                y_true.append(targets.cpu())
                total += outputs.size(0)

        y_pred, y_true = np.concatenate(y_pred,axis=0), np.concatenate(y_true,axis=0)
        y_pred, y_true = self.scaler.inverse_transform(y_pred), self.scaler.inverse_transform(y_true)
        avg_loss = total_loss / total
        rmse = self.rmse.compute(y_true, y_pred)
        evm = self.evm.compute(y_true, y_pred)
        return avg_loss, rmse, evm