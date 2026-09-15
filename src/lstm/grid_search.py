import torch
import torch.nn as nn
from itertools import product
from torch.utils.data import DataLoader
from pa_model import SlidingWindowDataset
from .lstm import PaLSTM
from .trainer_lstm import TrainerLSTM
from .results_storage import ResultsStorage

class GridSearchLSTM():
    def __init__(self, train, val, param_grid, device):
        self.train = train
        self.val = val
        self.train_norm, self.scaler_x, self.scaler_y = train.normalize()
        self.val_norm, _, _ = val.normalize(self.scaler_x, self.scaler_y)
        self.device = device
        self.grid_combinations = list(product(param_grid["windows_size"],param_grid["hidden_size"],param_grid["num_layers"],param_grid["learning_rate"],param_grid["dropout"],param_grid["batch_size"]))

    def run(self, save_path):
        # Executa o grid search sobre os hiperparâmetros e salva os resultados dos melhores modelos em arquivos
        n_combinations = len(self.grid_combinations)
        best_metric, best_model, best_params = None, None, None
        storage = ResultsStorage(save_path)

        for run_id, (ws, hs, nl, lr, dr, bs) in enumerate(self.grid_combinations):
            # Cria o modelo LSTM com os hiperparâmetros atuais
            model = PaLSTM(input_size=4, hidden_size=hs, num_layers=nl, dropout=dr).to(self.device)
            model.to(torch.float64)

            # Configurando Dataloader com Sliding Window Dataset para treino e validação
            train_loader = DataLoader(SlidingWindowDataset(self.train_norm, ws), batch_size=bs, shuffle=True)
            val_loader = DataLoader(SlidingWindowDataset(self.val_norm, ws), batch_size=bs, shuffle=False)

            print(f"Combination ({run_id+1}/{n_combinations}): WS = {ws}, HS = {hs}, NL = {nl}, LR = {lr}, DR = {dr}, BS = {bs}\n")

            # Configurando classe para treinar lstm
            trainer = TrainerLSTM(model, nn.MSELoss(), lr, self.scaler_y, device=self.device, early_stopping=True, verbose=True)

            # Treinando modelo e coletando valores
            metric, model, history = trainer.fit(train_loader, val_loader)
            params = {"ws":ws, "hs":hs, "nl":nl, "lr":lr, "dr":dr, "bs":bs}

            storage.save(params,history)

            # Atualiza melhores valores
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_model = model
                best_params = params

        return best_metric, best_model, best_params