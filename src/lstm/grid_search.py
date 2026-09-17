import torch
import torch.nn as nn
from itertools import product
from torch.utils.data import DataLoader
from pa_model import SlidingWindowDataset
from .lstm import PaLSTM
from .trainer_lstm import TrainerLSTM
from .results_storage import ResultsStorage

class GridSearchLSTM():
    def __init__(self, train, val, param_grid, device, seed=42):
        self.train_norm, self.scaler_x, self.scaler_y = train.normalize()
        self.val_norm, _, _ = val.normalize(self.scaler_x, self.scaler_y)
        self.device = device
        self.seed = seed
        self.grid_combinations = list(product(param_grid["window_size"],param_grid["hidden_size"],param_grid["num_layers"],param_grid["learning_rate"],param_grid["dropout"],param_grid["batch_size"]))

    def _build_model(self, hs, nl, dr):
        model = PaLSTM(input_size=4, hidden_size=hs, num_layers=nl, dropout=dr).to(self.device)
        return model.to(torch.float64)

    def run(self, save_path):
        # Executa o grid search sobre os hiperparâmetros e salva os resultados dos melhores modelos em arquivos.
        # Combinações já concluídas em save_path são puladas, permitindo retomar uma busca interrompida.
        n_combinations = len(self.grid_combinations)
        best_metric, best_model, best_params = None, None, None
        storage = ResultsStorage(save_path)

        # Retoma o melhor resultado de uma execução anterior, se houver
        previous_best = storage.load_best(self.device)
        if previous_best is not None:
            best_metric, best_params, state_dict = previous_best
            best_model = self._build_model(best_params["hs"], best_params["nl"], best_params["dr"])
            best_model.load_state_dict(state_dict)

        for run_id, (ws, hs, nl, lr, dr, bs) in enumerate(self.grid_combinations):
            params = {"ws":ws, "hs":hs, "nl":nl, "lr":lr, "dr":dr, "bs":bs}

            if storage.is_done(params):
                print(f"Combination ({run_id+1}/{n_combinations}) already done, skipping\n")
                continue

            # Mesma semente para toda combinação: inicialização dos pesos e embaralhamento reprodutíveis
            torch.manual_seed(self.seed)

            # Cria o modelo LSTM com os hiperparâmetros atuais
            model = self._build_model(hs, nl, dr)

            # Configurando Dataloader com Sliding Window Dataset para treino e validação
            generator = torch.Generator().manual_seed(self.seed)
            train_loader = DataLoader(SlidingWindowDataset(self.train_norm, ws), batch_size=bs, shuffle=True, generator=generator)
            val_loader = DataLoader(SlidingWindowDataset(self.val_norm, ws), batch_size=bs, shuffle=False)

            print(f"Combination ({run_id+1}/{n_combinations}): WS = {ws}, HS = {hs}, NL = {nl}, LR = {lr}, DR = {dr}, BS = {bs}\n")

            # Configurando classe para treinar lstm
            trainer = TrainerLSTM(model, nn.MSELoss(), lr, self.scaler_y, device=self.device, early_stopping=True, verbose=True)

            # Treinando modelo e coletando valores
            metric, model, history = trainer.fit(train_loader, val_loader)

            # Atualiza e salva os melhores valores antes de marcar a combinação como concluída
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_model = model
                best_params = params
                storage.save_best(best_model, best_params, best_metric, self.scaler_x, self.scaler_y)

            storage.save(params,history)

        return best_metric, best_model, best_params
