import torch
from itertools import product

from pa_model import Dataset, SWDataset, WindowLoader
from lstm import ResultsStorage
from .model import DpdLSTM
from .trainer_dpd import TrainerDPD


class GridSearchDPD():
    # Same logic as GridSearchLSTM, for the DPD: each combination trains a DpdLSTM through the frozen PA LSTM.
    #   - pa, wp, scaler_x, scaler_y: the PA twin and the scalers it was trained with (load_pa_twin)
    #   - gain: complex linear gain G; the target of each sample is G·x
    def __init__(self, train, val, pa, wp, scaler_x, scaler_y, gain, param_grid, device, seed=42):
        self.pa, self.wp = pa, wp
        self.scaler_x, self.scaler_y = scaler_x, scaler_y
        self.device = device
        self.seed = seed

        # Input x normalized with scaler_x (the PA input space, where the DPD works) and
        # target G·x normalized with scaler_y (the PA output space, where the loss is computed)
        self.train_norm, _, _ = Dataset(train.x, gain * train.x).normalize(scaler_x, scaler_y)
        self.val_norm, _, _ = Dataset(val.x, gain * val.x).normalize(scaler_x, scaler_y)

        self.grid_combinations = list(product(param_grid["window_size"], param_grid["hidden_size"], param_grid["num_layers"],
                                              param_grid["learning_rate"], param_grid["dropout"], param_grid["batch_size"]))

    def _build_model(self, hs, nl, dr):
        return DpdLSTM(hidden_size=hs, num_layers=nl, dropout=dr).to(self.device).to(torch.float32)

    def run(self, save_path):
        n_combinations = len(self.grid_combinations)
        best_metric, best_model, best_params = None, None, None
        storage = ResultsStorage(save_path, key_metric="val_nmse")

        previous_best = storage.load_best(self.device)
        if previous_best is not None:
            best_metric, best_params, state_dict = previous_best
            best_model = self._build_model(best_params["hs"], best_params["nl"], best_params["dr"])
            best_model.load_state_dict(state_dict)

        for run_id, (ws, hs, nl, lr, dr, bs) in enumerate(self.grid_combinations):
            params = {"ws": ws, "hs": hs, "nl": nl, "lr": lr, "dr": dr, "bs": bs}

            if storage.is_done(params):
                print(f"Combination ({run_id+1}/{n_combinations}) already done, skipping\n")
                continue

            torch.manual_seed(self.seed)
            dpd = self._build_model(hs, nl, dr)

            # Each window has ws + wp - 1 samples of x: enough for wp outputs of the DPD (window ws each),
            # which form the window of the PA LSTM. The target is G·x at the last instant
            window = ws + self.wp - 1
            generator = torch.Generator().manual_seed(self.seed)
            train_loader = WindowLoader(SWDataset(self.train_norm, window, device=self.device), batch_size=bs, shuffle=True, generator=generator)
            val_loader = WindowLoader(SWDataset(self.val_norm, window, device=self.device), batch_size=bs, shuffle=False)

            print(f"Combination ({run_id+1}/{n_combinations}): WS = {ws}, HS = {hs}, NL = {nl}, LR = {lr}, DR = {dr}, BS = {bs}\n")

            trainer = TrainerDPD(dpd, self.pa, ws, self.wp, lr, self.scaler_x, self.scaler_y, device=self.device,
                                 early_stopping=True, verbose=True)
            metric, dpd, history = trainer.fit(train_loader, val_loader)

            if best_metric is None or metric < best_metric:
                best_metric, best_model, best_params = metric, dpd, params
                storage.save_best(best_model, best_params, best_metric, self.scaler_x, self.scaler_y)

            storage.save(params, history)

        return best_metric, best_model, best_params
