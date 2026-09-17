import json
from pathlib import Path
import joblib
import pandas as pd
import torch

class ResultsStorage:
    def __init__(self, results_dir, param_keys=("ws", "hs", "nl", "lr", "dr", "bs"), key_metric="val_evm"):
        self.results_dir = Path(results_dir)
        self.history_dir = self.results_dir / "histories"
        self.summary_path = self.results_dir / "summary.csv"
        self.best_model_path = self.results_dir / "best_model.pt"
        self.best_params_path = self.results_dir / "best_params.json"
        self.scalers_path = self.results_dir / "scalers.joblib"
        self.param_keys = list(param_keys)
        self.key_metric = key_metric
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def history_path(self, params) -> Path:
        # Builds the file name from the hyperparameters, e.g. ws10_hs32_nl1_lr0.001_dr0.2_bs64.csv
        name = "_".join(f"{k}{params[k]:g}" for k in self.param_keys)
        return self.history_dir / f"{name}.csv"

    def is_done(self, params) -> bool:
        # The history is written last in save(), so its existence means the combination is complete
        return self.history_path(params).exists()

    def save(self, params: dict, history: dict):
        df_history = pd.DataFrame(history)
        df_history.index.name = "epoch"

        # Appends the hyperparameters and the metrics from the best epoch to the summary
        best_epoch = int(df_history[self.key_metric].idxmin())
        row = {k: params[k] for k in self.param_keys}
        row["best_epoch"] = best_epoch
        row.update(df_history.loc[best_epoch].to_dict())

        pd.DataFrame([row]).to_csv(
            self.summary_path,
            mode="a",
            header=not self.summary_path.exists(),
            index=False,
        )

        # Saves the full history of this combination in its own file
        df_history.to_csv(self.history_path(params))

    def save_best(self, model, params: dict, metric, scaler_x, scaler_y):
        # Saves the weights, hyperparameters and scalers needed to reuse the best model
        torch.save(model.state_dict(), self.best_model_path)
        with open(self.best_params_path, "w") as f:
            json.dump({"params": params, self.key_metric: metric}, f, indent=4, default=float)
        joblib.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, self.scalers_path)

    def load_best(self, device=torch.device("cpu")):
        # Returns (metric, params, state_dict), or None if no best model was saved yet
        if not self.best_params_path.exists():
            return None
        with open(self.best_params_path) as f:
            best = json.load(f)
        state_dict = torch.load(self.best_model_path, map_location=device)
        return best[self.key_metric], best["params"], state_dict

    def load_scalers(self):
        scalers = joblib.load(self.scalers_path)
        return scalers["scaler_x"], scalers["scaler_y"]

    def load_summary(self) -> pd.DataFrame:
        return pd.read_csv(self.summary_path)

    def load_history(self, params) -> pd.DataFrame:
        # params can be a dict or a row from the summary
        return pd.read_csv(self.history_path(params), index_col="epoch")
