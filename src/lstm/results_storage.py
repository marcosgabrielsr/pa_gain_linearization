from pathlib import Path
import pandas as pd

class ResultsStorage:
    def __init__(self, results_dir, param_keys=("ws", "hs", "nl", "lr", "dr", "bs"), key_metric="val_evm"):
        self.results_dir = Path(results_dir)
        self.history_dir = self.results_dir / "histories"
        self.summary_path = self.results_dir / "summary.csv"
        self.param_keys = list(param_keys)
        self.key_metric = key_metric
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def history_path(self, params) -> Path:
        # Builds the file name from the hyperparameters, e.g. ws10_hs32_nl1_lr0.001_dr0.2_bs64.csv
        name = "_".join(f"{k}{params[k]:g}" for k in self.param_keys)
        return self.history_dir / f"{name}.csv"

    def save(self, params: dict, history: dict):
        # Saves the full history of this combination in its own file
        df_history = pd.DataFrame(history)
        df_history.index.name = "epoch"
        df_history.to_csv(self.history_path(params))

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

    def load_summary(self) -> pd.DataFrame:
        return pd.read_csv(self.summary_path)

    def load_history(self, params) -> pd.DataFrame:
        # params can be a dict or a row from the summary
        return pd.read_csv(self.history_path(params), index_col="epoch")
