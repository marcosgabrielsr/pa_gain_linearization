# utils/earlystopping.py
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_metric = None
        self.best_model = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_metric, model):
        if self.best_metric is None or val_metric < self.best_metric - self.delta:
            self.best_metric = val_metric
            self.best_model = model
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed")