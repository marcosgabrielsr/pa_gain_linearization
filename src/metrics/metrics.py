from abc import ABC, abstractmethod
import numpy as np

class Metrics(ABC):

    @abstractmethod
    def compute(self, y_true, y_pred):
        ...

class RMSE(Metrics):

    def compute(self, y_true, y_pred):
        rmse = np.sqrt(np.mean(np.abs(y_true - y_pred) ** 2))
        return rmse


class EVM(Metrics):

    def compute(self, y_true, y_pred):
        epsilon = np.finfo(np.float64).eps
        num = RMSE().compute(y_true, y_pred)
        den = np.sqrt(np.mean(np.abs(y_true) ** 2))
        evm =  100 * (num / (den + epsilon))
        return evm

class RSQR(Metrics):

    def compute(self, y_true, y_pred):
        epsilon = np.finfo(np.float64).eps
        ss_res = np.sum(np.abs(y_true - y_pred) ** 2)
        ss_tot = np.sum(np.abs(y_true - np.mean(y_true, axis=0)) ** 2)
        r2 = 1 - ss_res / (ss_tot + epsilon)
        return r2