from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import welch


def as_complex(y):
    """Aceita um vetor complexo (N,) ou um array real (N, 2) = [I, Q] e devolve o vetor complexo."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        return y
    return y[:, 0] + 1j * y[:, 1]


def _psd_linear(sig, fs, nperseg):
    """PSD linear (Welch, janela Hann) de um sinal complexo em banda base, com f de -fs/2 a fs/2."""
    sig = as_complex(sig)
    nperseg = min(nperseg, len(sig))
    f, p = welch(sig, fs=fs, window="hann", nperseg=nperseg, return_onesided=False, detrend=False)
    return np.fft.fftshift(f), np.fft.fftshift(p)


def psd(sig, fs, nperseg=2048):
    """PSD em dB/Hz. Retorna (f [Hz], P [dB/Hz])."""
    f, p = _psd_linear(sig, fs, nperseg)
    return f, 10 * np.log10(p + 1e-30)


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

class AIC(Metrics):
    """ Akaike Information Criterion, assuming Gaussian residuals: n * ln(SSE / n) + 2k """

    def __init__(self, n_params):
        self.n_params = n_params

    def compute(self, y_true, y_pred):
        epsilon = np.finfo(np.float64).eps
        n = np.size(y_true)
        mse = np.mean(np.abs(y_true - y_pred) ** 2)
        aic = n * np.log(mse + epsilon) + 2 * self.n_params
        return aic


class BIC(Metrics):
    """ Bayesian Information Criterion, assuming Gaussian residuals: n * ln(SSE / n) + k * ln(n) """

    def __init__(self, n_params):
        self.n_params = n_params

    def compute(self, y_true, y_pred):
        epsilon = np.finfo(np.float64).eps
        n = np.size(y_true)
        mse = np.mean(np.abs(y_true - y_pred) ** 2)
        bic = n * np.log(mse + epsilon) + self.n_params * np.log(n)
        return bic


class NMSE(Metrics):
    """ Normalized Mean Squared Error, in dB: 10·log10( Σ|y_true - y_pred|² / Σ|y_true|² ) """

    def compute(self, y_true, y_pred):
        epsilon = np.finfo(np.float64).eps
        num = np.sum(np.abs(y_true - y_pred) ** 2)
        den = np.sum(np.abs(y_true) ** 2)
        return 10 * np.log10(num / (den + epsilon) + epsilon)


class ACEPR(Metrics):
    """ Adjacent Channel Error Power Ratio, in dB (worst side):
        10·log10( error power in the adjacent channel / signal power in the main channel )
        - main channel:      |f| <= bw/2
        - adjacent channels: bw/2 < |f| <= 3·bw/2 (same width as the main channel)
        Only meaningful for signals in temporal order. """

    def __init__(self, fs, bw, nperseg=2048):
        assert 1.5 * bw <= fs / 2, "adjacent channel goes beyond fs/2"
        self.fs = fs
        self.bw = bw
        self.nperseg = nperseg

    def compute(self, y_true, y_pred):
        y_true, y_pred = as_complex(y_true), as_complex(y_pred)
        f, p_err = _psd_linear(y_true - y_pred, self.fs, self.nperseg)
        _, p_sig = _psd_linear(y_true, self.fs, self.nperseg)
        bw = self.bw
        main  = np.abs(f) <= bw / 2
        lower = (f >= -1.5 * bw) & (f < -bw / 2)
        upper = (f >  bw / 2) & (f <= 1.5 * bw)
        p_main = np.sum(p_sig[main])                 # uniform bins: sum is proportional to the integral
        return 10 * np.log10(max(np.sum(p_err[lower]), np.sum(p_err[upper])) / p_main)
