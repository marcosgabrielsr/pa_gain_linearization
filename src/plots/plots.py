"""
Gráficos dos sinais do PA (entrada x, saída medida, saída prevista pela LSTM).

Todas as funções de uma série aceitam:
    - color: cor da série (ex.: COLOR_DATASET, COLOR_LSTM)
    - ax:    eixo onde desenhar. Sem ax, a função cria a figura e chama plt.show();
             com ax, só desenha (quem chamou decide quando mostrar). Serve para sobrepor séries.
    - label: nome da série na legenda (a legenda só aparece se houver label)
"""

import numpy as np
import matplotlib.pyplot as plt

from metrics.metrics import psd

EPS = 1e-15

# Cores padrão: dados medidos em azul, previsão da LSTM em laranja
COLOR_DATASET = "tab:blue"
COLOR_LSTM = "tab:orange"
COLOR_ERROR = "tab:gray"

SCATTER = dict(marker=".", ls="none", ms=3, alpha=.3, rasterized=True)


def to_decibel(v):
    """complexo -> magnitude em dB"""
    return 20 * np.log10(np.abs(v) + EPS)


def _axes(ax, figsize=(10, 5)):
    """Devolve (ax, created): cria uma figura nova se ax for None."""
    if ax is not None:
        return ax, False
    _, ax = plt.subplots(figsize=figsize)
    return ax, True


def _finish(ax, created, title, label):
    ax.set_title(title)
    ax.grid(True, alpha=.3)
    ax.set_axisbelow(True)
    if label is not None:
        ax.legend(markerscale=4, frameon=False)
    if created:
        plt.show()


# ---------------------------------------------------------------------------
# Gráficos de uma série
# ---------------------------------------------------------------------------

def plot_AM_AM(x, y, db=True, title="AM/AM Graph", x_label="|x|", y_label="|y|", ax=None, color=None, label=None):
    """Magnitude de saída em função da magnitude de entrada."""
    ax, created = _axes(ax)
    if db:
        ax.plot(to_decibel(x), to_decibel(y), color=color, label=label, **SCATTER)
        ax.set_xlabel(f"{x_label} (dB)"); ax.set_ylabel(f"{y_label} (dB)")
    else:
        ax.plot(np.abs(x), np.abs(y), color=color, label=label, **SCATTER)
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    _finish(ax, created, title, label)


def plot_AM_PM(x, y, db=True, degrees=True, remove_offset=True, title="AM/PM Graph", ax=None, color=None, label=None):
    """Diferença de fase (saída - entrada) em função da magnitude de entrada."""
    # Phase difference computed via y * conj(x): already wrapped to [-π, π]
    dphi = np.angle(y * np.conj(x))
    if remove_offset:
        # Remove the constant phase rotation (delay/PA offset) using the circular mean
        dphi = np.angle(np.exp(1j * dphi) * np.exp(-1j * np.angle(np.mean(np.exp(1j * dphi)))))
    if degrees:
        dphi = np.degrees(dphi); unit = "°"
    else:
        unit = "rad"

    ax, created = _axes(ax)
    if db:
        ax.plot(to_decibel(x), dphi, color=color, label=label, **SCATTER)
        ax.set_xlabel("|x| (dB)")
    else:
        ax.plot(np.abs(x), dphi, color=color, label=label, **SCATTER)
        ax.set_xlabel("|x|")
    ax.set_ylabel(f"∠y − ∠x ({unit})")
    _finish(ax, created, title, label)


def plot_PM_PM(x, y, degrees=True, title="PM/PM Graph", ax=None, color=None, label=None):
    """Fase de saída em função da fase de entrada."""
    phi_x, phi_y = np.angle(x), np.angle(y)
    if degrees:
        phi_x, phi_y = np.degrees(phi_x), np.degrees(phi_y); unit = "°"
    else:
        unit = "rad"

    ax, created = _axes(ax)
    ax.plot(phi_x, phi_y, color=color, label=label, **SCATTER)
    ax.set_xlabel(f"∠x ({unit})"); ax.set_ylabel(f"∠y ({unit})")
    _finish(ax, created, title, label)


def plot_constellation(y, title="Constellation", ax=None, color=None, label=None):
    """Diagrama I x Q das amostras no tempo.
    Num sinal OFDM aparece como uma nuvem circular (os pontos QAM só aparecem após demodular)."""
    ax, created = _axes(ax, figsize=(6, 6))
    ax.plot(np.real(y), np.imag(y), color=color, label=label, **SCATTER)
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.set_aspect("equal", "datalim")
    _finish(ax, created, title, label)


# ---------------------------------------------------------------------------
# PSD (várias séries)
# ---------------------------------------------------------------------------

def plot_PSD(signals, fs, bw=None, title="PSD", ax=None, colors=None):
    """PSD (Welch) de sinais complexos em banda base.
        - signals: {label: sinal complexo}
        - colors:  {label: cor} (opcional)
        - bw:      desenha (tracejado) as bordas do canal principal e dos adjacentes
    """
    colors = colors or {}
    ax, created = _axes(ax)
    for name, sig in signals.items():
        f, p = psd(sig, fs)
        ax.plot(f / 1e6, p, lw=1, label=name, color=colors.get(name))
    if bw is not None:
        for edge in (-1.5 * bw, -bw / 2, bw / 2, 1.5 * bw):
            ax.axvline(edge / 1e6, color="k", ls="--", lw=.8)
    ax.set_xlabel("Frequency (MHz)"); ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(title); ax.grid(True, alpha=.3); ax.legend(frameon=False)
    if created:
        plt.show()


# ---------------------------------------------------------------------------
# Comparação dataset x LSTM numa única grade
# ---------------------------------------------------------------------------

def plot_comparison(x, y_true, y_pred, fs=None, bw=None,
                    labels=("Dataset", "LSTM"), colors=(COLOR_DATASET, COLOR_LSTM),
                    title="Dataset vs LSTM"):
    """Grade 2x3: AM/AM, AM/PM, PM/PM na primeira linha; constelação da saída e PSD (2 colunas)
    na segunda, com as duas séries sobrepostas. Sem fs (ex.: dados embaralhados), a PSD fica de fora."""
    fig, axes = plt.subplot_mosaic([["amam", "ampm", "pmpm"], ["const", "psd", "psd"]],
                                   figsize=(18, 10), constrained_layout=True)
    a_amam, a_ampm, a_pmpm, a_const, a_psd = (axes[k] for k in ("amam", "ampm", "pmpm", "const", "psd"))
    series = ((y_true, labels[0], colors[0]), (y_pred, labels[1], colors[1]))

    for y, label, color in series:
        plot_AM_AM(x, y, ax=a_amam, color=color, label=label, title="AM/AM")
        plot_AM_PM(x, y, ax=a_ampm, color=color, label=label, title="AM/PM")
        plot_PM_PM(x, y, ax=a_pmpm, color=color, label=label, title="PM/PM")
        plot_constellation(y, ax=a_const, color=color, label=label, title="Constellation (output)")

    if fs is not None:
        plot_PSD({labels[0]: y_true, labels[1]: y_pred, "Error": y_true - y_pred}, fs, bw, ax=a_psd,
                 colors={labels[0]: colors[0], labels[1]: colors[1], "Error": COLOR_ERROR}, title="PSD")
    else:
        a_psd.set_visible(False)

    fig.suptitle(title)
    plt.show()
