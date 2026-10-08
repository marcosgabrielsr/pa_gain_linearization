"""
Emulador bit-exato da LSTM em ponto fixo.

Convenção de shapes:
    N = número de janelas processadas em paralelo (batch)
    F = features de entrada por instante   (best_model: F = 2, só I e Q da entrada do PA)
    H = hidden_size                        (best_model: H = 16)
    T = comprimento da janela              (best_model: T = 10)
    L = camadas LSTM empilhadas            (best_model: L = 2)
    2 = saída da rede: (I, Q) previstos

Ordem das portas no nn.LSTM do PyTorch: i, f, g, o (blocos de H linhas).
A camada 0 recebe x_t (F,); a camada l > 0 recebe o h da camada l-1 no mesmo instante.
A Linear usa o h da última camada no último instante.

Processar as N janelas em paralelo não muda nenhum bit: cada janela é
independente, e o NumPy só faz a mesma conta N vezes de uma vez.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ACC_BITS = 48  # largura do acumulador (como o DSP48 da Xilinx); usado para os biases


# ---------------------------------------------------------------------------
# Configuração dos formatos de ponto fixo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QFormat:
    """### Formato de ponto fixo com sinal (complemento de dois)
        - total_bits: bits totais, incluindo o sinal
        - frac_bits:  bits fracionários
    """
    total_bits: int
    frac_bits: int

    @property
    def int_bits(self) -> int:
        """Bits inteiros, incluindo o sinal (ex.: Q3.13 -> 3)."""
        return self.total_bits - self.frac_bits

    @property
    def qmin(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def qmax(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    def __str__(self) -> str:
        return f"Q{self.int_bits}.{self.frac_bits}"


@dataclass(frozen=True)
class FixedConfig:
    """### Formato de cada sinal da LSTM
        - x:    entradas normalizadas     medido ~ ±2,1           -> Q3.13
        - w:    pesos                     medido -3,55 .. +2,05   -> Q4.12 (folga: [-8, 8))
        - gate: saídas de sigmoid/tanh    sempre em (-1, 1)       -> Q1.15
        - h:    estado oculto             sempre em (-1, 1)       -> Q1.15
        - c:    estado de célula          pior caso |c| <= T = 10 -> Q5.11
        - out:  saída da Linear (I, Q)    a medir                 -> Q3.13
        Os biases não usam `w`: são guardados direto no formato do acumulador
        (ver quantize_weights).
    """
    x: QFormat = QFormat(16, 13)
    w: QFormat = QFormat(16, 12)
    gate: QFormat = QFormat(16, 15)
    h: QFormat = QFormat(16, 15)
    c: QFormat = QFormat(16, 11)
    out: QFormat = QFormat(16, 13)


# ---------------------------------------------------------------------------
# Bloco 0 — Pesos
# ---------------------------------------------------------------------------

def weights_from_state_dict(sd: dict) -> dict:
    """### Organiza um state_dict (arrays NumPy) na estrutura usada aqui
        #### estrutura:
            - "layers": lista com L dicionários, um por camada l:
                - W_ih: (4H, F) na camada 0, (4H, H) nas demais
                - W_hh: (4H, H)
                - b:    (4H,)  bias_ih + bias_hh somados
            - "W_fc": (2, H)
            - "b_fc": (2,)
    """
    layers = []
    l = 0
    while f"lstm.weight_ih_l{l}" in sd:
        layers.append({
            "W_ih": np.asarray(sd[f"lstm.weight_ih_l{l}"], dtype=np.float64),
            "W_hh": np.asarray(sd[f"lstm.weight_hh_l{l}"], dtype=np.float64),
            "b": np.asarray(sd[f"lstm.bias_ih_l{l}"], dtype=np.float64)
                 + np.asarray(sd[f"lstm.bias_hh_l{l}"], dtype=np.float64),
        })
        l += 1
    return {
        "layers": layers,
        "W_fc": np.asarray(sd["fc.weight"], dtype=np.float64),
        "b_fc": np.asarray(sd["fc.bias"], dtype=np.float64),
    }


def get_weights_from_pth(file: Path) -> dict:
    """### Lê o .pt (state_dict salvo em cuda:0) e devolve a estrutura de weights_from_state_dict"""
    import torch
    sd = torch.load(file, map_location="cpu")
    return weights_from_state_dict({k: v.numpy() for k, v in sd.items()})


def quantize_weights(weights: dict, cfg: FixedConfig) -> dict:
    """### Quantiza os pesos
        - Matrizes em cfg.w.
        - Biases no formato do acumulador ao qual são somados: assim a soma
          acontece sem nenhum shift e sem perder precisão.
        - acc_frac de cada camada = w.frac + max(frac da entrada, h.frac):
          W_ih·x e W_hh·h são alinhados nesse formato antes de somar.
        #### Retorno:
            - mesma estrutura, dtype int64, mais "acc_frac" em cada camada e "acc_frac_fc"
    """
    layers_q = []
    for l, layer in enumerate(weights["layers"]):
        x_fmt = cfg.x if l == 0 else cfg.h
        acc_frac = cfg.w.frac_bits + max(x_fmt.frac_bits, cfg.h.frac_bits)
        layers_q.append({
            "W_ih": quantize(layer["W_ih"], cfg.w),
            "W_hh": quantize(layer["W_hh"], cfg.w),
            "b": quantize(layer["b"], QFormat(ACC_BITS, acc_frac)),
            "acc_frac": acc_frac,
        })
    acc_frac_fc = cfg.w.frac_bits + cfg.h.frac_bits
    return {
        "layers": layers_q,
        "W_fc": quantize(weights["W_fc"], cfg.w),
        "b_fc": quantize(weights["b_fc"], QFormat(ACC_BITS, acc_frac_fc)),
        "acc_frac_fc": acc_frac_fc,
    }


# ---------------------------------------------------------------------------
# Bloco 1 — Referência em float
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lstm_step_float(
    x_t: np.ndarray, h: np.ndarray, c: np.ndarray, layer: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """### Um passo de UMA camada LSTM em float
        #### shapes:
            - x_t: (N, F) na camada 0, (N, H) nas demais
            - h, c: (N, H)
        #### Retorno:
            - (h_novo, c_novo), ambos (N, H)
    """
    H = h.shape[1]
    z = x_t @ layer["W_ih"].T + h @ layer["W_hh"].T + layer["b"]   # (N, 4H)
    i = sigmoid(z[:, 0 * H:1 * H])
    f = sigmoid(z[:, 1 * H:2 * H])
    g = np.tanh(z[:, 2 * H:3 * H])
    o = sigmoid(z[:, 3 * H:4 * H])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)
    return h_new, c_new


def forward_float(windows: np.ndarray, weights: dict) -> np.ndarray:
    """### Percorre as janelas a partir de h = c = 0 e aplica a Linear no último h
        #### shapes:
            - windows: (N, T, F)
        #### Retorno:
            - y: (N, 2)
    """
    N, T, _ = windows.shape
    H = weights["layers"][0]["W_hh"].shape[1]
    hs = [np.zeros((N, H)) for _ in weights["layers"]]
    cs = [np.zeros((N, H)) for _ in weights["layers"]]
    for t in range(T):
        inp = windows[:, t, :]
        for l, layer in enumerate(weights["layers"]):
            hs[l], cs[l] = lstm_step_float(inp, hs[l], cs[l], layer)
            inp = hs[l]
    return hs[-1] @ weights["W_fc"].T + weights["b_fc"]


# ---------------------------------------------------------------------------
# Bloco 2 — Ferramentas de quantização
# ---------------------------------------------------------------------------

def saturate(q: np.ndarray, fmt: QFormat) -> np.ndarray:
    """### Trava os inteiros na faixa de fmt (saturação, nunca overflow)"""
    return np.clip(q, fmt.qmin, fmt.qmax)


def quantize(x: np.ndarray, fmt: QFormat) -> np.ndarray:
    """### Float -> inteiro: multiplica por 2^frac, arredonda (meio para cima) e satura
        #### Retorno: mesmo shape de x, dtype int64
    """
    q = np.floor(np.asarray(x, dtype=np.float64) * (1 << fmt.frac_bits) + 0.5)
    return saturate(q, fmt).astype(np.int64)


def dequantize(q: np.ndarray, frac_bits: int) -> np.ndarray:
    """### Inteiro -> float: divide por 2^frac_bits"""
    return np.asarray(q, dtype=np.float64) / (1 << frac_bits)


def align(q: np.ndarray, from_frac: int, to_frac: int) -> np.ndarray:
    """### Aumenta os bits fracionários (shift à esquerda): não perde precisão"""
    assert to_frac >= from_frac
    return q << (to_frac - from_frac)


def matvec_fixed(v_q: np.ndarray, W_q: np.ndarray) -> np.ndarray:
    """### v · Wᵀ em precisão total, SEM shift (o acumulador de um DSP)
        #### shapes:
            - v_q: (N, n) int64 com frac fv
            - W_q: (m, n) int64 com frac fw
        #### Retorno: (N, m) int64 com frac fv + fw
    """
    return v_q @ W_q.T


def requantize(acc: np.ndarray, acc_frac: int, fmt: QFormat) -> np.ndarray:
    """### Reduz o acumulador a fmt: shift à direita com arredondamento + saturação
        Em hardware: somar 2^(s-1) e descartar os s bits de baixo.
        O >> do NumPy em int64 é aritmético (preserva o sinal), como no HDL.
    """
    s = acc_frac - fmt.frac_bits
    if s > 0:
        r = (acc + (1 << (s - 1))) >> s
    else:
        r = acc << (-s)
    return saturate(r, fmt)


def mult_fixed(a_q: np.ndarray, a_frac: int, b_q: np.ndarray, b_frac: int, fmt: QFormat) -> np.ndarray:
    """### Produto elemento a elemento (⊙) reduzido ao formato fmt"""
    return requantize(a_q * b_q, a_frac + b_frac, fmt)


# ---------------------------------------------------------------------------
# Bloco 4 — Medir faixas (definido antes do Bloco 3, que o usa)
# ---------------------------------------------------------------------------

@dataclass
class RangeTracker:
    """### Registra mín/máx (em float) de cada sinal interno e compara com o formato"""
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)
    fmts: dict[str, QFormat | None] = field(default_factory=dict)

    def update(self, name: str, values: np.ndarray, fmt: QFormat | None = None) -> None:
        self.mins[name] = min(self.mins.get(name, np.inf), float(np.min(values)))
        self.maxs[name] = max(self.maxs.get(name, -np.inf), float(np.max(values)))
        self.fmts[name] = fmt

    @staticmethod
    def int_bits_needed(lo: float, hi: float) -> int:
        """Menor k com 2^k > max(|lo|, |hi|), mais 1 bit de sinal."""
        m = max(abs(lo), abs(hi))
        k = 0
        while (1 << k) <= m:
            k += 1
        return k + 1

    def report(self) -> None:
        print(f"{'sinal':<14}{'mín':>10}{'máx':>10}{'bits int':>10}  formato")
        for name in self.mins:
            lo, hi, fmt = self.mins[name], self.maxs[name], self.fmts[name]
            need = self.int_bits_needed(lo, hi)
            if fmt is None:
                tag = "(pré-ativação: entra na LUT)"
            elif need > fmt.int_bits:
                tag = f"{fmt}  <-- SATURA"
            else:
                tag = f"{fmt}  ok"
            print(f"{name:<14}{lo:>10.3f}{hi:>10.3f}{need:>10d}  {tag}")


# ---------------------------------------------------------------------------
# Bloco 3 — Passo e forward em ponto fixo
# ---------------------------------------------------------------------------

def lstm_step_fixed(
    x_t: np.ndarray, h: np.ndarray, c: np.ndarray,
    layer_q: dict, x_fmt: QFormat, cfg: FixedConfig,
    tracker: RangeTracker | None = None, tag: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """### Um passo de UMA camada LSTM usando só inteiros
        #### shapes (int64):
            - x_t: (N, F) em cfg.x na camada 0; (N, H) em cfg.h nas demais (formato em x_fmt)
            - h:   (N, H) em cfg.h
            - c:   (N, H) em cfg.c
        #### Retorno: (h_novo em cfg.h, c_novo em cfg.c)
        Versão 1 das ativações: dequantiza a pré-ativação, aplica sigmoid/tanh
        em float e quantiza em cfg.gate. Isso é o que uma LUT ideal faria; a LUT
        real entra depois, para medir o erro dela separadamente.
    """
    H = h.shape[1]
    fw, g_frac = cfg.w.frac_bits, cfg.gate.frac_bits
    acc_frac = layer_q["acc_frac"]

    # 1) Pré-ativações: W_ih·x e W_hh·h em precisão total, alinhados e somados ao bias
    ax = align(matvec_fixed(x_t, layer_q["W_ih"]), fw + x_fmt.frac_bits, acc_frac)
    ah = align(matvec_fixed(h, layer_q["W_hh"]), fw + cfg.h.frac_bits, acc_frac)
    acc = ax + ah + layer_q["b"]                                     # (N, 4H), frac acc_frac
    z = dequantize(acc, acc_frac)

    # 2) Ativações (versão 1: "LUT ideal")
    i = quantize(sigmoid(z[:, 0 * H:1 * H]), cfg.gate)
    f = quantize(sigmoid(z[:, 1 * H:2 * H]), cfg.gate)
    g = quantize(np.tanh(z[:, 2 * H:3 * H]), cfg.gate)
    o = quantize(sigmoid(z[:, 3 * H:4 * H]), cfg.gate)

    # 3) c_novo = f ⊙ c + i ⊙ g  (os dois produtos têm frac diferentes: alinhar antes de somar)
    fc = f * c                                   # frac g_frac + c_frac
    ig = i * g                                   # frac 2·g_frac
    common = max(g_frac + cfg.c.frac_bits, 2 * g_frac)
    acc_c = align(fc, g_frac + cfg.c.frac_bits, common) + align(ig, 2 * g_frac, common)
    c_new = requantize(acc_c, common, cfg.c)

    # 4) h_novo = o ⊙ tanh(c_novo)
    tc = quantize(np.tanh(dequantize(c_new, cfg.c.frac_bits)), cfg.gate)
    h_new = mult_fixed(o, g_frac, tc, g_frac, cfg.h)

    if tracker is not None:
        for k, name in enumerate("ifgo"):
            tracker.update(f"{tag}pre_{name}", z[:, k * H:(k + 1) * H])
        tracker.update(f"{tag}c", acc_c / (1 << common), cfg.c)    # antes da saturação
        tracker.update(f"{tag}h", dequantize(h_new, cfg.h.frac_bits), cfg.h)
    return h_new, c_new


def forward_fixed(
    windows: np.ndarray, weights_q: dict, cfg: FixedConfig,
    tracker: RangeTracker | None = None,
) -> np.ndarray:
    """### Par da forward_float, em ponto fixo
        #### shapes:
            - windows: (N, T, F) float, quantizadas aqui para cfg.x
        #### Retorno:
            - y: (N, 2) float, dequantizado de cfg.out
    """
    N, T, _ = windows.shape
    H = weights_q["layers"][0]["W_hh"].shape[1]
    if tracker is not None:
        tracker.update("x", windows, cfg.x)
    xq = quantize(windows, cfg.x)
    hs = [np.zeros((N, H), dtype=np.int64) for _ in weights_q["layers"]]
    cs = [np.zeros((N, H), dtype=np.int64) for _ in weights_q["layers"]]
    for t in range(T):
        inp, inp_fmt = xq[:, t, :], cfg.x
        for l, layer_q in enumerate(weights_q["layers"]):
            hs[l], cs[l] = lstm_step_fixed(inp, hs[l], cs[l], layer_q, inp_fmt, cfg,
                                           tracker, tag=f"L{l}_")
            inp, inp_fmt = hs[l], cfg.h

    acc = matvec_fixed(hs[-1], weights_q["W_fc"]) + weights_q["b_fc"]
    acc_frac = weights_q["acc_frac_fc"]
    if tracker is not None:
        tracker.update("out", dequantize(acc, acc_frac), cfg.out)
    return dequantize(requantize(acc, acc_frac, cfg.out), cfg.out.frac_bits)


# ---------------------------------------------------------------------------
# Bloco 5 — Dados, métricas e comparação
# ---------------------------------------------------------------------------

def load_data(csv: Path, T: int, split=(0.6, 0.2, 0.2)):
    """### Lê o CSV, normaliza com média/desvio do TREINO e monta as janelas do TESTE
        Janela n: X[n-T+1 .. n]  ->  alvo Y[n]   (montada só dentro do conjunto de teste)
        #### Retorno:
            - windows: (N, T, 2) entrada normalizada
            - y_true:  (N, 2)    alvo normalizado
            - y_mean, y_std: (2,) para desnormalizar a saída
    """
    data = np.genfromtxt(csv, delimiter=",", names=True)
    X = np.column_stack([data["Xreal"], data["Ximg"]])
    Y = np.column_stack([data["Yreal"], data["Yimg"]])
    n_tr = int(split[0] * len(X))
    n_va = int((split[0] + split[1]) * len(X))
    x_mean, x_std = X[:n_tr].mean(0), X[:n_tr].std(0)       # StandardScaler usa ddof=0
    y_mean, y_std = Y[:n_tr].mean(0), Y[:n_tr].std(0)
    Xs = (X[n_va:] - x_mean) / x_std
    Ys = (Y[n_va:] - y_mean) / y_std
    idx = np.arange(T - 1, len(Xs))
    windows = np.stack([Xs[n - T + 1:n + 1] for n in idx])
    return windows, Ys[idx], y_mean, y_std


def nmse_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return 10 * np.log10(np.sum(e ** 2) / np.sum(y_true ** 2))


def evm_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return 100 * np.sqrt(np.sum(e ** 2) / np.sum(y_true ** 2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("best_model.pt"))
    ap.add_argument("--data", type=Path, default=Path("dadosIniciais.csv"))
    ap.add_argument("--T", type=int, default=10)
    args = ap.parse_args(argv)

    weights = get_weights_from_pth(args.model)
    run(weights, args.data, args.T, FixedConfig())


def run(weights: dict, data: Path, T: int, cfg: FixedConfig) -> None:
    """### Compara float x ponto fixo no conjunto de teste"""
    windows, y_true_s, y_mean, y_std = load_data(data, T)

    y_float_s = forward_float(windows, weights)
    tracker = RangeTracker()
    y_fixed_s = forward_fixed(windows, quantize_weights(weights, cfg), cfg, tracker)

    # Métricas no domínio original (desnormalizado)
    y_true = y_true_s * y_std + y_mean
    y_float = y_float_s * y_std + y_mean
    y_fixed = y_fixed_s * y_std + y_mean

    print(f"Formatos: x={cfg.x} w={cfg.w} gate={cfg.gate} h={cfg.h} c={cfg.c} out={cfg.out}")
    print(f"Janelas de teste: {len(windows)}\n")
    print(f"{'':<22}{'NMSE (dB)':>10}{'EVM (%)':>10}")
    print(f"{'float vs. real':<22}{nmse_db(y_true, y_float):>10.2f}{evm_percent(y_true, y_float):>10.3f}")
    print(f"{'ponto fixo vs. real':<22}{nmse_db(y_true, y_fixed):>10.2f}{evm_percent(y_true, y_fixed):>10.3f}")
    print(f"{'ponto fixo vs. float':<22}{nmse_db(y_float, y_fixed):>10.2f}{evm_percent(y_float, y_fixed):>10.3f}")
    print("  (a última linha é só o erro de quantização)\n")
    tracker.report()


if __name__ == "__main__":
    main()