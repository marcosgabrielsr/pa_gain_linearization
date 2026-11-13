# `src/plots/plots.py` — gráficos

Funções de gráfico dos sinais do PA. Antes ficavam definidas dentro do `grid_search_lstm.ipynb`; foram movidas para cá em 2026-11-13.

```python
from plots import (plot_AM_AM, plot_AM_PM, plot_PM_PM, plot_constellation, plot_PSD,
                   plot_comparison, COLOR_DATASET, COLOR_LSTM, COLOR_ERROR)
```

## Funções

| Função | Eixos | Observação |
|---|---|---|
| `plot_AM_AM(x, y, db=True)` | \|x\| × \|y\| | compressão de ganho |
| `plot_AM_PM(x, y, db=True, remove_offset=True)` | \|x\| × (∠y − ∠x) | rotação de fase com a amplitude; `remove_offset` tira a rotação constante (média circular) |
| `plot_PM_PM(x, y)` | ∠x × ∠y | |
| `plot_constellation(y)` | I × Q | amostras **no tempo** (ver abaixo) |
| `plot_PSD(signals, fs, bw=None, colors=None)` | f (MHz) × PSD (dB/Hz) | várias séries: `signals = {label: sinal}`; `bw` desenha as bordas dos canais |
| `plot_comparison(x, y_true, y_pred, fs=None, bw=None)` | grade | todas as anteriores, com dataset e LSTM sobrepostos |

## Argumentos comuns: `color`, `ax`, `label`

- **`color`**: a cor da série. As cores padrão do projeto:

  | Constante | Cor | Uso |
  |---|---|---|
  | `COLOR_DATASET` | azul (`tab:blue`) | dados medidos |
  | `COLOR_LSTM` | laranja (`tab:orange`) | previsão da LSTM |
  | `COLOR_ERROR` | cinza (`tab:gray`) | erro (na PSD) |

- **`ax`**: sem `ax`, a função cria a figura e já mostra (`plt.show()`), como antes. Com `ax`, ela só desenha naquele eixo, e assim dá para **sobrepor** séries:
  ```python
  fig, ax = plt.subplots(figsize=(10, 5))
  plot_AM_AM(x, y_true, ax=ax, color=COLOR_DATASET, label="Dataset")
  plot_AM_AM(x, y_pred, ax=ax, color=COLOR_LSTM,    label="LSTM")
  plt.show()
  ```
- **`label`**: nome da série na legenda. A legenda só aparece se houver `label`.

## Constelação: por que é uma "nuvem"

`plot_constellation` mostra I × Q de cada amostra **no tempo**. Num sinal OFDM, cada amostra é a soma de centenas de subportadoras, e o resultado parece uma nuvem circular (quase gaussiana), não os 256 pontos do 256QAM.

Para ver os pontos QAM, seria preciso **demodular**: sincronizar no prefixo cíclico, aplicar a FFT por símbolo e pegar as subportadoras ativas. No APA_200MHz, isso não é possível no conjunto de teste: um símbolo OFDM tem 32.768 amostras mais o prefixo cíclico, e o teste tem só 19.662 amostras.

Ainda assim, o gráfico é útil. A compressão do PA aparece como um **achatamento na borda da nuvem**, e dá para comparar visualmente o espalhamento do dataset com o da LSTM.

## `plot_comparison`

Uma figura só, com as duas séries sobrepostas (dataset em azul, LSTM em laranja):

```
┌─────────┬─────────┬─────────┐
│  AM/AM  │  AM/PM  │  PM/PM  │
├─────────┼─────────┴─────────┤
│ Constel.│        PSD        │
└─────────┴───────────────────┘
```

- Na PSD também aparece o erro (cinza), com as bordas do canal principal e dos adjacentes tracejadas.
- Sem `fs` (nosso PA, com dados embaralhados), o painel da PSD fica oculto.
- Como a série da LSTM é desenhada por cima, onde as duas coincidem só se vê o laranja. As regiões em que o azul aparece são onde o modelo erra.
