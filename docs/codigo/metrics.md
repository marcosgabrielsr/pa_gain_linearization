# `src/metrics/metrics.py` — métricas

Todas as métricas seguem a mesma interface: `Metrica(...).compute(y_true, y_pred)`. `y_true` e `y_pred` podem ser vetores complexos `(N,)` ou arrays reais `(N, 2)` = [I, Q] (é o formato que o `TrainerLSTM` usa).

| Classe | Fórmula | Unidade | Melhor |
|---|---|---|---|
| `RMSE` | √mean\|e\|² | unid. do sinal | menor |
| `EVM` | 100 · RMS(e) / RMS(y_true) | % | menor |
| `RSQR` | 1 − SS_res / SS_tot | – | maior (→ 1) |
| `AIC(n_params)` | n·ln(MSE) + 2k | – | menor |
| `BIC(n_params)` | n·ln(MSE) + k·ln(n) | – | menor |
| `NMSE` | 10·log10( Σ\|e\|² / Σ\|y_true\|² ) | dB | mais negativo |
| `ACEPR(fs, bw)` | 10·log10( P_erro(canal adjacente) / P_sinal(canal principal) ) | dB | mais negativo |

Aqui, e = y_true − y_pred. `NMSE` e `ACEPR` foram adicionados em 2026-11-12.

## Funções auxiliares

- `as_complex(y)`: converte `(N, 2)` [I, Q] em complexo; um vetor que já é complexo passa direto.
- `psd(sig, fs, nperseg=2048)`: densidade espectral de potência pelo método de Welch (janela Hann), em dB/Hz. Retorna `(f, P)` com f de −fs/2 a fs/2 (sinal complexo em banda base: o espectro tem os dois lados).

## ACEPR em detalhe

*Adjacent Channel Error Power Ratio*: mede quanto do **erro do modelo** cai fora da banda do sinal. É importante para PA/DPD porque a distorção não linear "espalha" o espectro para os canais vizinhos.

```
canal principal:    |f| <= bw/2
canais adjacentes:  bw/2 < |f| <= 3·bw/2   (lado inferior e superior, mesma largura do principal)
ACEPR = 10·log10( max(P_erro_inferior, P_erro_superior) / P_sinal_principal )
```

- Retorna o **pior lado** (o maior dos dois).
- As potências são somas dos bins da PSD linear. Como os bins são uniformes, a razão entre somas é igual à razão entre integrais.
- `nperseg = 2048`: com fs = 983,04 MHz, dá uma resolução de cerca de 480 kHz. Um `nperseg` maior dá mais resolução, mas uma curva mais ruidosa.
- **Só faz sentido com dados em ordem temporal** (APA_200MHz). No nosso PA (`expData.csv` embaralhado, [ADR-0001](../adr/0001-manter-dados-embaralhados.md)), o "espectro" não tem significado físico.
- Para o APA: `fs = spec["input_signal_fs"]` (983,04 MHz) e `bw = spec["bw_main_ch"]` (200 MHz), lidos de `data/raw/APA_200MHz/spec.json`.

## Uso no `grid_search_lstm.ipynb` (seção "3. Analysing Results")

1. Uma célula de métricas imprime, juntos, EVM, RMSE, R², NMSE e, no APA, o ACEPR.
2. Os gráficos AM/AM, AM/PM e PM/PM vêm em seguida.
3. A **PSD** fica no fim (`plot_PSD`): saída medida, previsão da LSTM e erro, com as bordas dos canais tracejadas.
