# `src/pa_model/dataset.py` — leitura e preparo dos dados

A classe `Dataset` guarda dois vetores complexos: `x` (entrada do PA) e `y` (saída do PA).

| Método | O que faz |
|---|---|
| `Dataset.from_csv(caminho)` | Lê um CSV único com `Xreal, Ximg, Yreal, Yimg`. É o formato do nosso PA (`expData.csv`) |
| `Dataset.from_split_csv(pasta)` | Lê um dataset **já dividido** no formato do OpenDPD e devolve `(train, val, test)`. Adicionado em 2026-11-09 ([ADR-0005](../adr/0005-dataset-apa-200mhz.md)) |
| `ds.split(train=0.6, val=0.2)` | Corta por índice em treino/validação/teste, sem embaralhar. Usado só com `from_csv` |
| `ds.normalize(scaler_x, scaler_y)` | `StandardScaler` em I e Q. Sem scalers, ajusta novos (use no treino); com scalers, só aplica (use em validação e teste) |

## `from_split_csv`

```python
train, val, test = Dataset.from_split_csv(DATA_RAW / "APA_200MHz")
```

Espera, dentro da pasta, os arquivos:

```
train_input.csv  train_output.csv
val_input.csv    val_output.csv
test_input.csv   test_output.csv
```

Cada arquivo tem as colunas `I, Q`. O `_input` é a entrada do PA (x) e o `_output`, a saída (y). As linhas `n` de `_input` e `_output` são a mesma amostra.

- A ordem temporal é mantida: nada é embaralhado.
- O split é o oficial do dataset, para que os resultados sejam comparáveis com os do OpenDPD.
- Os dados do APA já vêm normalizados (máx = 1), mas o `normalize()` continua sendo aplicado no pipeline, igual ao do nosso PA.

## Uso no `grid_search_lstm.ipynb`

```python
DATASET = "APA_200MHz"   # ou "pa_proprio"
```

| `DATASET` | Carga | Split | `RUN_DIR` (onde o grid search salva) |
|---|---|---|---|
| `"pa_proprio"` | `expData.csv` (`from_csv`) | `ds.split()` | `results/v2/` |
| `"APA_200MHz"` | `from_split_csv` | oficial | `results/APA_200MHz/` |

A seção de análise lê o melhor modelo de:
- `RUN_DIR`, no APA;
- `results/`, no nosso PA. Esse é o modelo antigo, até a comparação entre o `v2` e o backup ([ADR-0004](../adr/0004-otimizar-auxiliares-manter-grid-search.md)).
