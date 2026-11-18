# `notebooks/dpd_lstm.ipynb`

Notebook principal do DPD, agora para o **APA_200MHz** ([ADR-0007](../adr/0007-dpd-do-apa-com-memoria-sem-back-off.md)). Até 2026-11-18, a Etapa 1 usava a LSTM do nosso PA (`results/best_model.pt`, `expData.csv`). Essa versão continua no histórico do git.

O código do DPD fica no pacote [`src/dpd/`](dpd.md). O notebook só chama as funções e analisa os resultados.

## Pré-requisitos (na máquina do orientador)

1. `git pull` e `pip install -e .` na raiz. O pacote `dpd` é novo, e uma instalação *editable* antiga pode não encontrá-lo.
2. O grid search do PA do APA já rodado (`grid_search_lstm.ipynb` com `DATASET = "APA_200MHz"`). Ele gera, em `results/APA_200MHz/`, os arquivos `best_model.pt`, `best_params.json` e `scalers.joblib`.
3. O [`apa_linear_gain.ipynb`](apa_linear_gain.md) rodado, que gera `results/APA_200MHz/linear_gain.json`.
4. Recomendado: `python scripts/check_dpd.py` antes do grid (ver abaixo).

## Variáveis principais

| Nome | Valor | O que é |
|---|---|---|
| `PA_DIR` | `results/APA_200MHz` | LSTM do PA e `linear_gain.json` |
| `DPD_DIR` | `results/APA_200MHz/dpd` | resultados do grid search do DPD |
| `WP` | janela da LSTM do PA (de `best_params.json`) | |
| `WD` | janela do melhor DPD | |
| `FS`, `BW` | 983,04 MHz e 200 MHz (do `spec.json`) | para o ACEPR e a PSD |

---

## Etapa 1 — Carregar o gêmeo digital do PA e validá-lo

**Por quê:** o DPD é treinado *através* da LSTM do PA. Antes, é preciso garantir que ela foi carregada como no treino e saber como se comporta fora da faixa medida, porque o DPD vai levar a entrada do PA para lá.

- **1.2 Dados:** `Dataset.from_split_csv` usa o split oficial (60/20/20), em ordem temporal, o mesmo do grid search do PA.
- **1.3 LSTM congelada:** `load_pa_twin(PA_DIR, device)` carrega os pesos, a janela `WP` e os **scalers do treino do PA** (`scalers.joblib`), zera o dropout e congela os pesos ([dpd.md](dpd.md#load_pa_twinrun_dir-device--pa-wp-scaler_x-scaler_y)).
- **1.4 Validação:** NMSE e EVM da LSTM no teste, contra o y medido, com a saída alinhada por `[WP-1:]`. O resultado deve ficar próximo ao do `grid_search_lstm.ipynb`. Uma diferença grande indica janela, scaler ou pesos errados.
- **1.5 Extrapolação:** AM/AM linear com o PA medido, a LSTM e a LSTM com x × 1,3. A linha tracejada marca o max|x| do treino.

| À direita da linha tracejada, a curva... | Significado | Consequência para o DPD |
|---|---|---|
| achata (satura) | fisicamente plausível | o DPD não consegue passar da saturação |
| continua subindo | o modelo "acredita" que o PA não satura | o DPD pode explorar esse erro, com um u grande que no PA real não funcionaria |
| oscila ou cai de forma estranha | extrapolação ruim | os picos do DPD não são confiáveis |

## Etapa 2 — Ganho linear G

Lê `linear_gain.json` ([ADR-0006](../adr/0006-ganho-linear-do-apa.md)): G = 1,1660 + j0,0057.

Também conta quantas amostras de treino têm |G·x| acima do max|y| de treino. Nessas amostras, o alvo está fora do que o PA entregou nas medições, e o DPD terá de extrapolar. Não há back-off, por decisão do Marcos ([ADR-0007](../adr/0007-dpd-do-apa-com-memoria-sem-back-off.md)).

## Etapa 3 — Grid search do DPD

```python
gsearch = GridSearchDPD(train, val, pa, WP, scaler_x, scaler_y, G, param_grid, device)
best_metric, best_dpd, best_params = gsearch.run(DPD_DIR)
```

- A grade (36 combinações) e o motivo de cada valor estão no [ADR-0008](../adr/0008-grid-search-do-dpd-sem-baseline.md).
- Assim como no grid do PA, combinações já feitas em `DPD_DIR` são puladas. Dá para interromper e retomar.
- Como cada passo atravessa o DPD **e** a LSTM do PA, cada combinação demora mais que no grid do PA.

## Etapa 4 — Resultados

1. **Summary:** todas as combinações, ordenadas pela NMSE de validação.
2. **Melhor DPD:** carregado de `DPD_DIR` (`ResultsStorage.load_best`), com o histórico:
   - NMSE de treino e de validação por época;
   - **max |u|** por época, com uma linha no max|x| de treino. Quanto mais acima da linha, mais o DPD depende da extrapolação da LSTM.
3. **Teste, sem DPD × com DPD**, sempre contra o alvo G·x:
   - sem DPD: `run_pa(test.x)[WD-1:]` (a LSTM do PA sozinha);
   - com DPD: `run_dpd(test.x)`;
   - os dois cortados para começar em x[WD+WP−2], para comparar as mesmas amostras (`x_t`, `target = G·x_t`);
   - métricas: NMSE (dB), EVM (%) e ACEPR (dB). Um DPD que funciona **diminui** as três;
   - max|u| no teste.
4. **AM/AM e AM/PM** (eixo |x| linear): alvo G·x (cinza), sem DPD (azul) e com DPD (laranja). Com o DPD funcionando, a curva laranja se aproxima da reta do alvo e a AM/PM fica perto de ∠G.
5. **PSD:** com o DPD, a potência fora do canal principal (bordas tracejadas) deve cair em direção à do alvo.
6. **Característica do DPD** (|u| × |x|, quadrado): acima da diagonal, o DPD está **expandindo** para compensar a compressão do PA. A linha pontilhada vermelha marca o max|x| de treino, e o que fica acima dela é a extrapolação.

## `scripts/check_dpd.py`

Teste rápido do pipeline na máquina do orientador, antes do grid:

```bash
python scripts/check_dpd.py                 # ws=5, hs=16, nl=1, 3 épocas
python scripts/check_dpd.py --ws 10 --epochs 5
```

1. O gêmeo é determinístico em `train()` (dropout zerado)?
2. O DPD não treinado é a identidade (u = x, e DPD+PA = PA)?
3. Treina **uma** combinação como o `GridSearchDPD` e imprime o tempo por época e o histórico. Serve para estimar o tempo do grid inteiro: tempo por época × épocas (até 50) × 36.
