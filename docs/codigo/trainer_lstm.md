# `src/lstm/trainer_lstm.py` e `src/lstm/earlystopping.py` — treino da LSTM

Otimizados em 2026-11-05 ([ADR-0004](../adr/0004-otimizar-auxiliares-manter-grid-search.md)). A interface pública não mudou: `TrainerLSTM(...)`, `fit(train_loader, val_loader)` e `evaluate(val_loader)` recebem e devolvem o mesmo que antes.

## Fluxo

```
fit()
 └─ para cada época:
     ├─ _fit(train_loader)     -> model.train()  + _run_epoch(train=True)
     ├─ evaluate(val_loader)   -> model.eval()   + no_grad + _run_epoch(train=False)
     ├─ guarda as 6 métricas de treino e de validação no history
     └─ early stopping (pela val_evm)
```

Cada `_run_epoch` devolve `(avg_loss, rmse, evm, rsqr, aic, bic)`.

## O que mudou e por quê

### 1. Sincronização com a GPU: uma vez por época, não por lote

A GPU trabalha **de forma assíncrona**: a CPU enfileira as operações e segue em frente. Qualquer coisa que precise do **valor** de um tensor na CPU (`.item()`, `.cpu()`) obriga a CPU a esperar a GPU terminar tudo o que está na fila.

| Antes (por lote) | Agora |
|---|---|
| `total_loss += loss.item() * n` | `total_loss += loss.detach().double() * n` (continua na GPU) |
| `y_pred.append(outputs.detach().cpu())` | `y_pred.append(outputs.detach())` (continua na GPU) |
| – | no fim da época: `torch.cat(y_pred).cpu()` e `(total_loss / total).item()`, uma vez só |

- `.detach()`: separa o tensor do grafo de gradientes. Sem isso, o grafo de todos os lotes ficaria preso na memória.
- `.double()`: a soma da loss é feita em `float64`, como era antes, quando o `float` do Python já é 64 bits.

### 2. Um único laço para treino e validação

`_fit` e `evaluate` tinham o mesmo laço e as mesmas 8 linhas de métricas. Agora:
- `_run_epoch(loader, train)` percorre o loader e só executa `zero_grad/backward/step` se `train=True`;
- `_metrics(y_true, y_pred)` desnormaliza com o `scaler` de saída e calcula RMSE, EVM, R², AIC e BIC em NumPy `float64`, igual a antes.

As métricas de **treino** continuam sendo calculadas, para os históricos manterem as mesmas 12 colunas.

### 3. Histórico montado com laço

```python
history = {f"{split}_{k}": [] for split in ("train", "val") for k in KEYS}
for k in KEYS:
    history[f"train_{k}"].append(train_results[k])
    history[f"val_{k}"].append(val_results[k])
```

As colunas e a ordem são as mesmas de antes (`train_avg_loss ... train_bic, val_avg_loss ... val_bic`).

### 4. Early stopping: copiar só quando melhora

Antes, `check_early_stop(val, copy.deepcopy(self.model))` copiava o modelo **toda época**, porque o argumento é avaliado antes da chamada. Agora o trainer passa `self.model`, e o `EarlyStopping` faz `copy.deepcopy(model)` só quando a métrica melhora. O resultado é o mesmo.

## O que ficou de fora (de propósito)

- Precisão mista (AMP), `torch.compile`, `cudnn.benchmark`: mudariam os números ou a reprodutibilidade e impediriam comparar com o backup.

## Validação

`scripts/check_trainer.py` (rodar na raiz do projeto, na máquina com GPU):

1. confere se o `WindowLoader` e o `DataLoader` geram os mesmos lotes;
2. treina a melhor combinação (ws10, hs16, nl2, bs128) por 5 épocas, como o `GridSearchLSTM`, e mede o tempo por época;
3. compara o histórico com `results_backup_2026-11-05/histories/ws10_hs16_nl2_lr0.001_dr0_bs128.csv`.

Para medir o ganho de tempo, rode também com `--dataloader` (mesmo trainer novo, mas com o `DataLoader` antigo).
