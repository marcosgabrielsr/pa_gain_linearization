# `src/dpd/` — DPD treinado através do gêmeo digital do PA

Pacote do DPD do APA_200MHz ([ADR-0007](../adr/0007-dpd-do-apa-com-memoria-sem-back-off.md)). Segue a mesma estrutura do pacote `lstm` (modelo, trainer, grid search) e reaproveita `PaLSTM`, `EarlyStopping`, `ResultsStorage`, `SWDataset` e `WindowLoader`.

```python
from dpd import DpdLSTM, GridSearchDPD, TrainerDPD, load_pa_twin, cascade, run_pa, run_dpd
```

| Arquivo | Conteúdo |
|---|---|
| `model.py` | `DpdLSTM`: o DPD |
| `chain.py` | `load_pa_twin`, `cascade`, `run_pa`, `run_dpd`: o gêmeo digital e a cadeia DPD → PA |
| `trainer_dpd.py` | `TrainerDPD`: treino de uma combinação |
| `grid_search_dpd.py` | `GridSearchDPD`: o grid search |

## A ideia (aprendizado direto, [ADR-0002](../adr/0002-dpd-por-aprendizado-direto-com-gemeo-digital.md))

```
x ──► DPD (treinável) ──u──► PA_LSTM (congelada) ──► ŷ        loss = MSE(ŷ, G·x)
```

- Não sabemos qual deveria ser u (é o que queremos descobrir), mas sabemos o que queremos **depois** do PA: G·x.
- Não dá para fazer *backpropagation* pelo PA físico. Por isso usamos a LSTM do PA no lugar dele. O gradiente da loss **atravessa a LSTM congelada** (os pesos dela não mudam) e atualiza só os pesos do DPD.
- O treino **não usa o y medido**: só x, G e a LSTM do PA. O y medido serviu para treinar a LSTM do PA.

## `DpdLSTM(hidden_size, num_layers, dropout=0.0)` — `model.py`

```
u[n] = x[n] + f(x[n−W+1 … n])
```

- `f` é uma `PaLSTM`: recebe uma janela de W amostras de x, com shape (B, W, 2), e devolve uma correção em I/Q, com shape (B, 2).
- **Conexão residual:** o DPD aprende só a **correção** em torno da identidade, que é mais fácil do que aprender o sinal inteiro.
- **A camada final começa em zero** (`nn.init.zeros_` em `f.fc`). Antes do treino, f = 0 e u = x: o DPD começa como a identidade, e o treino parte de "sem DPD". Como a entrada da última camada não é zero, o gradiente dela já é diferente de zero no primeiro passo, e o resto da rede passa a receber gradiente em seguida.
- Com W = 1, o DPD é **sem memória** (só olha a amostra atual). Ele entra no grid como referência.

## `chain.py`

### `load_pa_twin(run_dir, device)` → `(pa, wp, scaler_x, scaler_y)`

Carrega a melhor LSTM de um grid search do PA (`best_model.pt`, `best_params.json`, `scalers.joblib`) como gêmeo digital:

- `requires_grad_(False)` em todos os pesos: **congelada**;
- `pa.lstm.dropout = 0.0` e `pa.train()`: na GPU, o *backward* da LSTM do cuDNN só é permitido em `train()`. Se o melhor modelo do PA tiver `dr > 0` (a grade do APA testou 0 e 0,5), o `train()` ligaria o dropout e o gêmeo responderia de forma aleatória. Zerar o dropout não muda os pesos, porque ele só age durante o treino do próprio PA;
- os **scalers são os do treino do PA**. O DPD trabalha no espaço normalizado da **entrada** do PA (`scaler_x`), e a loss é calculada no espaço da **saída** (`scaler_y`).

### `cascade(dpd, pa, x_windows, wd, wp)` → `(ŷ, u)`: a cadeia no treino

Para produzir **um** ŷ[n], a LSTM do PA precisa de uma janela de `wp` valores de u. Cada um desses u precisa de uma janela de `wd` valores de x. Por isso, cada amostra do lote é uma janela de x com **`wd + wp − 1`** amostras:

```
x[n−wd−wp+2 … n]   (wd+wp−1 amostras)
   │ unfold: wp sub-janelas de wd amostras
   ▼
DPD ──► u[n−wp+1 … n]   (wp amostras)
   │ uma janela da LSTM do PA
   ▼
PA_LSTM ──► ŷ[n]   ◄── comparado com G·x[n]
```

- `x_windows.unfold(1, wd, 1)`: no eixo do tempo de cada janela, pega fatias de `wd` com passo 1, ou seja, `wp` sub-janelas. O shape passa de (B, wd+wp−1, 2) para (B, wp, 2, wd); depois de `permute`, fica (B, wp, wd, 2).
- As B·wp sub-janelas passam pelo DPD **de uma vez** (`reshape(B*wp, wd, 2)`), e o resultado volta para (B, wp, 2), que é a janela da LSTM do PA.
- Tudo é operação de tensor, então o gradiente atravessa a cadeia inteira.

### `run_pa` e `run_dpd`: a cadeia num sinal inteiro (análise)

Recebem um sinal complexo em unidades físicas e devolvem sinais complexos em unidades físicas (normalizam na entrada e desnormalizam na saída).

| Função | Saída | Alinhamento |
|---|---|---|
| `run_pa(pa, x, wp, ...)` | ŷ do PA sozinho | ŷ[k] ↔ x[k + wp − 1] |
| `run_dpd(dpd, pa, x, wd, wp, ...)` | (u, ŷ) | u[k] ↔ x[k + wd − 1]; ŷ[k] ↔ x[k + wd + wp − 2] |

Para comparar "sem DPD" com "com DPD" nas mesmas amostras, os dois são cortados para começar em x[wd + wp − 2] (ver a Etapa 4 em [dpd_lstm.md](dpd_lstm.md)).

Testes feitos na CPU, com uma LSTM de PA de pesos aleatórios (2026-11-18):

- o DPD não treinado dá u = x, e `run_dpd` = `run_pa` deslocado;
- `cascade` (treino) e `run_dpd` (análise) dão o mesmo ŷ, amostra por amostra;
- o gêmeo dá a mesma saída duas vezes em `train()`.

## `TrainerDPD` — `trainer_dpd.py`

Mesma estrutura do `TrainerLSTM` (`fit`, `_fit`, `evaluate`, `_run_epoch`, early stopping), com estas diferenças:

- o otimizador (Adam) recebe **só os parâmetros do DPD**;
- cada passo chama `cascade`, e a loss é o MSE entre ŷ e o alvo G·x (normalizado);
- métricas por época, para treino e validação:

| Chave | O que mede |
|---|---|
| `avg_loss` | MSE no espaço normalizado |
| `nmse` | NMSE (dB) entre ŷ e G·x, em unidades físicas. **Métrica de escolha** e do early stopping. |
| `evm` | EVM (%) entre ŷ e G·x |
| `max_u` | maior \|u\| em unidades físicas: mostra o quanto o DPD extrapola além da faixa medida |

- `_abs_physical(u)` desfaz o `StandardScaler` dentro do PyTorch (u·scale + mean) para calcular |u| sem sair da GPU.
- O ACEPR não é calculado por época, porque o lote de treino é embaralhado e o espectro exige ordem temporal. Ele é calculado no teste, no notebook.

## `GridSearchDPD` — `grid_search_dpd.py`

```python
gsearch = GridSearchDPD(train, val, pa, WP, scaler_x, scaler_y, G, param_grid, device)
best_metric, best_dpd, best_params = gsearch.run(DPD_DIR)
```

Mesma lógica do `GridSearchLSTM`: a mesma `param_grid`, combinações já feitas puladas, retomada do melhor anterior, semente fixa e `ResultsStorage` (summary, histórico por combinação, melhor modelo e scalers).

- **Alvo:** `Dataset(train.x, G * train.x).normalize(scaler_x, scaler_y)`. O x vai para o espaço da entrada do PA, e G·x para o espaço da saída. O `normalize` com scalers dados só aplica a transformação, sem reajustar.
- **Janelas:** `SWDataset(..., window = ws + wp − 1)`. O `SWDataset` já entrega o alvo do último instante da janela, que é exatamente G·x[n].
- **`ResultsStorage(save_path, key_metric="val_nmse")`:** escolhe a menor NMSE de validação.
- `ws` na grade é a janela **do DPD**. A janela do PA (`wp`) é fixa, vinda do modelo do PA.
