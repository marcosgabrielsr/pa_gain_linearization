# ADR-0004: Otimizar os auxiliares do treino e manter a lógica do grid search

- **Data:** 2026-11-05
- **Status:** Aceita (validação pendente: `scripts/check_trainer.py`)
- **Decidido por:** Marcos

## Contexto

O grid search da LSTM do PA (`src/lstm/grid_search.py`) testa 360 combinações. Cada época tem cerca de 290 mil janelas de treino. Dois gargalos foram identificados:

1. **`SWDataset` + `DataLoader`:** para cada lote, o `DataLoader` chama `__getitem__` 128 vezes em Python, empilha os resultados (*collate*) e copia o lote para a GPU. Com uma LSTM pequena (H ≤ 16), a GPU passa a maior parte do tempo esperando a CPU.
2. **`TrainerLSTM`:** `loss.item()` e `outputs.cpu()` dentro do laço obrigam a CPU a esperar a GPU **a cada lote**. Além disso, o `EarlyStopping` recebia `copy.deepcopy(model)` em toda época, mesmo sem melhora.

Também foi proposto trocar o grid search pelo Optuna (busca bayesiana, *pruning*, persistência em SQLite).

## Decisão

- **Manter a lógica do grid search** (`itertools.product`, `ResultsStorage`, critério de melhor modelo), porque ela é mais fácil de interpretar nesta fase. O Optuna fica para o futuro.
- **Otimizar apenas os auxiliares**, sem mudar o que é calculado:
  - `SWDataset` monta todas as janelas com `unfold` (uma *view*, sem cópia), já no `device`;
  - o novo `WindowLoader` substitui o `DataLoader` e pega cada lote com uma única indexação;
  - o `TrainerLSTM` acumula loss e previsões no `device` e sincroniza uma vez por época; código duplicado foi unificado;
  - o `EarlyStopping` copia o modelo só quando há melhora.
- **Fazer backup** dos resultados antigos (`results_backup_2026-11-05/`) e rodar o novo grid search em `results/v2/`, para comparar.

## Alternativas consideradas

- **Optuna:** é mais rápido e tem menos código próprio, mas é menos transparente. Adiado.
- **`DataLoader` com `num_workers > 0` e `pin_memory`:** reduz o gargalo, mas não o elimina; a montagem continua amostra a amostra.
- **Precisão mista (AMP), `torch.compile`, `cudnn.benchmark`:** mudariam os números ou a reprodutibilidade, o que impediria comparar com o backup. Descartados por ora.

## Consequências

- O `WindowLoader` reproduz a **mesma ordem de lotes** do `DataLoader` (mesmo `generator`; ver [codigo/sd_window.md](../codigo/sd_window.md)). Os históricos novos devem bater com os do backup, a menos de diferenças mínimas de arredondamento da GPU.
- Os históricos mantêm as mesmas 12 colunas, e o `scripts/read_histories.py` continua funcionando.
- O `SWDataset` continua aceitando índice inteiro e funcionando com `DataLoader`, então os notebooks antigos não quebram.
- **Atenção:**
  - `results/v2/` fica **dentro** de `results/`. `scripts/read_histories.py` usa `rglob`, então, sem `-d`, mistura históricos antigos e novos. Use `-d results/v2` ou `-d results_backup_2026-11-05`.
  - A seção "3. Analysing Results" do `grid_search_lstm.ipynb` continua lendo `results/best_params.json` e `results/best_model.pt` (os **antigos**). O mesmo vale para o `dpd_lstm.ipynb`. Trocar para `v2` é uma decisão posterior, depois de comparar os resultados.

## Referências

- [codigo/sd_window.md](../codigo/sd_window.md), [codigo/trainer_lstm.md](../codigo/trainer_lstm.md)
- `scripts/check_trainer.py`
