# `src/pa_model/sd_window.py` — janelas deslizantes

Contém o `SWDataset` (as janelas) e o `WindowLoader` (a iteração em lotes). Otimizado em 2026-11-05 ([ADR-0004](../adr/0004-otimizar-auxiliares-manter-grid-search.md)).

## A janela

A LSTM prevê a saída do PA no instante t a partir das últimas W entradas:

```
janela i:  x[i], x[i+1], ..., x[i+W-1]   ->   alvo: y[i+W-1]
```

Para N amostras, há N − W + 1 janelas (com `target_dim = 1`). A definição é a mesma da versão antiga; só a implementação mudou.

## `SWDataset`

```python
self.x       = torch.tensor(..., device=device)                          # (N, 2) = [I, Q]
self.windows = self.x.unfold(0, window_size, 1).transpose(1, 2)[:n]      # (N-W+1, W, 2)
self.window_targets = self.targets[start:start + n]                      # (N-W+1, 2)
```

- **`device`:** com `device="cuda"`, a série inteira vai para a GPU uma única vez (cerca de 2 MB no treino). O padrão é `"cpu"`, igual ao comportamento antigo.
- **`unfold(0, W, 1)`:** no eixo 0 (tempo), pega fatias de tamanho W com passo 1. O resultado é uma ***view***: as janelas se sobrepõem na memória e nada é copiado. Para N = 290 mil e W = 10, uma cópia ocuparia 10× mais memória.
- **`transpose(1, 2)`:** de `(N-W+1, 2, W)` para `(N-W+1, W, 2)`, que é o formato `(batch, tempo, features)` da LSTM (`batch_first=True`).
- **Alvo:** a janela i começa em `x[i]`, e o alvo é `y[i+W-1]`, por isso `start = W - 1`.
- **`target_dim > 1`** (não usado hoje): a saída é `(n, target_dim, 2)`, igual à versão antiga.

`__getitem__(idx)` aceita:
- um **inteiro**, que devolve uma janela `(W, 2)` e o alvo `(2,)`. Isso mantém o `SWDataset` compatível com `DataLoader`;
- um **tensor de índices**, que devolve um lote inteiro `(B, W, 2)` e `(B, 2)` numa só operação. É assim que o `WindowLoader` o usa.

## `WindowLoader`

Substitui o `DataLoader` para um `SWDataset`:

```python
idx = torch.randperm(n, generator=self.generator)     # ordem aleatória das janelas
for batch_idx in idx.to(device).split(batch_size):    # fatias de batch_size índices
    yield self.dataset[batch_idx]                     # uma indexação = um lote
```

**Por que é mais rápido:** o `DataLoader` faz, por lote, `batch_size` chamadas Python a `__getitem__`, empilha os resultados (*collate*) e o trainer copia o lote para a GPU. O `WindowLoader` faz uma única indexação de tensor, que já está na GPU.

### Mesma ordem de lotes que o `DataLoader`

Para comparar os resultados novos com o backup, os lotes precisam sair **na mesma ordem** do código antigo. Internamente, a cada época, o `DataLoader(shuffle=True, generator=g)`:

1. sorteia uma "semente base" para os workers: `torch.empty((), dtype=torch.int64).random_(generator=g)`. Isso consome um número do gerador;
2. gera a ordem: `torch.randperm(n, generator=g)` (no `RandomSampler`).

O `WindowLoader` repete os dois passos, na mesma ordem. O passo 1 não tem outra função aqui; existe só para manter a sequência do gerador idêntica. Isso depende da implementação interna do PyTorch, por isso o `scripts/check_trainer.py` **verifica** a igualdade lote a lote na máquina do orientador.

Sem `shuffle`, a ordem é `0, 1, 2, ...`, igual ao `DataLoader(shuffle=False)`.

## Uso

```python
from pa_model import SWDataset, WindowLoader

train_loader = WindowLoader(SWDataset(train_norm, ws, device=device), batch_size=bs, shuffle=True, generator=generator)
for inputs, targets in train_loader:     # inputs (B, W, 2), targets (B, 2), já no device
    ...
```
