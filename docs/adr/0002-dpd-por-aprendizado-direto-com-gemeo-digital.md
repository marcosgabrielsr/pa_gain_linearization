# ADR-0002: DPD por aprendizado direto (DLA) usando a LSTM como gêmeo digital

- **Data:** 2026-11-05
- **Status:** Proposta (confirmar na Etapa 4, ao treinar)
- **Decidido por:** Marcos, com recomendação do Claude

## Contexto

O DPD (pré-distorcedor digital) é um bloco colocado antes do PA, treinado para que

```
PA(DPD(x)) ≈ G·x
```

onde G é o ganho linear alvo, calculado por mínimos quadrados na zona linear (−20 a 0 dB) em `notebooks/pa_gain_linearization.ipynb` (`get_linear_gain`).

Não é possível fazer *backpropagation* através do PA físico. Porém, já existe um modelo diferenciável dele: a LSTM `results/best_model.pt` (2 camadas, `hidden_size` = 16, janela = 10, EVM de validação ≈ 0,43%).

## Decisão

Treinar o DPD pela **arquitetura de aprendizado direto (DLA) baseada em modelo**:

```
x ──► DPD_θ (treinável) ──u──► PA_LSTM (congelado) ──► ŷ
loss = média |ŷ − G·x|²
```

O gradiente da loss atravessa a LSTM congelada e atualiza apenas os pesos θ do DPD.

## Alternativas consideradas

- **Aprendizado indireto (ILA):** treinar um pós-distorcedor que mapeia y/G → x e copiá-lo para a frente do PA.
  Prós: não precisa de modelo do PA e é clássico na literatura.
  Contras: é sensível a ruído na saída medida, só é exato para sistemas inversíveis e não aproveita o modelo que já temos.
- **DPD polinomial (ex.: *memory polynomial*, GMP) por mínimos quadrados:**
  Prós: simples e interpretável.
  Contras: foge da linha de redes neurais do projeto. Pode servir como *baseline* futura.

## Consequências

- A qualidade do DPD fica **limitada pela qualidade do modelo do PA**. Em particular, o DPD vai aumentar a amplitude para compensar a compressão, chegando a regiões onde a LSTM nunca viu dados (extrapolação). Isso é verificado na Etapa 1 (célula 1.6 do notebook).
- **Saturação:** se |G·x| for maior que a saída máxima do PA, nenhum DPD resolve. É preciso escolher G e/ou um *back-off* na entrada (Etapa 2).
- **Detalhe técnico:** na GPU, o *backward* de uma LSTM do cuDNN só é permitido em modo `train()`. Como o modelo tem `dropout = 0.0`, `pa.train()` não altera o cálculo e resolve o problema.
- O DPD deve operar no espaço **normalizado** da entrada do PA, então os scalers fazem parte do pipeline.

## Referências

- [docs/codigo/dpd_lstm.md](../codigo/dpd_lstm.md)
- `papers/an-overview-of-rf-power-amplifier-digital-predistortion-techniques-for-wireless-communication-systems.pdf` (DLA × ILA)
