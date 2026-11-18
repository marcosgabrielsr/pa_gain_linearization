# ADR-0008: Hiperparâmetros do DPD por grid search, sem baseline

- **Data:** 2026-11-18
- **Status:** Aceita
- **Decidido por:** Marcos

## Contexto

O DPD ([ADR-0007](0007-dpd-do-apa-com-memoria-sem-back-off.md)) tem hiperparâmetros: janela, tamanho, número de camadas, taxa de aprendizado, dropout e batch. Para o modelo do PA, eles foram escolhidos por grid search (`GridSearchLSTM`).

## Decisão

Ir **direto para o grid search** (`GridSearchDPD`), com a mesma lógica do grid search do PA (pula combinações já feitas, salva o melhor modelo e o histórico de cada combinação). A melhor combinação é a de menor **NMSE de validação** entre PA_LSTM(DPD(x)) e G·x.

Grade inicial (36 combinações):

| Parâmetro | Valores |
|---|---|
| `window_size` | 1, 5, 10 (1 = sem memória, como referência) |
| `hidden_size` | 8, 16, 32 |
| `num_layers` | 1, 2 |
| `learning_rate` | 0,001; 0,0005 |
| `dropout` | 0 (fixo) |
| `batch_size` | 128 (fixo) |

## Alternativas consideradas

- **Treinar uma configuração antes (baseline):** facilitaria encontrar erros no pipeline com um treino só. Foi descartado por decisão do Marcos. Para reduzir o risco, foi criado o `scripts/check_dpd.py`, que testa o pipeline (identidade, gêmeo determinístico, uma combinação por poucas épocas) antes do grid.

## Consequências

- **Dropout fixo em 0:** o modelo é pequeno, o alvo G·x não tem ruído de medição, e o dropout deixaria aleatória uma função que deveria ser determinística.
- O custo por combinação é maior que no grid do PA, porque cada passo passa pelo DPD **e** pela LSTM do PA (forward e backward).
- Os resultados ficam em `results/APA_200MHz/dpd/`, separados do grid search do PA.

## Referências

- [codigo/dpd.md](../codigo/dpd.md), [ADR-0004](0004-otimizar-auxiliares-manter-grid-search.md)
