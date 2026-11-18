# ADR-0007: DPD do APA com memória (LSTM residual), sem back-off

- **Data:** 2026-11-18
- **Status:** Aceita
- **Decidido por:** Marcos, com recomendação do Claude

## Contexto

O DPD será feito para o PA do **APA_200MHz**, e não para o nosso PA. O treino segue a DLA do [ADR-0002](0002-dpd-por-aprendizado-direto-com-gemeo-digital.md): o DPD é treinado através da LSTM do APA (resultado do `grid_search_lstm.ipynb`, em `results/APA_200MHz/`), congelada.

Diferenças em relação ao nosso PA:

- os dados do APA estão em **ordem temporal** e o PA tem **memória** ([ADR-0005](0005-dataset-apa-200mhz.md)). O [ADR-0003](0003-dpd-sem-memoria.md) (DPD sem memória) não vale aqui;
- com o G do [ADR-0006](0006-ganho-linear-do-apa.md), o alvo nos picos (|G·x| até 1,166) passa do maior valor de saída medido (1,0).

## Decisão

1. O DPD é uma **LSTM com conexão residual**, `u[n] = x[n] + f(x[n−W+1 … n])`, com a camada final iniciada em zero (o DPD começa como a identidade). A classe `PaLSTM` é reaproveitada como `f`.
2. **Sem back-off e sem reduzir G:** o DPD pode pedir ao PA amplitudes maiores que as medidas. A LSTM do PA responde nessa região **extrapolando**.

## Alternativas consideradas

- **Back-off** (escalar a entrada para que o pico de G·x caiba no que o PA entrega): o alvo seria alcançável, mas o PA operaria com menos potência. Foi descartado por decisão do Marcos.
- **G menor:** mesmo efeito do back-off.
- **DPD sem memória (MLP):** não compensa os efeitos de memória do APA. Ainda assim, entra no grid como referência (`ws = 1`).

## Consequências

- As amostras com |G·x| acima do max|y| de treino são poucas: no conjunto completo, 41 de 98.304 (0,04 %), as com |x| > 0,857. O efeito no NMSE deve ser pequeno.
- Nessas amostras, o resultado depende de como a LSTM do PA extrapola. O DPD pode aprender a "explorar" um comportamento do modelo que o PA real não tem. Por isso, o trainer registra a **max |u|** em cada época, e o notebook mostra a característica |u| × |x| do DPD.
- **Detalhe técnico:** o *backward* da LSTM do cuDNN exige `train()`. Se a LSTM do APA tiver `dr > 0`, isso ligaria o dropout no gêmeo digital. Por isso, `load_pa_twin` zera o dropout da cópia congelada.
- O resultado vale para o **gêmeo digital**. Ele só vale para o PA real na medida em que a LSTM representa o PA, inclusive na região de extrapolação.

## Referências

- [codigo/dpd.md](../codigo/dpd.md), [codigo/dpd_lstm.md](../codigo/dpd_lstm.md)
- [ADR-0002](0002-dpd-por-aprendizado-direto-com-gemeo-digital.md), [ADR-0005](0005-dataset-apa-200mhz.md), [ADR-0006](0006-ganho-linear-do-apa.md)
