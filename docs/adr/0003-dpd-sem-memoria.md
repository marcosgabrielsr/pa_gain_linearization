# ADR-0003: DPD sem memória (amostra a amostra)

- **Data:** 2026-11-05
- **Status:** Proposta (confirmar na Etapa 3)
- **Decidido por:** Marcos, com recomendação do Claude
- **Escopo:** só o nosso PA. Não se aplica ao APA_200MHz, que tem memória ([ADR-0005](0005-dataset-apa-200mhz.md))

## Contexto

Pelo [ADR-0001](0001-manter-dados-embaralhados.md), os dados de treino estão embaralhados, e o modelo do PA é, na prática, **sem memória**: a saída depende essencialmente da amostra atual.

## Decisão

O DPD será uma rede **sem memória**: cada amostra x[n] é mapeada para u[n] olhando apenas para ela mesma. Ponto de partida: uma MLP pequena com conexão residual, `u = x + f(x)`, para começar próximo da identidade.

## Alternativas consideradas

- **DPD com memória (LSTM, ou MLP com janela de amostras):**
  Prós: seria necessário para um PA com efeito de memória.
  Contras: com dados embaralhados, não há memória real a compensar. Seria mais complexo sem ganho esperado.

## Consequências

- O modelo é simples de explicar, treinar e, futuramente, implementar em hardware.
- Ainda assim, a saída do DPD é passada ao PA_LSTM **em janelas de 10 amostras na mesma ordem do arquivo**, porque é esse o formato de entrada com que a LSTM foi treinada.
- Se o [ADR-0001](0001-manter-dados-embaralhados.md) for substituído (dados em ordem temporal), esta decisão deve ser revista.

## Referências

- [ADR-0001](0001-manter-dados-embaralhados.md), [ADR-0002](0002-dpd-por-aprendizado-direto-com-gemeo-digital.md)
