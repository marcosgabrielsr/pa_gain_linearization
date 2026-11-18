# ADR-0006: Ganho linear G do APA por mínimos quadrados em 0 < |x| < 0,7

- **Data:** 2026-11-18
- **Status:** Aceita
- **Decidido por:** Marcos, com recomendação do Claude

## Contexto

O DPD precisa de um alvo, PA(DPD(x)) ≈ G·x ([ADR-0002](0002-dpd-por-aprendizado-direto-com-gemeo-digital.md)). No nosso PA, G foi calculado numa zona linear escolhida olhando o gráfico (−20 < |x| < 0 dB). No APA_200MHz isso não serve: a escala é outra (pico em 0 dB, dados normalizados) e o `spec.json` não traz o ganho nem o P1dB do PA.

Fatos levantados em [apa_linear_gain.ipynb](../codigo/apa_linear_gain.md):

- a curva de ganho por faixa tem um platô e começa a comprimir perto do pico;
- o P1dB **não aparece** nos dados válidos: a queda de 1 dB só acontece nas 19 amostras com |x| ≥ −1 dB;
- a fase (AM/PM) se afasta do platô antes da magnitude;
- no AM/AM linear, a nuvem acompanha uma reta pela origem até |x| ≈ 0,7, e fica abaixo dela a partir daí.

## Decisão

G é o **ganho complexo por mínimos quadrados**, G = Σ y·x* / Σ|x|² (`get_linear_gain`), calculado nas amostras com **0 < |x| < 0,7**. No APA, G = 1,1660 + j0,0057 (|G| = 1,334 dB, ∠G = 0,28°). O valor é salvo em `results/APA_200MHz/linear_gain.json`, que o `dpd_lstm.ipynb` lê.

## Alternativas consideradas

- **Limite superior pela tolerância de compressão** (0,1 a 0,3 dB em magnitude, ou 1° a 5° em fase, a partir de um G₀ do platô): é mais rigoroso, mas exige escolher duas tolerâncias e uma região de referência. A magnitude de G quase não muda com o limite (1,37 a 1,39 dB), mas a fase muda (de 4,9° a 1,2°).
- **Reta só em magnitude (`fit_slope`, m = 1,1700):** é a reta vista no AM/AM, mas não tem fase, e o alvo do DPD é complexo.
- **Todas as amostras:** os picos comprimidos puxariam G para baixo (|G| = 1,1628).

## Consequências

- G é simples de explicar: "a reta pela origem que melhor se ajusta à parte reta do AM/AM".
- O intervalo 0 < |x| < 0,7 inclui a região em que a fase já começa a girar. Por isso ∠G (0,28°) fica abaixo da fase do platô de baixa amplitude (~5,8°). Se o DPD precisar de um alvo com a fase do platô, este ADR deve ser revisto.
- |G| · max|x| = 1,166 é maior que o max|y| medido (1,0): o alvo nos picos está fora do alcance do PA ([ADR-0007](0007-dpd-do-apa-com-memoria-sem-back-off.md)).

## Referências

- [codigo/apa_linear_gain.md](../codigo/apa_linear_gain.md), [diário 2026-11-17](../diario/2026-11-17.md), [diário 2026-11-18](../diario/2026-11-18.md)
