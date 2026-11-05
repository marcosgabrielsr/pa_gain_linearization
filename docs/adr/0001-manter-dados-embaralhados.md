# ADR-0001: Manter os dados repetidos e embaralhados (`expData.csv`)

- **Data:** 2026-11-05
- **Status:** Aceita
- **Decidido por:** orientador (orientação), confirmado por Marcos

## Contexto

Os dados medidos do PA estão em `data/raw/dadosIniciais.csv` (48.384 amostras I/Q de entrada e saída, em ordem temporal). O `data/processed/expData.csv`, usado para treinar a LSTM do PA, é gerado na célula 7 de `notebooks/pa_gain_linearization.ipynb`:

```python
df = df.loc[df.index.repeat(10)]          # repete cada amostra 10×  -> 483.840 linhas
df = df.sample(frac=1, random_state=42)   # embaralha todas as linhas
```

Depois, `Dataset.split()` corta 60/20/20 (treino/validação/teste) por índice.

Essa escolha tem duas consequências técnicas:

1. **A memória do PA se perde.** A LSTM recebe janelas de 10 amostras consecutivas, mas, após o embaralhamento, as 9 anteriores são amostras aleatórias, sem relação temporal com a atual. O modelo tende a aprender um mapa **estático** (sem memória).
2. **O teste vaza para o treino.** Cada amostra tem 10 cópias. A chance de nenhuma das outras 9 cair no treino (60%) é 0,4⁹ ≈ 0,03%. Ou seja, quase todo o teste já foi visto no treino, e as métricas de validação e teste (ex.: EVM de validação de 0,43%) são otimistas.

## Decisão

Manter o `expData.csv` como está (repetido e embaralhado) nesta fase do projeto, por orientação do orientador.

A justificativa do orientador: os dados vêm de um sistema de comunicação, cujo sinal varia de qualquer forma, e embaralhar expõe o modelo a uma diversidade maior de situações.

## Alternativas consideradas

- **Usar `dadosIniciais.csv` na ordem original, com split temporal (sem repetir nem embaralhar).**
  Prós: preserva a memória do PA e dá métricas sem vazamento.
  Contras: exige retreinar o grid search da LSTM e contraria a orientação atual.

## Consequências

- O modelo do PA (`results/best_model.pt`) é tratado como **sem memória**. Daí vem o [ADR-0003](0003-dpd-sem-memoria.md) (DPD sem memória).
- As métricas de teste devem ser lidas como **otimistas**, por causa do vazamento.
- Efeitos de memória (ex.: assimetria espectral, ACPR) não serão capturados pelo modelo nem corrigidos pelo DPD.
- Fica em aberto: entender melhor a motivação com o orientador e reavaliar esta decisão se os resultados do DPD exigirem.

## Referências

- [notebooks/pa_gain_linearization.ipynb](../../notebooks/pa_gain_linearization.ipynb), célula 7
- [src/pa_model/dataset.py](../../src/pa_model/dataset.py) (`split`, `normalize`)
- [diario/2026-11-05.md](../diario/2026-11-05.md)
