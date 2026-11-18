# `notebooks/apa_linear_gain.ipynb` — ganho linear do APA_200MHz

Calcula o **ganho linear G** do PA do dataset APA_200MHz (OpenDPD, [SOURCE.md](../../data/raw/APA_200MHz/SOURCE.md)). G é o alvo do DPD: PA(DPD(x)) ≈ G·x ([dpd_lstm.md](dpd_lstm.md)).

Este notebook reúne o que foi desenvolvido entre 2026-11-17 e 2026-11-18 para o APA. O [`pa_gain_linearization.ipynb`](../../notebooks/pa_gain_linearization.ipynb) voltou a ser só do nosso PA (máscara fixa −20 < |x| < 0 dB, escolhida olhando o gráfico).

## Seções do notebook

| Seção | O que faz |
|---|---|
| 3. Loading Data | junta treino, validação e teste num único `ds`, porque o ganho usa todas as amostras |
| 4. AM/AM | AM/AM em dB e linear; AM/AM normalizado (quadrado) |
| 5. Gain per Band | técnica do ponto de compressão de ganho: faixas de \|x\|, ganho por faixa, curva de ganho |
| 6. Linear Gain | reta pela origem ŷ = m·x nas magnitudes, com 0 < \|x\| < 0,7 |
| 7. Complex Gain G | o mesmo intervalo com números complexos (`get_linear_gain`): magnitude **e** fase |
| 8. Saving G | grava `results/APA_200MHz/linear_gain.json`, lido pelo `dpd_lstm.ipynb` |

Os dados do APA são **normalizados** (pico de |x| = 1, ou seja, 0 dB). Por isso, o ganho calculado é um ganho normalizado, e as amplitudes são relativas ao pico. O `spec.json` não traz o ganho físico, nem o P1dB do PA.

## Ganho complexo G e o arquivo `linear_gain.json`

O valor usado pelo DPD é o **G complexo** (`get_linear_gain`), calculado nas mesmas amostras da reta (`fit_mask`, 0 < |x| < 0,7). Decisão registrada no [ADR-0006](../adr/0006-ganho-linear-do-apa.md).

| | Valor no APA |
|---|---|
| G | 1,1660 + j0,0057 |
| \|G\| | 1,1661 (1,334 dB) |
| ∠G | 0,28° |
| Amostras usadas | 97.635 de 98.304 |
| \|G\| · max\|x\| | 1,166, acima do max\|y\| medido (1,0): nos picos, o DPD vai extrapolar ([ADR-0007](../adr/0007-dpd-do-apa-com-memoria-sem-back-off.md)) |

O `m` da reta em magnitude (1,1700) é um pouco maior que |G| (1,1661): por causa da memória e da AM/PM, os termos y·x* se cancelam parcialmente na soma complexa (ver a seção da `fit_slope`).

Formato do `linear_gain.json`: `dataset`, `gi`, `gq` (partes real e imaginária de G), `abs`, `abs_db`, `angle_deg`, `fit_range` e `n_samples`.

**Onde rodar:** a pasta `results/` não é versionada (`.gitignore`). Rode este notebook **na máquina do orientador**, antes do `dpd_lstm.ipynb`, para que o arquivo exista lá. É rápido (não usa a GPU).

## `plot_normalized_AM_AM(x, y, db=True, title)`

AM/AM com os dois eixos **normalizados pelo pico** (|x|/max|x| e |y|/max|y|), num **gráfico quadrado, com a mesma escala nos dois eixos**. O objetivo é enxergar a relação linear entre x e y.

- **Por que quadrado e com a mesma escala:** com `set_aspect("equal")` e os mesmos limites nos dois eixos, 1 unidade em x ocupa o mesmo espaço que 1 unidade em y. Uma relação linear aparece então como uma **reta a 45°**, e a compressão como o afastamento dessa inclinação. Num gráfico retangular (10 × 5), a mesma reta aparece "deitada", e a inclinação engana.
- `db=True` (padrão): os dois eixos em dB, com 0 dB no pico. A zona linear é uma reta de **inclinação 1** (cada dB a mais na entrada dá 1 dB a mais na saída). A região de baixa amplitude aparece espalhada (ruído).
- `db=False`: os dois eixos lineares, de 0 a 1. A zona linear é uma reta que passa pela origem, e a compressão aparece como a curva "entortando para baixo" perto de 1.
- Serve para **comparar PAs diferentes** (o nosso e o APA) na mesma escala.
- No **APA**, a normalização não muda os valores, porque os dados do OpenDPD já vêm normalizados (max|x| = max|y| = 1).

## Reta pela origem: `fit_slope(x, y, x_min=0, x_max=0.7)`

Ajusta por mínimos quadrados uma reta que **passa pela origem**, ŷ = m·x, sobre as **magnitudes** |x| e |y|. É a reta que se vê no AM/AM com eixos lineares.

Dedução: queremos o m que minimiza E(m) = Σ (yₙ − m·xₙ)². Derivando e igualando a zero:

```
dE/dm = −2 Σ xₙ·(yₙ − m·xₙ) = 0   ⟹   m = Σ xₙ·yₙ / Σ xₙ²      (com xₙ = |xₙ|, yₙ = |yₙ|)
```

Como a reta é forçada a passar pela origem, não há termo b, e a solução é só essa razão.

- Só entram no ajuste as amostras com `x_min < |x| < x_max`. A função devolve `m` e a máscara `mask` dessas amostras.
- **Por que limitar o intervalo:** as amostras comprimidas (perto do pico) puxariam a reta para baixo. No APA: m = 1,1673 com todas as amostras, 1,1700 com |x| < 0,7, 1,1757 com |x| < 0,5.
- **Magnitude × complexo:** o `get_linear_gain` faz a mesma conta com números complexos, G = Σ y·x* / Σ|x|², e com isso ajusta ganho **e** fase. O m em magnitude sai um pouco maior que |G| (~0,3 %): por causa da memória e da AM/PM, os termos y·x* não apontam todos na mesma direção e se cancelam parcialmente na soma complexa. Para o alvo do DPD, que inclui a fase, vale o G complexo.

### Resultado no APA_200MHz (0 < |x| < 0,7)

- **m = 1,1700 (1,364 dB)**, com 97.635 das 98.304 amostras.
- Dentro do intervalo, a nuvem fica centrada na reta. Acima de |x| ≈ 0,7, os pontos ficam abaixo dela: é a compressão.

### `plot_AM_AM(..., slope=None, fit_range=None)`

A `plot_AM_AM` ganhou dois argumentos opcionais (sem eles, o gráfico é igual ao de antes):

- `slope`: desenha a reta ŷ = slope·x (vermelha, tracejada). Em dB, a reta vira y_dB = x_dB + 20·log10(m), ou seja, uma reta de inclinação 1 deslocada.
- `fit_range = (x_min, x_max)`: desenha linhas verticais pontilhadas nos limites do ajuste. Em dB, um limite ≤ 0 é ignorado, porque |x| = 0 não tem valor em dB (viraria −300 dB por causa do EPS).

No notebook, a célula após o AM/AM linear define `X_MIN, X_MAX = 0, 0.7`, chama `fit_slope` e plota a reta sobre o AM/AM com `db=False`.

## Zona linear: técnica do ponto de compressão de ganho

Para o nosso PA, a zona linear foi escolhida olhando o gráfico (máscara fixa `-20 < |x| < 0 dB`). Para o APA, ela será encontrada automaticamente:

1. **Dividir as amostras em faixas de |x| (dB)** (`getting_samples`) e **calcular o ganho G_k de cada faixa** (`band_gain`). Ambas implementadas.
2. Ganho de pequeno sinal G₀: o platô das faixas de baixa amplitude. Pendente.
3. Limite superior: a primeira faixa em que |G_k| cai mais que uma tolerância (0,1 a 1 dB) abaixo de G₀. Esse é o início da compressão; com 1 dB, é o P1dB. Pendente.
4. Limite inferior: descartar as faixas com poucas amostras ou dominadas por ruído. Pendente.
5. G final por mínimos quadrados, só nas amostras da zona encontrada (`get_linear_gain`).

Ganho de uma faixa k (passo 1), por mínimos quadrados:

```
G_k = Σ yₙ·xₙ* / Σ |xₙ|²      (n = amostras da faixa k; xₙ* = conjugado de xₙ)
```

Para uma amostra só, isso é y/x. Com várias, é a média de yₙ/xₙ ponderada por |xₙ|². O conjugado "alinha" as fases das entradas antes de somar. Calcular por faixa, em vez de amostra a amostra, é necessário no APA, porque ele tem **memória**: o mesmo |x| dá saídas diferentes, e o y/x individual vira uma nuvem.

## `getting_samples(x, y, bw=0.5)`

Divide as amostras em faixas de largura `bw` dB de **amplitude de entrada**. O `y` de cada amostra vai junto com o seu `x`.

```python
edges, bands = getting_samples(ds.x, ds.y, bw=0.5)
x_k, y_k = bands[k]        # amostras com edges[k] <= |x|dB < edges[k+1]
```

| Retorno | Shape | Conteúdo |
|---|---|---|
| `edges` | (K+1,) | bordas das faixas, em dB |
| `bands` | lista de K tuplas | `(x_k, y_k)`, as amostras de cada faixa; uma faixa vazia tem arrays vazios |

Como a grade é montada:

- **Alinhada em múltiplos de `bw`** (…, −30,0, −29,5, …). A primeira borda é o múltiplo de `bw` logo abaixo de min|x| (`floor`). A última é o múltiplo **estritamente acima** de max|x| (`floor + 1`).
- **Nenhuma amostra fica de fora.** A primeira versão usava `np.arange(min + bw, max + 0.001, bw)`. Como (max − min) quase nunca é múltiplo de `bw`, a grade parava antes do máximo: no APA, parava em −0,39 dB, e as 4 amostras de pico ficavam sem faixa. O acréscimo fixo de 0,001 não resolvia isso, porque não tem relação com o tamanho do passo.
- O número de faixas é calculado com `floor` sobre múltiplos inteiros de `bw`, em vez de `np.arange` com um stop em ponto flutuante, o que evita perder ou ganhar uma borda por arredondamento.
- `np.digitize(x_db, edges) - 1` dá o índice da faixa de cada amostra. A convenção é o intervalo fechado à esquerda e aberto à direita: [edges[k], edges[k+1]).

### Resultado no APA_200MHz (bw = 0,5 dB, 98.304 amostras)

| | Valor |
|---|---|
| \|x\| | de −55,39 a 0,00 dB |
| Bordas | de −55,5 a +0,5 dB |
| Número de faixas | 112 (7 vazias, todas na região de baixa amplitude) |
| Amostras nas faixas | 98.304 de 98.304 |

Distribuição: só ~1 % das amostras está abaixo de −30 dB, e ~0,1 % abaixo de −40 dB; a mediana é −11,6 dB. As faixas mais baixas terão poucas amostras e serão descartadas no passo 4.

## `band_gain(edges, bands)`

Calcula, para cada faixa devolvida por `getting_samples`, o ganho complexo por mínimos quadrados, G_k = Σ y·x* / Σ|x|², e o separa em magnitude e fase.

```python
centers, counts, gain_db, phase_deg = band_gain(*getting_samples(ds.x, ds.y, bw=0.5))
```

| Retorno | Shape | Conteúdo |
|---|---|---|
| `centers` | (K,) | centro de cada faixa, (edges[k] + edges[k+1]) / 2, em dB; é o eixo x do gráfico |
| `counts` | (K,) | número de amostras da faixa; é o critério do limite inferior |
| `gain_db` | (K,) | \|G_k\| em dB; mostra o platô e a compressão |
| `phase_deg` | (K,) | ∠G_k em graus; é a AM/PM média |

Detalhes:

- **Faixas vazias ficam `NaN`** em vez de serem removidas. Assim, a posição k dos quatro arrays é sempre a faixa k, e eles podem ser filtrados juntos, com a mesma máscara. O matplotlib não desenha pontos `NaN`.
- A conta usa **`np.sum`/`np.abs`**, e não o `sum`/`abs` do Python. Com um array vazio, o `sum` do Python devolve o inteiro 0, e 0/0 entre inteiros dá `ZeroDivisionError`. Por isso uma primeira versão quebrava no APA. Além disso, o `np.sum` é bem mais rápido.
- `centers` é calculado de forma vetorizada: `edges[:-1]` são as bordas de baixo, e `edges[1:]` as de cima.

### Resultado no APA_200MHz (bw = 0,5 dB, agrupado de 2 em 2 dB para resumir)

| \|x\| (dB) | amostras | \|G\| (dB) | ∠G (°) |
|---|---|---|---|
| −40 | 56 | 1,38 | 19,4 |
| −30 | 565 | 1,27 | 2,6 |
| −20 | 5.023 | 1,34 | 5,7 |
| −10 | 16.253 | 1,41 | 1,7 |
| −6 | 6.005 | 1,26 | −4,0 |
| −2 | 145 | 0,59 | −12,3 |

- Abaixo de cerca de −34 dB, o ganho e a fase oscilam: são poucas amostras e há ruído.
- Entre −30 e −8 dB, o ganho forma um platô de ~1,3 a 1,4 dB.
- Acima de −8 dB, o ganho comprime.
- A fase começa a cair antes do ganho, em torno de −14 dB: o APA tem uma AM/PM forte. O critério de fase pode definir o limite superior antes do de magnitude.

## `plot_band_gain(centers, counts, gain_db, phase_deg, min_count=50, title)`

Plota a saída da `band_gain` em três painéis empilhados, com o mesmo eixo x (centro da faixa, em dB):

| Painel | Conteúdo | Para que serve |
|---|---|---|
| 1 | \|G_k\| (dB) | ver o platô (zona linear) e a queda (compressão) |
| 2 | ∠G_k (°) | ver a AM/PM média |
| 3 | número de amostras (escala log) | ver onde há poucas amostras e o G_k não é confiável |

- `sharex=True`: os três painéis compartilham o eixo x. Uma instabilidade no ganho pode ser comparada diretamente com o número de amostras da mesma faixa.
- O painel de amostras usa **escala log**, porque os valores vão de 1 a ~4.000; numa escala linear, as faixas pequenas não apareceriam. As faixas vazias são passadas como `NaN` e não são desenhadas, já que log(0) não existe.
- `height_ratios = [2, 2, 1]`: o painel de amostras é só uma referência, então fica menor.
- **`min_count` (limite inferior, passo 4)**: as faixas com menos de `min_count` amostras não são confiáveis, porque o ruído não se cancela na média. A máscara `valid = counts >= min_count` esconde essas faixas nos painéis de ganho e fase (elas viram `NaN`). No painel de amostras, as faixas usadas ficam em azul e as descartadas em cinza, com uma linha tracejada em `min_count`.
- **Só existe limite mínimo, não máximo.** Mais amostras deixam o G_k *mais* confiável. Um teto, por exemplo de 1000, descartaria de −20 a −5 dB, que é o miolo do platô e o início da compressão: 87 % das amostras.

No notebook, a célula logo após o teste da `getting_samples` define `MIN_COUNT = 50` e chama `band_gain` e `plot_band_gain`.

### O que o gráfico mostra no APA_200MHz

- Abaixo de cerca de −35 dB, com menos de ~30 amostras por faixa, o ganho pula entre −3 e +27 dB, e a fase entre −150° e +180°: o ruído domina.
- Entre −30 e −8 dB, o ganho é praticamente constante (o platô).
- Perto de 0 dB, o ganho cai ~1,4 dB e a fase ~20°.
- Sem limite inferior, os pontos de ruído **dominam a escala do eixo y**, e a queda de ~1 dB da compressão aparece achatada.

### Com `min_count = 50` (64 faixas, de −33,75 a −1,75 dB, 99,5 % das amostras)

- **De −34 a −25 dB** (50 a ~300 amostras por faixa), o ganho ainda oscila entre 0,8 e 2,7 dB, e a fase entre 1° e 13°. Um mínimo maior (100 a 300) cortaria esse trecho.
- **De −24 a −8 dB**, há um platô: o ganho fica em ~1,3 a 1,4 dB. Ele sobe levemente, ~0,1 dB, até −10 dB.
- **A partir de −8 dB**, o ganho comprime até ~0,7 dB em −1,75 dB.
- **A fase** fica estável em ~5,8° de −24 a −16 dB e cai a partir de ~−15 dB, até −11°. Ela sai do platô antes do ganho.
