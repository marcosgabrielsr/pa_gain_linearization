# `notebooks/dpd_lstm.ipynb`

Notebook principal do DPD. Cada etapa do roteiro (ver [README](../README.md#roteiro-do-dpd)) entra como uma nova seção.

**Pré-requisitos:** pacote instalado (`pip install -e .` na raiz) para os imports `pa_model`, `lstm` e `metrics`; `results/best_model.pt`, `results/best_params.json` e `data/processed/expData.csv` presentes. Execução na máquina com GPU (funciona também em CPU, mais devagar).

---

## Etapa 1 — Carregar o gêmeo digital do PA e validá-lo

**Por quê:** a DLA ([ADR-0002](../adr/0002-dpd-por-aprendizado-direto-com-gemeo-digital.md)) treina o DPD *através* do modelo do PA. Antes de qualquer treino, precisamos garantir duas coisas:

1. o modelo foi carregado exatamente como no treino (mesmos dados, mesma normalização);
2. sabemos como ele se comporta **fora** da faixa de dados medida, porque o DPD vai empurrar a entrada do PA para lá.

### 1.1 Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
```

- `device`: usa a GPU se houver; senão, a CPU.
- `manual_seed(42)`: torna reprodutível qualquer aleatoriedade futura (ex.: inicialização dos pesos do DPD). É a mesma semente do grid search.

### 1.2 Dados e normalização

```python
ds = Dataset.from_csv(DATA_PROCESSED / "expData.csv")
train, val, test = ds.split()
_, scaler_x, scaler_y = train.normalize()
```

- `Dataset.from_csv` lê as colunas `Xreal, Ximg, Yreal, Yimg` e monta dois vetores complexos: `x` (entrada do PA) e `y` (saída).
- `split()` corta por índice: 60% treino, 20% validação, 20% teste. É o mesmo corte do grid search.
- `train.normalize()` ajusta um `StandardScaler` por sinal (um para x, outro para y), **só com o treino**. Cada componente (I e Q) vira `(valor − média) / desvio`.
  - Por que reajustar em vez de carregar? O ajuste é determinístico: mesmos dados de treino geram os mesmos scalers usados no grid search (`GridSearchLSTM.__init__`). O valor de retorno descartado (`_`) é o próprio treino normalizado, que não usamos aqui.

Duas funções fazem a ponte entre o "mundo físico" e o "mundo da rede":

| Função | Entrada | Saída |
|---|---|---|
| `to_tensor(z, scaler)` | complexo `(N,)` em unidades físicas | tensor `(N, 2)` = [I, Q] normalizado, já no `device` |
| `to_complex(t, scaler)` | tensor `(N, 2)` normalizado | complexo `(N,)` em unidades físicas |

Detalhes de `to_complex`:
- `.detach()` desliga o tensor do grafo de gradientes, que não é necessário para análise;
- `.cpu().numpy()` traz o tensor para a CPU, porque o scikit-learn não trabalha com tensores da GPU;
- `.astype(np.float64)` evita perder precisão ao desnormalizar.

**Regra:** use `scaler_x` com sinais de **entrada** do PA (x, e futuramente u) e `scaler_y` com sinais de **saída** (y, ŷ).

### 1.3 Carregar a LSTM congelada

```python
params = json.load(open(RESULTS / "best_params.json"))["params"]
W = params["ws"]
pa = PaLSTM(2, params["hs"], params["nl"], params["dr"]).to(device)
pa.load_state_dict(torch.load(RESULTS / "best_model.pt", map_location=device, weights_only=True))
pa.eval()
for p in pa.parameters():
    p.requires_grad_(False)
```

- `best_params.json` guarda os hiperparâmetros da melhor combinação (`ws` = 10, `hs` = 16, `nl` = 2, `dr` = 0,0). A arquitetura é recriada com eles. O `.pt` contém **só os pesos** (`state_dict`), não a estrutura da rede.
- `map_location=device`: o arquivo foi salvo em `cuda:0`; isso permite carregá-lo em qualquer dispositivo.
- `weights_only=True`: carrega apenas tensores, que é mais seguro do que desserializar objetos Python arbitrários.
- `pa.eval()`: modo de inferência (desliga o dropout; aqui ele já é 0).
- `requires_grad_(False)`: **congela os pesos**. O otimizador do DPD nunca vai alterá-los. O gradiente, porém, continua atravessando a rede até a **entrada**, e é isso que permite treinar o DPD na Etapa 4.

### 1.4 `pa_forward(u)`: passar um sinal pelo PA

```python
def pa_forward(u):
    windows = u.unfold(0, W, 1).transpose(1, 2)    # (N-W+1, W, 2)
    return pa(windows)
```

A LSTM não recebe uma amostra, e sim uma **janela** de W = 10 amostras, e prevê a saída do **último** instante (igual ao `SWDataset`).

- `u.unfold(0, W, 1)`: no eixo 0 (tempo), pega fatias de tamanho W com passo 1. A partir de `(N, 2)`, gera `(N-W+1, 2, W)`.
- `.transpose(1, 2)`: reordena para `(N-W+1, W, 2)`, o formato `(batch, tempo, features)` esperado pela LSTM (`batch_first=True`).
- **Alinhamento:** a saída k corresponde à amostra k+W−1. As primeiras W−1 = 9 amostras não têm janela completa e ficam sem saída. Por isso as referências são cortadas com `[W-1:]`.

Por que não usar `SWDataset` + `DataLoader`? Na Etapa 4, `u` será a **saída do DPD**, e o gradiente precisa passar pela montagem das janelas. `unfold` é uma operação de tensor diferenciável; um `DataLoader` quebraria o grafo.

### 1.5 Validação: EVM no teste

```python
x_test = to_tensor(test.x, scaler_x)
with torch.no_grad():
    y_hat = to_complex(pa_forward(x_test), scaler_y)
x_ref, y_ref = test.x[W-1:], test.y[W-1:]
print(f"EVM do PA_LSTM no teste: {EVM().compute(y_ref, y_hat):.3f} %")
```

- `torch.no_grad()`: não monta o grafo de gradientes, o que economiza memória (aqui só estamos avaliando).
- EVM (*Error Vector Magnitude*): `100 · RMS(y_ref − ŷ) / RMS(y_ref)`, em unidades físicas.
- **Esperado:** valor próximo ao EVM de teste do `grid_search_lstm.ipynb`. Uma diferença grande indicaria um desalinhamento (janela, scaler ou pesos errados).
- Lembrete ([ADR-0001](../adr/0001-manter-dados-embaralhados.md)): com o vazamento treino→teste, esse EVM é otimista.

### 1.6 Extrapolação: o modelo além da faixa de treino

```python
alpha = 1.3
y_ext = to_complex(pa_forward(to_tensor(alpha * test.x, scaler_x)), scaler_y)
```

- O sinal de teste é multiplicado por 1,3 **em unidades físicas** (antes de normalizar), simulando uma entrada 30% maior que qualquer uma medida.
- O gráfico AM/AM (|y| × |x|, em **escala linear**, onde a saturação fica visível) sobrepõe:
  - PA medido;
  - PA_LSTM (deve coincidir com o medido);
  - PA_LSTM com x × 1,3;
  - linha tracejada no máx |x| do treino.

**Como interpretar a região à direita da linha tracejada:**

| Comportamento | Significado | Consequência para o DPD |
|---|---|---|
| A curva achata (satura) | Fisicamente plausível | Ok. O DPD só deve evitar pedir mais do que a saturação permite |
| A curva continua subindo | O modelo "acredita" que o PA nunca satura | O DPD pode explorar esse erro, gerando um u enorme que no PA real não funcionaria → limitar \|u\| |
| A curva oscila ou cai de forma estranha | Extrapolação ruim da rede | Limitar \|u\| à faixa medida |

Esse resultado orienta a Etapa 2 (escolha de G e do back-off).

### Resultados da Etapa 1

> A preencher após a execução: EVM da célula 1.5 e análise do gráfico da célula 1.6.
