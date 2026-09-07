# Estrutura do Projeto

Documento de referência da organização do repositório `pa-gain-linearization`.

---

## Princípio de organização

A estrutura separa três categorias com ciclos de vida distintos:

| Categoria | Onde vive | Muda quando |
|---|---|---|
| **Código reutilizável** | `src/pa_model/` | A lógica do pipeline muda |
| **Execução e análise** | `scripts/`, `notebooks/` | O experimento muda |
| **Dados e artefatos** | `data/`, `results/` | A cada rodada |

Três regras derivam disso:

1. **Notebook não contém lógica.** Ele importa, chama e plota. Classe ou função com mais de poucas linhas pertence ao pacote.
2. **Artefato não mora dentro do pacote.** `.pth`, CSVs e figuras são saída, não código.
3. **Uma pasta só nasce quando dói não ter.** Hierarquia com um arquivo por diretório é custo sem benefício.

---

## Árvore

```
pa-gain-linearization/
├── pyproject.toml
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── pa_gain_linearization.ipynb
├── scripts/
│   └── run_grid_search.py
├── results/
│   ├── checkpoints/
│   └── logs/
└── src/
    └── pa_model/
        ├── __init__.py
        ├── paths.py
        ├── dataset.py
        ├── models.py
        ├── trainer.py
        ├── search.py
        └── metrics.py
```

---

## Raiz

### `pyproject.toml`

Único arquivo necessário para o Python reconhecer o projeto como pacote instalável. Contém metadados (`name`, `version`, `requires-python`), dependências e a instrução `where = ["src"]`.

É o que faz `pip install -e .` funcionar e, por consequência, o que elimina qualquer manipulação de `sys.path` no notebook. Também abriga a configuração de ferramentas como `ruff`, `black` e `pytest`.

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "pa-gain-linearization"
version = "0.1.0"
description = "Modelagem e linearização de ganho de amplificador de potência"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "torch",
    "scikit-learn",
    "pandas",
]

[tool.setuptools.packages.find]
where = ["src"]
```

### `README.md`

Função concreta, não cerimonial: registrar o comando de setup, o formato esperado dos dados de medição do PA e como reproduzir uma rodada. É o que impede que, meses depois, haja dúvida sobre o layout do CSV bruto.

### `.gitignore`

Impede que `results/`, `data/`, `__pycache__/` e o `.egg-info` gerado pela instalação editável entrem no repositório.

```gitignore
__pycache__/
*.egg-info/
.ipynb_checkpoints/
results/
data/
.venv/
```

---

## `data/` — entrada

### `data/raw/`

Medições do PA como saíram do instrumento. **Somente leitura**: nenhum script escreve aqui.

Isso funciona como ponto de restauração. Se o pré-processamento estiver errado, reprocessa-se a partir do original em vez de perder a medição.

### `data/processed/`

Saída do pré-processamento: janelas deslizantes, splits temporais 60/20/20, dados normalizados.

Tudo aqui é regenerável a partir de `raw/` mais o código — e por isso fica fora do Git.

> A separação existe porque só uma das duas pastas é irrecuperável.

---

## `src/` — código

### `src/`

Container neutro, **não importável**. Não possui `__init__.py` e não é um pacote.

Sua função é garantir que, após `pip install -e .`, o import resolva para o pacote instalado e não para uma pasta que por acaso está no diretório atual. Evita o falso positivo clássico: *"funciona porque eu estava na raiz do projeto"*.

### `src/pa_model/__init__.py`

Marca o pacote e define sua API pública.

Recomendação: manter **vazio** no início. Reexportar nomes cria dois caminhos válidos para a mesma classe (`pa_model.Dataset` e `pa_model.dataset.Dataset`), e ambos acabam sendo usados sem consistência.

### `src/pa_model/paths.py`

Centraliza a resolução de caminhos ancorada na localização do próprio arquivo, não no diretório de trabalho.

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "results" / "checkpoints"
LOGS = ROOT / "results" / "logs"
```

O `parents[2]` sobe de `paths.py` → `pa_model` → `src` → raiz. Se a profundidade do arquivo mudar, esse índice muda junto.

Parece supérfluo até o dia em que o mesmo código roda do notebook e do script, com diretórios de trabalho diferentes.

### `src/pa_model/dataset.py`

Classes `Dataset` e `SlidingWindowDataset`.

Ficam juntas porque a segunda depende conceitualmente da primeira e ambas mudam pelo mesmo motivo: alteração no formato dos dados ou na janela temporal. Coisas que mudam juntas ficam juntas.

### `src/pa_model/models.py`

Definição da arquitetura: `nn.LSTM`, inicialização de `h_0`/`c_0` com shape `(num_layers, batch_size, hidden_size)`, extração do último timestep e camada linear de saída.

Apenas estrutura da rede, sem nada de treino. A separação importa porque o modelo é o que se troca ao comparar contra os baselines (MP, VARMAX, feedforward) mantendo o resto do pipeline idêntico.

### `src/pa_model/trainer.py`

Classe `TrainerLSTM`: loop de épocas, early stopping, salvamento de checkpoint, logging.

Depende de `models.py`, mas não o contrário. Essa direção única de dependência é o que permite avaliar um `.pth` salvo sem instanciar o treinador.

### `src/pa_model/search.py`

Classe `GridSearchLSTM`. Separada do trainer porque opera num nível acima: orquestra múltiplos treinos.

É também onde vivem estratégias alternativas de busca (random search, successive halving). Trocar a estratégia não deve exigir alteração em `trainer.py`.

### `src/pa_model/metrics.py`

NMSE, EVM, RMSE. Funções puras, sem estado.

São importadas tanto pelo trainer quanto pelo notebook de avaliação — o que já justifica não deixá-las dentro de nenhum dos dois. Pela pureza, são o candidato mais óbvio a teste automatizado quando isso se tornar necessário.

---

## Execução

### `scripts/run_grid_search.py`

Ponto de entrada para execução longa. Herda o papel do antigo `main.py`.

Precisa ser executável sem interação: `python scripts/run_grid_search.py` roda do início ao fim, sobrevive a desconexão e retoma de onde parou. Um grid de milhares de combinações não cabe num kernel de notebook.

### `notebooks/pa_gain_linearization.ipynb`

Exploração, plotagem e análise de resultados já produzidos.

Importa, chama e plota — nenhuma definição de classe. Isso mantém o notebook curto o suficiente para ser lido e mantém a lógica onde pode ser testada e reutilizada.

Para edições no pacote refletirem sem reiniciar o kernel:

```python
%load_ext autoreload
%autoreload 2

from pa_model.dataset import Dataset
from pa_model.metrics import nmse
```

---

## `results/` — saída

### `results/checkpoints/`

Arquivos `.pth` (`rmse.pth`, `evm.pth`, `avg_loss.pth`).

Pasta dedicada facilita nomear rodadas por timestamp e evita sobrescrita acidental. Torna explícito que os checkpoints são artefatos versionados por rodada, avaliados no conjunto de teste uma única vez.

### `results/logs/`

CSVs de métricas por época e por combinação de hiperparâmetros.

Separados dos checkpoints por terem ciclo de vida diferente: logs são lidos constantemente durante a análise, checkpoints raramente. Num diretório homogêneo, um CSV de 0 bytes — sintoma de falha no logging com early stopping — salta aos olhos numa listagem.

---

## Setup

```bash
# na raiz do projeto
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

**Verificação:** o kernel do Jupyter precisa apontar para o mesmo ambiente do `pip`. Numa célula do notebook:

```python
import sys
print(sys.executable)
```

Compare com `which python` no terminal. Se divergirem, a instalação foi para outro ambiente e o import continuará falhando — o que parece erro de configuração do `pyproject.toml`, mas não é.

---

## O que ainda não existe (e por quê)

| Item | Quando adicionar |
|---|---|
| `tests/` | Quando houver função pura cuja regressão silenciosa custe caro. `metrics.py` é o primeiro candidato. |
| `configs/` | Quando o grid não couber mais num dicionário em `search.py`. Antes disso, é indireção sem ganho. |
| CI | Depois de `tests/`, não antes. |
| `models/`, `training/` como pacotes | Quando um único arquivo por tema deixar de ser suficiente. |
