# Documentação do projeto — DPD para o PA

Esta pasta registra **o que foi feito**, **o que cada código faz** e **por que** cada decisão foi tomada.

## Estrutura

| Pasta | Conteúdo | Quando escrever |
|---|---|---|
| [diario/](diario/) | Diário de bordo: um arquivo por dia de trabalho (`AAAA-MM-DD.md`) | Ao fim de cada sessão: o que foi feito, resultados, dúvidas, próximos passos |
| [adr/](adr/) | ADRs (*Architecture Decision Records*): uma decisão por arquivo, numerada | Sempre que uma escolha afetar o rumo do projeto (dados, arquitetura, métricas...) |
| [codigo/](codigo/) | Explicação de cada código/notebook, bloco a bloco | Ao criar ou alterar um código relevante |

### Diário × ADR

- O **diário** é cronológico e informal: "hoje rodei X, deu Y, não entendi Z".
- O **ADR** é atemporal e formal: "decidimos A em vez de B, por estes motivos, com estas consequências". Um ADR não é apagado. Se a decisão mudar, cria-se um ADR novo que o **substitui**, e o antigo recebe o status `Substituída por ADR-XXXX`.

## Índice de ADRs

| Nº | Título | Status |
|---|---|---|
| [0001](adr/0001-manter-dados-embaralhados.md) | Manter os dados repetidos e embaralhados (`expData.csv`) | Aceita |
| [0002](adr/0002-dpd-por-aprendizado-direto-com-gemeo-digital.md) | DPD por aprendizado direto (DLA) usando a LSTM como gêmeo digital | Proposta |
| [0003](adr/0003-dpd-sem-memoria.md) | DPD sem memória (amostra a amostra) | Proposta |
| [0004](adr/0004-otimizar-auxiliares-manter-grid-search.md) | Otimizar os auxiliares do treino e manter a lógica do grid search | Aceita |
| [0005](adr/0005-dataset-apa-200mhz.md) | Adicionar o APA_200MHz (OpenDPD) como um segundo PA | Aceita |

Modelo para novos ADRs: [adr/0000-modelo.md](adr/0000-modelo.md).

## Índice de códigos documentados

| Código | Documentação |
|---|---|
| [notebooks/dpd_lstm.ipynb](../notebooks/dpd_lstm.ipynb) | [codigo/dpd_lstm.md](codigo/dpd_lstm.md) |
| [src/pa_model/sd_window.py](../src/pa_model/sd_window.py) | [codigo/sd_window.md](codigo/sd_window.md) |
| [src/lstm/trainer_lstm.py](../src/lstm/trainer_lstm.py), [earlystopping.py](../src/lstm/earlystopping.py) | [codigo/trainer_lstm.md](codigo/trainer_lstm.md) |
| [src/pa_model/dataset.py](../src/pa_model/dataset.py) | [codigo/dataset.md](codigo/dataset.md) |

## Roteiro do DPD

| Etapa | Descrição | Situação |
|---|---|---|
| 0 | Decidir sobre os dados | Concluída ([ADR-0001](adr/0001-manter-dados-embaralhados.md)) |
| 1 | Carregar a LSTM do PA congelada e validá-la | Código pronto, aguardando execução |
| 2 | Definir o ganho alvo G e o back-off | Pendente |
| 3 | Criar o modelo do DPD | Pendente |
| 4 | Treinar o DPD (DLA) | Pendente |
| 5 | Avaliar sem e com DPD (AM/AM, AM/PM, EVM, NMSE) | Pendente |
