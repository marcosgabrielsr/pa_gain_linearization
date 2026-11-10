# ADR-0005: Adicionar o APA_200MHz (OpenDPD) como um segundo PA

- **Data:** 2026-11-09
- **Status:** Aceita
- **Decidido por:** Marcos

## Contexto

Queremos treinar a LSTM também com o dataset `APA_200MHz` do [OpenDPD](https://github.com/lab-emi/OpenDPD/tree/main/datasets/APA_200MHz). Ele é bem diferente do dataset do nosso PA:

| | Nosso PA (`dadosIniciais.csv`) | APA_200MHz |
|---|---|---|
| Dispositivo | o nosso PA | outro PA (APA) |
| Sinal | – | LTE com 5 portadoras de 20 MHz, 256QAM, banda de 200 MHz, PAPR de 10 dB, amostrado a 983 MHz |
| Amostras | 48.384 | 98.304 (58.980 / 19.662 / 19.662) |
| Arquivos | um CSV `Xreal, Ximg, Yreal, Yimg` | 6 CSVs já divididos, colunas `I, Q` |
| Escala | valores brutos | normalizado (máx \|x\| = máx \|y\| = 1) |
| Ganho de pequeno sinal | – | ≈ 1,17 ∠ 4,7° |
| Ordem | embaralhada ([ADR-0001](0001-manter-dados-embaralhados.md)) | temporal (amostras vizinhas correlacionadas) |

Um sinal de 200 MHz de banda passando por um PA real quase certamente tem **efeito de memória**.

## Decisão

1. **Modelo separado.** O APA ganha a sua própria LSTM, treinada com o mesmo código, com resultados em `results/APA_200MHz/`. Os dois datasets **não** são misturados.
2. **Ordem temporal e split oficial.** Usamos os arquivos `train/val/test` como vêm, sem repetir e sem embaralhar. O [ADR-0001](0001-manter-dados-embaralhados.md) vale **só para o nosso PA**.
3. **Dados versionados.** Ficam em `data/raw/APA_200MHz/`, sem modificação, junto com a licença (Apache-2.0) e o `SOURCE.md` (fonte, commit `76015bf`, citação, SHA-256).

## Alternativas consideradas

- **Juntar os dois datasets num treino só:** são dispositivos diferentes, com ganhos e escalas diferentes. O modelo não representaria nenhum dos dois, e um DPD precisa do modelo de **um** PA. Rejeitada.
- **Repetir e embaralhar como no nosso PA:** ficaria consistente com o ADR-0001, mas descartaria a memória e impediria a comparação com os resultados do OpenDPD. Rejeitada.
- **Baixar os dados por script em vez de versionar:** deixaria o repositório menor (cerca de 7 MB), mas dependeria do repositório externo continuar igual. Rejeitada.

## Consequências

- O código de treino não muda. Só foi criado o `Dataset.from_split_csv(pasta)`, e o `grid_search_lstm.ipynb` escolhe o dataset pela variável `DATASET`.
- No APA, a janela (`ws`) deve fazer diferença de verdade no grid search.
- O [ADR-0003](0003-dpd-sem-memoria.md) (DPD sem memória) **não se aplica** ao APA. O DPD dele precisará de uma decisão própria.
- `scripts/read_histories.py` usa `rglob`. Sem `-d results/APA_200MHz`, ele mistura os históricos dos dois PAs.
- Qualquer resultado publicado com esses dados deve citar o OpenDPD (ver `data/raw/APA_200MHz/SOURCE.md`).

## Referências

- `data/raw/APA_200MHz/SOURCE.md`
- [codigo/dataset.md](../codigo/dataset.md)
- Y. Wu et al., "OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion", ISCAS 2024. https://doi.org/10.1109/ISCAS58744.2024.10558162
