# APA_200MHz — origem dos dados

- **Fonte:** OpenDPD — https://github.com/lab-emi/OpenDPD/tree/main/datasets/APA_200MHz
- **Commit:** `76015bfa5ae2e664341b51f52735a73338865f18` (baixado em 2026-11-09)
- **Licença:** Apache-2.0 (cópia em [LICENSE](LICENSE))
- **Citação:** Y. Wu et al., "OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion", ISCAS 2024. https://doi.org/10.1109/ISCAS58744.2024.10558162

Arquivos copiados sem modificação: `train/val/test_input.csv`, `train/val/test_output.csv` (colunas `I,Q`) e `spec.json` (descrição do sinal). Os scripts e imagens do diretório original não foram copiados.

## Sinal (de `spec.json`)

LTE com 5 portadoras de 20 MHz (TM3.1a, 256QAM), dispositivo APA. Gerado a 491,52 MHz, transmitido e capturado a 983,04 MHz, com banda efetiva de 200 MHz e PAPR de 10 dB.

## SHA-256

```
2ca703c9e7eb39839db1fb01f91081a86e18535e53751c587cc71d2d71e9c625  train_input.csv
e760adf3908ed1be1e610c46f056e88bad6107a81cc8b01d91306727316b5930  train_output.csv
39bb15c9bd92549d1653498c140caff5cb2f20edffd433eafa46b4a81c491981  val_input.csv
02f67574444c7a8ba321cde1ea919c07fef1c99fb9d25678befc019c6b6645e2  val_output.csv
5027d3d69391ed22ad79c410831bdfed47b25045088dda0756801cf591c947bf  test_input.csv
991c1f97f38e57c7f3614dd3f23c411791e03d5eb5eae67492455cf9e94820f6  test_output.csv
edaf447bc2d59c2cf10df4ad5bececf057bd8822a45864ec8959ddea1a0ebf9a  spec.json
```
