"""
Valida as otimizações do SWDataset/WindowLoader e do TrainerLSTM (2026-11-05).

1. Ordem dos lotes: WindowLoader e DataLoader, com a mesma semente, devem gerar lotes idênticos.
2. Treina UMA combinação exatamente como o GridSearchLSTM.run faz, mede o tempo por época
   e compara o histórico com o do backup (gerado com o código antigo).

Uso (na raiz do projeto, na máquina com GPU):
    python scripts/check_trainer.py                         # melhor combinação, 5 épocas
    python scripts/check_trainer.py --dataloader            # mesmo treino com o DataLoader (para comparar o tempo)
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pa_model import Dataset, SWDataset, WindowLoader, DATA_PROCESSED, ROOT
from lstm import PaLSTM, TrainerLSTM


def check_batch_order(data, ws, bs, seed, n_samples=20000, epochs=2):
    """True se WindowLoader e DataLoader geram exatamente os mesmos lotes, época após época."""
    sub = Dataset(data.x[:n_samples], data.y[:n_samples])
    ds = SWDataset(sub, ws)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    wl = WindowLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        for (xa, ya), (xb, yb) in zip(dl, wl, strict=True):
            if not (torch.equal(xa, xb) and torch.equal(ya, yb)):
                return False
    return True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=int, default=10)
    ap.add_argument("--hs", type=int, default=16)
    ap.add_argument("--nl", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--dr", type=float, default=0.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataloader", action="store_true", help="treinar com o DataLoader antigo em vez do WindowLoader")
    ap.add_argument("--backup", type=Path, default=ROOT / "results_backup_2026-11-05")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = Dataset.from_csv(DATA_PROCESSED / "expData.csv")
    train, val, _ = ds.split()
    train_norm, scaler_x, scaler_y = train.normalize()
    val_norm, _, _ = val.normalize(scaler_x, scaler_y)

    # 1) Ordem dos lotes
    same = check_batch_order(train_norm, args.ws, args.bs, args.seed)
    print(f"\n1) Ordem dos lotes WindowLoader x DataLoader: {'IGUAL' if same else 'DIFERENTE'}")

    # 2) Treino de uma combinação, igual ao GridSearchLSTM.run
    torch.manual_seed(args.seed)
    model = PaLSTM(input_size=2, hidden_size=args.hs, num_layers=args.nl, dropout=args.dr).to(device).to(torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    if args.dataloader:
        train_loader = DataLoader(SWDataset(train_norm, args.ws), batch_size=args.bs, shuffle=True, generator=generator)
        val_loader = DataLoader(SWDataset(val_norm, args.ws), batch_size=args.bs, shuffle=False)
    else:
        train_loader = WindowLoader(SWDataset(train_norm, args.ws, device=device), batch_size=args.bs, shuffle=True, generator=generator)
        val_loader = WindowLoader(SWDataset(val_norm, args.ws, device=device), batch_size=args.bs, shuffle=False)

    trainer = TrainerLSTM(model, nn.MSELoss(), args.lr, scaler_y, n_epochs=args.epochs, device=device, early_stopping=True)

    sync(device)
    t0 = time.perf_counter()
    _, _, history = trainer.fit(train_loader, val_loader)
    sync(device)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(history)
    df.index.name = "epoch"
    loader_name = "DataLoader" if args.dataloader else "WindowLoader"
    print(f"\n2) {loader_name}: {elapsed / len(df):.2f} s por época ({len(df)} épocas)\n")
    print(df.to_string())

    # 3) Comparação com o histórico do backup (mesmo nome de arquivo que o ResultsStorage usa)
    params = {"ws": args.ws, "hs": args.hs, "nl": args.nl, "lr": args.lr, "dr": args.dr, "bs": args.bs}
    name = "_".join(f"{k}{v:g}" for k, v in params.items())
    path = args.backup / "histories" / f"{name}.csv"
    if not path.exists():
        print(f"\n3) Histórico do backup não encontrado: {path}")
        return

    old = pd.read_csv(path, index_col="epoch")
    n = min(len(df), len(old))
    rel = ((df.iloc[:n] - old.iloc[:n][df.columns]).abs() / old.iloc[:n][df.columns].abs()).max()
    print(f"\n3) Maior diferença relativa em relação ao backup ({path.name}, {n} épocas):")
    print(rel.to_string())


if __name__ == "__main__":
    main()
