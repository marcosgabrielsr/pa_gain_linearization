"""
Checks the DPD pipeline before running the whole grid search (dpd_lstm.ipynb), on the machine with the GPU.

1. Loads the PA LSTM of the APA (results/APA_200MHz) as a frozen twin and the gain G (linear_gain.json).
2. Checks that the twin is deterministic in train() mode and that the untrained DPD is the identity.
3. Trains ONE combination exactly like GridSearchDPD.run (a few epochs) and prints the time per epoch.

Usage (at the project root):
    python scripts/check_dpd.py                       # ws=5, hs=16, nl=1, 3 epochs
    python scripts/check_dpd.py --ws 10 --epochs 5
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from pa_model import Dataset, SWDataset, WindowLoader, DATA_RAW, RESULTS
from dpd import DpdLSTM, TrainerDPD, load_pa_twin, run_pa, run_dpd


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", type=int, default=5)
    ap.add_argument("--hs", type=int, default=16)
    ap.add_argument("--nl", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    pa_dir = RESULTS / "APA_200MHz"
    train, val, test = Dataset.from_split_csv(DATA_RAW / "APA_200MHz")
    pa, wp, scaler_x, scaler_y = load_pa_twin(pa_dir, device)
    gain = json.load(open(pa_dir / "linear_gain.json"))
    G = gain["gi"] + 1j * gain["gq"]
    print(f"PA LSTM window = {wp}, G = {G:.4f}")

    # 1) The twin must give the same output twice (dropout zeroed in train() mode)
    xs = torch.randn(64, wp, 2, device=device)
    with torch.no_grad():
        print("1) PA twin deterministic in train():", torch.equal(pa(xs), pa(xs)))

    # 2) Untrained DPD = identity: DPD + PA must give the same output as the PA alone
    torch.manual_seed(args.seed)
    dpd = DpdLSTM(args.hs, args.nl).to(device)
    x = test.x[:2000]
    u, y_dpd = run_dpd(dpd, pa, x, args.ws, wp, scaler_x, scaler_y, device)
    y_pa = run_pa(pa, x, wp, scaler_x, scaler_y, device)[args.ws - 1:]
    print("2) Untrained DPD is the identity:", np.allclose(u, x[args.ws - 1:], atol=1e-4), np.allclose(y_dpd, y_pa, atol=1e-4))

    # 3) One combination, like GridSearchDPD.run
    train_norm, _, _ = Dataset(train.x, G * train.x).normalize(scaler_x, scaler_y)
    val_norm, _, _ = Dataset(val.x, G * val.x).normalize(scaler_x, scaler_y)
    window = args.ws + wp - 1
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = WindowLoader(SWDataset(train_norm, window, device=device), batch_size=args.bs, shuffle=True, generator=generator)
    val_loader = WindowLoader(SWDataset(val_norm, window, device=device), batch_size=args.bs, shuffle=False)

    trainer = TrainerDPD(dpd, pa, args.ws, wp, args.lr, scaler_x, scaler_y, n_epochs=args.epochs, device=device, early_stopping=True)
    sync(device)
    t0 = time.perf_counter()
    _, _, history = trainer.fit(train_loader, val_loader)
    sync(device)
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(history)
    df.index.name = "epoch"
    print(f"\n3) {elapsed / len(df):.2f} s per epoch ({len(df)} epochs)\n")
    print(df.to_string())


if __name__ == "__main__":
    main()
