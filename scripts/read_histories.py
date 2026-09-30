import argparse
from pathlib import Path
import pandas as pd
from pa_model import RESULTS

# Accepted column suffixes for each metric (R² may be saved as "rsqr" or "r2")
METRICS = {
    "evm": ("evm",),
    "rmse": ("rmse",),
    "rsqr": ("rsqr", "r2"),
}

# For R² the best value is the highest one, for the others it is the lowest
MAXIMIZE = {"rsqr"}


def find_column(df, split, metric):
    for suffix in METRICS[metric]:
        column = f"{split}_{suffix}"
        if column in df.columns:
            return column
    return None


def read_histories(results_dir, split, metric):
    rows = []
    for path in sorted(Path(results_dir).rglob("histories/*.csv")):
        df = pd.read_csv(path, index_col="epoch")
        column = find_column(df, split, metric)
        if column is None:
            print(f"Skipping {path.name}: no '{split}_{metric}' column")
            continue

        best_epoch = df[column].idxmax() if metric in MAXIMIZE else df[column].idxmin()
        rows.append({"combination": path.stem, "best_epoch": int(best_epoch), column: df.loc[best_epoch, column]})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prints the best value of a metric for every history in the results folder")
    parser.add_argument("-m", "--metric", choices=METRICS.keys(), default="evm", help="metric to rank by (default: evm)")
    parser.add_argument("-s", "--split", choices=("val", "train"), default="val", help="split to read the metric from (default: val)")
    parser.add_argument("-n", "--top", type=int, default=None, help="number of combinations to show (default: all)")
    parser.add_argument("-d", "--results-dir", type=Path, default=RESULTS, help=f"results folder (default: {RESULTS})")
    args = parser.parse_args()

    df = read_histories(args.results_dir, args.split, args.metric)
    if df.empty:
        print(f"No histories with '{args.split}_{args.metric}' found in {args.results_dir}")
        return

    column = df.columns[-1]
    df = df.sort_values(column, ascending=args.metric not in MAXIMIZE).reset_index(drop=True)
    if args.top is not None:
        df = df.head(args.top)

    print(df.to_string())
    best = df.iloc[0]
    print(f"\nBest {column}: {best[column]:.6f} ({best['combination']}, epoch {best['best_epoch']})")


if __name__ == "__main__":
    main()
