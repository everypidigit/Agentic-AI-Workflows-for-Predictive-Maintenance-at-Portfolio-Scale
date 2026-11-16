"""Regenerate the positive binary recommendation chart from the counts CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("#0072B2", "#E69F00", "#009E73")
DATASETS = ("WAGO", "Siemens", "SMC")
PIPELINES = ("P1", "P2", "P3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/pipelines/binary_results.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/pipelines/figures/fig01_positive_recommendations_large_text",
                        help="Output path without an extension")
    parser.add_argument("--font-size", type=float, default=13)
    parser.add_argument("--width", type=float, default=7.1, help="Width in inches")
    parser.add_argument("--height", type=float, default=3.4, help="Height in inches")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8-sig") as file:
        counts = {(row["dataset"], row["pipeline"]): (int(row["true_count"]), int(row["n"]))
                  for row in csv.DictReader(file)}

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": args.font_size,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(args.width, args.height), layout="constrained")
    positions = range(len(DATASETS))
    bar_width = 0.24
    for index, (pipeline, color) in enumerate(zip(PIPELINES, COLORS)):
        values = []
        for dataset in DATASETS:
            positives, total = counts[dataset, pipeline]
            values.append(100 * positives / total)
        bars = ax.bar([x + (index - 1) * bar_width for x in positions],
                      values, bar_width, label=pipeline, color=color,
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f", padding=3)

    ax.set_xticks(positions, [f"{d}\n(n = {counts[d, 'P1'][1]})" for d in DATASETS])
    ax.set_ylabel("Positive recommendations (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1), borderaxespad=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = args.output.with_name(f"{args.output.name}.{extension}")
        fig.savefig(path, dpi=args.dpi)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
