"""Plot the paper's binary disagreement figure from aggregate counts."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("WAGO", "Siemens", "SMC")
PIPELINES = ("P1", "P2", "P3")


def main():
    """Render and save the heatmaps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/pipelines/binary_disagreement.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/pipelines/figures/fig03_binary_disagreement")
    parser.add_argument("--font-size", type=float, default=9)
    parser.add_argument("--width", type=float, default=7.1)
    parser.add_argument("--height", type=float, default=2.55)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as file:
        counts = {
            (row["dataset"], row["pair"]): (int(row["disagreement_count"]),
                                             int(row["n_comparable"]))
            for row in csv.DictReader(file)
        }

    plt.rcParams.update({"font.size": args.font_size, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height),
                             layout="constrained")
    for ax, dataset in zip(axes, DATASETS):
        cells = [[None if a == b else counts[dataset, "-".join(sorted((a, b)))]
                  for b in PIPELINES] for a in PIPELINES]
        rates = [[100 * cell[0] / cell[1] if cell else 0 for cell in row]
                 for row in cells]
        plot = ax.imshow(rates, vmin=0, vmax=40, cmap="Blues")
        ax.set_title(dataset)
        ax.set_xticks(range(3), PIPELINES)
        ax.set_yticks(range(3), PIPELINES)
        for i, row in enumerate(cells):
            for j, cell in enumerate(row):
                label = "—" if cell is None else f"{rates[i][j]:.1f}%\n{cell[0]}/{cell[1]}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=args.font_size - 2,
                        color="white" if rates[i][j] > 22 else "#111111")
    fig.colorbar(plot, ax=axes, label="Binary disagreement (%)", shrink=0.82, pad=0.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = args.output.with_name(f"{args.output.name}.{extension}")
        fig.savefig(path, dpi=args.dpi)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
