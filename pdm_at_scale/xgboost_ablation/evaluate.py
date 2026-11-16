"""Score XGBoost and the pipelines against the labels, for the ablation.

With family folds, every product is predicted by XGBoost while its family is held
out, so all the methods are scored on all products, on the SMC products and per fold
(mean, standard deviation, minimum and maximum). With a random split, the methods
are scored on the test part.

The binary predictions are scored with the counts (TP, TN, FP, FN), the accuracy and
F1; the ordinal ones with the share of products at the right level, where a missing
level (``NO_MAJORITY`` of Pipeline 2) counts as wrong.

The pipeline outputs are listed in ``data/pipelines/pipelines.json`` as pipeline name
→ CSV file relative to that JSON, with the columns ``product_id``,
``ManufacturerProductDesignation``, ``binary`` and ``ordinal``.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import numpy.typing as npt

from pdm_at_scale import label_smc
from pdm_at_scale.xgboost_ablation import common, train_xgboost

#: Name of XGBoost in the outputs.
XGBOOST = "XGBoost"

#: Metrics in the order of the outputs.
METRICS: Sequence[str] = ("n", "tp", "tn", "fp", "fn", "accuracy", "f1", "level_acc")

#: Metrics which are counts.
COUNTS = frozenset(["n", "tp", "tn", "fp", "fn"])

#: Metrics summarized over the folds.
FOLD_METRICS: Sequence[str] = ("f1", "accuracy")

Ints = npt.NDArray[np.int64]
Bools = npt.NDArray[np.bool_]


@dataclasses.dataclass(frozen=True)
class Method:
    """Hold the predictions of one method in the dataset order (-1: missing level)."""

    name: str
    label: Bools
    level: Ints


def level_index(level: str) -> int:
    """Return the position of the level, or -1 if it is not a level."""
    values = [ordinal.value for ordinal in common.LEVELS]
    return values.index(level) if level in values else -1


def load_xgboost(
    path: pathlib.Path, rows: Sequence[common.DatasetRow], columns: Sequence[str]
) -> Method:
    """Load the predictions of XGBoost."""
    _, records = common.read_csv(path, columns)
    by_id = {record["product_id"]: record for record in records}
    if by_id.keys() != {row.product_id for row in rows}:
        raise common.AblationError(f"{path} does not match the dataset")
    ordered = [by_id[row.product_id] for row in rows]
    return Method(
        name=XGBOOST,
        label=np.array([r["pred_label"] == "true" for r in ordered], dtype=np.bool_),
        level=np.array([level_index(r["pred_level"]) for r in ordered], dtype=np.int64),
    )


def load_pipeline(
    name: str, path: pathlib.Path, rows: Sequence[common.DatasetRow]
) -> Method:
    """Load the binary and ordinal outputs of one pipeline."""
    header, records = common.read_csv(path)
    needed = ("product_id", "ManufacturerProductDesignation", "binary", "ordinal")
    if not all(column in header for column in needed):
        raise common.AblationError(f"{path}: expected the columns {list(needed)}")

    by_id = {record["product_id"]: record for record in records}
    labels: list[bool] = []
    levels: list[int] = []
    for row in rows:
        record = by_id.get(row.product_id)
        if record is None:
            raise common.AblationError(f"{path}: no output for {row.product_id!r}")
        if common.normalize_whitespace(
            record["ManufacturerProductDesignation"]
        ) != common.normalize_whitespace(row.designation):
            raise common.AblationError(
                f"{path}: the designation of {row.product_id!r} differs"
            )
        binary = record["binary"].strip().upper()
        ordinal = record["ordinal"].strip().upper()
        if binary not in ("TRUE", "FALSE") or (
            level_index(ordinal) < 0 and ordinal != "NO_MAJORITY"
        ):
            raise common.AblationError(f"{path}: invalid output of {row.product_id!r}")
        labels.append(binary == "TRUE")
        levels.append(level_index(ordinal))

    return Method(
        name=name,
        label=np.array(labels, dtype=np.bool_),
        level=np.array(levels, dtype=np.int64),
    )


def score(
    rows: Sequence[common.DatasetRow], method: Method, index: Ints
) -> dict[str, float]:
    """Score the method on the products at the given positions."""
    truth = np.array([row.gt_label for row in rows], dtype=np.bool_)[index]
    truth_level = np.array(
        [level_index(row.gt_ordinal.value) for row in rows], dtype=np.int64
    )[index]
    pred = method.label[index]

    tp = float((truth & pred).sum())
    tn = float((~truth & ~pred).sum())
    fp = float((~truth & pred).sum())
    fn = float((truth & ~pred).sum())
    return {
        "n": float(len(index)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(index),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else math.nan,
        "level_acc": float((method.level[index] == truth_level).mean()),
    }


def summarize(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Compute the mean, the sample standard deviation, the minimum and the maximum."""
    array = np.array(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)),
        float(array.min()),
        float(array.max()),
    )


def check_digest(description: Mapping[str, str], name: str) -> pathlib.Path:
    """Check that an input of the run has not changed since the training."""
    path = pathlib.Path(description["path"])
    if not path.is_file() or common.sha256_of(path) != description["sha256"]:
        raise common.AblationError(f"The {name} {path} changed since the training")
    return path


def latex_table(
    results: Mapping[tuple[str, str], Mapping[str, float]],
    per_fold: Mapping[str, Mapping[str, Sequence[float]]],
) -> str:
    """Render the results with family folds as one LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Predictions on all "
        + str(int(results[("all", XGBOOST)]["n"]))
        + r" labelled products, each predicted by XGBoost while its family is held "
        r"out, and on the "
        + str(int(results[("smc", XGBOOST)]["n"]))
        + r" SMC products. F1 (folds) is the mean $\pm$ standard deviation over the "
        r"folds; the level accuracy counts a missing level as wrong.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\hline",
        r" & \multicolumn{7}{c}{All products} & \multicolumn{2}{c}{SMC} \\",
        r"Method & F1 & F1 (folds) & Acc. & TP & TN & FP & FN & Acc. & Level acc. \\",
        r"\hline",
    ]
    for method, metrics in per_fold.items():
        everything, smc = results[("all", method)], results[("smc", method)]
        mean, std, _, _ = summarize(metrics["f1"])
        cells = [
            f"{everything['f1']:.2f}",
            f"{mean:.2f} $\\pm$ {std:.2f}",
            f"{everything['accuracy']:.2f}",
            *(str(int(everything[metric])) for metric in ("tp", "tn", "fp", "fn")),
            f"{smc['accuracy']:.2f}",
            format(Decimal(str(smc["level_acc"])).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ), ".2f"),
        ]
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main(prog: str) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_dir", default="results/xgboost_ablation/family_seed0")
    parser.add_argument(
        "--pipelines",
        help="JSON list of the pipeline outputs; without it, XGBoost only",
    )
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    output_dir = run_dir / "evaluation"

    try:
        params_path = run_dir / "params.json"
        if not params_path.is_file():
            raise common.AblationError(f"{params_path} does not exist; train first")
        params = json.loads(params_path.read_text(encoding="utf-8"))
        rows = common.read_dataset(check_digest(params["inputs"]["dataset"], "dataset"))
        split_path = check_digest(params["inputs"]["split"], "split")
        family = params["design"] == "family"

        methods = [
            load_xgboost(
                run_dir / "predictions.csv",
                rows,
                train_xgboost.FAMILY_PREDICTION_COLUMNS
                if family
                else train_xgboost.PREDICTION_COLUMNS,
            )
        ]
        if args.pipelines is not None:
            config_path = pathlib.Path(args.pipelines)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            methods += [
                load_pipeline(name, config_path.parent / file, rows)
                for name, file in config.items()
            ]

        sets: dict[str, Ints] = {}
        fold_of_row: Ints | None = None
        if family:
            folds = common.read_folds(split_path, rows)
            fold_of_row = np.array(
                [folds[row.product_id][1] for row in rows], dtype=np.int64
            )
            sets["all"] = np.arange(len(rows), dtype=np.int64)
            sets["smc"] = np.array(
                [
                    i
                    for i, row in enumerate(rows)
                    if label_smc.find_rule(row.product_id) is not None
                ],
                dtype=np.int64,
            )
        else:
            parts = common.read_split(split_path, rows)
            sets["test"] = np.array(
                [i for i, row in enumerate(rows) if parts[row.product_id] == "test"],
                dtype=np.int64,
            )
    except (common.AblationError, OSError, ValueError, KeyError) as exception:
        print(exception, file=sys.stderr)
        return 1

    results = {
        (eval_set, method.name): score(rows, method, index)
        for eval_set, index in sets.items()
        for method in methods
    }
    common.write_csv(
        output_dir / "metrics.csv",
        ("eval_set", "method", *METRICS),
        [
            {
                "eval_set": eval_set,
                "method": method,
                **{metric: f"{values[metric]:.6g}" for metric in METRICS},
            }
            for (eval_set, method), values in results.items()
        ],
    )

    for eval_set in sets:
        print(f"\n{eval_set}:")
        print(" " * 12 + "".join(f"{metric:>10}" for metric in METRICS))
        for method in methods:
            values = results[(eval_set, method.name)]
            cells = [
                f"{int(values[m]):>10}" if m in COUNTS else f"{values[m]:>10.3f}"
                for m in METRICS
            ]
            print(f"{method.name:12}" + "".join(cells))

    if fold_of_row is not None:
        fold_numbers = sorted(set(fold_of_row.tolist()))
        per_fold = {
            method.name: {
                metric: [
                    score(rows, method, np.flatnonzero(fold_of_row == fold))[metric]
                    for fold in fold_numbers
                ]
                for metric in FOLD_METRICS
            }
            for method in methods
        }
        common.write_csv(
            output_dir / "folds.csv",
            ("method", "metric", *(f"fold{f}" for f in fold_numbers), "mean", "std")
            + ("min", "max"),
            [
                {
                    "method": method,
                    "metric": metric,
                    **{
                        f"fold{f}": f"{v:.6g}"
                        for f, v in zip(fold_numbers, values, strict=True)
                    },
                    **{
                        key: f"{v:.6g}"
                        for key, v in zip(
                            ("mean", "std", "min", "max"),
                            summarize(values),
                            strict=True,
                        )
                    },
                }
                for method, metrics in per_fold.items()
                for metric, values in metrics.items()
            ],
        )
        (output_dir / "table.tex").write_text(
            latex_table(results, per_fold), encoding="utf-8"
        )

        print("\nPer fold (mean ± std, min, max):")
        for name, fold_metrics in per_fold.items():
            for metric, fold_values in fold_metrics.items():
                mean, std, low, high = summarize(fold_values)
                print(
                    f"  {name:12}{metric:10}{mean:.3f} ± {std:.3f} "
                    f"(min {low:.3f}, max {high:.3f})"
                )

    print(f"\nWrote the evaluation to {output_dir}")
    return 0


def entry_point() -> int:
    """Provide an entry point for the console script."""
    return main(prog="pdm-at-scale-ablation-evaluate")


if __name__ == "__main__":
    sys.exit(main(prog="evaluate.py"))
