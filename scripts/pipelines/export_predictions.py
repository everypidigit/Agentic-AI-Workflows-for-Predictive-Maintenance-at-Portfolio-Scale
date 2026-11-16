"""Combine completed pipeline outputs into the evaluator's prediction CSVs."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {"wago": "WAGO", "siemens": "Siemens", "smc": "SMC"}
INPUT_FIELDS = ("product_id", "ManufacturerProductDesignation")
ORDINAL = {"NONE", "LOW", "MODERATE", "HIGH"}


def read_rows(path):
    """Read a nonempty CSV with consistent columns."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if (
            not rows
            or not reader.fieldnames
            or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or any(None in row or None in row.values() for row in rows)
        ):
            raise ValueError(f"{path}: expected a nonempty, regular CSV")
        return reader.fieldnames, rows


def export_rows(path, source, dataset, binary, ordinal, pipeline):
    """Normalize label names while preserving additional pipeline columns."""
    header, rows = read_rows(path)
    required = {*INPUT_FIELDS, binary, ordinal}
    if not required <= set(header):
        raise ValueError(f"{path}: missing fields {sorted(required - set(header))}")
    if [tuple(row[key] for key in INPUT_FIELDS) for row in rows] != source:
        raise ValueError(
            f"{path}: product IDs, designations or order differ from inputs"
        )
    allowed = ORDINAL | {"NO_MAJORITY"} if pipeline == 2 else ORDINAL
    exported = []
    for row in rows:
        if row[binary] not in {"TRUE", "FALSE"} or row[ordinal] not in allowed:
            raise ValueError(f"{path}: invalid labels for {row['product_id']}")
        exported.append(
            {
                "dataset": dataset,
                **{key: row[key] for key in INPUT_FIELDS},
                "binary": row[binary],
                "ordinal": row[ordinal],
                **{key: value for key, value in row.items() if key not in required},
            }
        )
    return exported


def main():
    """Validate and export all three pipelines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline1-dir", required=True, type=Path)
    parser.add_argument("--pipelines23-dir", required=True, type=Path)
    parser.add_argument("--pipeline3-dir", type=Path, help="Use a standalone P3 run")
    parser.add_argument(
        "--inputs-dir", type=Path, default=ROOT / "data/pipelines/inputs"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/pipelines")
    args = parser.parse_args()

    try:
        config = json.loads((ROOT / "data/pipelines/pipelines.json").read_text())
        results = {pipeline: [] for pipeline in (1, 2, 3)}
        for dataset, label in DATASETS.items():
            input_path = args.inputs_dir / f"{dataset}_product_designations.csv"
            header, rows = read_rows(input_path)
            if not set(INPUT_FIELDS) <= set(header):
                raise ValueError(f"{input_path}: missing product input columns")
            source = [tuple(row[key] for key in INPUT_FIELDS) for row in rows]
            if len({product_id for product_id, _ in source}) != len(source):
                raise ValueError(f"{input_path}: duplicate product IDs")
            p3_dir = (
                args.pipeline3_dir / dataset / "SYSTEM_CONTEXT"
                if args.pipeline3_dir
                else args.pipelines23_dir / "pipeline_3" / dataset
            )
            specs = [
                (
                    args.pipeline1_dir / dataset / "results/experiment_1_combined.csv",
                    "binary",
                    "ordinal",
                ),
                (
                    args.pipelines23_dir
                    / "pipeline_2"
                    / dataset
                    / "results/experiment_2_majority_vote.csv",
                    "binary_majority_vote",
                    "ordinal_majority_vote",
                ),
                (
                    p3_dir / "results/experiment_3_final_judgment.csv",
                    "final_binary",
                    "final_ordinal",
                ),
            ]
            for pipeline, (path, binary, ordinal) in enumerate(specs, 1):
                results[pipeline].extend(
                    export_rows(path, source, label, binary, ordinal, pipeline)
                )

        outputs = {
            args.output_dir / config[f"Pipeline {p}"]: rows
            for p, rows in results.items()
        }
        for path, rows in outputs.items():
            if path.exists():
                raise ValueError(f"{path} already exists. Choose a new --output-dir.")
            if any(row.keys() != rows[0].keys() for row in rows):
                raise ValueError(f"{path}: source headers differ between datasets")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, rows in outputs.items():
            with path.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{path}: {len(rows)} products")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
