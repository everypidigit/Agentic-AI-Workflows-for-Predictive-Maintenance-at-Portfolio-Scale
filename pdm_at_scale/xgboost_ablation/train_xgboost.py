"""Train XGBoost on the embeddings, for the ablation.

We train a binary classifier on ``gt_label`` and a regressor on the ordinal score of
``gt_ordinal`` in [0, 1]; the regressor output is clipped to [0, 1] and rounded to
the nearest level for the comparison with the pipelines.

With family folds, the models are trained once per fold on the other folds with 300
trees and predict the products of the fold, so that each product is predicted once
by models which have not seen its family. With a random split, the models are
trained on the train part with early stopping on the validation part.
"""

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import xgboost

from pdm_at_scale.xgboost_ablation import common

#: Columns of the predictions with a random split.
PREDICTION_COLUMNS: Sequence[str] = (
    "product_id",
    "part",
    "pred_label",
    "pred_score",
    "pred_level",
)

#: Columns of the predictions with family folds.
FAMILY_PREDICTION_COLUMNS: Sequence[str] = (
    "product_id",
    "group",
    "fold",
    "pred_label",
    "pred_score",
    "pred_level",
)

#: Hyperparameters of both models.
HYPERPARAMETERS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "random_state": 0,
    "tree_method": "hist",
}

#: Trees and early stopping with a random split, which has validation data.
RANDOM_TREES: dict[str, Any] = {"n_estimators": 2000, "early_stopping_rounds": 50}

#: Trees with family folds, which have no validation data.
FAMILY_TREES: dict[str, Any] = {"n_estimators": 300, "early_stopping_rounds": None}

Floats = npt.NDArray[np.float64]
Bools = npt.NDArray[np.bool_]


def load_embeddings(
    path: pathlib.Path, rows: Sequence[common.DatasetRow]
) -> npt.NDArray[np.float32]:
    """Load the embeddings in the order of the dataset."""
    if not path.is_file():
        raise common.AblationError(f"The file {path} does not exist")

    with np.load(path, allow_pickle=False) as archive:
        index = {str(pid): i for i, pid in enumerate(archive["product_ids"])}
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)

    if any(row.product_id not in index for row in rows):
        raise common.AblationError(f"{path} lacks embeddings of some products")
    return embeddings[[index[row.product_id] for row in rows]]


def fit_and_predict(
    estimator: xgboost.XGBModel,
    x: npt.NDArray[np.float32],
    y: npt.NDArray[Any],
    train: Bools,
    val: Bools | None,
    predict: Bools,
) -> Floats:
    """Fit the estimator, with early stopping on ``val`` if given, and predict."""
    if val is None:
        estimator.fit(x[train], y[train], verbose=False)
        trees = (0, int(estimator.get_params()["n_estimators"]))
    else:
        estimator.fit(x[train], y[train], eval_set=[(x[val], y[val])], verbose=False)
        trees = (0, int(estimator.best_iteration) + 1)

    if isinstance(estimator, xgboost.XGBClassifier):
        return np.asarray(
            estimator.predict_proba(x[predict], iteration_range=trees)[:, 1],
            dtype=np.float64,
        )
    return np.asarray(estimator.predict(x[predict], iteration_range=trees), np.float64)


def main(prog: str) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--dataset", default="data/xgboost_ablation/dataset.csv")
    parser.add_argument("--embeddings", default="data/xgboost_ablation/embeddings.npz")
    parser.add_argument(
        "--split", default="data/xgboost_ablation/split_family_seed0.csv"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    embeddings_path = pathlib.Path(args.embeddings)
    split_path = pathlib.Path(args.split)
    output_dir = pathlib.Path(
        "results/xgboost_ablation"
    ) / split_path.stem.removeprefix("split_")

    if (output_dir / "predictions.csv").exists() and not args.overwrite:
        print(f"{output_dir} already contains a run; pass --overwrite", file=sys.stderr)
        return 1

    try:
        rows = common.read_dataset(dataset_path)
        design = common.design_of(split_path)
        x = load_embeddings(embeddings_path, rows)
        rounds: list[tuple[Bools, Bools | None, Bools]]
        locations: list[dict[str, str]]
        if design == "random":
            parts = common.read_split(split_path, rows)
            part_of_row = np.array([parts[row.product_id] for row in rows])
            rounds = [
                (
                    part_of_row == "train",
                    part_of_row == "val",
                    np.ones(len(rows), dtype=np.bool_),
                )
            ]
            locations = [{"part": parts[row.product_id]} for row in rows]
            columns, trees = PREDICTION_COLUMNS, RANDOM_TREES
        else:
            folds = common.read_folds(split_path, rows)
            fold_of_row = np.array([folds[row.product_id][1] for row in rows])
            rounds = [
                (fold_of_row != fold, None, fold_of_row == fold)
                for fold in sorted(set(fold_of_row.tolist()))
            ]
            locations = [
                {
                    "group": folds[row.product_id][0],
                    "fold": str(folds[row.product_id][1]),
                }
                for row in rows
            ]
            columns, trees = FAMILY_PREDICTION_COLUMNS, FAMILY_TREES
    except common.AblationError as exception:
        print(exception, file=sys.stderr)
        return 1

    labels = np.array([int(row.gt_label) for row in rows], dtype=np.int64)
    scores = np.array([row.gt_score for row in rows], dtype=np.float64)

    p_true = np.full(len(rows), np.nan)
    pred_score = np.full(len(rows), np.nan)
    for train, val, predict in rounds:
        p_true[predict] = fit_and_predict(
            xgboost.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **HYPERPARAMETERS,
                **trees,
            ),
            x,
            labels,
            train,
            val,
            predict,
        )
        pred_score[predict] = np.clip(
            fit_and_predict(
                xgboost.XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    **HYPERPARAMETERS,
                    **trees,
                ),
                x,
                scores,
                train,
                val,
                predict,
            ),
            0.0,
            1.0,
        )

    common.write_csv(
        output_dir / "predictions.csv",
        columns,
        [
            {
                "product_id": row.product_id,
                **locations[i],
                "pred_label": "true" if p_true[i] >= 0.5 else "false",
                "pred_score": f"{pred_score[i]:.6f}",
                "pred_level": common.level_of_score(float(pred_score[i])).value,
            }
            for i, row in enumerate(rows)
        ],
    )
    (output_dir / "params.json").write_text(
        json.dumps(
            {
                "design": design,
                "inputs": {
                    name: {"path": str(path), "sha256": common.sha256_of(path)}
                    for name, path in (
                        ("dataset", dataset_path),
                        ("embeddings", embeddings_path),
                        ("split", split_path),
                    )
                },
                "hyperparameters": {**HYPERPARAMETERS, **trees},
                "xgboost": xgboost.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Trained {len(rounds)} times into {output_dir}")
    return 0


def entry_point() -> int:
    """Provide an entry point for the console script."""
    return main(prog="pdm-at-scale-ablation-train")


if __name__ == "__main__":
    sys.exit(main(prog="train_xgboost.py"))
