"""Split the products for training and testing.

The SMC labels are assigned per product family, and the products of a family have
nearly the same text. A random split therefore puts near-twins with the same label
into training and test. With ``--by_family``, we split whole families into five
folds instead, so that each product is predicted by a model which has not seen its
family. The SMC products are grouped by their labelling rule (see ``label_smc.py``),
since sibling families share a rule (e.g., IZS and IZT are both ionizers); the WAGO
products are grouped by their ``ManufacturerProductFamily``.

Without ``--by_family``, the products are split randomly into train, validation and
test (50/25/25).
"""

import argparse
import collections
import pathlib
import random
import sys
from collections.abc import Mapping, Sequence

from sklearn.model_selection import train_test_split

from pdm_at_scale import label_smc
from pdm_at_scale.model import Ordinal
from pdm_at_scale.xgboost_ablation import build_dataset, common

#: Number of family folds.
FOLDS = 5


def split_randomly(rows: Sequence[common.DatasetRow], seed: int) -> dict[str, str]:
    """Assign each product randomly to train, validation or test (50/25/25)."""
    ids = [row.product_id for row in rows]
    train_ids, rest_ids = train_test_split(
        ids, train_size=0.5, random_state=seed, shuffle=True
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=0.5, random_state=seed, shuffle=True
    )
    return {
        **{str(pid): "train" for pid in train_ids},
        **{str(pid): "val" for pid in val_ids},
        **{str(pid): "test" for pid in test_ids},
    }


def group_products(
    rows: Sequence[common.DatasetRow], wago_environment: pathlib.Path
) -> dict[str, str]:
    """Assign each product to its family group."""
    families = dict(
        build_dataset.extract_wago_nameplate_texts(
            build_dataset.read_environment(wago_environment),
            "ManufacturerProductFamily",
        )
    )

    groups: dict[str, str] = {}
    for row in rows:
        if row.product_id in families:
            family = families[row.product_id]
            groups[row.product_id] = (
                f"family:{family}"
                if family is not None
                else f"product:{row.product_id}"
            )
            continue

        rule = label_smc.find_rule(row.product_id)
        if rule is None:
            raise common.AblationError(
                f"{row.product_id!r} is neither in {wago_environment} nor matched "
                f"by an SMC labelling rule"
            )
        groups[row.product_id] = f"rule:{rule.product_id_pattern.pattern}"
    return groups


def assign_folds(
    rows: Sequence[common.DatasetRow], groups: Mapping[str, str], seed: int
) -> dict[str, int]:
    """Assign each group to a fold, balancing the labels over the folds.

    The negative products (``NONE`` and ``LOW``) are balanced together, the positive
    ones per level. The classes are handled from the rarest one. Within a class, the
    largest group goes first into the fold with the fewest products of that class,
    and then with the fewest products overall. The seed only orders groups of equal
    size.
    """
    members: dict[str, list[common.DatasetRow]] = collections.defaultdict(list)
    for row in rows:
        members[groups[row.product_id]].append(row)

    def class_of(row: common.DatasetRow) -> Ordinal:
        return row.gt_ordinal if row.gt_label else Ordinal.NONE

    groups_of_class: dict[Ordinal, list[str]] = collections.defaultdict(list)
    for group, group_rows in members.items():
        classes = {class_of(row) for row in group_rows}
        if len(classes) != 1:
            raise common.AblationError(f"The group {group!r} mixes labels")
        groups_of_class[classes.pop()].append(group)

    rng = random.Random(seed)
    per_class = {a_class: [0] * FOLDS for a_class in groups_of_class}
    total = [0] * FOLDS
    fold_of_group: dict[str, int] = {}

    for a_class in sorted(
        groups_of_class,
        key=lambda c: (sum(len(members[g]) for g in groups_of_class[c]), c.value),
    ):
        names = sorted(groups_of_class[a_class])
        rng.shuffle(names)
        names.sort(key=lambda g: -len(members[g]))
        for group in names:
            fold = min(range(FOLDS), key=lambda f: (per_class[a_class][f], total[f], f))
            fold_of_group[group] = fold
            per_class[a_class][fold] += len(members[group])
            total[fold] += len(members[group])

    return {row.product_id: fold_of_group[groups[row.product_id]] for row in rows}


def main(prog: str) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--dataset", default="data/xgboost_ablation/dataset.csv")
    parser.add_argument("--by_family", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wago_environment", default="data/wago/environment.json")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    design = "family" if args.by_family else "random"
    output = dataset_path.parent / f"split_{design}_seed{args.seed}.csv"

    try:
        rows = common.read_dataset(dataset_path)
        if args.by_family:
            groups = group_products(rows, pathlib.Path(args.wago_environment))
            folds = assign_folds(rows, groups, seed=args.seed)
    except common.AblationError as exception:
        print(exception, file=sys.stderr)
        return 1

    if args.by_family:
        common.write_csv(
            output,
            common.FOLD_COLUMNS,
            [
                {
                    "product_id": row.product_id,
                    "group": groups[row.product_id],
                    "fold": str(folds[row.product_id]),
                }
                for row in rows
            ],
        )
        fold_sizes = collections.Counter(folds.values())
        print(
            f"Split {len(rows)} products in {len(set(groups.values()))} groups into "
            f"{FOLDS} folds of {', '.join(str(fold_sizes[f]) for f in range(FOLDS))} "
            f"products to {output}"
        )
    else:
        parts = split_randomly(rows, seed=args.seed)
        common.write_csv(
            output,
            common.SPLIT_COLUMNS,
            [
                {"product_id": row.product_id, "part": parts[row.product_id]}
                for row in rows
            ],
        )
        part_sizes = collections.Counter(parts.values())
        print(
            f"Split {len(rows)} products into {part_sizes['train']} train, "
            f"{part_sizes['val']} validation and {part_sizes['test']} test to {output}"
        )
    return 0


def entry_point() -> int:
    """Provide an entry point for the console script."""
    return main(prog="pdm-at-scale-ablation-split")


if __name__ == "__main__":
    sys.exit(main(prog="split.py"))
