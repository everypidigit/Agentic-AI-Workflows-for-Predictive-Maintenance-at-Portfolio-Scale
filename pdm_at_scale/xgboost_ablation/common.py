"""Share the data model and the file helpers of the XGBoost ablation."""

import csv
import hashlib
import pathlib
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import pydantic

from pdm_at_scale import model
from pdm_at_scale.model import Ordinal

#: Columns of the labelled dataset.
DATASET_COLUMNS: Sequence[str] = (
    "product_id",
    "ManufacturerProductDesignation",
    "gt_label",
    "gt_ordinal",
)

#: Parts of a random split.
Part = Literal["train", "val", "test"]
PARTS: Sequence[Part] = ("train", "val", "test")

#: Designs of the split: a random split into parts, or folds of whole families.
Design = Literal["random", "family"]

#: Columns of a random split.
SPLIT_COLUMNS: Sequence[str] = ("product_id", "part")

#: Columns of a split into folds of whole families.
FOLD_COLUMNS: Sequence[str] = ("product_id", "group", "fold")

#: Ordinal levels in increasing order.
LEVELS: Sequence[Ordinal] = (
    Ordinal.NONE,
    Ordinal.LOW,
    Ordinal.MODERATE,
    Ordinal.HIGH,
)


class AblationError(Exception):
    """Signal an invalid input, which the script reports and exits on."""


class DatasetRow(pydantic.BaseModel):
    """Represent one labelled product of the ablation."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    product_id: str = pydantic.Field(min_length=1)
    designation: str = pydantic.Field(min_length=1)
    gt_label: bool
    gt_ordinal: Ordinal

    @pydantic.model_validator(mode="after")
    def _check_consistent_labels(self) -> "DatasetRow":
        if self.gt_label != (self.gt_ordinal in model.POSITIVE_ORDINALS):
            raise ValueError(
                f"The binary label {self.gt_label} of {self.product_id!r} "
                f"is inconsistent with its ordinal label {self.gt_ordinal.value}"
            )
        return self

    @property
    def gt_score(self) -> float:
        """Return the ordinal score in [0, 1]."""
        return model.ORDINAL_SCORE[self.gt_ordinal]


def normalize_whitespace(text: str) -> str:
    """Collapse the whitespace so that formatting differences do not matter."""
    return " ".join(text.split())


def level_of_score(score: float) -> Ordinal:
    """Map a score in [0, 1] to the nearest ordinal level."""
    return min(LEVELS, key=lambda level: abs(model.ORDINAL_SCORE[level] - score))


def sha256_of(path: pathlib.Path) -> str:
    """Compute the SHA-256 digest of the file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(
    path: pathlib.Path, expected_columns: Sequence[str] | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    """Read the CSV file and check its columns, if ``expected_columns`` is given."""
    if not path.is_file():
        raise AblationError(f"The file {path} does not exist")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    if expected_columns is not None and header != list(expected_columns):
        raise AblationError(
            f"{path}: expected the columns {list(expected_columns)}, got {header}"
        )

    return header, rows


def write_csv(
    path: pathlib.Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> None:
    """Write the rows as a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(path: pathlib.Path, rows: Sequence[DatasetRow]) -> None:
    """Write the labelled dataset."""
    write_csv(
        path,
        DATASET_COLUMNS,
        [
            {
                "product_id": row.product_id,
                "ManufacturerProductDesignation": row.designation,
                "gt_label": "true" if row.gt_label else "false",
                "gt_ordinal": row.gt_ordinal.value,
            }
            for row in rows
        ],
    )


def read_dataset(path: pathlib.Path) -> list[DatasetRow]:
    """Read and validate the labelled dataset."""
    _, records = read_csv(path, DATASET_COLUMNS)

    rows: list[DatasetRow] = []
    errors: list[str] = []
    for line, record in enumerate(records, start=2):
        try:
            if record["gt_label"] not in ("true", "false"):
                raise ValueError(f"invalid gt_label {record['gt_label']!r}")
            rows.append(
                DatasetRow(
                    product_id=record["product_id"],
                    designation=record["ManufacturerProductDesignation"],
                    gt_label=record["gt_label"] == "true",
                    gt_ordinal=Ordinal(record["gt_ordinal"]),
                )
            )
        except ValueError as exception:
            errors.append(f"line {line}: {exception}")

    ids = [row.product_id for row in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate product IDs")

    if len(errors) > 0:
        raise AblationError(f"{path}: " + "; ".join(errors[:10]))

    return rows


def design_of(path: pathlib.Path) -> Design:
    """Tell the design of a split file from its columns."""
    header, _ = read_csv(path)
    if header == list(SPLIT_COLUMNS):
        return "random"
    if header == list(FOLD_COLUMNS):
        return "family"
    raise AblationError(
        f"{path}: expected the columns {list(SPLIT_COLUMNS)} or "
        f"{list(FOLD_COLUMNS)}, got {header}"
    )


def _check_covers(
    path: pathlib.Path, ids: Sequence[str], dataset: Sequence[DatasetRow]
) -> None:
    if len(set(ids)) != len(ids):
        raise AblationError(f"{path}: duplicate product IDs")
    if set(ids) != {row.product_id for row in dataset}:
        raise AblationError(f"{path} does not match the products of the dataset")


def read_split(path: pathlib.Path, dataset: Sequence[DatasetRow]) -> dict[str, Part]:
    """Read a random split as product ID → part."""
    _, records = read_csv(path, SPLIT_COLUMNS)
    _check_covers(path, [record["product_id"] for record in records], dataset)

    split: dict[str, Part] = {}
    for record in records:
        if record["part"] not in PARTS:
            raise AblationError(f"{path}: invalid part {record['part']!r}")
        split[record["product_id"]] = cast(Part, record["part"])
    return split


def read_folds(
    path: pathlib.Path, dataset: Sequence[DatasetRow]
) -> dict[str, tuple[str, int]]:
    """Read the family folds as product ID → (group, fold)."""
    _, records = read_csv(path, FOLD_COLUMNS)
    _check_covers(path, [record["product_id"] for record in records], dataset)

    folds: dict[str, tuple[str, int]] = {}
    for record in records:
        if not record["fold"].isdigit():
            raise AblationError(f"{path}: invalid fold {record['fold']!r}")
        folds[record["product_id"]] = (record["group"], int(record["fold"]))

    folds_of_group: dict[str, set[int]] = {}
    for group, fold in folds.values():
        folds_of_group.setdefault(group, set()).add(fold)
    if any(len(numbers) > 1 for numbers in folds_of_group.values()):
        raise AblationError(f"{path}: a group is split over several folds")

    return folds
