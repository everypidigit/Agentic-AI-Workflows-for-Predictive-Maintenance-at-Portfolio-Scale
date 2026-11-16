"""Label the WAGO and SMC products for the XGBoost ablation.

The product IDs and designations come from the Digital Nameplates of the AAS
environments in ``data/``, which is what the pipelines read.

The SMC products get the labels of ``data/smc/ground_truth.json``. The WAGO products
are the negative case and get the ordinal label ``LOW``: they degrade, e.g., at the
contacts, but without measurable signs.
"""

import argparse
import collections
import json
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

import pydantic

from pdm_at_scale import label_smc, model
from pdm_at_scale.model import Ordinal
from pdm_at_scale.xgboost_ablation import common


def read_environment(path: pathlib.Path) -> Mapping[str, Any]:
    """Read an AAS environment as JSON."""
    if not path.is_file():
        raise common.AblationError(f"The file {path} does not exist")

    text = path.read_text(encoding="utf-8")
    if text.startswith("version https://git-lfs"):
        raise common.AblationError(
            f"{path} is a Git LFS pointer, not the data; run: git lfs pull"
        )

    environment = json.loads(text)
    if not isinstance(environment, dict):
        raise common.AblationError(f"{path}: expected a JSON object")
    return environment


def extract_wago_nameplate_texts(
    environment: Mapping[str, Any], id_short: str
) -> list[tuple[str, str | None]]:
    """Extract a text element of the WAGO Digital Nameplates, per product.

    The product ID of a WAGO shell is its ID-short, i.e., the WAGO article number.
    A product without the element or without an English text gets ``None``.
    """
    submodels_by_id = {
        submodel["id"]: submodel for submodel in environment["submodels"]
    }

    result: list[tuple[str, str | None]] = []
    for shell in environment["assetAdministrationShells"]:
        nameplate = next(
            submodels_by_id[reference["keys"][0]["value"]]
            for reference in shell["submodels"]
            if submodels_by_id[reference["keys"][0]["value"]]["idShort"] == "Nameplate"
        )
        element = next(
            (e for e in nameplate["submodelElements"] if e["idShort"] == id_short),
            None,
        )
        value = (element or {}).get("value")
        text: str | None
        if isinstance(value, str):
            # NOTE: A Property has a plain value rather than language strings.
            text = value
        else:
            text = next((s["text"] for s in value or [] if s["language"] == "en"), None)
        result.append((shell["idShort"], text or None))

    return result


def read_ground_truth(path: pathlib.Path) -> model.GroundTruth:
    """Read and validate the ground-truth labels.

    The file contains the computed field ``gt_ordinal_score``, which the data model
    does not accept as input, so we drop it.
    """
    if not path.is_file():
        raise common.AblationError(f"The file {path} does not exist")

    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        item.pop("gt_ordinal_score", None)

    try:
        return model.GroundTruth.model_validate(data)
    except pydantic.ValidationError as exception:
        raise common.AblationError(f"{path}: {exception}") from exception


def build(
    smc_environment: pathlib.Path,
    wago_environment: pathlib.Path,
    smc_ground_truth: pathlib.Path,
) -> list[common.DatasetRow]:
    """Label the products of both environments."""
    entries = {
        entry.product_id: entry for entry in read_ground_truth(smc_ground_truth).root
    }

    try:
        smc = label_smc.extract_products(read_environment(smc_environment))
        wago = extract_wago_nameplate_texts(
            read_environment(wago_environment), "ManufacturerProductDesignation"
        )
    except (KeyError, StopIteration) as exception:
        raise common.AblationError(
            f"Unexpected structure of an environment: {exception!r}"
        ) from exception

    if {pid for pid, _ in smc} != entries.keys():
        raise common.AblationError(
            f"The products of {smc_environment} differ from {smc_ground_truth}"
        )

    rows: list[common.DatasetRow] = []
    for product_id, designation in smc:
        entry = entries[product_id]
        if common.normalize_whitespace(designation) != common.normalize_whitespace(
            entry.product_designation
        ):
            raise common.AblationError(
                f"The designation of {product_id!r} differs between "
                f"{smc_environment} and {smc_ground_truth}"
            )
        rows.append(
            common.DatasetRow(
                product_id=product_id,
                designation=designation,
                gt_label=entry.gt_label,
                gt_ordinal=entry.gt_ordinal,
            )
        )

    for product_id, wago_designation in wago:
        if wago_designation is None:
            raise common.AblationError(f"{product_id!r} has no English designation")
        rows.append(
            common.DatasetRow(
                product_id=product_id,
                designation=wago_designation,
                gt_label=False,
                gt_ordinal=Ordinal.LOW,
            )
        )

    ids = [row.product_id for row in rows]
    if len(set(ids)) != len(ids):
        raise common.AblationError("Duplicate product IDs across the environments")

    return rows


def main(prog: str) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--smc_environment", default="data/smc/sample-smc.json")
    parser.add_argument("--wago_environment", default="data/wago/environment.json")
    parser.add_argument("--smc_ground_truth", default="data/smc/ground_truth.json")
    parser.add_argument("--output", default="data/xgboost_ablation/dataset.csv")
    args = parser.parse_args()

    try:
        rows = build(
            smc_environment=pathlib.Path(args.smc_environment),
            wago_environment=pathlib.Path(args.wago_environment),
            smc_ground_truth=pathlib.Path(args.smc_ground_truth),
        )
    except common.AblationError as exception:
        print(exception, file=sys.stderr)
        return 1

    common.write_dataset(pathlib.Path(args.output), rows)

    counter = collections.Counter(row.gt_ordinal for row in rows)
    print(
        f"Labelled {len(rows)} products to {args.output}: "
        + ", ".join(f"{level.value} {counter[level]}" for level in common.LEVELS)
    )
    return 0


def entry_point() -> int:
    """Provide an entry point for the console script."""
    return main(prog="pdm-at-scale-ablation-dataset")


if __name__ == "__main__":
    sys.exit(main(prog="build_dataset.py"))
