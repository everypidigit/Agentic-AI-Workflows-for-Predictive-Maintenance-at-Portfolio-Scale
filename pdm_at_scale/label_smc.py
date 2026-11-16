"""Label the SMC products as PdM candidates by their product series.

The SMC sample is the positive case of the experiment, and the WAGO sample is
the negative one.

We follow the target of the paper: a product is a potential and meaningful PdM
candidate if, in at least one realistic use, it wears out or degrades in a way
which shows measurable signs early enough to predict an upcoming failure. The
OEM does not know the application at the time of the screening, so the
criticality, the downtime costs and the price of the product are deliberately
*not* part of the target, and neither is whether a sensor is already built in
(the paper assumes that system-level measurements are available). The mere
possibility of a failure is not enough, as almost any product would then
qualify.

The ordinal labels follow one rubric:

* ``HIGH``: a progressive degradation is the dominant failure mode, its
  precursors are measurable with common sensing, and they appear early enough
  to plan the maintenance before the functional failure.
* ``MODERATE``: a plausible progressive degradation with measurable
  precursors, but the lead time is uncertain or the maintenance is mostly
  condition-based servicing (e.g., cleaning) rather than preventing a failure.
* ``LOW``: a degradation mechanism exists, but its precursors are hardly
  measurable or appear too late to act on.
* ``NONE``: no credible measurable precursor of the failure modes beyond the
  generic ageing of any product.

We label the product series rather than the individual products, since all the
products of a series share the same degradation mechanisms; size, stroke and
price do not change them. The rules are applied in order, and the first
matching rule wins. A product without a matching rule is an error, so that
a new series in the sample is not labelled silently.
"""

import argparse
import collections
import dataclasses
import json
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from pdm_at_scale import model
from pdm_at_scale.model import Ordinal


@dataclasses.dataclass(frozen=True)
class Rule:
    """Label the products whose ID matches the pattern."""

    product_id_pattern: re.Pattern[str]
    gt_ordinal: Ordinal
    rationale: str

    def matches(self, product_id: str) -> bool:
        """Check whether the rule applies to the product."""
        return self.product_id_pattern.match(product_id) is not None


_CYLINDER_PRECURSORS = (
    "which shows as rising leakage, longer stroke times and higher air "
    "consumption well before the cylinder stalls or misses its end position."
)

RULES: Sequence[Rule] = (
    Rule(
        product_id_pattern=re.compile(r"^IZE11"),
        gt_ordinal=Ordinal.NONE,
        rationale=(
            "Monitor for electrostatic sensors (IZE11x), i.e., a display and "
            "switching electronics without moving parts or consumables. Apart "
            "from the generic ageing of any electronics, it has no degradation "
            "with measurable precursors."
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^IZ[ST]4"),
        gt_ordinal=Ordinal.MODERATE,
        rationale=(
            "Corona-discharge ionizer (IZS40-42, IZT40-42): the emitter needles "
            "get contaminated and erode progressively, which shows as a drifting "
            "ion balance and a longer discharge time before the static "
            "neutralization fails. The usual action is condition-based cleaning "
            "or replacing the emitters rather than preventing a breakdown."
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^MH"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Pneumatic air gripper: the piston seals and the finger guides wear "
            "progressively, which shows as rising leakage, slower opening and "
            "closing, and a falling gripping force well before the gripper "
            "fails to hold the workpiece."
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^C(D)?UJ"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Mini free-mount pneumatic cylinder (CUJ): the piston and rod seals "
            "wear progressively, and the return spring of single-acting variants "
            "fatigues, " + _CYLINDER_PRECURSORS
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^C(D)?U\d"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Single-acting free-mount pneumatic cylinder (CU): the piston and rod "
            "seals wear and the return spring fatigues progressively, "
            + _CYLINDER_PRECURSORS
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^C(D)?UW"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Double-acting double-rod pneumatic cylinder (CUW): the piston and "
            "both rod seals wear progressively, " + _CYLINDER_PRECURSORS
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^C(D)?QM"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Double-acting compact pneumatic cylinder with guide rod (CQM): the "
            "piston and rod seals and the guide wear progressively, "
            + _CYLINDER_PRECURSORS
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^R(D)?Q"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Double-acting compact pneumatic cylinder with air cushion (RQ): the "
            "piston, rod and cushion seals wear progressively, which shows as "
            "rising leakage, longer stroke times and harder end-of-stroke "
            "impacts well before the cylinder stalls or gets damaged."
        ),
    ),
    Rule(
        product_id_pattern=re.compile(r"^MGP"),
        gt_ordinal=Ordinal.HIGH,
        rationale=(
            "Compact guide cylinder (MGP): the piston and rod seals and the "
            "guide bushings wear progressively, which shows as rising leakage, "
            "longer stroke times and a growing play of the guide well before "
            "the cylinder stalls or loses its positioning accuracy."
        ),
    ),
)


def find_rule(product_id: str) -> Rule | None:
    """Find the first rule which applies to the product."""
    return next((rule for rule in RULES if rule.matches(product_id)), None)


def extract_products(environment: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Extract the product IDs and designations from the Digital Nameplates."""
    submodels_by_id = {
        submodel["id"]: submodel for submodel in environment["submodels"]
    }

    result: list[tuple[str, str]] = []
    for shell in environment["assetAdministrationShells"]:
        product_id: str = next(
            specific_asset_id["value"]
            for specific_asset_id in shell["assetInformation"]["specificAssetIds"]
            if specific_asset_id["name"] == "manufacturerPartId"
        )

        nameplate = submodels_by_id[
            f"urn:smart-assets.smc.com:sm:nameplate:{product_id}"
        ]

        designation_element = next(
            element
            for element in nameplate["submodelElements"]
            if element["idShort"] == "ManufacturerProductDesignation"
        )
        designation: str = next(
            lang_string["text"]
            for lang_string in designation_element["value"]
            if lang_string["language"] == "en"
        )

        result.append((product_id, designation))

    return result


def main(prog: str) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--environment",
        help="Path to the SMC AAS environment as JSON",
        default="data/smc/sample-smc.json",
    )
    parser.add_argument(
        "--output",
        help="Path to the JSON file with the ground-truth labels",
        default="data/smc/ground_truth.json",
    )
    args = parser.parse_args()

    environment_path = pathlib.Path(args.environment)
    output_path = pathlib.Path(args.output)

    environment = json.loads(environment_path.read_text(encoding="utf-8"))

    entries: list[model.GroundTruthEntry] = []
    errors: list[str] = []
    for product_id, designation in extract_products(environment):
        rule = find_rule(product_id)
        if rule is None:
            errors.append(f"No rule matches the product {product_id!r}")
            continue

        entries.append(
            model.GroundTruthEntry(
                product_id=product_id,
                product_designation=designation,
                gt_label=rule.gt_ordinal in model.POSITIVE_ORDINALS,
                gt_ordinal=rule.gt_ordinal,
                gt_explanation=rule.rationale,
            )
        )

    if len(errors) > 0:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    ground_truth = model.GroundTruth(entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        ground_truth.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    counter = collections.Counter(entry.gt_ordinal.value for entry in entries)
    positives = sum(1 for entry in entries if entry.gt_label)
    print(
        f"Labelled {len(entries)} products to {output_path}: "
        f"{positives} positive, {len(entries) - positives} negative; "
        + ", ".join(f"{ordinal.value}: {counter[ordinal.value]}" for ordinal in Ordinal)
    )

    return 0


def entry_point() -> int:
    """Provide an entry point for the console script."""
    return main(prog="pdm-at-scale-label-smc")


if __name__ == "__main__":
    sys.exit(main(prog="label_smc.py"))
