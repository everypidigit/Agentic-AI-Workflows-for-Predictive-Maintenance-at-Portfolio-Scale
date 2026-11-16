"""Define the data model shared by the experiments."""

import enum

import pydantic


class Ordinal(str, enum.Enum):
    """Rate how strongly a product supports PdM, as in the paper."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


#: Map the ordinal levels to equidistant scores in [0, 1].
ORDINAL_SCORE: dict[Ordinal, float] = {
    Ordinal.NONE: 0.0,
    Ordinal.LOW: 1.0 / 3.0,
    Ordinal.MODERATE: 2.0 / 3.0,
    Ordinal.HIGH: 1.0,
}

#: Ordinal levels which count as a positive binary label.
POSITIVE_ORDINALS = frozenset([Ordinal.MODERATE, Ordinal.HIGH])


class GroundTruthEntry(pydantic.BaseModel):
    """Represent the ground-truth labels of a single product.

    The binary label is ``True`` if the product is a potential and meaningful
    PdM candidate, i.e., in at least one realistic use, it wears out or
    degrades in a way which shows measurable signs early enough to predict an
    upcoming failure.

    The ordinal label rates how strongly the product supports PdM. It must be
    consistent with the binary label: ``MODERATE`` and ``HIGH`` are positive,
    ``NONE`` and ``LOW`` negative. The ordinal score maps the ordinal label to
    [0, 1] for comparisons on a numeric scale.

    The explanation states in one or two sentences whether and why the product
    is suitable for PdM, as the qualitative assessment in the paper.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    product_id: str = pydantic.Field(min_length=1)
    product_designation: str = pydantic.Field(min_length=1)
    gt_label: bool
    gt_ordinal: Ordinal
    gt_explanation: str = pydantic.Field(min_length=1)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def gt_ordinal_score(self) -> float:
        """Map the ordinal label to [0, 1]."""
        return round(ORDINAL_SCORE[self.gt_ordinal], 4)

    @pydantic.model_validator(mode="after")
    def _check_consistent_labels(self) -> "GroundTruthEntry":
        if self.gt_label != (self.gt_ordinal in POSITIVE_ORDINALS):
            raise ValueError(
                f"The binary label {self.gt_label} of {self.product_id!r} "
                f"is inconsistent with its ordinal label {self.gt_ordinal.value}"
            )
        return self


class GroundTruth(pydantic.RootModel[list[GroundTruthEntry]]):
    """Represent the ground-truth labels of a dataset."""

    @pydantic.model_validator(mode="after")
    def _check_unique_product_ids(self) -> "GroundTruth":
        seen: set[str] = set()
        for entry in self.root:
            if entry.product_id in seen:
                raise ValueError(f"Duplicate product ID: {entry.product_id!r}")
            seen.add(entry.product_id)
        return self
