"""Embed the product texts with the BAAI embedding model, for the XGBoost ablation.

We use bge-large-en-v1.5 as in our earlier work on semantic matching (Heppner et
al., 2025). The text of a product is ``<product ID>: <designation>``, which is what
the pipelines read. The model revision is resolved and recorded, so that the
embeddings can be reproduced; this needs access to Hugging Face.
"""

import argparse
import json
import pathlib
import sys

import huggingface_hub
import numpy as np
import sentence_transformers

from pdm_at_scale.xgboost_ablation import common

#: Embedding model on Hugging Face.
MODEL = "BAAI/bge-large-en-v1.5"


def main(prog: str) -> int:
    """Execute the main routine."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--dataset", default="data/xgboost_ablation/dataset.csv")
    parser.add_argument("--output", default="data/xgboost_ablation/embeddings.npz")
    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset)
    output = pathlib.Path(args.output)

    try:
        rows = common.read_dataset(dataset_path)
    except common.AblationError as exception:
        print(exception, file=sys.stderr)
        return 1

    revision = huggingface_hub.HfApi().model_info(MODEL).sha
    encoder = sentence_transformers.SentenceTransformer(MODEL, revision=revision)
    embeddings = np.asarray(
        encoder.encode(
            [f"{row.product_id}: {row.designation}" for row in rows],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        product_ids=np.array([row.product_id for row in rows], dtype=np.str_),
        embeddings=embeddings,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": revision,
                "text": "{product_id}: {designation}",
                "sentence-transformers": sentence_transformers.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Embedded {len(rows)} products ({embeddings.shape[1]} dimensions) to {output}"
    )
    return 0


def entry_point() -> int:
    """Provide an entry point for the console script."""
    return main(prog="pdm-at-scale-ablation-embed")


if __name__ == "__main__":
    sys.exit(main(prog="embed.py"))
