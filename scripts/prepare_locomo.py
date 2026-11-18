"""Seed LoCoMo conversations into memory and write probe labels.

python scripts/prepare_locomo.py            # -> collection locomo_v1, data/locomo/*.jsonl/.parquet
"""

from __future__ import annotations

import argparse

from amic.app import build_components
from amic.contracts.config import Settings
from amic.eval.datasets import edu, locomo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="locomo_v1")
    ap.add_argument("--per-conv", type=int, default=35)
    a = ap.parse_args()
    convs = locomo.load()
    c = build_components(Settings(chroma_collection=a.collection), llm=object())  # type: ignore[arg-type]
    n = locomo.seed(c, convs)
    ps = locomo.probes(convs, a.per_conv)
    edu.save_jsonl(locomo.PATH.parent / "probes.jsonl", ps)
    df = locomo.labels_frame(ps)
    df.to_parquet(locomo.PATH.parent / "labels.parquet", index=False)
    print(f"seeded {n} turns from {len(convs)} conversations; {len(ps)} probes")
    print(df.category.value_counts().to_dict())


if __name__ == "__main__":
    main()
