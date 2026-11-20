#!/usr/bin/env python
"""Skip-heavy copy of the Arab Barometer proxy: ten questions gated on a rare answer, as in the practice UNHCR file.

    python scripts/make_arab_skipheavy.py bench/arab_barometer_real bench/rs41_arab_barometer_real bench/rs41_arab_skipheavy

The practice UNHCR file has ten questions asked only of the 0.1-0.4 % of respondents who gave one answer to a
hidden question. The same pattern is imposed by rule. The parent is the question (neither gated nor a parent) of the
full file with an option whose share is closest to 0.43 %; the children are the first ten other such questions, in
schema order, with 3 to 10 options. Respondents who did not give the parent's rare answer get "not asked" on the
children; the others keep their answers. Nothing else changes. The parent is a hidden question, so no skip rule can
be verified on the facts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARENT_SHARE = 0.0043
N_CHILDREN = 10


def free_questions(items):
    parents = {item["gate"]["parent"] for item in items.values() if item.get("gate")}
    return [name for name, item in items.items()
            if item["class"] == "PREDICT" and not item.get("gate") and name not in parents]


def main():
    full, src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    full_items = json.loads((full / "schema.json").read_text())["items"]
    full_df = pd.read_parquet(full / "respondents.parquet")
    free = free_questions(full_items)
    _, parent, option = min((abs(np.log(share / PARENT_SHARE)), name, option) for name in free
                            for option, share in full_df[name].value_counts(normalize=True).items())
    children = [name for name in free if name != parent and 3 <= len(full_items[name]["values"]) <= 10][:N_CHILDREN]

    schema = json.loads((src / "schema.json").read_text())
    df = pd.read_parquet(src / "respondents.parquet")
    items, not_asked = schema["items"], schema["gated_value"]
    asked = (df[parent] == option).to_numpy()
    for child in children:
        df[child] = np.where(asked, df[child], not_asked).astype(object)
        items[child]["gate"] = {"parent": parent, "observed_if": [option]}
        items[child]["question"] += f" [asked only if {parent} = {option}]"
    schema["dataset"]["description"] += f" [skip-heavy copy: {N_CHILDREN} items gated on {parent} = {option}]"
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False))
    df.to_parquet(out / "respondents.parquet", index=False)
    print(f"{out.name}: {N_CHILDREN} questions gated on {parent} = {option!r}; {int(asked.sum())} of {len(df)} asked")


if __name__ == "__main__":
    main()
