#!/usr/bin/env python
"""Low-loss recoding of the Arab Barometer proxy, closer to UNHCR's regime of near-constant answers.

    python scripts/make_arab_lowloss.py bench/rs41_arab_barometer_real bench/rs41_arab_lowloss

Every question that is neither gated nor the parent of a gated question becomes a yes/no question about one rare
answer ("Yes": the respondent gave that answer). The answer is fixed by rule: question n (schema order) aims at a
share of 1 %, 3 % or 6 % (n mod 3) and takes, among its options with at least 0.8 % of the Train answers, the one
whose share is closest to the aim in log ratio. Gated questions and their parents are kept, so the question count
and the routing are unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_SHARES = (0.01, 0.03, 0.06)
MIN_SHARE = 0.008


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    schema = json.loads((src / "schema.json").read_text())
    df = pd.read_parquet(src / "respondents.parquet")
    items = schema["items"]
    parents = {item["gate"]["parent"] for item in items.values() if item.get("gate")}
    free = [name for name, item in items.items()
            if item["class"] == "PREDICT" and not item.get("gate") and name not in parents]
    train = df["role"] == "TRAIN"
    for n, name in enumerate(free):
        shares = df.loc[train, name].value_counts(normalize=True)
        candidates = shares[shares >= MIN_SHARE]
        option = candidates.index[np.argmin(np.abs(np.log(candidates.values / TARGET_SHARES[n % 3])))]
        df[name] = np.where(df[name].isna(), None, np.where(df[name] == option, "Yes", "No")).astype(object)
        items[name]["values"] = ["No", "Yes"]
        items[name]["question"] = f"{items[name]['question']} [answer = {option}]"
    schema["dataset"]["description"] = schema["dataset"].get("description", "") + " [low-loss recoding]"
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(schema, indent=1))
    df.to_parquet(out / "respondents.parquet", index=False)
    print(f"{out.name}: {len(free)} of {len(items)} items recoded to yes/no")


if __name__ == "__main__":
    main()
