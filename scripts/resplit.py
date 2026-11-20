#!/usr/bin/env python
"""Copy a dataset with new roles: a fresh random split (outer confirmation of a selected change) or one
group held out (mechanics test for unseen groups).

    python scripts/resplit.py bench/afrobarometer bench/rs1_afrobarometer --seed 1 --dev 0.2
    python scripts/resplit.py sandboxes/unicef_real bench/unicef_unseen_zambia --column Country --value Zambia

--seed: every row is TRAIN or DEV at random (DEV share --dev), so the development phase scores a split
no earlier experiment has seen; with --dev 0.2 it holds about 3x as many people as the bench's DEV (6%).
--test: also hold out this share as TEST (rows TRAIN, DEV, TEST), so the final phase (visible TRAIN + DEV,
scored on TEST) can be confirmed on a split no earlier experiment has seen.
--column/--value: every person of that group in DEV, everyone else in TRAIN.
Rows are written TRAIN first, then DEV, as the organizers deliver them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("source", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--seed", type=int)
parser.add_argument("--dev", type=float, default=0.2, help="DEV share of a random split")
parser.add_argument("--test", type=float, default=0.0, help="TEST share of a random split (final-phase confirmation)")
parser.add_argument("--column")
parser.add_argument("--value")
args = parser.parse_args()

frame = pd.read_parquet(args.source / "respondents.parquet")
if args.seed is not None:
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(frame))
    n_dev, n_test = round(args.dev * len(frame)), max(1, round(args.test * len(frame)))   # the kit needs >= 1 TEST row
    held = np.zeros(len(frame), bool)
    held[order[:n_dev]] = True
    test = np.zeros(len(frame), bool)
    test[order[n_dev:n_dev + n_test]] = True
    what = f"random split, seed {args.seed}"
else:
    held = (frame[args.column].astype(str) == args.value).to_numpy()
    test = np.zeros(len(frame), bool)
    what = f"{args.column}={args.value}"
train = ~held & ~test
frame["role"] = np.select([held, test], ["DEV", "TEST"], "TRAIN")
frame = pd.concat([frame[train], frame[held], frame[test]], ignore_index=True)   # rows run TRAIN, DEV, TEST
schema = json.loads((args.source / "schema.json").read_text())
schema["split"] = {"n_train": int(train.sum()), "n_dev": int(held.sum()), "n_test": int(test.sum())}
args.out.mkdir(parents=True, exist_ok=True)
frame.to_parquet(args.out / "respondents.parquet", index=False)
(args.out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1))
print(f"{args.out}: {what}; {int(train.sum())} TRAIN, {int(held.sum())} DEV, {int(test.sum())} TEST")
