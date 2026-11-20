#!/usr/bin/env python
"""Make a sandbox look like the real files where the official generator is too clean.

    python scripts/degrade_sandbox.py sandboxes/unhcr sandboxes/unhcr_real

The official sandbox obeys every gate exactly, which rewards hard-coding gate
logic. The real schema descriptions say otherwise, so this
script injects the documented pathologies into an existing sandbox:

  1. Soft gates: each gated item gets an agreement rate drawn to match UNHCR's
     reported spread (median about 0.83, range 0.29-0.98). The disagreeing rows
     flip in both directions: asked -> NA_GATED, and skipped -> a drawn answer.
  2. Round coding (UNHCR only): in the multi-select families 6.4, 6.6, 6.11,
     7A.7 and 7B.1, round-1 rows (wave == "1") carry "0" where round 2 carries
     the sentinel.

  3. --panel F (optional): a share F of people become repeat interviews of another
     person: same known facts except `wave`, and each answer copied with probability
     PANEL_KEEP (otherwise left as it was). This mimics a panel re-contacted across rounds,
     to test that a nearest-neighbour member can exploit it. It does not say whether the
     real file is a panel.

The output directory has the same format as the input, so evaluate.py takes it
as it is. It is still synthetic: use it to choose *gate handling*, and nothing else.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SENTINEL = "NA_GATED"
PANEL_KEEP = 0.7
ROUND_FAMILIES = re.compile(r"^(s6_4|s6_6|s6_11|s7a_7|s7b_1)_")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel", type=float, default=0.0,
                        help="share of people turned into repeat interviews of another person")
    parser.add_argument("--no-gates", action="store_true", help="skip the soft-gate degradation (source already has it)")
    parser.add_argument("--add-wave", type=float, default=0.0,
                        help="add a GIVEN survey round `wave`: '1' with this probability, else '2' (UNHCR: 0.61)")
    parser.add_argument("--round-families", default=None,
                        help="regex of items whose round-1 skips are written as --skip-value (UNHCR round coding)")
    parser.add_argument("--skip-value", default="0")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src, out = Path(args.source), Path(args.out)
    schema = json.loads((src / "schema.json").read_text())
    frame = pd.read_parquet(src / "respondents.parquet")

    if args.add_wave:
        frame.insert(1, "wave", np.where(rng.random(len(frame)) < args.add_wave, "1", "2"))
        schema["items"] = {"wave": {"question": "Survey round.", "class": "GIVEN", "values": ["1", "2"], "gate": None},
                           **schema["items"]}
    families = re.compile(args.round_families) if args.round_families else ROUND_FAMILIES
    coded = 0
    if args.round_families:
        for name, rec in schema["items"].items():
            if rec["class"] == "PREDICT" and rec.get("gate") and families.match(name) and args.skip_value in rec["values"]:
                hit = (frame["wave"] == "1") & (frame[name] == SENTINEL)
                frame.loc[hit, name] = args.skip_value
                coded += 1
        print(f"round coding: {coded} items, round-1 skips written as {args.skip_value!r}")

    agreements = []
    for name, rec in schema["items"].items():
        if args.no_gates:
            break
        gate = rec.get("gate")
        if rec["class"] != "PREDICT" or not gate or gate["parent"] not in frame:
            continue
        agreement = float(np.clip(rng.beta(3.7, 1.0), 0.29, 0.98))   # median 0.5 ** (1 / 3.7) = 0.83
        agreements.append(agreement)
        flip = rng.random(len(frame)) > agreement
        asked = frame[name] != SENTINEL
        answered = frame.loc[asked, name]
        # Asked but marked skipped.
        frame.loc[flip & asked, name] = SENTINEL
        # Skipped but holding an answer, drawn from the item's answered distribution.
        target = flip & ~asked
        if target.any() and len(answered):
            frame.loc[target, name] = rng.choice(answered.to_numpy(), target.sum())
        if "wave" in frame and ROUND_FAMILIES.match(name) and "0" in rec["values"]:
            frame.loc[(frame["wave"] == "1") & (frame[name] == SENTINEL), name] = "0"

    if args.panel > 0:
        given = [k for k, r in schema["items"].items() if r["class"] == "GIVEN" and k in frame]
        predict = [k for k, r in schema["items"].items() if r["class"] == "PREDICT" and k in frame]
        n = len(frame)
        repeats = rng.choice(n, int(args.panel * n), replace=False)
        sources = rng.choice(np.setdiff1d(np.arange(n), repeats), len(repeats))
        for col in given:
            if col != "wave":
                frame.loc[repeats, col] = frame.loc[sources, col].to_numpy()
        if "wave" in frame:
            frame.loc[repeats, "wave"] = np.where(frame.loc[sources, "wave"].to_numpy() == "1", "2", "1")
        for col in predict:
            keep = rng.random(len(repeats)) < PANEL_KEEP
            frame.loc[repeats[keep], col] = frame.loc[sources[keep], col].to_numpy()

    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "respondents.parquet", index=False)
    (out / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=1))
    print(f"{src} -> {out}: {len(agreements)} gated items degraded, "
          f"median agreement {np.median(agreements):.2f}" if agreements else "no gated items")


if __name__ == "__main__":
    main()
