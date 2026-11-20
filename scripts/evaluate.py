#!/usr/bin/env python
"""Score a submission in-process on local datasets (sandboxes or the real-survey bench).

    python scripts/evaluate.py submission                      # the 3 sandboxes
    python scripts/evaluate.py submission --phase 2 --data sandboxes/unhcr
    python scripts/evaluate.py submission --data bench/afrobarometer bench/wgm_vaccine   # real-survey bench
    python scripts/evaluate.py submission --by-item unicef

A dataset is a directory holding respondents.parquet (with a `role` column) and
schema.json, the format make_sandbox.py writes.

Reuses the official starter-kit code for loading, masking, output validation,
flooring and scoring, so the numbers match `starter-kit/score.py`. Skill is measured
against the crowd marginal of the visible rows, R (the official reference since Oct 26):
0 = `baseline/marginal_counts`, 1 = perfect. Both references are saved with --save. It skips the
venv install and the network cut, which is what makes it fast; use
`scripts/package.py --full` for the faithful rehearsal. An illegal output
(wrong count, width, order, NaN...) raises here exactly as it would on the grader.

Sandbox answers are invented: a sandbox score says the code is legal and fast,
not how it will do on the leaderboard.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "starter-kit"
DEFAULT_DATA = [ROOT / "sandboxes" / name for name in ("unicef", "world_bank", "unhcr")]

sys.path.insert(0, str(KIT))
import make_sandbox  # noqa: E402
import score as kit  # noqa: E402


def load_submission(submission_dir):
    """Import <submission_dir>/main.py as a fresh module, as the grader does."""
    submission_dir = Path(submission_dir).resolve()
    spec = importlib.util.spec_from_file_location("submission_main", submission_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(submission_dir))
    return module


def staged_frame(schema, masked):
    """What predict() receives on the grader: a JSON round trip, None for blanks."""
    columns = ["respondent_id"] + make_sandbox.generated_items(schema)
    rows = [[None if pd.isna(v) else v for v in row]
            for row in masked[columns].to_numpy(dtype=object)]
    payload = json.loads(json.dumps({"columns": columns, "data": rows}))
    return pd.DataFrame(payload["data"], columns=payload["columns"])


def save_predictions(path, schema, vectors, truth, cells, uniform, reference):
    """Per-cell scored distributions, for paired comparisons and calibration (scripts/analyze.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {item: make_sandbox.options_for(schema, item) for item in {c[2] for c in cells}}
    np.savez_compressed(
        path,
        respondent=np.array([c[1] for c in cells], dtype=object),
        item=np.array([c[2] for c in cells], dtype=object),
        truth=np.array([options[c[2]].index(t) for c, t in zip(cells, truth)]),
        width=np.array([len(v) for v in vectors]),
        probs=np.concatenate(vectors), uniform=uniform, reference=reference)


def subsample(schema, respondents, frac, roles=("TRAIN",)):
    """Learning curves: keep a random share of the people in each role (fixed seed, so a smaller share is a
    subset of a larger one). TEST people are never dropped."""
    if frac >= 1.0:
        return schema, respondents
    split, drop = dict(schema["split"]), []
    for role in roles:
        rows = respondents.index[respondents["role"] == role]
        drop.append(pd.Index(rows).to_series().sample(frac=1 - frac, random_state=0).index)
        split[f"n_{role.lower()}"] = int(len(rows) - len(drop[-1]))
    return {**schema, "split": split}, respondents.drop(drop[0].append(drop[1:])).reset_index(drop=True)


def evaluate(module, data_dir, phase, config, train_frac=1.0, save=None, sequential=False, vis_frac=1.0):
    schema = make_sandbox.load_schema(str(Path(data_dir) / "schema.json"), config)
    respondents = kit.load_frames(str(data_dir), schema)
    schema, respondents = subsample(schema, respondents, train_frac)
    schema, respondents = subsample(schema, respondents, vis_frac, ("TRAIN", "DEV"))
    masked, cells, truth = kit.sample_rows(schema, respondents, phase)

    clock = getattr(module, "_CLOCK", None)
    if clock is not None and not sequential:  # each dataset gets one call's share of the run
        clock.update(start=time.time(), calls=0, seen=set())   # each dataset simulates the first call
    started = time.time()
    vectors = module.predict(staged_frame(schema, masked),
                             json.loads(json.dumps(schema, ensure_ascii=False)))
    seconds = time.time() - started

    kit.check(schema, vectors, cells)
    floor = config["scoring"]["floor"]
    scored = kit.floored(vectors, schema, cells, floor)
    reference = kit.marginal_reference(schema, masked, cells, truth, floor)
    result = kit.score(schema, config, scored, truth, cells, reference)
    uniform = kit.uniform_reference(schema)
    if save is not None:
        save_predictions(save, schema, scored, truth, cells, uniform, reference)
    _, crowd = kit.baselines(schema, truth, cells, floor)
    diag = getattr(module, "LAST_DIAGNOSTICS", {})
    if save is not None and "lab" in diag:     # every member's output (SIMBENCH_LAB), for analyze.py blend
        lab = diag.pop("lab")
        hidden = respondents[respondents[kit.ROLE_COLUMN] == kit.PHASE_ROLES[phase]["hidden"]]
        lab["Y_hid"] = np.column_stack([
            hidden[item].map({o: i for i, o in enumerate(make_sandbox.options_for(schema, item))}).to_numpy()
            for item in lab["targets"]])
        lab["uniform"], lab["reference"] = uniform, reference
        np.savez_compressed(save.with_name(save.stem + "_lab.npz"), **lab)
    return {"skill": result["skill"], "se": result["std_error"] / reference, "reference": reference,
            # submissions report their in-run skill against the uniform guess: same loss, put on R
            "cv_skill": 1.0 - (1.0 - diag.get("cv_skill", float("nan"))) * uniform / reference,
            "crowd_oracle": 1.0 + crowd / reference, "seconds": seconds,
            "by_item": result["by_item"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="directory holding main.py")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--data", nargs="+", type=Path, default=DEFAULT_DATA,
                        help="dataset directories (default: the three sandboxes)")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="keep only this share of TRAIN rows (learning curves)")
    parser.add_argument("--vis-frac", type=float, default=1.0,
                        help="keep only this share of TRAIN and DEV rows (learning curves in the final phase)")
    parser.add_argument("--save", action="store_true",
                        help="store per-cell predictions in results/<name>/<dataset>.npz")
    parser.add_argument("--name", help="results folder name (default: the submission's folder name)")
    parser.add_argument("--sequential", action="store_true",
                        help="one clock for all datasets, as on the grader (three calls in one run); "
                             "default: every dataset simulates the first call")
    parser.add_argument("--lab", action="store_true",
                        help="with --save: also store every member's out-of-fold and held-out output "
                             "(results/<name>/<dataset>_lab.npz) for `analyze.py blend`")
    parser.add_argument("--cache", metavar="DIR",
                        help="experiments (v46+): save every fitted member to DIR/<dataset>.pt, or reuse it if present, "
                             "so the submission's own combiner can be re-run on member subsets (SIMBENCH_DROP) or variants")
    parser.add_argument("--cache-out", metavar="DIR",
                        help="experiments (v47+, with SIMBENCH_AUGMENT=1): save the cached members plus the newly "
                             "fitted SIMBENCH_EXTRA ones to DIR/<dataset>.pt")
    parser.add_argument("--by-item", metavar="NAME",
                        help="also print the per-item log score for this dataset")
    args = parser.parse_args()

    args.data = [d for d in args.data if not d.name.startswith("_")]   # e.g. bench/_raw
    missing = [d for d in args.data if not (d / "respondents.parquet").exists()]
    if missing:
        sys.exit(f"no dataset at {missing[0]} (sandboxes: run scripts/make_sandboxes.sh)")
    config = make_sandbox.load_config(str(KIT / "config.yml"))
    if args.lab:
        os.environ["SIMBENCH_LAB"] = "1"
    import_started = time.time()
    module = load_submission(args.submission)
    import_seconds = time.time() - import_started

    skills, total = [], 0.0
    print(f"{'dataset':<16} {'skill':>7} {'±se':>6} {'cv':>7} {'crowd*':>7} {'seconds':>8} {'R':>7}")
    for data_dir in args.data:
        instrument = data_dir.name
        name = args.name or Path(args.submission).name
        save = (ROOT / "results" / name / f"{instrument}.npz") if args.save else None
        if args.cache:
            Path(args.cache).mkdir(parents=True, exist_ok=True)
            os.environ["SIMBENCH_CACHE_FILE"] = str(Path(args.cache) / f"{instrument}.pt")
        if args.cache_out:
            Path(args.cache_out).mkdir(parents=True, exist_ok=True)
            os.environ["SIMBENCH_CACHE_OUT"] = str(Path(args.cache_out) / f"{instrument}.pt")
        r = evaluate(module, data_dir, args.phase, config, args.train_frac, save, args.sequential,
                     args.vis_frac)
        skills.append(r["skill"])
        total += r["seconds"]
        print(f"{instrument:<16} {r['skill']:7.4f} {r['se']:6.4f} "
              f"{r['cv_skill']:7.4f} {r['crowd_oracle']:7.4f} {r['seconds']:8.1f} {r['reference']:7.4f}")
        if args.by_item == instrument:
            print(r["by_item"].sort_values("mean").to_string())
    budget = config["phases"][args.phase]["timeout_seconds"]
    print(f"{'MEAN':<16} {sum(skills) / len(skills):7.4f} {'':6} {'':7} {'':7} {total:8.1f}"
          f"  (grader budget {budget}s; import {import_seconds:.1f}s)")
    print("skill = 1 - loss / R, R = loss of the visible rows' crowd marginal (nats/cell; 0 = marginal_counts).")
    print("cv = the submission's own in-run cross-validated skill on visible rows (if it reports one);")
    print("     it should track `skill`. crowd* = crowd marginal fitted on the hidden rows (an oracle)")


if __name__ == "__main__":
    main()
