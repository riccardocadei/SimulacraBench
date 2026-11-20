#!/usr/bin/env python
"""From the crowd marginal to the full pipeline: the four steps of the report's Fig. 1b, on one dataset.

    python experiments/ladder.py world_bank bench/afrowb76
    python experiments/ladder.py unicef bench/wgm_unicef --arms marginal offshelf --out results/ladder

Every arm runs the scored code (submission/main.py) in the final phase, with members switched off:

  marginal   the crowd marginal alone (each question's answer shares among the visible respondents)
  offshelf   TabPFN-3.5 fast with its published weights, in context on all visible respondents, alone
  finetuned  the fine-tuned TabPFN member alone
  full       the unmodified pipeline

"Alone" means the pool puts all its weight on that member and the geometric pool is skipped; the post-processing
(verified skip rules, floor compensation) is the pipeline's own in every arm. The first argument says which
competition survey's settings to use (unicef, world_bank or unhcr); the dataset is a directory in the competition
format (scripts/build_bench.py writes the public surveys this way). The run-time budget is lifted, so fine-tuning
stops at its cap of passes instead of following the clock. Needs a GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as ev  # noqa: E402

ARMS = ("marginal", "offshelf", "finetuned", "full")
OTHER_MEMBERS = ("fit_mlp", "fit_linear", "fit_latent", "fit_tabfm", "fit_tabpfn_family")
PATCHED = OTHER_MEMBERS + ("fit_tabpfn", "linear_mixture", "family_mixture", "fit_log_pool", "LOUD_FINAL")


class SwitchedOff(RuntimeError):
    """Raised by a member that an arm removes; the pipeline skips a member that fails."""


def switched_off(*args, **kwargs):
    raise SwitchedOff()


def all_weight_on_last(P, strength=None):
    """Stands in for the linear pool: all weight on the last member (the crowd marginal is always first)."""
    weights = np.zeros(P.shape[-1])
    weights[-1] = 1.0
    return weights, np.tile(weights, (P.shape[0], 1))


def offshelf_tabpfn(M):
    """TabPFN-3.5 fast as published: one in-context fit per question on all visible respondents."""
    def member(d):
        import tabpfn.checkpoint
        from tabpfn import TabPFNClassifier
        tabpfn.checkpoint.Checkpoint.is_safetensors = property(lambda self: True)
        M.align_class_orders(False)
        model = TabPFNClassifier(model_path=M.model_file(M.TABPFN_REPO, M.TABPFN_FAST), device=str(M.DEVICE),
                                 n_estimators=M.TABPFN_ESTIMATORS[2 if d.phase2 else 1],
                                 categorical_features_indices=d.nominal_cols or None, random_state=M.SEED)
        X_vis, X_hid = d.codes[d.vis_idx], d.codes[d.hid_idx]
        p_vis = {k: np.zeros((len(d.vis_idx), len(idx), k)) for k, idx in d.buckets.items()}
        p_hid = {k: np.zeros((len(d.hid_idx), len(idx), k)) for k, idx in d.buckets.items()}
        for k, idx in d.buckets.items():
            for jj, j in enumerate(idx):
                y = d.Y[:, j]
                for fold in range(M.K_FOLDS):      # out-of-fold slots: crowd shares (the pool is forced anyway)
                    held = d.folds == fold
                    p_vis[k][held, jj] = M.marginal(y[~held], k)
                if np.unique(y).size < 2:
                    p_hid[k][:, jj] = M.marginal(y, k)
                    continue
                model.fit(X_vis, y)
                p_hid[k][:, jj][:, model.classes_] = model.predict_proba(X_hid)
        d.finetuned = False
        as_tensors = lambda p: {k: torch.as_tensor(v, dtype=torch.float32, device=M.DEVICE) for k, v in p.items()}
        return M.as_member_output(as_tensors(p_vis), as_tensors(p_hid))
    return member


def set_arm(M, original, arm):
    for name, value in original.items():
        setattr(M, name, value)
    if arm == "full":
        return
    for name in OTHER_MEMBERS:
        setattr(M, name, switched_off)
    M.linear_mixture = all_weight_on_last
    M.family_mixture = lambda P, family, strength, n_vis: all_weight_on_last(P)[1]
    M.fit_log_pool = switched_off
    if arm == "marginal":
        M.fit_tabpfn, M.LOUD_FINAL = switched_off, False      # LOUD_FINAL would stop a run without TabPFN
    elif arm == "offshelf":
        M.fit_tabpfn, M.LOUD_FINAL = offshelf_tabpfn(M), False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("survey", choices=["unicef", "world_bank", "unhcr"], help="whose settings to use")
    parser.add_argument("data", type=Path, help="dataset directory (respondents.parquet + schema.json)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--out", type=Path, help="directory for one JSON file per arm")
    args = parser.parse_args()

    M = ev.load_submission(args.submission)
    M.instrument = lambda schema: args.survey          # public surveys are not recognised by name
    M.TOTAL_SECONDS = {1: 1e5, 2: 1e5}                 # no clock: fine-tuning runs to its cap of passes
    M.DEBUG = False                                    # a switched-off member is skipped, not raised
    original = {name: getattr(M, name) for name in PATCHED}
    config = ev.make_sandbox.load_config(str(ev.KIT / "config.yml"))
    expected = {"marginal": ["marginal"], "offshelf": ["marginal", "tabpfn"], "finetuned": ["marginal", "tabpfn"]}

    for arm in args.arms:
        set_arm(M, original, arm)
        M.FT_STATS.clear()
        M.LAST_DIAGNOSTICS.clear()
        started = time.time()
        result = ev.evaluate(M, args.data, 2, config)
        members = list((M.LAST_DIAGNOSTICS.get("weights") or {}).keys())
        if arm in expected and members != expected[arm]:
            sys.exit(f"{arm}: members {members}, expected {expected[arm]} (a member failed)")
        if arm in ("finetuned", "full") and not M.LAST_DIAGNOSTICS.get("finetuned"):
            sys.exit(f"{arm}: TabPFN was not fine-tuned")
        record = dict(arm=arm, survey=args.survey, data=args.data.name, skill=result["skill"], se=result["se"],
                      members=len(members), passes=M.FT_STATS.get("passes"), seconds=round(time.time() - started))
        print(f"{arm:10s} skill {record['skill']:.4f} (se {record['se']:.4f})  members {record['members']:2d}  "
              f"fine-tuning passes {record['passes']}  {record['seconds']} s", flush=True)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{args.survey}_{args.data.name}_{arm}.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
