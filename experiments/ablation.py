#!/usr/bin/env python
"""Skill lost when one part of the pipeline is removed: the report's Table 1, on one dataset.

    python experiments/ablation.py unicef bench/rs41_wgm_unicef
    python experiments/ablation.py world_bank bench/rs41_afrowb76 --arms networks geometric_pool

The scored code (submission/main.py) is fitted once in the final phase, and every member's output is kept in
memory. Each arm then repeats the run with one part removed:

  tabpfn_member  the fine-tuned TabPFN gets no weight in either pool (its students stay)
  networks       the multi-task networks get no weight in either pool
  regressions    the multinomial regressions get no weight in either pool
  latent_classes the latent-class model gets no weight in either pool
  second_model   the pooled-question TabPFN or TabFM gets no weight in either pool
  crowd_marginal the crowd marginal gets no weight in either pool
  geometric_pool the geometric pool is skipped; the linear pool stands alone
  optimism       no optimism correction of the fine-tuned member when the pool weights are fitted
  floor          no compensation of the grader's probability floor
  distillation   the students are refitted without the fine-tuned TabPFN as teacher
  finetuning     TabPFN is used with its published weights; every member is refitted

All arms but the last two reuse the stored members, so their difference from the full run is exact. The last two
refit members, and two fits differ slightly from run to run (GPU kernels are not deterministic). The run-time budget is
lifted, so fine-tuning stops at its cap of passes. Needs a GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate as ev  # noqa: E402

ARMS = ("tabpfn_member", "networks", "regressions", "latent_classes", "second_model", "crowd_marginal",
        "geometric_pool", "optimism", "floor", "distillation", "finetuning")
DROPS = {"tabpfn_member": ("tabpfn",), "networks": ("networks",), "regressions": ("regressions",),
         "latent_classes": ("latent",), "second_model": ("pooled_tabpfn", "tabfm"), "crowd_marginal": ("marginal",)}
MEMBER_FITS = {"fit_tabpfn": "tabpfn", "fit_mlp": "networks", "fit_tabfm": "tabfm", "fit_linear": "regressions",
               "fit_latent": "latent", "fit_tabpfn_family": "pooled_tabpfn"}
STUDENTS = ("networks", "regressions")
POOLS = ("linear_mixture", "family_mixture", "fit_log_pool")
SETTINGS = ("FT_DOSE_C", "FT_DOSE_BY", "FLOOR", "DISTILL_LAMBDA", "DISTILL_LAMBDA_BY", "LOUD_FINAL")


class Pipeline:
    """The scored module with its member fits stored, and switches for the arms."""

    def __init__(self, submission, survey):
        self.M = M = ev.load_submission(submission)
        M.instrument = lambda schema: survey           # public surveys are not recognised by name
        M.TOTAL_SECONDS = {1: 1e5, 2: 1e5}             # no clock: fine-tuning runs to its cap of passes
        M.DEBUG = False
        self.original = {name: getattr(M, name) for name in tuple(MEMBER_FITS) + POOLS + SETTINGS}
        self.store = {}                                # (kind, arguments) -> member output, kept across arms
        self.order = []                                # kinds of the members fitted in this run, in pool order
        self.dropped = ()                              # kinds that get no pool weight in this arm
        self.refit = ()                                # kinds fitted afresh in this arm
        self.plain_tabpfn = False
        self.no_pool = False
        for name, kind in MEMBER_FITS.items():
            setattr(M, name, self.stored(name, kind))
        M.linear_mixture, M.family_mixture, M.fit_log_pool = self.linear_mixture, self.family_mixture, self.log_pool

    def stored(self, name, kind):
        fit = self.original[name]

        def member(d, *args):
            key = (kind, self.plain_tabpfn, tuple(a for a in args if not isinstance(a, dict)))
            if key in self.store and kind not in self.refit:
                output, finetuned, stats = self.store[key]
            else:
                if kind == "tabpfn" and self.plain_tabpfn:
                    d.ft_seconds = 0.0                 # no fine-tuning: the published weights, in context
                output = fit(d, *args)
                finetuned, stats = getattr(d, "finetuned", False), dict(self.M.FT_STATS)
                if kind not in self.refit:
                    self.store[key] = (output, finetuned, stats)
            if kind == "tabpfn":
                d.finetuned = finetuned
                self.M.FT_STATS.clear()
                self.M.FT_STATS.update(stats)
            self.order.append(kind)
            return output
        return member

    def kept(self, n_members):
        """Indices of the members that keep their pool weight (member 0 is the crowd marginal)."""
        kinds = ["marginal"] + self.order
        if len(kinds) != n_members or not self.dropped:
            return None                                # already reduced, or nothing to drop
        return [i for i, kind in enumerate(kinds) if kind not in self.dropped]

    def linear_mixture(self, P, strength):
        keep = self.kept(P.shape[-1])
        if keep is None:
            return self.original["linear_mixture"](P, strength)
        w, W = self.original["linear_mixture"](P[..., keep], strength)
        w_all, W_all = np.zeros(P.shape[-1]), np.zeros((P.shape[0], P.shape[-1]))
        w_all[keep], W_all[:, keep] = w, W
        return w_all, W_all

    def family_mixture(self, P, family, strength, n_vis):
        keep = self.kept(P.shape[-1])
        if keep is None:
            return self.original["family_mixture"](P, family, strength, n_vis)
        W_all = np.zeros((P.shape[0], P.shape[-1]))
        W_all[:, keep] = self.original["family_mixture"](P[..., keep], family, strength, n_vis)
        return W_all

    def log_pool(self, d, oof_buckets, hid_buckets, w_init):
        if self.no_pool:
            raise RuntimeError("geometric pool switched off")     # the pipeline keeps the linear pool
        keep = self.kept(len(oof_buckets))
        if keep is None:
            return self.original["fit_log_pool"](d, oof_buckets, hid_buckets, w_init)
        return self.original["fit_log_pool"](d, [oof_buckets[i] for i in keep], [hid_buckets[i] for i in keep],
                                             np.asarray(w_init)[keep])

    def set_arm(self, arm):
        M = self.M
        for name in SETTINGS:
            setattr(M, name, self.original[name])
        self.order, self.dropped, self.refit, self.plain_tabpfn, self.no_pool = [], (), (), False, False
        if arm in DROPS:
            self.dropped = DROPS[arm]
        elif arm == "geometric_pool":
            self.no_pool = True
        elif arm == "optimism":
            M.FT_DOSE_C, M.FT_DOSE_BY = 0.0, {}
        elif arm == "floor":
            M.FLOOR = 0.0
        elif arm == "distillation":
            M.DISTILL_LAMBDA, M.DISTILL_LAMBDA_BY, self.refit = 0.0, {}, STUDENTS
        elif arm == "finetuning":
            self.plain_tabpfn, self.refit, M.LOUD_FINAL = True, STUDENTS, False   # LOUD_FINAL stops untuned runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("survey", choices=["unicef", "world_bank", "unhcr"], help="whose settings to use")
    parser.add_argument("data", type=Path, help="dataset directory (respondents.parquet + schema.json)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--out", type=Path, help="directory for the JSON record")
    args = parser.parse_args()

    pipeline = Pipeline(args.submission, args.survey)
    config = ev.make_sandbox.load_config(str(ev.KIT / "config.yml"))
    records = []
    for arm in ["full"] + args.arms:
        pipeline.set_arm(arm)
        started = time.time()
        result = ev.evaluate(pipeline.M, args.data, 2, config)
        diagnostics = pipeline.M.LAST_DIAGNOSTICS
        weights = diagnostics.get("weights") or {}
        if arm == "full":
            full = result["skill"]
            if not diagnostics.get("finetuned"):
                sys.exit("full: TabPFN was not fine-tuned")
        if len(weights) != 1 + len(pipeline.order):
            sys.exit(f"{arm}: {len(weights)} members in the pool, {1 + len(pipeline.order)} fitted (a member failed)")
        if (arm == "geometric_pool") == ("pool_share" in diagnostics):
            sys.exit(f"{arm}: the geometric pool {'still ran' if arm == 'geometric_pool' else 'did not run'}")
        records.append(dict(arm=arm, survey=args.survey, data=args.data.name, skill=result["skill"],
                            lost_1e4=round((full - result["skill"]) * 1e4, 2), finetuned=bool(diagnostics.get("finetuned")),
                            seconds=round(time.time() - started)))
        print(f"{arm:15s} skill {result['skill']:.5f}   lost {records[-1]['lost_1e4']:+6.2f} e-4   "
              f"{records[-1]['seconds']} s", flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"ablation_{args.survey}_{args.data.name}.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
