#!/usr/bin/env python
"""Draws the report's figure "Priors against data" from the saved curves.

    python experiments/priors_figure.py                      # writes priors_vs_data.pdf
    python experiments/priors_figure.py --out figure.png

experiments/data/priors_vs_data.json holds, per survey and number of respondents with answers, the skill of TabPFN
used in context and what each prior adds to it (gain and its bootstrap standard error over respondents):

  group_ceiling    the most any group-level prior can add: the true answer shares of each country x gender x age
                   cell, computed on the whole file (no standard error)
  answer_prior     a prior we could build: a country table written by Claude (Wellcome Global Monitor), or a
                   TabPFN pre-tuned on 33 other Afrobarometer countries (Afrobarometer)
  structure_prior  orderings of the facts, and each region replaced by its coordinates

The curves were measured once, on one split, with an earlier build of the pipeline; this script only redraws them.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "priors_vs_data.json"
SERIES = (("group_ceiling", "most any group-level prior can add (true answer shares per country x gender x age)",
           "#1f77d4", "--"),
          ("structure_prior", "prior on the structure of the facts (orderings, region centroids)", "#e8601c", "-"),
          ("answer_prior", "answer prior we built (Claude country table; TabPFN pre-tuned on 33 other countries)",
           "#1aa36a", "-"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("priors_vs_data.pdf"))
    args = parser.parse_args()
    curves = json.loads(DATA.read_text())

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, (survey, rows) in zip(axes.flat, curves.items()):
        sizes = [row["respondents"] for row in rows]
        for key, label, color, style in SERIES:
            points = [(row["respondents"], row[key]) for row in rows if key in row]
            if not points:
                continue
            x = [n for n, _ in points]
            gain = [value if key == "group_ceiling" else value[0] for _, value in points]
            error = [0 if key == "group_ceiling" else value[1] for _, value in points]
            ax.errorbar(x, gain, yerr=error, color=color, linestyle=style, lw=1.6, marker="o", ms=3.5,
                        elinewidth=0.8, label=label)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xticks(sizes, [f"{n / 1000:.1f}k" if n >= 1000 else str(n) for n in sizes], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(survey, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("people with answers (log scale)", fontsize=8, y=0.14)
    fig.supylabel("skill added over data only", fontsize=8)
    handles = {label: handle for ax in axes.flat for handle, label in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=1, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0.01, 0.16, 1, 1))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
