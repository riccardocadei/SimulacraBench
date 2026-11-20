# SimulacraBench entry: a fine-tuned tabular foundation model in a stacked ensemble

Code of team `riccardocadei` for [SimulacraBench](https://simulacrabench.org/) (NeurIPS 2026 competition track).
The task: for respondents held out of a survey (UNHCR, UNICEF, World Bank), predict a distribution over every
answer from a few facts that are always known about them and the full answers of several thousand other respondents.

**Test-phase result** (leaderboard, two decimals shown): skill 0.49 overall; UNICEF 0.34, World Bank 0.24, UNHCR 0.88.

The method and its evidence are described in the technical report submitted to the competition.

## Method in one paragraph

Each survey is handled in one `predict()` call, inside the organizers' run (3,600 s on one GPU for the three
surveys). TabPFN-3.5, a pretrained tabular foundation model, is fine-tuned at run time on all questions of the survey.
Multi-task networks and multinomial regressions are distilled from it; a latent-class model, a pooled-question TabPFN
(or TabFM on UNICEF) and the crowd marginal join them. A linear and a geometric pool, fitted per question on
out-of-fold predictions, combine the 12-14 members. Two rules finish: skip rules that the visible rows confirm, and
compensation of the grader's probability floor. A time planner keeps the run inside the budget.

## Repository layout

| Path | Content |
|---|---|
| `submission/main.py` | The scored submission, unmodified (single file; its docstring describes every step) |
| `submission/requirements.txt`, `models.txt` | Pinned packages and the model the grader prefetches (TabFM) |
| `submission/NOTICE`, `submission/weights/LICENSE-*` | Licences of the shipped weights |
| `SHA256SUMS` | sha256 of every file in the uploaded zip |
| `scripts/setup.sh` | Clones the organizers' starter kit at the pinned commit and writes practice data |
| `scripts/fetch_weights.sh` | Downloads the pretrained weights and checks them (uses `make_fp16.py`, `weights_hash.py`) |
| `scripts/package.py`, `verify_zip.py` | Build and validate the submission zip; check its files against `SHA256SUMS` |
| `scripts/evaluate.py` | Scores a submission locally with the starter kit's own scorer |
| `scripts/build_bench.py`, `resplit.py`, `make_afrowb76.py` | Convert public surveys into the competition format (method selection only) |
| `experiments/` | Scripts that reproduce the report's evidence with the scored code (see its README) |
| `environment.lock.txt` | The Python 3.13 environment used for development and timing |

## Reproduce the submission zip

```bash
conda create -n simbench python=3.13 -y && conda activate simbench
pip install -r environment.lock.txt

bash scripts/setup.sh                              # 1. starter kit (pinned) + practice data
HF_TOKEN=... bash scripts/fetch_weights.sh         # 2. weights into submission/weights/, sha256 checked
python scripts/package.py submission               # 3. dist/submission.zip, validated by the starter kit

python scripts/verify_zip.py dist/submission.zip   # 4. every file equals the scored zip (SHA256SUMS)
```

Notes:

- TabPFN-3.5 is gated on Hugging Face: accept Prior Labs' licence on its model page before step 2. Step 2 needs a
  GPU for a load check of the 16-bit copy of the 219M checkpoint, which is built locally.
- The zip's own checksum is not reproducible (zip files store file times); the files inside are, hence `SHA256SUMS`.
  The uploaded zip had md5 `ec18518ed04fb4b8376d41c948d84943`.
- TabFM is not shipped: the grader fetches it at the revision pinned in `models.txt`.
- `weights/tabicl-classifier-v2-20260212.ckpt` is in the zip but the final submission does not use it.

## Run it locally

```bash
python scripts/evaluate.py submission                       # development phase on the three practice surveys
python scripts/evaluate.py submission --phase 2 --sequential --data sandboxes/unhcr sandboxes/world_bank sandboxes/unicef
```

The practice data have invented answers, so these runs test the code and its timing, not the method. The final
phase (`--phase 2`) takes about 3,200 s on an H100 and at most 13.2 GB of RAM and 35 GB of GPU memory.

Runs are not bit-reproducible: all seeds are fixed, but the number of fine-tuning passes follows the clock and GPU
kernels are not deterministic. Set `SIMBENCH_FT_EPOCHS` to pin the passes.

## Public surveys used for method selection

No competition answers ever left the organizers' run. Every design choice was selected on three public surveys
converted to the competition format. They are not copied here; each is available from its publisher:

| Survey | Stands in for | Where to get it | Licence |
|---|---|---|---|
| Afrobarometer Round 9, merged file | World Bank | [afrobarometer.org/data/merged-data](https://www.afrobarometer.org/data/merged-data/) | free with attribution |
| Arab Barometer Wave VIII | UNHCR | [arabbarometer.org/survey-data/data-downloads](https://www.arabbarometer.org/survey-data/data-downloads/) | free for research with citation |
| Wellcome Global Monitor 2020 | UNICEF | [wellcome.org/reports/wellcome-global-monitor-covid-19/2020](https://wellcome.org/reports/wellcome-global-monitor-covid-19/2020) | CC BY-NC-ND 4.0 |

`scripts/build_bench.py` downloads the Afrobarometer and Wellcome files itself and converts all three. The Arab
Barometer file sits behind a short form: download the Wave VIII SPSS file by hand and save it as
`bench/_raw/arab_barometer_w8.sav` before running the script. `experiments/README.md` gives the commands for the
splits used in the report.

## About `submission/main.py`

The file is byte-identical to the one that was scored, so it was not edited for release. Two things follow:

- Its first line still says "v102", the version in which the docstring was last rewritten; the scored build is
  internally called v106.
- Comments carry short tags such as `H47` or `U35`. They are the ids of the experiments that justified a setting;
  the report's appendix describes each adopted and each rejected change.

## Licence

Code: MIT (`LICENSE`). Pretrained weights are under their own licences (`submission/NOTICE`): TabPFN-3.5
Non-Commercial, TabICL BSD-3-Clause, TabFM Non-Commercial. No survey data are included.
