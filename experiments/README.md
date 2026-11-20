# Experiments

Scripts that reproduce the report's evidence with the scored code (`submission/main.py`). They run on public
surveys converted to the competition format, never on competition answers.

| Script | Report | What it does |
|---|---|---|
| `ladder.py` | Table 1 | Skill of the crowd marginal, TabPFN-3.5 off the shelf, the fine-tuned TabPFN alone, and the full pipeline |

## Data

The public surveys come from their publishers (links and licences in the main README). Build them once, after
`scripts/setup.sh`; place the Arab Barometer file at `bench/_raw/arab_barometer_w8.sav` first:

```bash
python scripts/build_bench.py                       # Afrobarometer (downloaded) and Arab Barometer
python scripts/build_bench.py --only wgm_unicef     # Wellcome Global Monitor, UNICEF-like questions
```

Each split of the report holds out 20 % of the respondents as the scored set (seeds 41, 43 and 45):

```bash
python scripts/resplit.py bench/wgm_unicef bench/rs41_wgm_unicef --seed 41 --dev 0.06 --test 0.2
python scripts/resplit.py bench/arab_barometer_real bench/rs41_arab_barometer_real --seed 41 --dev 0.06 --test 0.2
python scripts/resplit.py bench/afrobarometer bench/rs41_afrobarometer --seed 41 --dev 0.06 --test 0.2
python scripts/make_afrowb76.py bench/rs41_afrobarometer bench/rs41_afrowb76 1041   # World Bank's shape: 76 questions
```

## Table 1: from the crowd marginal to the full pipeline

```bash
python experiments/ladder.py unicef     bench/rs41_wgm_unicef
python experiments/ladder.py unhcr      bench/rs41_arab_barometer_real
python experiments/ladder.py world_bank bench/rs41_afrowb76
```

The first argument selects the settings of one competition survey; the second is its stand-in dataset. Each arm is
a full final-phase run with members switched off (see the script's docstring). A GPU is needed. The UNICEF
stand-in takes about 9 minutes on an H100; the other two are larger (roughly half an hour to two hours) and were
not rerun with this script.

Check, UNICEF stand-in, split 41, on an H100:

| Arm | This script | Report's run of the same split |
|---|---|---|
| `marginal` | 0.0002 | 0.0002 |
| `offshelf` | 0.0963 | 0.0958 |
| `finetuned` | 0.0967 | 0.0967 |
| `full` | 0.0971 | 0.0971 |

Table 1 of the report is the mean of three splits (and two fine-tuning seeds), measured on an earlier build of
the pipeline, before the World Bank changes; single runs differ from it by the split-to-split spread (about
0.002 on this dataset) and are not bit-reproducible.
