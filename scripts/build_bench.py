#!/usr/bin/env python
"""Build the real-survey bench: public surveys in the competition's dataset format.

    python scripts/build_bench.py                     # download (once) + build all
    python scripts/build_bench.py --only wgm_vaccine  # one dataset
    python scripts/evaluate.py submission --data bench/afrobarometer bench/wgm_vaccine bench/arab_barometer

The official sandbox invents its dependencies, so it cannot tell model families
apart. These datasets carry real survey structure instead: real marginals, real
demographic gradients, real routing. Each one is written exactly as
starter-kit/make_sandbox.py writes a sandbox (respondents.parquet with a `role`
column + schema.json), and is checked with the kit's own loader.

Sources (raw files land in bench/_raw/, everything under bench/ is git-ignored;
never commit or share it -- the licences do not allow redistribution):

  afrobarometer  Afrobarometer Round 9 merged file (2021-23). Free for use with
                 attribution; downloaded without login. Proxies World Bank
                 (South Africa, Lesotho, Nigeria) and UNICEF (Kenya, Mali, Zambia).
  wgm_vaccine    Wellcome Global Monitor 2020 (Gallup), CC BY-NC-ND 4.0.
                 Proxies UNICEF: trust in science/health workers, vaccine intent.
  arab_barometer Arab Barometer Wave VIII (2023-24). Free for research with citation;
                 no redistribution. Proxies UNHCR via the host populations of Iraq,
                 Jordan and Lebanon (Egypt is not in Wave VIII). Never downloaded: the
                 site asks for name, email and country behind a reCAPTCHA, so the .sav
                 is placed at AB_PATH by hand, and the build skips it when absent.

Conversion rules, shared by every source (see `finish`): all values are
strings; genuine non-response is an ordinary level; routing skips are
NA_GATED on gated items and nothing else; levels with fewer than MIN_COUNT
respondents merge into "Other"; near-constant PREDICT items are dropped; roles
are a random 80/6/14 split. Each source is also written in variants with other role
assignments (file order, interview date; see `finish`), and Arab Barometer gets two
degraded copies (soft gates, panel; see `degrade_arab_barometer`).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "starter-kit"
BENCH = ROOT / "bench"
RAW = BENCH / "_raw"

sys.path.insert(0, str(KIT))
import make_sandbox  # noqa: E402
import score as kit  # noqa: E402

URLS = {
    "afrobarometer_r9.sav": "https://www.afrobarometer.org/wp-content/uploads/2025/06/"
                            "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav",
    "wgm2020.sav": "https://wellcome.org/sites/default/files/2021-11/wgm-full-wave2-public-file.sav",
}

GATED = "NA_GATED"
OTHER = "Other"
NOT_RECORDED = "Not recorded"
MIN_COUNT = 20           # levels rarer than this merge into OTHER
MAX_MODAL_SHARE = 0.98   # PREDICT items whose top level exceeds this are dropped
SHARES = {"n_dev": 0.06, "n_test": 0.14}  # the rest is TRAIN
SEED = 0
VERSION = "bench-1"
MAX_MISSING_DATES = 0.05  # date variants are skipped when more interview dates are missing


# ------------------------------------------------------------------ helpers --

def download(filename):
    """Fetch a raw file into bench/_raw once; later runs reuse it."""
    path = RAW / filename
    if path.exists():
        return path
    RAW.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    request = urllib.request.Request(URLS[filename], headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            print(f"  downloading {filename} ...", flush=True)
            with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as fh:
                shutil.copyfileobj(response, fh, length=1 << 20)
            partial.rename(path)
            return path
        except OSError as exc:
            print(f"  attempt {attempt + 1} failed: {exc}")
            time.sleep(5)
    sys.exit(f"could not download {URLS[filename]}")


def read_sav(path, columns):
    """Codes, value labels and variable labels, keeping user-missing codes (DK, refused...)."""
    frame, meta = pyreadstat.read_sav(str(path), usecols=columns, user_missing=True)
    return frame, meta.variable_value_labels, meta.column_names_to_labels


def labelled(codes, labels):
    """Codes -> label strings, in code order. NaN becomes an explicit level."""
    def name(code):
        if pd.isna(code):
            return NOT_RECORDED
        return " ".join(labels.get(code, f"Code {code:g}").split())
    values = codes.map(name)
    order = [name(c) for c in sorted(set(codes.dropna()))]
    if values.eq(NOT_RECORDED).any():
        order.append(NOT_RECORDED)
    return values, list(dict.fromkeys(order))


def band_age(age, edges, top_code):
    """Integer ages -> closed bands; codes above `top_code` are non-response."""
    labels = [f"{lo}-{hi - 1}" for lo, hi in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    bins = pd.cut(age, edges + [top_code + 1], right=False, labels=labels)
    values = bins.astype(object).where(bins.notna(), "Don't know / refused")
    return values, labels + ["Don't know / refused"]


def question_text(label):
    """Drop the 'Q40b. ' numbering prefix from an SPSS variable label."""
    return re.sub(r"^Q\d+\w*\.\s*", "", label or "").strip()


# ------------------------------------------------------------------ builder --

def split_counts(n):
    """Role counts for n respondents in the SHARES proportions."""
    split = {key: int(round(share * n)) for key, share in SHARES.items()}
    return {"n_train": n - sum(split.values()), **split}


def date_order(dates, groups):
    """Rows ordered TRAIN, DEV, TEST, each in interview-date order, plus the role counts.

    Within each group the earliest rows are TRAIN, the next DEV and the latest TEST,
    in the SHARES proportions; the overall split is the sum over groups. Ties are broken
    at random (SEED), and a missing date sorts first, so it lands in TRAIN.
    """
    key = np.random.default_rng(SEED).random(len(dates))
    when = dates.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
    when[dates.isna().to_numpy()] = -np.inf
    role = np.empty(len(dates), dtype=int)
    groups = np.asarray(groups)
    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        rows = rows[np.lexsort((key[rows], when[rows]))]
        counts = split_counts(len(rows))
        role[rows] = np.repeat([0, 1, 2], [counts["n_train"], counts["n_dev"], counts["n_test"]])
    split = dict(zip(("n_train", "n_dev", "n_test"), np.bincount(role, minlength=3).tolist()))
    return np.lexsort((key, when, role)), split


def date_variants(name, data, dates, country):
    """`datewithin_<name>` and `dateglobal_<name>`: (variant, frame, split, order) tuples.

    datewithin: in each country the earliest 80% are TRAIN, the next 6% DEV, the last 14%
    TEST -- later interviews in the same countries. dateglobal: the same by global date,
    so "later" may also mean another country. Rows run TRAIN, DEV, TEST, each by date, and
    ids follow that order, as a delivery numbered in collection order would.
    """
    if dates is None:
        return []
    missing = dates.isna().mean()
    if missing > MAX_MISSING_DATES:
        print(f"{name}: date variants skipped, {missing:.0%} of interview dates missing")
        return []
    countries = data[country].to_numpy()
    plans = []
    if dates.groupby(countries).nunique().max() > 1:
        plans.append((f"datewithin_{name}", countries))
    else:
        print(f"datewithin_{name}: skipped, interview dates do not vary within any country")
    plans.append((f"dateglobal_{name}", np.zeros(len(dates))))
    variants = []
    for variant, groups in plans:
        order, split = date_order(dates, groups)
        frame = make_sandbox.assign_roles({"split": split},
                                          pd.DataFrame(data).iloc[order].reset_index(drop=True).astype(object))
        variants.append((variant, frame, split, order))
    return variants


def describe_date_split(variant, frame, dates, country):
    """Print the date range of each role and the country mix of DEV and TEST."""
    print(f"  {variant}:")
    for role in ("TRAIN", "DEV", "TEST"):
        mask = (frame["role"] == role).to_numpy()
        span = dates[mask]
        mix = ""
        if role != "TRAIN":
            shares = frame.loc[mask, country].value_counts(normalize=True)
            mix = ", ".join(f"{c} {s:.0%}" for c, s in shares.items())
        print(f"    {role:5s} n={mask.sum():5d}  {span.min():%Y-%m-%d} .. {span.max():%Y-%m-%d}  {mix}")


def merge_sparse(values, order, protected=()):
    """Fold levels with fewer than MIN_COUNT respondents into OTHER."""
    counts = values.value_counts()
    sparse = {v for v in order if counts.get(v, 0) < MIN_COUNT and v not in protected}
    if len(sparse) <= 1 and OTHER not in order:  # renaming one level to OTHER gains nothing
        sparse = {v for v in sparse if counts.get(v, 0) == 0}
    kept = [v for v in order if v not in sparse and counts.get(v, 0) > 0]
    mapping = {v: (v if v not in sparse else OTHER) for v in order}
    merged = values.map(lambda v: mapping.get(v, v))
    if (merged == OTHER).any() and OTHER not in kept:
        kept.append(OTHER)
    return merged, kept, mapping


def finish(name, columns, orders, given, predict, questions, gates, description,
           dates=None, country=None, all_variants=True):
    """Apply the shared conversion rules, then write and validate bench/<name>/.

    columns    {item: Series of label strings; GATED where the routing skipped}
    orders     {item: canonical level order (without GATED)}
    gates      {child: (parent, [observed_if labels])}
    dates      interview dates (datetime Series, NaT if missing), row-aligned with columns;
               they only order the date variants and are never written as an item
    country    the GIVEN item the datewithin variant splits within
    all_variants  False: write bench/<name>/ only (no id-, file- or date-order variants)
    """
    parents = {parent for parent, _ in gates.values()}
    data, items, dropped = {}, {}, []
    for item in given + predict:
        values = columns[item]
        if GATED in set(values) and item not in gates:
            raise ValueError(f"{name}: {item} holds {GATED} but has no gate")
        merged, levels, mapping = merge_sparse(values, orders[item])
        if item in predict and item not in parents:
            modal = merged.value_counts(normalize=True).iloc[0]
            if modal > MAX_MODAL_SHARE or len(merged.unique()) < 2:
                dropped.append(item)
                continue
        data[item] = merged
        items[item] = {"question": questions[item],
                       "class": "GIVEN" if item in given else "PREDICT",
                       "values": levels, "gate": None, "_mapping": mapping}
    for child, (parent, observed_if) in gates.items():
        if child in items:
            mapping = items[parent]["_mapping"]
            items[child]["gate"] = {"parent": parent, "observed_if": list(dict.fromkeys(
                mapping.get(v, v) for v in observed_if if mapping.get(v, v) in items[parent]["values"]))}
    for rec in items.values():
        del rec["_mapping"]

    n = len(next(iter(data.values())))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    frame = pd.DataFrame(data).iloc[perm].reset_index(drop=True).astype(object)
    split = split_counts(n)
    frame = make_sandbox.assign_roles({"split": split}, frame)

    # Variants. `<name>`: rows shuffled, roles random, ids carry no information.
    # `idorder_<name>`: same rows and roles, ids in the source file's order.
    # `fileorder_<name>`: rows NOT shuffled, so roles follow the source file's order (the
    # kit's assign_roles makes the first rows TRAIN and the last TEST). This mimics a
    # delivery built without shuffling: held-out people come from the end of the file,
    # e.g. the last countries or regions -- a deliberate distribution-shift stress test.
    # `datewithin_<name>`, `dateglobal_<name>`: roles by interview date (see date_variants).
    ordered = make_sandbox.assign_roles({"split": split}, pd.DataFrame(data).astype(object))
    sequential = np.arange(1, n + 1)
    variants = [(name, frame, sequential, split), (f"idorder_{name}", frame, perm + 1, split),
                (f"fileorder_{name}", ordered, sequential, split)]
    if not all_variants:
        variants = variants[:1]
    dated = date_variants(name, data, dates, country) if all_variants else []
    variants += [(variant, frame, sequential, split) for variant, frame, split, _ in dated]
    for variant, frame, ids, split in variants:
        schema = {"dataset": {"n_rows": n, "version": VERSION, "description": description},
                  "items": items, "split": split, "gated_value": GATED}
        frame = frame.copy()
        frame.insert(0, "respondent_id", [f"R{i:06d}" for i in ids])
        out = BENCH / variant
        out.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out / "respondents.parquet", index=False)
        with open(out / "schema.json", "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        # The kit's own checks: schema validity, declared values, role counts.
        config = make_sandbox.load_config(str(KIT / "config.yml"))
        kit.load_frames(str(out), make_sandbox.load_schema(str(out / "schema.json"), config))
    if frame.isna().any().any():
        raise ValueError(f"{name}: null cells")
    n_given = sum(rec["class"] == "GIVEN" for rec in items.values())
    n_gated = sum(rec["gate"] is not None for rec in items.values())
    widths = [len(rec["values"]) + bool(rec["gate"]) for rec in items.values() if rec["class"] == "PREDICT"]
    print(f"{name}: {n} rows, {n_given} GIVEN, {len(widths)} PREDICT ({n_gated} gated), "
          f"options mean {np.mean(widths):.1f} max {max(widths)}; dropped {len(dropped)} near-constant")
    for variant, frame, _, order in dated:
        describe_date_split(variant, frame, dates.iloc[order].reset_index(drop=True), country)


def convert(frame, value_labels, var_labels, items, gates, skip_pattern):
    """Label every item; on gated items, cells matching `skip_pattern` become GATED."""
    columns, orders, questions = {}, {}, {}
    for item in items:
        values, order = labelled(frame[item], value_labels.get(item, {}))
        if item in gates:
            skipped = values.str.contains(skip_pattern, case=False, regex=True)
            values = values.where(~skipped, GATED)
            order = [v for v in order if not re.search(skip_pattern, v, re.I)]
        columns[item], orders[item] = values, order
        questions[item] = question_text(var_labels.get(item))
    return columns, orders, questions


# ------------------------------------------------------------------ sources --

AFRO_COUNTRIES = ["Kenya", "Lesotho", "Mali", "Nigeria", "South Africa", "Zambia"]
AFRO_GIVEN = ["COUNTRY", "REGION", "URBRUR", "age_band", "Q100", "Q94", "Q95", "Q93A",
              "Q101", "Q2", "Q103"]
# Questionnaire modules kept as PREDICT (Q15-Q36 on democracy/institutions, Q46-Q50,
# Q74-Q88 on media, foreign influence and identity are left out to stay near 150).
AFRO_PREDICT = (
    ["Q3", "Q4A", "Q4B", "Q5A", "Q5B", "Q6A", "Q6B", "Q6C", "Q6D", "Q6E", "Q7A", "Q7B", "Q8",
     "Q9A", "Q9B", "Q9C", "Q10A", "Q10B", "Q10C", "Q11A", "Q11B", "Q11C", "Q11D", "Q11E", "Q13",
     "Q23", "Q30", "Q31"]
    + [f"Q37{c}" for c in "ABCDEFGHIJK"] + [f"Q38{c}" for c in "ABCDEFGHIJK"] + ["Q39A", "Q39B"]
    + [f"Q40{c}" for c in "ABCD"] + [f"Q41{c}" for c in "ABCDEFGH"] + [f"Q42{c}" for c in "ABCD"]
    + [f"Q43{c}" for c in "ABCDE"] + ["Q45PT1", "Q45PT2", "Q45PT3"]
    + ["Q51A", "Q51B", "Q52A", "Q52B", "Q53A", "Q54", "Q55A", "Q55B", "Q55C"]
    + ["Q57A", "Q57B", "Q58A", "Q58B", "Q58C", "Q59", "Q60", "Q61A", "Q61B", "Q61C", "Q62",
       "Q64A", "Q64B", "Q65A", "Q65B"]
    + ["Q66A", "Q66B", "Q67A", "Q67B", "Q68A", "Q68B", "Q69", "Q70A", "Q70B", "Q70C", "Q70D",
       "Q71", "Q72A", "Q72B", "Q72D"]
    + ["Q86A", "Q86B", "Q86C", "Q86D", "Q86E", "Q86F", "Q87A", "Q87B", "Q87C", "Q87D", "Q89A"]
    + [f"Q90{c}" for c in "ABCDEFGHI"] + ["Q91A", "Q91B", "Q91C", "Q92A", "Q92B", "Q93B", "Q93C"]
)
_YES = ["Yes"]
AFRO_GATES = {
    **{f"Q40{c}": ("Q40A", _YES) for c in "BCD"},
    **{f"Q41{c}": ("Q41A", _YES) for c in "BCDEFGH"},
    **{f"Q42{c}": ("Q42A", _YES) for c in "BCD"},
    **{f"Q43{c}": ("Q43A", _YES) for c in "BC"},
    "Q43E": ("Q43D", ["Once or twice", "A few times", "Often"]),
    "Q58B": ("Q58A", ["No", "Don't know"]),
    "Q58C": ("Q58B", ["Very unlikely", "Somewhat unlikely"]),
    **{c: ("Q67A", _YES) for c in ["Q67B", "Q68A", "Q68B", "Q69", "Q70A", "Q70B", "Q70C", "Q70D"]},
    "Q90G": ("Q90F", ["Yes (personally owns)"]),
    "Q92B": ("Q92A", _YES),
}


def build_afrobarometer(rest=False):
    """rest: the other 33 countries of the same round, as a pretraining corpus; never scored."""
    path = download("afrobarometer_r9.sav")
    raw = [c for c in AFRO_GIVEN + AFRO_PREDICT if c != "age_band"] + ["Q1"]
    frame, value_labels, var_labels = read_sav(path, raw + ["DATEINTR"])
    country = frame["COUNTRY"].map(value_labels["COUNTRY"])
    frame = frame[country.isin(AFRO_COUNTRIES) != rest].reset_index(drop=True)
    dates = pd.to_datetime(frame["DATEINTR"], errors="coerce")  # date of interview

    items = [c for c in AFRO_GIVEN + AFRO_PREDICT if c != "age_band"]
    columns, orders, questions = convert(frame, value_labels, var_labels, items, AFRO_GATES,
                                         r"^(?:not applicable|no contact)$")
    columns["age_band"], orders["age_band"] = band_age(frame["Q1"], [18, 25, 35, 45, 55, 65], 120)
    questions["age_band"] = "Age of respondent (banded)"
    questions["REGION"] = "Province or region"

    finish("afrobarometer_rest" if rest else "afrobarometer", columns, orders, AFRO_GIVEN, AFRO_PREDICT,
           questions, AFRO_GATES,
           "Afrobarometer Round 9 (2021-2023, face-to-face, adults 18+), merged public file, "
           + (f"every country except {', '.join(AFRO_COUNTRIES)} (pretraining corpus). " if rest else
              f"restricted to {', '.join(AFRO_COUNTRIES)}. ") + "Real-survey proxy for the World Bank "
           "(South Africa, Lesotho, Nigeria) and UNICEF (Kenya, Mali, Nigeria, Zambia) instruments: "
           "cheap demographics (country, region, urban/rural, age band, gender, education, "
           "religion, employment, race, home language, dwelling type) as GIVEN; lived poverty, "
           "participation, trust, corruption, public-service contact, most important problems, "
           "COVID-19 vaccination, climate change, social trust, assets and services as PREDICT. "
           "Gates follow the questionnaire routing (e.g. service-quality items asked only after "
           "contact with the service); NA_GATED marks the file's 'Not applicable'/'No contact' "
           "codes. Levels with fewer than 20 respondents are merged into 'Other'. "
           "Licence: Afrobarometer data are free to use with attribution; do not redistribute.",
           dates=dates, country="COUNTRY", all_variants=not rest)


WGM_COUNTRIES = ["Bangladesh", "Brazil", "Egypt", "El Salvador", "Iraq", "Kenya", "Mali",
                 "Nepal", "Nigeria", "Zambia"]
WGM_GIVEN = ["COUNTRYNEW", "age_band", "Gender", "Education", "Household_Income",
             "Subjective_Income", "EMP_2010"]
WGM_PREDICT = (
    ["W1", "W2", "W3", "W4"] + [f"W5{c}" for c in "ABCDEFG"] + ["W6", "W7A", "W7B", "W7C",
     "W8", "W9", "W10", "W11A", "W11B", "MH2A", "MH2B", "W13", "W14", "W15"]
    + [f"W15_1{c}" for c in "ABCDE"] + ["W15_2A", "W15_2B", "MH1"]
    + [f"MH3{c}" for c in "ABCD"] + ["MH4A", "MH4B", "MH5", "MH6", "MH7A", "MH7C"]
    + [f"MH8{c}" for c in "ABCDEFGH"] + [f"MH9{c}" for c in "ABCDEFGH"]
    + ["W27", "W28", "W29", "W30", "WP21757", "WP21758", "WP21759", "WP21760", "WP21761",
       "WP21768"]
)
WGM_GATES = {
    "W14": ("W13", _YES), "W15": ("W13", _YES),
    "W28": ("W27", _YES), "W29": ("W27", _YES),
    "MH7C": ("MH7A", _YES),
    **{f"MH8{c}": ("MH7A", _YES) for c in "ABCDEFGH"},
    **{f"MH9{c}": (f"MH8{c}", _YES) for c in "ABCDEFGH"},
}


def build_wgm_vaccine():
    path = download("wgm2020.sav")
    raw = [c for c in WGM_GIVEN + WGM_PREDICT if c != "age_band"] + ["Age"]
    frame, value_labels, var_labels = read_sav(path, raw + ["FIELD_DATE"])
    frame = frame[frame["COUNTRYNEW"].isin(WGM_COUNTRIES)].reset_index(drop=True)
    # "Study Completion Date": one month per country, so only the global variant is built.
    dates = pd.to_datetime(frame["FIELD_DATE"], errors="coerce")
    value_labels["Education"][99.0] = "DK/Refused"  # the file leaves this code unlabelled

    items = [c for c in WGM_GIVEN + WGM_PREDICT if c not in ("age_band", "COUNTRYNEW")]
    # Routing skips are blank in this file; ungated blanks (e.g. W30 in two
    # countries) stay an ordinary NOT_RECORDED level.
    columns, orders, questions = convert(frame, value_labels, var_labels, items, WGM_GATES,
                                         f"^{NOT_RECORDED}$")
    columns["COUNTRYNEW"], orders["COUNTRYNEW"] = frame["COUNTRYNEW"], sorted(WGM_COUNTRIES)
    questions["COUNTRYNEW"] = "Country"
    columns["age_band"], orders["age_band"] = band_age(frame["Age"], [15, 25, 35, 45, 55, 65], 99)
    questions["age_band"] = "Age of respondent (banded)"
    for item in questions:
        questions[item] = re.sub(r"\s*\[Note:.*\]$", "", questions[item])

    finish("wgm_vaccine", columns, orders, WGM_GIVEN, WGM_PREDICT, questions, WGM_GATES,
           "Wellcome Global Monitor 2020 (Gallup World Poll, adults 15+), public file, restricted "
           f"to {', '.join(WGM_COUNTRIES)} (the UNICEF instrument's countries present in the "
           "file, plus Bangladesh and El Salvador standing in for Pakistan and Guatemala). "
           "Real-survey proxy for the UNICEF Faith & Immunisation instrument: 7 GIVEN "
           "demographics (country, age band, gender, education, income quintile, subjective "
           "income, employment) and PREDICT items on knowledge of and trust in science, "
           "doctors, government and religion, COVID-19 impact and willingness to be vaccinated, "
           "climate change, mental health and social media. Gates follow the routing "
           "(climate items after 'heard of climate change', social-media items after use, "
           "anxiety/depression coping items after having felt anxious). Levels with fewer than "
           "20 respondents are merged into 'Other'. Licence: CC BY-NC-ND 4.0; do not redistribute.",
           dates=dates, country="COUNTRYNEW")


AB_PATH = RAW / "arab_barometer_w8.sav"   # downloaded by hand (form + reCAPTCHA) and renamed
AB_COUNTRIES = ["Iraq", "Jordan", "Lebanon"]  # Egypt, the fourth UNHCR host, is not in Wave VIII
AB_GIVEN = ["COUNTRY", "Q1", "Q13", "age_band", "Q1002", "Q1003", "Q1010", "Q1005", "hh_size",
            "Q1012", "Q1012B2", "Q1001F", "Q1001G"]
# Multi-select batteries, one 0/1 dummy per option.
AB_EMIGRATE_WHY = [f"Q104A_2_{i}" for i in range(1, 8)]
AB_EMIGRATE_WHERE = [f"Q104B_{i}" for i in (1, 2, 3, 5, 13, 14, 15, 16, 17, 18, 19, 22)]
AB_FOOD_COPING = [f"Q130_{i}" for i in range(1, 6)]
AB_DISCRIMINATION = [f"Q873_{i}" for i in (1, 2, 3, 4, 5, 6, 7, 9, 11, 12)]
AB_SOCIAL_MEDIA = [f"Q412A_{i}" for i in (1, 3, 4, 5, 11)]
# Questionnaire modules kept as PREDICT: items asked of everyone in all three countries,
# plus routed follow-ups. Split-ballot items (half the sample each) and country-specific
# modules (Gaza, foreign powers, democracy batteries) are left out.
AB_PREDICT = (
    # economy, household finances, food insecurity
    ["Q2061A", "Q262", "Q101", "Q102", "Q114", "Q118_1", "Q1016", "Q116", "Q1025", "Q1017",
     "Q867A", "Q867B", "Q112_2", "Q112B", "Q129A_1", "Q129A_2"] + AB_FOOD_COPING
    # emigration intentions, attitudes to migrants
    + ["Q104"] + AB_EMIGRATE_WHY + AB_EMIGRATE_WHERE
    + ["Q104C", "Q907", "Q916_1", "Q916_2", "Q916_3", "Q886"]
    # trust, safety, government performance, politics
    + ["Q103", "Q201A_1", "Q201A_2", "Q201A_3", "Q201A_7", "Q201A_31C",
       "Q201B_6", "Q201B_4", "Q201B_13", "Q201B_12"]
    + ["Q105A", "Q277_1", "Q277_2", "Q549_2", "Q549_3"]
    + [f"Q204A_{i}" for i in range(1, 10)]
    + ["Q204_2", "Q204_3", "Q204_11", "Q204_20", "Q210", "Q211_2", "Q276"]
    + ["Q404", "Q501D", "Q266", "Q521_1", "Q521_4", "Q521_6", "Q514A", "Q516A", "Q511",
       "Q512", "Q301A", "Q725_7", "Q738"]
    # religion
    + ["Q609", "Q610_1", "Q610_8", "Q606_2", "Q606_4", "Q606_8", "Q1012A_MUSLIM", "Q605",
       "Q610_6A"]
    # discrimination, services and environment, gender, media
    + AB_DISCRIMINATION + ["Q882", "Q274A", "Q274B", "Q540B_1", "Q540B_2", "Q540B_3", "Q554"]
    + ["Q601_3", "Q601_4", "Q601_5", "Q627_1", "Q627_2", "Q627_3", "Q625"]
    + ["Q409", "Q424"] + AB_SOCIAL_MEDIA + ["Q421"]
    # work and housing follow-ups to GIVEN facts
    + ["Q1006", "Q1006A", "Q1021", "Q1005B", "Q1005C", "Q1030", "Q1035", "Q1001H", "Q1010B1"]
)
_INTERNET_USER = ["Throughout the day", "At least once daily", "Several times a week",
                  "Once a week", "Less than once a week"]
AB_GATES = {
    **{c: ("Q104", _YES) for c in AB_EMIGRATE_WHY + AB_EMIGRATE_WHERE + ["Q104C"]},
    **{c: ("Q409", _INTERNET_USER) for c in ["Q424"] + AB_SOCIAL_MEDIA},
    **{c: ("Q1012", ["Muslim"]) for c in ["Q1012A_MUSLIM", "Q605", "Q610_6A"]},
    **{c: ("Q1005", ["Employed", "Self-Employed"]) for c in ["Q1006", "Q1006A", "Q1021"]},
    "Q1005B": ("Q1005", ["Retired"]),
    "Q1005C": ("Q1005", ["Unemployed"]),
    "Q1030": ("Q1005", ["A housewife", "A student", "Unemployed"]),
    "Q1035": ("Q1030", _YES),
    "Q1001H": ("Q1001G", ["Own"]),
    "Q1010B1": ("Q1010", ["Married", "Divorced", "Separated", "Widowed"]),
}
# Battery items whose SPSS label is only the sub-item get the stem they hang off.
AB_STEMS = {
    "Q104A_2_": "Reason for considering emigration", "Q104B_": "Country you would emigrate to",
    "Q130_": "Food coping action taken by your household",
    "Q873_": "Basis on which people in your country face discrimination",
    "Q412A_": "Social media platform you actively use",
    "Q129A_": "In the past 30 days, how often did your household",
    "Q916_": "Should migrant domestic workers be allowed", "Q201": "Trust in",
    "Q277_": "How often does this occur in your neighbourhood",
    "Q549_": "How concerned are you about threats to", "Q204A_": "Satisfaction with",
    "Q204_": "Government performance on", "Q521_": "Extent to which this freedom is guaranteed",
    "Q725_": "Evaluation of the foreign policies of", "Q610_": "How often do you",
    "Q606_": "Agree or disagree", "Q601_": "Agree or disagree",
    "Q627_": "How widespread is harassment of women", "Q540B_": "Concern about",
}


def build_arab_barometer():
    if not AB_PATH.exists():
        print(f"arab_barometer: skipped, {AB_PATH.relative_to(ROOT)} not found (download it by "
              "hand from arabbarometer.org; the form cannot be scripted)")
        return
    raw = [c for c in AB_GIVEN + AB_PREDICT if c not in ("age_band", "hh_size")]
    frame, value_labels, var_labels = read_sav(AB_PATH, raw + ["Q1001", "Q1014C", "DATE"])
    country = frame["COUNTRY"].map(value_labels["COUNTRY"])
    frame = frame[country.isin(AB_COUNTRIES)].reset_index(drop=True)
    dates = pd.to_datetime(frame["DATE"], format="%Y-%m-%d", errors="coerce")  # "today's date"
    for item in AB_EMIGRATE_WHY + AB_EMIGRATE_WHERE + AB_FOOD_COPING + AB_DISCRIMINATION \
            + AB_SOCIAL_MEDIA:  # 0 is labelled "Other" (or not at all), 1 by the option
        value_labels[item] = {0.0: "Not selected", 1.0: "Selected"}

    # Routing skips are blank (system-missing) in this file; ungated blanks stay an
    # ordinary NOT_RECORDED level.
    columns, orders, questions = convert(frame, value_labels, var_labels, raw, AB_GATES,
                                         f"^{NOT_RECORDED}$")
    columns["age_band"], orders["age_band"] = band_age(frame["Q1001"], [18, 25, 35, 45, 55, 65], 120)
    columns["hh_size"], orders["hh_size"] = band_age(frame["Q1014C"], [1, 3, 5, 7], 97)
    for item in raw:  # labels here are numbered "Q201A_1 ..." without the trailing period
        text = re.sub(r"^Q\w+?(?:\.\s*|\s+)", "", var_labels.get(item) or "")
        stem = next((s for prefix, s in AB_STEMS.items() if item.startswith(prefix)), None)
        questions[item] = " ".join((f"{stem}: {text}" if stem else text).split())
    questions.update({
        "COUNTRY": "Country", "Q1": "Governorate", "age_band": "Age of respondent (banded)",
        "hh_size": "Household size, including the respondent (banded)",
        "Q118_1": "Most important issue (first choice)",
        "Q882": "Have you personally experienced discrimination in your country?",
    })

    finish("arab_barometer", columns, orders, AB_GIVEN, AB_PREDICT, questions, AB_GATES,
           "Arab Barometer Wave VIII (2023-2024, face-to-face, adults 18+), public file, "
           f"restricted to {', '.join(AB_COUNTRIES)}. Real-survey proxy for the UNHCR ERPIS "
           "instrument (Syrian refugees in Egypt, Iraq, Jordan, Lebanon): the host populations "
           "of the same countries (Egypt is not in Wave VIII). GIVEN: country, governorate, "
           "urban/rural/camp, age band, gender, education, marital status, employment status, "
           "household size band, religion, ethnicity, household head, rent/own. PREDICT: "
           "economic situation and household finances, food insecurity and coping, emigration "
           "intentions, reasons and destinations, attitudes to migrants, trust in institutions, "
           "neighbourhood safety, government performance and services, politics, religiosity, "
           "discrimination, gender attitudes, internet use, and work/housing follow-ups. Gates "
           "follow the questionnaire routing (emigration reasons, destinations and irregular "
           "migration only after having considered emigrating; social-media items only for "
           "internet users; denomination and Islamic-practice items only for Muslims; job items "
           "by employment status; mortgage only for owners); NA_GATED marks the file's blank "
           "(system-missing) skip codes. Split-ballot and country-specific items are left out. "
           "Levels with fewer than 20 respondents are merged into 'Other'. Licence: Arab "
           "Barometer data are free for research use with citation; do not redistribute.",
           dates=dates, country="COUNTRY")
    degrade_arab_barometer()


def degrade_arab_barometer():
    """UNHCR-like proxies with real answers: soft gates at UNHCR rates (`arab_barometer_real`),
    plus 40% repeat interviews (`arab_barometer_panel`), via scripts/degrade_sandbox.py.
    Its round-coding rule needs UNHCR item names and a `wave` column, so it does not fire here."""
    script, source = ROOT / "scripts" / "degrade_sandbox.py", BENCH / "arab_barometer"
    sys.stdout.flush()
    for out, extra in (("arab_barometer_real", []), ("arab_barometer_panel", ["--panel", "0.4"])):
        subprocess.run([sys.executable, str(script), str(source), str(BENCH / out),
                        "--seed", str(SEED), *extra], check=True)
        config = make_sandbox.load_config(str(KIT / "config.yml"))
        kit.load_frames(str(BENCH / out), make_sandbox.load_schema(str(BENCH / out / "schema.json"), config))


WGMU_COUNTRIES = WGM_COUNTRIES + ["Ghana", "Uganda", "Tanzania", "Cameroon", "Senegal", "Ethiopia", "Philippines",
                                  "Indonesia", "Peru", "Morocco"]
WGMU_PREDICT = ["W4", "W5B", "W5C", "W5E", "W5G", "W6", "W30", "WP21768", "WP21757", "W13", "W14", "W15"]


def build_wgm_unicef():
    """UNICEF-shaped proxy (Oct 19): 12 PREDICT items (trust in health care, government, scientists, doctors and
    traditional healers, science vs religion, COVID-19 vaccination and impact, climate items, two gated on a hidden
    answer), the same 7 GIVEN facts, 20 countries (about 20,000 people), like UNICEF's 12 / 7 / 19,846."""
    path = download("wgm2020.sav")
    raw = [c for c in WGM_GIVEN + WGMU_PREDICT if c != "age_band"] + ["Age"]
    frame, value_labels, var_labels = read_sav(path, raw)
    frame = frame[frame["COUNTRYNEW"].isin(WGMU_COUNTRIES)].reset_index(drop=True)
    value_labels["Education"][99.0] = "DK/Refused"
    gates = {k: v for k, v in WGM_GATES.items() if k in WGMU_PREDICT}
    items = [c for c in WGM_GIVEN + WGMU_PREDICT if c not in ("age_band", "COUNTRYNEW")]
    columns, orders, questions = convert(frame, value_labels, var_labels, items, gates, f"^{NOT_RECORDED}$")
    columns["COUNTRYNEW"], orders["COUNTRYNEW"] = frame["COUNTRYNEW"], sorted(WGMU_COUNTRIES)
    questions["COUNTRYNEW"] = "Country"
    columns["age_band"], orders["age_band"] = band_age(frame["Age"], [15, 25, 35, 45, 55, 65], 99)
    questions["age_band"] = "Age of respondent (banded)"
    for item in questions:
        questions[item] = re.sub(r"\s*\[Note:.*\]$", "", questions[item])
    finish("wgm_unicef", columns, orders, WGM_GIVEN, WGMU_PREDICT, questions, gates,
           "Wellcome Global Monitor 2020, UNICEF-shaped proxy: 12 PREDICT items, 7 GIVEN facts, "
           f"{len(WGMU_COUNTRIES)} countries. Licence: CC BY-NC-ND 4.0; do not redistribute.",
           all_variants=False)


BUILDERS = {"afrobarometer": build_afrobarometer, "wgm_unicef": build_wgm_unicef, "afrobarometer_rest": lambda: build_afrobarometer(rest=True),
            "wgm_vaccine": build_wgm_vaccine,
            "arab_barometer": build_arab_barometer}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(BUILDERS), nargs="+",
                        help="build these datasets only (default: all)")
    args = parser.parse_args()
    for name in args.only or [n for n in BUILDERS if n not in ("afrobarometer_rest", "wgm_unicef")]:
        BUILDERS[name]()


if __name__ == "__main__":
    main()
