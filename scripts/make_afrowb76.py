"""afrowb76 = afrobarometer split shaped to real World Bank's J and n: every > 10-option PREDICT question plus
narrow ones (fixed random order, seed 0) until 76 questions incl. gate parents; then TRAIN+DEV thinned to 5,715
(TRAIN and DEV by the same factor, seed given), TEST kept.
    python scripts/make_afrowb76.py <src afrobarometer split dir> <out dir> <seed>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
src, out, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
schema = json.load(open(src / "schema.json")); df = pd.read_parquet(src / "respondents.parquet")
it = schema["items"]
pred = [k for k, v in it.items() if v["class"] == "PREDICT" and k in df.columns]
wide = [k for k in pred if len(it[k]["values"]) > 10]
narrow = [k for k in pred if k not in wide]
order = np.random.default_rng(0).permutation(narrow).tolist()
def close(keep):
    while True:
        new = {it[k]["gate"]["parent"] for k in keep if it[k].get("gate")} & set(pred) - keep
        if not new:
            return keep
        keep |= new
keep = close(set(wide))
for k in order:
    if len(keep) >= 76:
        break
    cand = close(keep | {k})
    if len(cand) <= 76:
        keep = cand
drop = [k for k in pred if k not in keep]
for k in drop:
    del it[k]
df = df.drop(columns=drop)
sp = schema["split"]; ntr, nde, nte = sp["n_train"], sp["n_dev"], sp["n_test"]
assert ntr + nde + nte == len(df)
f = 5715 / (ntr + nde); rng = np.random.default_rng(seed)
ktr = round(f * ntr); kde = 5715 - ktr
rows = np.concatenate([np.sort(rng.choice(ntr, ktr, replace=False)), ntr + np.sort(rng.choice(nde, kde, replace=False)),
                       np.arange(ntr + nde, len(df))])
df = df.iloc[rows].reset_index(drop=True)
schema["split"] = {"n_train": int(ktr), "n_dev": int(kde), "n_test": int(nte)}
if "n_rows" in schema.get("dataset", {}):
    schema["dataset"]["n_rows"] = len(df)
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "respondents.parquet", index=False)
json.dump(schema, open(out / "schema.json", "w"), indent=1, ensure_ascii=False)
print(f"{out}: {len(keep)} questions ({len(wide)} wide), {ktr} TRAIN + {kde} DEV, {nte} TEST")
