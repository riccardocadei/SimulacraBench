"""Usage: python scripts/make_fp16.py <tabpfn-v3.5-20260909.safetensors> <out fp16 file>  (needs a GPU for the load check)
Write the 220M TabPFN-3.5 default checkpoint as fp16 safetensors (the bundled file) and check TabPFN loads it
into an fp32 model whose weights equal the fp16-rounded ones used in the lab arms."""
import sys, os, numpy as np, torch, safetensors
from safetensors.torch import load_file, save_file
src, out = sys.argv[1], sys.argv[2]
sd = load_file(src)
meta = safetensors.safe_open(src, "pt").metadata()
save_file({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, out, metadata=meta)
print("wrote", out, os.path.getsize(out) / 1e6, "MB")
import tabpfn.checkpoint
from tabpfn import TabPFNClassifier
tabpfn.checkpoint.Checkpoint.is_safetensors = property(lambda self: True)
os.environ["TABPFN_MODEL_CACHE_SIZE"] = "0"
rng = np.random.default_rng(0)
X, y = rng.normal(size=(300, 6)).astype(np.float32), rng.integers(0, 3, 300)
p = {}
for tag, path in (("fp16", out), ("fp32", src)):
    clf = TabPFNClassifier(model_path=path, device="cuda", n_estimators=1, random_state=0)
    clf.fit(X, y)
    m = clf.models_[0] if isinstance(clf.models_, (list, tuple)) else clf.models_
    dt = {str(t.dtype) for t in m.state_dict().values() if t.is_floating_point()}
    if tag == "fp16":
        ref = {k: v.half().float() for k, v in sd.items() if v.is_floating_point()}
        msd = m.state_dict()
        diff = max(float((msd[k].float().cpu() - ref[k]).abs().max()) for k in ref if k in msd)
        print("fp16 file -> model dtypes", dt, "max |w - round16(w)|", diff, "matched", sum(k in msd for k in ref), len(ref))
    p[tag] = clf.predict_proba(X[:50])
print("max |p16 - p32|", float(np.abs(p["fp16"] - p["fp32"]).max()))
