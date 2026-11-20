#!/usr/bin/env bash
# Fetch the pretrained weights a submission ships (not redistributed in this repo) and check their sha256.
#   HF_TOKEN=... bash scripts/fetch_weights.sh submission
# TabPFN-3.5 is gated on Hugging Face: accept Prior Labs' licence on its model page first. The fp16 file is our
# derivative of the 220M default checkpoint (scripts/make_fp16.py; a GPU is used for its load check).
# TabFM is not shipped: the grader prefetches it from models.txt (google/tabfm-1.0.0-pytorch).
set -euo pipefail
SUB=${1:-submission}; W=$SUB/weights; mkdir -p "$W"
get() { python -c "from huggingface_hub import hf_hub_download as h; print(h('$1', '$2', revision='$3'))"; }
REV_TABPFN=06bf2ba35c80a92a3b9abb436b99cf49e7a0365e   # Prior-Labs/tabpfn_3_5 snapshot used for every check
cp -L "$(get Prior-Labs/tabpfn_3_5 tabpfn-v3.5-fast-20260909.safetensors $REV_TABPFN)" "$W/"
cp -L "$(get jingang/TabICL tabicl-classifier-v2-20260212.ckpt main)" "$W/"
SRC=$(get Prior-Labs/tabpfn_3_5 tabpfn-v3.5-20260909.safetensors $REV_TABPFN)
echo "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3  $SRC" | sha256sum -c -
python scripts/make_fp16.py "$SRC" "$W/tabpfn-v3.5-20260909-fp16.safetensors"
(cd "$W" && sha256sum -c - <<'SUMS'
14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550  tabpfn-v3.5-fast-20260909.safetensors
bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0  tabicl-classifier-v2-20260212.ckpt
SUMS
)
# the fp16 file's bytes depend on the safetensors writer (header layout), so it is checked by content
test "$(python scripts/weights_hash.py "$W/tabpfn-v3.5-20260909-fp16.safetensors")" = \
  f8c177f4305cb03eac95d4fbf550fa06ce6f604e107375bfe35d58d80772afdf && echo "tabpfn-v3.5-20260909-fp16.safetensors: content OK"
