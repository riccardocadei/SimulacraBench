#!/usr/bin/env bash
# Write practice data for every schema into sandboxes/<instrument>/ (git-ignored).
# Usage: scripts/make_sandboxes.sh [seed]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:-0}"
cd "$ROOT/starter-kit"
for name in sample unicef world_bank unhcr; do
  python make_sandbox.py --schema "data/$name.json" --out "$ROOT/sandboxes/$name" --seed "$SEED"
done
# Degraded copies with soft gates and UNHCR round coding.
for name in unicef world_bank unhcr; do
  python "$ROOT/scripts/degrade_sandbox.py" "$ROOT/sandboxes/$name" "$ROOT/sandboxes/${name}_real" --seed "$SEED"
done
# A UNHCR-shaped sandbox where 40% of people are repeat interviews of someone else (tests D7 mechanics only).
python "$ROOT/scripts/degrade_sandbox.py" "$ROOT/sandboxes/unhcr" "$ROOT/sandboxes/unhcr_panel" --panel 0.4 --seed "$SEED"
