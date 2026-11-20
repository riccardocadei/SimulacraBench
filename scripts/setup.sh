#!/usr/bin/env bash
# One-time setup of a fresh clone (any machine): the official starter kit at a pinned commit, and practice data.
# The starter kit is an external, read-only dependency, pinned to the commit we develop against.
# Usage: scripts/setup.sh            (then: conda activate simbench)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIT_URL=https://github.com/SituatedEvals/public
KIT_SHA=aa03186f8f083c7785a1a566c802c2077925e265

if [ ! -d "$ROOT/starter-kit/.git" ]; then
  git clone -q "$KIT_URL" "$ROOT/starter-kit"
fi
git -C "$ROOT/starter-kit" fetch -q origin
git -C "$ROOT/starter-kit" checkout -q "$KIT_SHA"
echo "starter-kit @ $(git -C "$ROOT/starter-kit" rev-parse --short HEAD)"

"$ROOT/scripts/make_sandboxes.sh"
