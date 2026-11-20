#!/usr/bin/env python
"""Check every file of a submission zip against SHA256SUMS.

    python scripts/verify_zip.py dist/submission.zip
"""
import hashlib
import sys
import zipfile
from pathlib import Path

expected = dict(line.split()[::-1] for line in (Path(__file__).resolve().parents[1] / "SHA256SUMS").read_text().splitlines())
archive = zipfile.ZipFile(sys.argv[1])
found = {}
for info in archive.infolist():
    digest = hashlib.sha256()
    with archive.open(info) as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    found[info.filename] = digest.hexdigest()
bad = sorted(name for name in expected.keys() | found.keys() if expected.get(name) != found.get(name))
print("\n".join(f"MISMATCH {name}" for name in bad) or f"OK: all {len(found)} files match SHA256SUMS")
sys.exit(1 if bad else 0)
