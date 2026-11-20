#!/usr/bin/env python
"""Zip a submission directory for upload and validate it.

    python scripts/package.py submission          # zip + contract check
    python scripts/package.py submission --full   # + official score.py on all 3

Writes dist/<name>.zip (zipped from *inside* the directory, so main.py is at the
root), then runs starter-kit/tools/check_submission_zip.py on it. With --full it
also runs the official starter-kit/score.py (fresh venv, network cut) on every
sandbox instrument -- the closest local rehearsal of the hosted run.
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "starter-kit"
SKIP = {"__pycache__", ".DS_Store", ".ipynb_checkpoints"}


def build_zip(submission: Path, strict: bool = False, dev_cap: int = 0, ft_dev: bool = False,
              full_dev: tuple = (), ft_dev_skip: tuple = ()) -> Path:
    out = ROOT / "dist" / (f"{submission.name}{'_strict' if strict else ''}{'_ftdev' if ft_dev else ''}"
                           f"{''.join('_full' + n for n in full_dev)}{''.join('_skip' + n for n in ft_dev_skip)}"
                           f"{f'_dev{dev_cap}' if dev_cap else ''}.zip")
    out.parent.mkdir(exist_ok=True)
    files = []                                   # os.walk follows a symlinked weights/ (shared, not copied)
    for root, _, names in os.walk(submission, followlinks=True):
        files += [Path(root) / n for n in names]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            rel = path.relative_to(submission)
            if any(part in SKIP for part in rel.parts) or path.suffix == ".zip":
                continue
            if (strict or dev_cap or ft_dev or full_dev or ft_dev_skip) and rel.as_posix() == "main.py":
                code = path.read_text()
                if strict:
                    assert "STRICT_DEFAULT = False" in code, "main.py has no STRICT_DEFAULT switch"
                    code = code.replace("STRICT_DEFAULT = False", "STRICT_DEFAULT = True", 1)
                if ft_dev:                     # rehearsal: the final's fine-tuning also runs in development
                    assert "FT_DEV = False" in code, "main.py has no FT_DEV switch"
                    code = code.replace("FT_DEV = False", "FT_DEV = True", 1)
                if ft_dev_skip:                # rehearsal: these surveys keep the development path without fine-tuning
                    assert "FT_DEV_SKIP = ()" in code, "main.py has no FT_DEV_SKIP switch"
                    code = code.replace("FT_DEV_SKIP = ()", f"FT_DEV_SKIP = {tuple(ft_dev_skip)!r}", 1)
                if full_dev:                   # probe: these surveys fine-tune TabPFN in development
                    assert "FULL_DEV = ()" in code, "main.py has no FULL_DEV switch"
                    code = code.replace("FULL_DEV = ()", f"FULL_DEV = {tuple(full_dev)!r}", 1)
                if dev_cap:                    # development budget only; the final's stays as written
                    assert "TOTAL_SECONDS = {1: 780," in code, "main.py has no TOTAL_SECONDS = {1: 780, ...}"
                    code = code.replace("TOTAL_SECONDS = {1: 780,", f"TOTAL_SECONDS = {{1: {dev_cap},", 1)
                zf.writestr("main.py", code)
                continue
            zf.write(path, rel.as_posix())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission")
    parser.add_argument("--full", action="store_true",
                        help="also run the official score.py on every sandbox instrument")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument("--strict", action="store_true",
                        help="development rehearsal: zip as <name>_strict.zip with STRICT_DEFAULT = True "
                             "(a failed or skipped member fails the run; never for the final)")
    parser.add_argument("--ft-dev", action="store_true",
                        help="deployment rehearsal: FT_DEV = True, so run-time fine-tuning also runs in development")
    parser.add_argument("--dev-cap", type=int, default=0,
                        help="development uploads: zip with this whole-run cap for the development phase "
                             "(e.g. 430); the final budget is unchanged")
    parser.add_argument("--full-dev", nargs="+", default=[], choices=["unicef", "world_bank", "unhcr"],
                        help="development probe: these surveys fine-tune TabPFN in development with all spare time (FULL_DEV)")
    parser.add_argument("--ft-dev-skip", nargs="+", default=[], choices=["unicef", "world_bank", "unhcr"],
                        help="with --ft-dev: these surveys keep the development path without fine-tuning (FT_DEV_SKIP)")
    args = parser.parse_args()

    submission = Path(args.submission).resolve()
    if not (submission / "main.py").exists():
        sys.exit(f"{submission} has no main.py")
    archive = build_zip(submission, args.strict, args.dev_cap, args.ft_dev, tuple(args.full_dev), tuple(args.ft_dev_skip))
    print(f"wrote {archive.relative_to(ROOT)} ({archive.stat().st_size / 1e6:.2f} MB)")

    subprocess.run([sys.executable, str(KIT / "tools" / "check_submission_zip.py"),
                    str(archive)], check=True)

    if args.full:
        for instrument in ("unicef", "world_bank", "unhcr"):
            print(f"--- score.py {instrument} (phase {args.phase})")
            subprocess.run([sys.executable, "score.py", "--submission", str(submission),
                            "--data", str(ROOT / "sandboxes" / instrument),
                            "--schema", f"data/{instrument}.json",
                            "--phase", str(args.phase)], cwd=KIT, check=True)


if __name__ == "__main__":
    main()
