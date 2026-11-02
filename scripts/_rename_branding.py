#!/usr/bin/env python3
"""One-off repo hygiene: rebrand the old internal project name to VERSE and
strip personal author tags in the bundled code. Conservative by design:

  - "SpatialEvol" / "SpatialEvo"  ->  "VERSE"   (branding only; verified not
    used as a Python identifier/import/class in this repo)
  - author tags  "Xinran(...)" / "Xinran:"  ->  "VERSE note"
  - PRESERVES the functional env-var prefix `SPATIALEVO_` (all-caps) so the
    trainer keeps reading its knobs; the top-level scripts export both names.
  - PRESERVES upstream names (ViLaSR / verl / EasyR1 / LLaMA-Factory) — these
    are attributed dependencies, not our branding.
  - Does NOT rewrite absolute paths (handled via configs/paths.env + top-level
    scripts, which override paths at runtime).

Usage: python scripts/_rename_branding.py [--apply]  (default: dry-run)
"""
import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIRS = ["self_evolve", "eval"]
EXTS = {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt", ".cfg", ".ini"}

# Order matters: longer first. All-caps SPATIALEVO_ is excluded via negative
# lookahead on the underscore+caps form (we only match the CamelCase spellings).
SUBS = [
    (re.compile(r"SpatialEvol"), "VERSE"),
    (re.compile(r"SpatialEvo(?!L)"), "VERSE"),   # CamelCase brand, not SPATIALEVO_
    (re.compile(r"Xinran\([^)]*\)\s*:?"), "VERSE note:"),
    (re.compile(r"\bXinran\b:"), "VERSE note:"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()
    total_files = 0
    total_hits = 0
    for d in TARGET_DIRS:
        for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if os.path.splitext(fn)[1] not in EXTS:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        txt = f.read()
                except (UnicodeDecodeError, IsADirectoryError):
                    continue
                new = txt
                hits = 0
                for rx, repl in SUBS:
                    new, n = rx.subn(repl, new)
                    hits += n
                if hits:
                    total_files += 1
                    total_hits += hits
                    rel = os.path.relpath(p, ROOT)
                    print(f"  {hits:4d}  {rel}")
                    if args.apply:
                        with open(p, "w", encoding="utf-8") as f:
                            f.write(new)
    mode = "APPLIED" if args.apply else "DRY-RUN (use --apply to write)"
    print(f"\n[{mode}] {total_hits} replacements across {total_files} files")


if __name__ == "__main__":
    main()
