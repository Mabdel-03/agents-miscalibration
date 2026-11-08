#!/usr/bin/env python3
"""Check this folder against MANIFEST.csv. Standard library only; run from anywhere."""
import csv
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    with open(os.path.join(ROOT, "MANIFEST.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing, changed = [], []
    for row in rows:
        path = os.path.join(ROOT, row["path"])
        if not os.path.exists(path):
            missing.append(row["path"])
        elif sha256(path) != row["sha256"]:
            changed.append(row["path"])
    listed = {row["path"] for row in rows} | {"MANIFEST.csv"}
    extra = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT).replace(os.sep, "/")
            if rel not in listed:
                extra.append(rel)
    print(f"{len(rows)} files listed; {len(missing)} missing, {len(changed)} changed, {len(extra)} unlisted")
    for label, paths in (("missing", missing), ("changed", changed), ("unlisted", extra)):
        for path in paths[:50]:
            print(f"  {label}: {path}")
    return 1 if missing or changed or extra else 0


if __name__ == "__main__":
    sys.exit(main())
