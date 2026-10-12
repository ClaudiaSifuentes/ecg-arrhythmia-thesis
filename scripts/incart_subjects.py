"""Build the INCART record -> subject map from the PhysioNet record headers.

    python scripts/incart_subjects.py            # needs internet (PhysioNet)
    python scripts/incart_subjects.py --local data/raw/incartdb

INCART: 75 records extracted from 32 Holter recordings. Each header comment
names its source recording as "| patient N |"; records are grouped by N. The map is written to configs/incart_subjects.json and picked up
by src/data/subjects.py. Expect 32 subjects.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import wfdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None, help="folder with I01.hea ... I75.hea")
    ap.add_argument("--out", default="configs/incart_subjects.json")
    args = ap.parse_args()

    comments = {}
    for i in range(1, 76):
        r = f"I{i:02d}"
        h = wfdb.rdheader(str(Path(args.local) / r)) if args.local else wfdb.rdheader(r, pn_dir="incartdb")
        comments[r] = " | ".join(c.strip() for c in (h.comments or []))
        print(r, comments[r])

    # Every INCART header names its source Holter as "| patient N |".
    mapping, groups = {}, {}
    for r in sorted(comments):
        m = re.search(r"\|\s*patient\s+(\d+)\s*\|", comments[r])
        if not m:
            raise SystemExit(f"{r}: no 'patient N' field in header comments: {comments[r]!r}")
        subject = f"inc_P{int(m.group(1)):02d}"
        mapping[f"inc_{r}"] = subject
        groups.setdefault(subject, []).append(r)
    groups = list(groups.values())

    print(f"\n{len(groups)} subjects inferred from 75 records (PhysioNet states 32 Holter recordings):")
    for g in groups:
        print("  ", g, "->", comments[g[0]])
    Path(args.out).write_text(json.dumps({"records": comments, "subject_of_record": mapping}, indent=1))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
