"""Build the INCART record -> subject map from the PhysioNet record headers.

    python scripts/incart_subjects.py            # needs internet (PhysioNet)
    python scripts/incart_subjects.py --local data/raw/incartdb

INCART: 75 records extracted from 32 Holter recordings. Records cut from the
same Holter recording share the header comments (age, sex, diagnoses) and are
consecutive, so consecutive records with identical comments are grouped as
one subject. The map is written to configs/incart_subjects.json and picked up
by src/data/subjects.py. Review the printed groups before trusting them.
"""

from __future__ import annotations

import argparse
import json
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

    mapping, groups = {}, []
    prev_r, prev_c = None, None
    for r in sorted(comments):
        if prev_r is not None and comments[r] == prev_c and comments[r]:
            subject = mapping[f"inc_{prev_r}"]
        else:
            subject = f"inc_S{len(groups) + 1:02d}"
            groups.append([])
        mapping[f"inc_{r}"] = subject
        groups[-1].append(r)
        prev_r, prev_c = r, comments[r]

    print(f"\n{len(groups)} subjects inferred from 75 records (PhysioNet states 32 Holter recordings):")
    for g in groups:
        print("  ", g, "->", comments[g[0]])
    Path(args.out).write_text(json.dumps({"records": comments, "subject_of_record": mapping}, indent=1))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
