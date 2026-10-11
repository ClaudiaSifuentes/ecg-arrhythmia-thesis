"""Download a PhysioNet beat database and build it with the project pipeline.

Needs internet access to PhysioNet (run it in Colab or locally).

    # MIT-BIH Supraventricular Arrhythmia DB (78 records, 128 Hz, SVEB-rich)
    python scripts/build_extra_db.py --db svdb --name svdb
    # INCART rebuilt with lead II (channel 1) instead of lead I (channel 0)
    python scripts/build_extra_db.py --db incartdb --name inc --channel 1 --fs 257

    # concatenate onto an existing processed dataset
    python scripts/build_extra_db.py --merge data/processed/mitbih_incart data/processed/svdb \
        --out data/processed/mitbih_incart_svdb

Same steps as the original builder (bandpass 0.5–40 Hz, resample to 360 Hz,
R-peak detection, 250-sample windows [R-100, R+150), 6 RR features, record
z-score). patient_id is prefixed with `{name}_`. Extra DBs only ever go to the
CV / training pool: the MIT-BIH test patients are untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INCART_EXCLUDE = {"I08", "I09", "I12", "I29", "I30", "I32", "I38", "I39", "I57", "I58", "I62"}
FILES = ("X_beats.npy", "X_rr.npy", "y_beats.npy", "patient_id.npy")


def build(db: str, name: str, fs: int, channel: int, out: Path) -> None:
    import wfdb

    from src.data.dataset_builder import build_dataset

    raw = Path("data/raw") / db
    raw.mkdir(parents=True, exist_ok=True)
    records = wfdb.get_record_list(db)
    if not all((raw / f"{r}.hea").exists() for r in records):
        wfdb.dl_database(db, dl_dir=str(raw))
    if db == "incartdb":
        records = [r for r in records if r not in INCART_EXCLUDE]
    build_dataset([{"name": name, "data_dir": str(raw), "records": records, "fs": fs, "channel": channel}], str(out))
    y = np.load(out / "y_beats.npy")
    pid = np.load(out / "patient_id.npy")
    print(f"{db}: {len(y)} beats, {len(np.unique(pid))} records, classes N/SVEB/VEB = {np.bincount(y, minlength=3)}")
    u, c = np.unique(pid[y == 1], return_counts=True)
    print("records with >=10 SVEB:", int((c >= 10).sum()), "| top:", sorted(zip(c, u), reverse=True)[:8])


def merge(srcs: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        parts = [np.load(Path(s) / f) for s in srcs]
        if f == "patient_id.npy":
            parts = [p.astype("<U16") for p in parts]
        np.save(out / f, np.concatenate(parts, axis=0))
    y = np.load(out / "y_beats.npy")
    print(f"merged {srcs} -> {out}: {len(y)} beats, classes {np.bincount(y, minlength=3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--name")
    ap.add_argument("--fs", type=int, default=128)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--merge", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.merge:
        merge(args.merge, Path(args.out))
    else:
        build(args.db, args.name, args.fs, args.channel, Path(args.out or f"data/processed/{args.name}_ch{args.channel}"))


if __name__ == "__main__":
    main()
