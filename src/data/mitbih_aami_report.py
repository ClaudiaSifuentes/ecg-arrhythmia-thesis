"""Generate AAMI EC57 (3-class) mapping report for MIT-BIH.

Inputs
- Local MIT-BIH WFDB directory (data/raw/mitdb)

Outputs
- Per-record distribution of AAMI classes (N / SVEB / VEB)
- Unknown/unmapped symbol counts
- Global totals

This is intended for:
- IEEE-style dataset justification
- Debugging annotation usage before beat segmentation/training

Compatibility
- Works with wfdb==4.3.x by chdir into local directory when calling rdann.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import wfdb

from src.data.aami_mapping import AAMI_N, AAMI_SVEB, AAMI_VEB, get_mapping_spec, map_symbols_to_aami
from src.data.mitbih_download import MITBIH_DB_NAME, get_logger, get_mitbih_record_list


@dataclass(frozen=True)
class RecordAamiStats:
    record_id: str
    n_total_ann: int
    n_mapped: int
    n_unknown: int
    aami_counts: Dict[str, int]  # keys: 'N','SVEB','VEB'
    unknown_symbol_counts: Dict[str, int]


@dataclass(frozen=True)
class AamiDatasetReport:
    db_name: str
    local_dir: str
    created_utc: str
    mapping_spec: Dict
    totals: Dict
    per_record: List[RecordAamiStats]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_in_dir(dir_path: Path):
    class _Cwd:
        def __init__(self, target: Path):
            self.target = str(target)
            self.prev: Optional[str] = None

        def __enter__(self):
            self.prev = os.getcwd()
            os.chdir(self.target)
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.prev is not None:
                os.chdir(self.prev)
            return False

    return _Cwd(dir_path)


def compute_aami_stats_for_record(local_dir: Path, record_id: str) -> RecordAamiStats:
    with _run_in_dir(local_dir):
        ann = wfdb.rdann(record_name=record_id, extension="atr", pn_dir=None)

    symbols = [str(s) for s in list(getattr(ann, "symbol", []) or [])]
    n_total = int(len(getattr(ann, "sample", [])))

    res = map_symbols_to_aami(symbols)

    aami_counter = Counter()
    unknown_counter = Counter()

    for s, c in zip(symbols, res.y):
        if c is None:
            unknown_counter[s] += 1
        elif c == AAMI_N:
            aami_counter["N"] += 1
        elif c == AAMI_SVEB:
            aami_counter["SVEB"] += 1
        elif c == AAMI_VEB:
            aami_counter["VEB"] += 1

    return RecordAamiStats(
        record_id=str(record_id),
        n_total_ann=n_total,
        n_mapped=res.mapped,
        n_unknown=res.unknown,
        aami_counts={"N": int(aami_counter["N"]), "SVEB": int(aami_counter["SVEB"]), "VEB": int(aami_counter["VEB"])},
        unknown_symbol_counts=dict(unknown_counter),
    )


def build_aami_dataset_report(
    local_dir: Path,
    records: Optional[Sequence[str]] = None,
    db_name: str = MITBIH_DB_NAME,
    logger: Optional[logging.Logger] = None,
) -> AamiDatasetReport:
    logger = logger or get_logger("mitbih.aami")
    local_dir = local_dir.resolve()

    if records is None:
        records = get_mitbih_record_list(db_name, logger=logger)

    per_record: List[RecordAamiStats] = []

    totals_aami = Counter()
    totals_unknown_symbols = Counter()
    totals = {
        "n_records": 0,
        "n_total_ann": 0,
        "n_mapped": 0,
        "n_unknown": 0,
        "aami_counts": {"N": 0, "SVEB": 0, "VEB": 0},
    }

    for rec in records:
        try:
            stats = compute_aami_stats_for_record(local_dir=local_dir, record_id=str(rec))
            per_record.append(stats)

            totals["n_records"] += 1
            totals["n_total_ann"] += stats.n_total_ann
            totals["n_mapped"] += stats.n_mapped
            totals["n_unknown"] += stats.n_unknown

            totals_aami.update(stats.aami_counts)
            totals_unknown_symbols.update(stats.unknown_symbol_counts)

            logger.info(
                "AAMI %s | N=%d SVEB=%d VEB=%d | unknown=%d",
                stats.record_id,
                stats.aami_counts["N"],
                stats.aami_counts["SVEB"],
                stats.aami_counts["VEB"],
                stats.n_unknown,
            )
        except Exception as e:
            logger.exception("Failed AAMI stats for %s: %s", rec, e)

    totals["aami_counts"] = {
        "N": int(totals_aami["N"]),
        "SVEB": int(totals_aami["SVEB"]),
        "VEB": int(totals_aami["VEB"]),
    }

    # Add unknown symbol top list (for debugging/reporting)
    totals["top_unknown_symbols"] = totals_unknown_symbols.most_common(20)

    return AamiDatasetReport(
        db_name=db_name,
        local_dir=str(local_dir),
        created_utc=_utc_now_iso(),
        mapping_spec=get_mapping_spec(),
        totals=totals,
        per_record=per_record,
    )


def export_aami_report_json(report: AamiDatasetReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


def export_aami_report_csv(report: AamiDatasetReport, out_path: Path) -> None:
    """Flat CSV: one row per record."""

    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "n_total_ann",
        "n_mapped",
        "n_unknown",
        "N",
        "SVEB",
        "VEB",
        "unknown_symbol_counts_json",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report.per_record:
            w.writerow(
                {
                    "record_id": r.record_id,
                    "n_total_ann": r.n_total_ann,
                    "n_mapped": r.n_mapped,
                    "n_unknown": r.n_unknown,
                    "N": r.aami_counts["N"],
                    "SVEB": r.aami_counts["SVEB"],
                    "VEB": r.aami_counts["VEB"],
                    "unknown_symbol_counts_json": json.dumps(r.unknown_symbol_counts, ensure_ascii=False),
                }
            )


def build_and_export_aami_report(
    local_dir: Path,
    out_dir: Path,
    records: Optional[Sequence[str]] = None,
    db_name: str = MITBIH_DB_NAME,
    logger: Optional[logging.Logger] = None,
) -> AamiDatasetReport:
    logger = logger or get_logger("mitbih.aami")
    report = build_aami_dataset_report(local_dir=local_dir, records=records, db_name=db_name, logger=logger)

    export_aami_report_json(report, out_dir / "mitbih_aami_report.json")
    export_aami_report_csv(report, out_dir / "mitbih_aami_report.csv")

    logger.info("AAMI report exported to %s", str(out_dir.resolve()))
    return report
