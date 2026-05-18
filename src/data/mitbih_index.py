"""MIT-BIH dataset indexing utilities.

Builds a per-record metadata table useful for:
- sanity checks (fs, signal length, number of annotations)
- research reporting (IEEE-style dataset description)
- debugging label distributions and downstream class mapping

Outputs
- JSON report (rich, nested)
- CSV table (flat, analysis-friendly)

Compatibility notes
- This project uses wfdb==4.3.x which reads local files by running from the
  directory containing the WFDB record files. We therefore temporarily chdir
  into the dataset directory when calling wfdb.rdheader/rdann.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import wfdb

from src.data.mitbih_download import MITBIH_DB_NAME, get_logger, get_mitbih_record_list


@dataclass(frozen=True)
class RecordIndexRow:
    record_id: str
    fs: Optional[int]
    sig_len: Optional[int]
    n_annotations: Optional[int]
    symbol_counts: Dict[str, int]


@dataclass(frozen=True)
class DatasetIndexReport:
    db_name: str
    local_dir: str
    created_utc: str
    n_records: int
    records: List[RecordIndexRow]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_in_dir(dir_path: Path):
    """Context manager to temporarily chdir into a directory."""

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


def index_record(local_dir: Path, record_id: str) -> RecordIndexRow:
    """Index a single record from local WFDB files."""

    with _run_in_dir(local_dir):
        header = wfdb.rdheader(record_name=record_id, pn_dir=None)
        ann = wfdb.rdann(record_name=record_id, extension="atr", pn_dir=None)

    fs = int(header.fs) if getattr(header, "fs", None) is not None else None
    sig_len = int(header.sig_len) if getattr(header, "sig_len", None) is not None else None

    symbols_raw = getattr(ann, "symbol", None)
    symbols: List[str]
    if symbols_raw is None:
        symbols = []
    else:
        # wfdb returns list-like; ensure plain list[str]
        symbols = [str(s) for s in list(symbols_raw)]

    symbol_counts = dict(Counter(symbols))

    samples_raw = getattr(ann, "sample", None)
    if samples_raw is None:
        n_annotations = 0
    else:
        # ann.sample is typically a numpy array
        n_annotations = int(len(samples_raw))

    return RecordIndexRow(
        record_id=str(record_id),
        fs=fs,
        sig_len=sig_len,
        n_annotations=n_annotations,
        symbol_counts=symbol_counts,
    )


def build_dataset_index(
    local_dir: Path,
    records: Optional[Sequence[str]] = None,
    db_name: str = MITBIH_DB_NAME,
    logger: Optional[logging.Logger] = None,
) -> DatasetIndexReport:
    """Build an index report for the dataset present in `local_dir`."""

    logger = logger or get_logger("mitbih.index")
    local_dir = local_dir.resolve()

    if records is None:
        # authoritative list from PhysioNet (unless offline); stable for mitdb
        records = get_mitbih_record_list(db_name, logger=logger)

    rows: List[RecordIndexRow] = []
    for rec in records:
        try:
            row = index_record(local_dir=local_dir, record_id=str(rec))
            rows.append(row)
            logger.info(
                "Indexed %s | fs=%s sig_len=%s n_ann=%s unique_symbols=%d",
                row.record_id,
                row.fs,
                row.sig_len,
                row.n_annotations,
                len(row.symbol_counts),
            )
        except Exception as e:
            logger.exception("Failed to index record %s: %s", rec, e)
            # keep going; partial index is still useful

    return DatasetIndexReport(
        db_name=db_name,
        local_dir=str(local_dir),
        created_utc=_utc_now_iso(),
        n_records=len(rows),
        records=rows,
    )


def export_index_json(report: DatasetIndexReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


def export_index_csv(report: DatasetIndexReport, out_path: Path) -> None:
    """Export a flat CSV (symbol counts are JSON-encoded per row)."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["record_id", "fs", "sig_len", "n_annotations", "symbol_counts_json"]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report.records:
            w.writerow(
                {
                    "record_id": r.record_id,
                    "fs": r.fs,
                    "sig_len": r.sig_len,
                    "n_annotations": r.n_annotations,
                    "symbol_counts_json": json.dumps(r.symbol_counts, ensure_ascii=False),
                }
            )


def build_and_export_dataset_index(
    local_dir: Path,
    out_dir: Path,
    records: Optional[Sequence[str]] = None,
    db_name: str = MITBIH_DB_NAME,
    logger: Optional[logging.Logger] = None,
) -> DatasetIndexReport:
    """Convenience wrapper: build index + export JSON/CSV."""

    logger = logger or get_logger("mitbih.index")
    report = build_dataset_index(local_dir=local_dir, records=records, db_name=db_name, logger=logger)

    export_index_json(report, out_dir / "mitbih_index.json")
    export_index_csv(report, out_dir / "mitbih_index.csv")

    logger.info("Index exported to %s", str(out_dir.resolve()))
    return report
