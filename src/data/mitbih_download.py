"""
Downloader + validator for MIT-BIH Arrhythmia Database (mitdb) via WFDB.

Features
- Download the 48 canonical MITDB records.
- Validate required WFDB files: .dat, .hea, .atr
- Integrity logging and export report (JSON + CSV).
- Robust error handling with retries and clear diagnostics.

Rationale
- PhysioNet does not guarantee ZIP endpoints for databases.
- WFDB's dl_database is the most stable way to fetch records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import inspect
import wfdb

MITBIH_DB_NAME = "mitdb"

# MIT-BIH Arrhythmia Database record list (48 records).
MITBIH_RECORDS_48: Tuple[str, ...] = (
    "100",
    "101",
    "102",
    "103",
    "104",
    "105",
    "106",
    "107",
    "108",
    "109",
    "111",
    "112",
    "113",
    "114",
    "115",
    "116",
    "117",
    "118",
    "119",
    "121",
    "122",
    "123",
    "124",
    "200",
    "201",
    "202",
    "203",
    "204",
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
    "215",
    "217",
    "219",
    "220",
    "221",
    "222",
    "223",
    "228",
    "230",
    "231",
    "232",
    "233",
    "234",
)

REQUIRED_EXTENSIONS: Tuple[str, ...] = (".dat", ".hea", ".atr")


class DownloadError(RuntimeError):
    pass


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileCheck:
    path: str
    exists: bool
    size_bytes: int
    ok_nonempty: bool
    sha256: Optional[str] = None


@dataclass(frozen=True)
class RecordValidation:
    record: str
    ok: bool
    errors: List[str]
    warnings: List[str]
    files: List[FileCheck]
    fs: Optional[int] = None
    n_sig: Optional[int] = None
    sig_len: Optional[int] = None
    n_annotations: Optional[int] = None
    annotation_range_ok: Optional[bool] = None


@dataclass(frozen=True)
class IntegrityReport:
    db_name: str
    physionet_dir: str
    local_dir: str
    created_utc: str
    records_requested: List[str]
    records_ok: List[str]
    records_failed: List[str]
    validations: List[RecordValidation]


def get_logger(name: str = "mitbih") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_check(path: Path, compute_sha256: bool) -> FileCheck:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    sha = _sha256_file(path) if (compute_sha256 and exists and size > 0) else None
    return FileCheck(
        path=str(path),
        exists=exists,
        size_bytes=size,
        ok_nonempty=(exists and size > 0),
        sha256=sha,
    )


def missing_required_files(local_dir: Path, record: str) -> List[Path]:
    missing: List[Path] = []
    for ext in REQUIRED_EXTENSIONS:
        p = local_dir / f"{record}{ext}"
        if not p.exists():
            missing.append(p)
    return missing


def _wfdb_dl_database(
    *,
    db_name: str,
    dl_dir: str,
    records: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Compatibility wrapper for `wfdb.dl_database`.

    Observed parameter names across WFDB versions:
    - `db_dir` (WFDB 4.x, e.g. 4.3.1)
    - `db_name` (some versions)
    - `physionet_dir` (some versions)

    This wrapper inspects the signature and calls the appropriate variant.
    """

    logger = logger or get_logger()

    try:
        sig = inspect.signature(wfdb.dl_database)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    # WFDB 4.x
    if "db_dir" in params:
        wfdb.dl_database(db_dir=db_name, dl_dir=dl_dir, records=list(records))
        return

    # Older/newer variants
    if "db_name" in params:
        wfdb.dl_database(db_name=db_name, dl_dir=dl_dir, records=list(records))
        return

    if "physionet_dir" in params:
        wfdb.dl_database(physionet_dir=db_name, dl_dir=dl_dir, records=list(records))
        return

    # Fallback for unknown signatures
    for kw in ("db_dir", "db_name", "physionet_dir"):
        try:
            wfdb.dl_database(**{kw: db_name}, dl_dir=dl_dir, records=list(records))
            return
        except TypeError:
            continue

    raise TypeError("Unsupported wfdb.dl_database signature; cannot determine database parameter name")


def download_records(
    out_dir: Path,
    records: Sequence[str] = MITBIH_RECORDS_48,
    physionet_dir: str = MITBIH_DB_NAME,
    overwrite: bool = False,
    retries: int = 3,
    retry_backoff_s: float = 2.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Download records using wfdb.dl_database.

    Files downloaded are stored flat in out_dir:
      {record}.hea, {record}.dat, {record}.atr
    """
    logger = logger or get_logger()
    ensure_dir(out_dir)

    try:
        logger.info("wfdb version: %s", getattr(wfdb, "__version__", "unknown"))
    except Exception:
        pass

    for rec in records:
        if not overwrite:
            missing = missing_required_files(out_dir, rec)
            if not missing:
                logger.info("Skip %s (already present)", rec)
                continue

        attempt = 0
        while True:
            attempt += 1
            try:
                logger.info("Downloading %s (attempt %d/%d)", rec, attempt, retries)
                _wfdb_dl_database(
                    db_name=physionet_dir,
                    dl_dir=str(out_dir),
                    records=[rec],
                    logger=logger,
                )
                missing_after = missing_required_files(out_dir, rec)
                if missing_after:
                    raise DownloadError(f"Downloaded but missing: {[str(p) for p in missing_after]}")
                break
            except Exception as e:
                if attempt >= retries:
                    raise DownloadError(f"Failed to download {rec} after {retries} attempts: {e}") from e
                sleep_s = retry_backoff_s * (2 ** (attempt - 1))
                logger.warning("Download error for %s: %s | retry in %.1fs", rec, e, sleep_s)
                time.sleep(sleep_s)


def _run_in_dir(dir_path: Path):
    """Context manager to temporarily chdir into a directory (wfdb 4.3.1 compatibility)."""

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


def validate_record(
    local_dir: Path,
    record: str,
    compute_sha256: bool = True,
    logger: Optional[logging.Logger] = None,
) -> RecordValidation:
    """
    Validate a record by:
    - required files exist and are non-empty
    - header readable
    - signal readable from .dat (sample slice)
    - annotations readable from .atr + range check

    NOTE (wfdb 4.3.1): rdheader/rdsamp/rdann do not accept `rd_dir`, so we
    temporarily `chdir` into `local_dir` to read local files.
    """
    logger = logger or get_logger()
    errors: List[str] = []
    warnings: List[str] = []

    files = [
        _file_check(local_dir / f"{record}{ext}", compute_sha256=compute_sha256)
        for ext in REQUIRED_EXTENSIONS
    ]

    missing = [f.path for f in files if not f.exists]
    if missing:
        errors.append(f"Missing required files: {missing}")

    empty = [f.path for f in files if f.exists and not f.ok_nonempty]
    if empty:
        errors.append(f"Empty files: {empty}")

    fs = None
    n_sig = None
    sig_len = None
    n_ann = None
    ann_range_ok = None

    if not errors:
        with _run_in_dir(local_dir):
            # Header check
            try:
                header = wfdb.rdheader(record_name=record, pn_dir=None)
                fs = int(header.fs) if header.fs is not None else None
                n_sig = int(header.n_sig) if header.n_sig is not None else None
                sig_len = int(header.sig_len) if header.sig_len is not None else None
                if fs is None:
                    warnings.append("Header fs is None")
            except Exception as e:
                errors.append(f"rdheader failed: {e}")

            # Signal check
            if not errors:
                try:
                    sig, _fields = wfdb.rdsamp(record_name=record, sampfrom=0, sampto=1000, pn_dir=None)
                    if sig is None or getattr(sig, "size", 0) == 0:
                        errors.append("rdsamp returned empty signal")
                except Exception as e:
                    errors.append(f"rdsamp failed: {e}")

            # Annotation check
            if not errors:
                try:
                    ann = wfdb.rdann(record_name=record, extension="atr", pn_dir=None)
                    n_ann = int(len(ann.sample)) if ann is not None and ann.sample is not None else 0
                    if n_ann == 0:
                        warnings.append("No annotations found")

                    if sig_len is not None and n_ann > 0:
                        mn = int(min(ann.sample))
                        mx = int(max(ann.sample))
                        ann_range_ok = (mn >= 0) and (mx < sig_len)
                        if not ann_range_ok:
                            warnings.append(
                                f"Annotation range out of bounds: min={mn}, max={mx}, sig_len={sig_len}"
                            )
                    else:
                        ann_range_ok = True
                except Exception as e:
                    errors.append(f"rdann failed: {e}")

    ok = len(errors) == 0
    if ok:
        logger.info("OK %s | fs=%s n_sig=%s sig_len=%s n_ann=%s", record, fs, n_sig, sig_len, n_ann)
    else:
        logger.error("FAIL %s | %s", record, errors)

    return RecordValidation(
        record=record,
        ok=ok,
        errors=errors,
        warnings=warnings,
        files=files,
        fs=fs,
        n_sig=n_sig,
        sig_len=sig_len,
        n_annotations=n_ann,
        annotation_range_ok=ann_range_ok,
    )


def validate_dataset(
    local_dir: Path,
    records: Sequence[str],
    compute_sha256: bool = True,
    logger: Optional[logging.Logger] = None,
) -> List[RecordValidation]:
    logger = logger or get_logger()
    ensure_dir(local_dir)

    validations: List[RecordValidation] = []
    for rec in records:
        validations.append(
            validate_record(
                local_dir=local_dir,
                record=rec,
                compute_sha256=compute_sha256,
                logger=logger,
            )
        )
    return validations


def build_report(
    local_dir: Path,
    physionet_dir: str,
    records_requested: Sequence[str],
    validations: Sequence[RecordValidation],
) -> IntegrityReport:
    ok = [v.record for v in validations if v.ok]
    failed = [v.record for v in validations if not v.ok]
    return IntegrityReport(
        db_name=MITBIH_DB_NAME,
        physionet_dir=physionet_dir,
        local_dir=str(local_dir),
        created_utc=_utc_now_iso(),
        records_requested=list(records_requested),
        records_ok=ok,
        records_failed=failed,
        validations=list(validations),
    )


def export_report_json(report: IntegrityReport, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


def export_report_csv(report: IntegrityReport, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    fieldnames = [
        "record",
        "ok",
        "fs",
        "n_sig",
        "sig_len",
        "n_annotations",
        "annotation_range_ok",
        "errors",
        "warnings",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v in report.validations:
            w.writerow(
                {
                    "record": v.record,
                    "ok": v.ok,
                    "fs": v.fs,
                    "n_sig": v.n_sig,
                    "sig_len": v.sig_len,
                    "n_annotations": v.n_annotations,
                    "annotation_range_ok": v.annotation_range_ok,
                    "errors": "|".join(v.errors),
                    "warnings": "|".join(v.warnings),
                }
            )


# NOTE: Some MITDB record numbers that appear in literature are NOT present
# in the PhysioNet `mitdb` file listing (e.g., 204). To avoid 404s, we can
# derive the authoritative list from wfdb/PhysioNet at runtime.
MITBIH_RECORDS_FALLBACK_48: Tuple[str, ...] = MITBIH_RECORDS_48


def get_mitbih_record_list(
    db_name: str = MITBIH_DB_NAME,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Return the authoritative record list from PhysioNet via WFDB.

    Falls back to a hardcoded list if the call fails (offline).
    """

    logger = logger or get_logger()
    try:
        records = wfdb.get_record_list(db_name)
        records = [str(r) for r in records]
        if len(records) >= 40:
            return records
        logger.warning("wfdb.get_record_list('%s') returned a small list (%d); using fallback", db_name, len(records))
        return list(MITBIH_RECORDS_FALLBACK_48)
    except Exception as e:
        logger.warning("Could not fetch record list via wfdb.get_record_list('%s'): %s | using fallback", db_name, e)
        return list(MITBIH_RECORDS_FALLBACK_48)


def download_validate_and_report(
    out_dir: Path,
    report_dir: Path,
    records: Sequence[str] = MITBIH_RECORDS_48,
    physionet_dir: str = MITBIH_DB_NAME,
    overwrite: bool = False,
    retries: int = 3,
    compute_sha256: bool = True,
    logger: Optional[logging.Logger] = None,
) -> IntegrityReport:
    logger = logger or get_logger()
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    # If caller did not override records explicitly, use authoritative list.
    if list(records) == list(MITBIH_RECORDS_48):
        records = get_mitbih_record_list(physionet_dir, logger=logger)

    download_records(
        out_dir=out_dir,
        records=records,
        physionet_dir=physionet_dir,
        overwrite=overwrite,
        retries=retries,
        logger=logger,
    )

    validations = validate_dataset(
        local_dir=out_dir,
        records=records,
        compute_sha256=compute_sha256,
        logger=logger,
    )

    report = build_report(
        local_dir=out_dir,
        physionet_dir=physionet_dir,
        records_requested=records,
        validations=validations,
    )

    export_report_json(report, report_dir / "mitbih_integrity_report.json")
    export_report_csv(report, report_dir / "mitbih_integrity_report.csv")

    return report