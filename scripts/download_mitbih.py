"""
Download + validate MIT-BIH (mitdb) and export integrity reports.

Usage:
  uv run -m scripts.download_mitbih --out-dir data/raw/mitdb --report-dir reports/integrity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.mitbih_download import (
    MITBIH_RECORDS_48,
    download_validate_and_report,
    get_logger,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and validate MIT-BIH (mitdb) via WFDB.")
    p.add_argument("--out-dir", type=str, default="data/raw/mitdb", help="Destination for WFDB files.")
    p.add_argument("--report-dir", type=str, default="reports/integrity", help="Where to store JSON/CSV reports.")
    p.add_argument("--overwrite", action="store_true", help="Redownload even if present.")
    p.add_argument("--retries", type=int, default=3, help="Retries per record.")
    p.add_argument("--no-sha256", action="store_true", help="Disable SHA256 computation.")
    p.add_argument("--records", type=str, nargs="*", default=list(MITBIH_RECORDS_48), help="Optional record list override.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("mitbih.cli")

    report = download_validate_and_report(
        out_dir=Path(args.out_dir),
        report_dir=Path(args.report_dir),
        records=args.records,
        overwrite=args.overwrite,
        retries=args.retries,
        compute_sha256=(not args.no_sha256),
        logger=logger,
    )

    if report.records_failed:
        logger.error("Validation failed for: %s", report.records_failed)
        logger.error("See reports in %s", args.report_dir)
        return 2

    logger.info("All records OK: %d", len(report.records_ok))
    logger.info("Reports saved in %s", args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())