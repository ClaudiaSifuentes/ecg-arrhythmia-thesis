"""Generate AAMI EC57 (N/SVEB/VEB) mapping report for MIT-BIH.

Usage:
  uv run -m scripts.report_aami_mitbih --data-dir data/raw/mitdb --out-dir reports/metadata

Outputs:
  - reports/metadata/mitbih_aami_report.json
  - reports/metadata/mitbih_aami_report.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.mitbih_download import get_logger
from src.data.mitbih_aami_report import build_and_export_aami_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MIT-BIH → AAMI EC57 (3-class) mapping + distribution report")
    p.add_argument("--data-dir", type=str, default="data/raw/mitdb", help="Directory containing MIT-BIH WFDB records")
    p.add_argument("--out-dir", type=str, default="reports/metadata", help="Where to save report JSON/CSV")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("mitbih.aami.cli")

    build_and_export_aami_report(
        local_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        logger=logger,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
