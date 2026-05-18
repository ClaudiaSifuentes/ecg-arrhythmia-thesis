"""Build MIT-BIH (mitdb) dataset index (per-record metadata).

Usage:
  uv run -m scripts.index_mitbih --data-dir data/raw/mitdb --out-dir reports/metadata

Outputs:
  - mitbih_index.json
  - mitbih_index.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.mitbih_download import get_logger
from src.data.mitbih_index import build_and_export_dataset_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index MIT-BIH dataset (per-record metadata + annotation symbol distribution).")
    p.add_argument("--data-dir", type=str, default="data/raw/mitdb", help="Directory containing WFDB MIT-BIH records.")
    p.add_argument("--out-dir", type=str, default="reports/metadata", help="Where to export JSON/CSV index reports.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("mitbih.index.cli")

    build_and_export_dataset_index(
        local_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        logger=logger,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
