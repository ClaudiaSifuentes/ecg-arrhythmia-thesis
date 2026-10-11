"""Record -> subject mapping for patient-wise splits.

Splitting by record is NOT splitting by patient when one subject contributes
several records:
- MIT-BIH: records 201 and 202 come from the same subject (MIT-BIH directory).
- INCART: 75 records extracted from 32 Holter recordings (2–3 records per
  subject). The mapping is read from configs/incart_subjects.json, produced by
  scripts/incart_subjects.py from the record headers. Records without a
  mapping fall back to their own id (and a warning is printed once).

All CV splitting (outer folds and inner validation) groups by subject.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SUBJECT_OF_RECORD: dict[str, str] = {
    "mit_202": "mit_201",
    "202": "201",
}

INCART_MAP_PATH = Path(__file__).resolve().parents[2] / "configs" / "incart_subjects.json"


def _load_map() -> dict[str, str]:
    m = dict(SUBJECT_OF_RECORD)
    if INCART_MAP_PATH.exists():
        m.update(json.loads(INCART_MAP_PATH.read_text())["subject_of_record"])
    return m


def subject_ids(patient_id: np.ndarray) -> np.ndarray:
    pid = np.asarray(patient_id).astype(str)
    m = _load_map()
    if np.char.startswith(pid, "inc_").any() and not INCART_MAP_PATH.exists():
        warnings.warn("configs/incart_subjects.json missing: INCART records grouped by record, "
                      "not by subject (run scripts/incart_subjects.py)")
    return np.array([m.get(p, p) for p in pid])
