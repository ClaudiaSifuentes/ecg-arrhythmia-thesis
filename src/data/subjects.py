"""Record -> subject mapping for patient-wise splits.

Splitting by record is NOT splitting by patient when one subject contributes
several records:
- MIT-BIH: records 201 and 202 come from the same subject (MIT-BIH directory).
- INCART: 75 records extracted from 32 Holter recordings (2–3 records per
  subject). The mapping below must be filled from the record headers
  (age / sex / diagnoses); records not listed fall back to their own id.

All CV splitting (outer folds and inner validation) groups by subject.
"""

from __future__ import annotations

import numpy as np

SUBJECT_OF_RECORD: dict[str, str] = {
    "mit_202": "mit_201",
    "202": "201",
}


def subject_ids(patient_id: np.ndarray) -> np.ndarray:
    pid = np.asarray(patient_id).astype(str)
    return np.array([SUBJECT_OF_RECORD.get(p, p) for p in pid])
