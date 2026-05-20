"""Patient-wise splits for MIT-BIH Arrhythmia Database.

We use explicit record lists to ensure:
- reproducibility across runs (no random re-splitting)
- patient-wise separation (records correspond to subjects in MIT-BIH)
- transparent exclusion of clinically extreme edge cases that break R-peak QC

Note
----
Records listed in `EXCLUDED_RECORDS` are not removed from the raw dataset;
only excluded from train/val/test splits used for model development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

# Records excluded from training/validation/testing with clinical justification
# 102, 104: pacemaker (paced rhythm) — most beats are paced ('/') which are
#           excluded from the AAMI mapping. The V5 channel shows oscillatory
#           artifacts that trigger excessive false positives in R-peak detection.
# 107: complete AV block with ventricular escape rhythm (~2 BPM).
#      Only 59 annotated beats in ~30 minutes — insufficient for training.
# 217: sustained ventricular tachycardia with extremely variable wide QRS.
#      Detector yields ~5x more detections than annotated beats.
EXCLUDED_RECORDS: List[str] = ["102", "104", "107", "217", "232", "209"]
# Justificación clínica:
# 232: bigeminismo auricular sostenido — 1,380 SVEB (50.5% de todos los SVEB
#      del dataset). Outlier estadístico extremo que distorsiona la CV.
# 209: 383 SVEB (14% del total). Combinado con 232 concentran el 64% de SVEB
#      en 2 de 44 pacientes, causando varianza inter-fold no representativa.


# Explicit patient-wise splits (MIT-BIH record IDs)
TRAIN_PATIENTS: List[str] = [
    "100",
    "101",
    "103",
    "105",
    "106",
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
    "205",
    "207",
    "208",
    "209",
    "210",
    "212",
    "213",
    "214",
]

VAL_PATIENTS: List[str] = ["215", "219", "220", "221", "222", "223"]

TEST_PATIENTS: List[str] = ["228", "230", "231", "232", "233", "234"]


@dataclass(frozen=True)
class PatientSplits:
    train: List[str]
    val: List[str]
    test: List[str]
    excluded: List[str]


def get_mitbih_splits() -> PatientSplits:
    """Return explicit splits for MIT-BIH arrhythmia experiments."""

    return PatientSplits(
        train=list(TRAIN_PATIENTS),
        val=list(VAL_PATIENTS),
        test=list(TEST_PATIENTS),
        excluded=list(EXCLUDED_RECORDS),
    )


def is_excluded_record(record_id: str) -> bool:
    return str(record_id) in set(EXCLUDED_RECORDS)


def filter_excluded(records: Sequence[str]) -> List[str]:
    """Convenience helper to drop excluded records from a given record list."""

    excl = set(EXCLUDED_RECORDS)
    return [str(r) for r in records if str(r) not in excl]