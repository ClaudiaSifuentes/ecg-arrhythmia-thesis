"""AAMI EC57 mapping utilities for MIT-BIH beat annotations.

Goal
- Map MIT-BIH annotation symbols (WFDB `ann.symbol`) into the simplified AAMI EC57
  classes used in this thesis:
    0 = N (Normal)
    1 = SVEB (Supraventricular ectopic beat)
    2 = VEB (Ventricular ectopic beat)

Notes
- MIT-BIH includes many annotation symbols that are not beats or not used for
  beat-level classification (e.g., '+', '~', '|', '"', '/', 'f', etc.).
- For L1 beat classification we typically keep only beat annotations.

This module is intentionally explicit and conservative (clinically safer):
- Only maps symbols we are confident are beats.
- Everything else becomes "unknown" and can be filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Final labels used by the project
AAMI_N = 0
AAMI_SVEB = 1
AAMI_VEB = 2


@dataclass(frozen=True)
class AamiMappingResult:
    """Holds mapping results for a list of symbols."""

    y: List[Optional[int]]
    mapped: int
    unknown: int


# --- Symbol sets ---
# Based on common MIT-BIH to AAMI groupings used in literature.
# Keep it minimal for your 3-class setting (N, SVEB, VEB).

# N group (normal / bundle branch block / paced / fusion of paced+normal)
N_SYMBOLS = {
    "N",  # Normal beat
    "L",  # Left bundle branch block
    "R",  # Right bundle branch block
    "e",  # Atrial escape beat (often grouped as N in some setups)
    "j",  # Nodal (junctional) escape beat
    # NOTE: '/' (paced) and 'f' (fusion of paced+normal) are intentionally excluded
    # from L1 mapping in this project so they fall into "unknown" and can be filtered out.
}

# SVEB group
SVEB_SYMBOLS = {
    "A",  # Atrial premature
    "a",  # Aberrated atrial premature
    "J",  # Nodal (junctional) premature
    "S",  # Supraventricular premature
}

# VEB group
VEB_SYMBOLS = {
    "V",  # Premature ventricular contraction
    "E",  # Ventricular escape beat
    "F",  # Fusion of ventricular and normal beat (include as VEB per project decision)
}


def map_symbol_to_aami_class(symbol: str) -> Optional[int]:
    """Map a single MIT-BIH symbol to {0,1,2} or None if not mapped."""

    if symbol in N_SYMBOLS:
        return AAMI_N
    if symbol in SVEB_SYMBOLS:
        return AAMI_SVEB
    if symbol in VEB_SYMBOLS:
        return AAMI_VEB
    return None


def map_symbols_to_aami(symbols: Sequence[str]) -> AamiMappingResult:
    """Map a list of annotation symbols to AAMI classes.

    Returns
    - y: list of class ids or None for unknown symbols
    - mapped / unknown: counts
    """

    y: List[Optional[int]] = []
    mapped = 0
    unknown = 0

    for s in symbols:
        c = map_symbol_to_aami_class(str(s))
        y.append(c)
        if c is None:
            unknown += 1
        else:
            mapped += 1

    return AamiMappingResult(y=y, mapped=mapped, unknown=unknown)


def get_mapping_spec() -> Dict[str, Dict[str, List[str]]]:
    """Return the current mapping spec (useful to log/export into reports)."""

    return {
        "labels": {
            "0": "N",
            "1": "SVEB",
            "2": "VEB",
        },
        "groups": {
            "N": sorted(N_SYMBOLS),
            "SVEB": sorted(SVEB_SYMBOLS),
            "VEB": sorted(VEB_SYMBOLS),
        },
    }
