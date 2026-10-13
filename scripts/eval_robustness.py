"""Robustness of CV models under signal perturbations (OOF patients only).

    python scripts/eval_robustness.py --exp E16_alpha3 [--exp E17_aug ...]

For every (seed, fold) checkpoint in models/cv/{exp}/ the held-out fold is
re-scored under each perturbation, with the same decision rule as the CV
(config). Pooled metrics per seed -> mean [min–max], for the whole pool and
MIT-BIH only. Output: reports/robustness/{exp}.md

Perturbations (beat windows only; RR features are kept clean, see the caveat
in src/data/domain_shift_sim.py):
  clean        no change
  polar_sim    simulate_polar_h10_degradation (130 Hz round trip, baseline
               wander, motion artifacts, tiered contact noise)
  shift8       R-peak centring error: each beat shifted by U{-8..8} samples
               (±22 ms), edge-padded — the deployed detector does not centre
               beats exactly on the annotation
  polar+shift  both
  nstdb_*      REAL noise from the MIT-BIH Noise Stress Test DB (src/data/nstdb_noise.py):
               em / ma / bw at 12 dB, and a random mix of the three at 6 dB
Also reports the 3-seed ensemble (mean probability), which is what is deployed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.domain_shift_sim import simulate_polar_h10_degradation  # noqa: E402
from src.models.fusion import FusionClassifier  # noqa: E402
from src.training.cv_protocol import is_mitbih, load_dev_data, outer_folds, predict_logits, softmax  # noqa: E402
from src.training.decision import decide_from_config  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402

KEYS = ["f1_macro", "sveb_se", "sveb_ppv", "sveb_f1", "veb_se", "veb_ppv", "veb_f1"]


def shift_beats(X: np.ndarray, max_shift: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.integers(-max_shift, max_shift + 1, size=len(X))
    idx = np.clip(np.arange(X.shape[1])[None, :] - s[:, None], 0, X.shape[1] - 1)
    return np.take_along_axis(X, idx, axis=1).astype(np.float32)


_NSTDB = None
NSTDB_PERTURB = {"nstdb_em12": (("em",), 12.0), "nstdb_ma12": (("ma",), 12.0), "nstdb_bw12": (("bw",), 12.0),
                 "nstdb_all6": (("bw", "em", "ma"), 6.0)}


def perturb(name: str, X: np.ndarray, seed: int) -> np.ndarray:
    global _NSTDB
    if name == "clean":
        return X
    if name in NSTDB_PERTURB:
        from src.data.nstdb_noise import add_nstdb_noise, load_nstdb_noise

        if _NSTDB is None:
            _NSTDB = load_nstdb_noise()
        types, snr = NSTDB_PERTURB[name]
        return add_nstdb_noise(X, _NSTDB, types, snr, seed)
    if name == "polar_sim":
        return simulate_polar_h10_degradation(X, seed=seed)
    if name == "shift8":
        return shift_beats(X, 8, seed)
    if name == "polar+shift":
        return shift_beats(simulate_polar_h10_degradation(X, seed=seed), 8, seed + 1)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--perturb", nargs="+",
                    default=["clean", "polar_sim", "nstdb_em12", "nstdb_ma12", "nstdb_bw12", "nstdb_all6"])
    args = ap.parse_args()
    out_dir = Path("reports") / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for exp in args.exp:
        cfg = json.loads((Path("reports") / "cv" / exp / "config.json").read_text())
        data = load_dev_data(cfg)
        fold = outer_folds(data, int(cfg["cv"]["n_splits"]), int(cfg["cv"]["split_seed"]), cfg["cv"].get("fold_map"))
        mit = is_mitbih(data.pid)
        K = int(cfg["cv"]["n_splits"])
        # perturbed copies are shared by every model so all see the same noise
        Xp = {}
        for pname in args.perturb:
            Xp[pname] = np.empty_like(data.X_beats)
            for k in range(K):
                te = fold == k
                Xp[pname][te] = perturb(pname, data.X_beats[te], seed=1000 + k)
        res = {p: {"all": [], "mit": [], "ens_all": None, "ens_mit": None} for p in args.perturb}
        ens_prob = {p: np.zeros((len(data.y), 3), dtype=np.float64) for p in args.perturb}
        for s in cfg["seeds"]:
            prob = {p: np.zeros((len(data.y), 3), dtype=np.float32) for p in args.perturb}
            for k in range(K):
                te = fold == k
                m = FusionClassifier(dropout=float(cfg["model"]["dropout"]), use_rr=bool(cfg["model"]["use_rr"]))
                m.load_state_dict(torch.load(Path("models") / "cv" / exp / f"seed{s}_fold{k}.pt", map_location="cpu"))
                m.to(device)
                for p in args.perturb:
                    prob[p][te] = softmax(predict_logits(m, Xp[p][te], data.X_rr[te]))
            for p in args.perturb:
                ens_prob[p] += prob[p] / len(cfg["seeds"])
                pred = decide_from_config(prob[p], cfg["decision"])
                res[p]["all"].append(class_metrics(data.y, pred))
                res[p]["mit"].append(class_metrics(data.y[mit], pred[mit]))
            print(f"[{exp}] seed {s} done", flush=True)
        for p in args.perturb:
            pred = decide_from_config(ens_prob[p], cfg["decision"])
            res[p]["ens_all"] = class_metrics(data.y, pred)
            res[p]["ens_mit"] = class_metrics(data.y[mit], pred[mit])

        lines = [f"# Robustez — {exp}", "", f"Regla: {cfg['decision']}. Pacientes OOF (cada modelo evalúa solo su fold).", ""]
        for view in ("all", "mit"):
            lines += [f"## Vista: {'pool completo' if view == 'all' else 'solo MIT-BIH'}", "",
                      "| perturbación | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
            base = {k: np.mean([r[k] for r in res["clean"][view]]) for k in KEYS} if "clean" in res else None
            for p in args.perturb:
                rs = res[p][view]
                cells = []
                for k in KEYS:
                    v = [r[k] for r in rs]
                    d = f" (Δ{np.mean(v) - base[k]:+.3f})" if base and p != "clean" else ""
                    cells.append(f"{np.mean(v):.3f} [{min(v):.3f}–{max(v):.3f}]{d}")
                lines.append(f"| {p} | " + " | ".join(cells) + " |")
            lines += ["", f"Ensamble de {len(cfg['seeds'])} semillas (lo que se despliega):", "",
                      "| perturbación | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
            for p in args.perturb:
                e = res[p][f"ens_{view}"]
                lines.append(f"| {p} | " + " | ".join(f"{e[k]:.3f}" for k in KEYS) + " |")
            lines.append("")
        (out_dir / f"{exp}.md").write_text("\n".join(lines))
        (out_dir / f"{exp}.json").write_text(json.dumps(res, default=float))
        print("\n".join(lines))


if __name__ == "__main__":
    main()
