"""Literature-comparable evaluation: de Chazal inter-patient split DS1 -> DS2.

    python scripts/eval_ds2.py --config configs/experiments/E16_alpha3.yaml [--mit_only] [--hybrid gbm]

Protocol (de Chazal et al. 2004; also Llamedo 2011, Zhang 2014, Mondéjar 2019)
- DS1 (train, 22 records) and DS2 (test, 22 records) of MIT-BIH; paced records
  102, 104, 107, 217 excluded.
- Training: DS1 MIT-BIH records (+ every non-MIT source of the config's data,
  e.g. INCART/SVDB, unless --mit_only). Early stopping on an inner validation
  of DS1 *subjects* (never DS2).
- Decision: argmax, and an SVEB threshold chosen on that DS1 inner validation
  (max SVEB F1). DS2 is only reported.
- Reported on DS2: Se / PPV / F1 for SVEB and VEB, also without record 232
  (≈75 % of DS2 SVEB beats), per seed and for the 3-seed ensemble.

--hybrid gbm|logreg (E21): also report CNN + RR-only model. The RR model is
trained on the same DS1 training beats as the CNN; the mixing weight w is
chosen on the DS1 inner validation (max macro-F1), like the threshold.

Known caveats (stated in the report):
- This project maps F (fusion) beats to VEB; most papers keep F separate, which
  shifts VEB Se/PPV slightly (DS2 has ≈390 F beats, mostly record 213).
- Records 201 (DS1) and 202 (DS2) are the same subject; kept as in the
  literature so the numbers stay comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.rr_context import build_rr_features  # noqa: E402
from src.data.subjects import subject_ids  # noqa: E402
from src.data.torch_datasets import load_processed_arrays  # noqa: E402
from src.models.fusion import FusionClassifier  # noqa: E402
from src.training.cv_protocol import inner_split, is_mitbih, load_config, predict_logits, softmax, train_model  # noqa: E402
from src.training.decision import decide  # noqa: E402
from src.training.metrics import class_metrics  # noqa: E402

DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
KEYS = ["f1_macro", "sveb_se", "sveb_ppv", "sveb_f1", "veb_se", "veb_ppv", "veb_f1"]
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mit_only", action="store_true", help="train on MIT-BIH DS1 only (as the literature)")
    ap.add_argument("--hybrid", choices=["gbm", "logreg"], default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tag = cfg["exp_id"] + ("_ds1mit" if args.mit_only else "_ds1ext")
    model_tag = tag
    if args.hybrid:
        tag += f"_hyb{args.hybrid}"
    out_dir = Path("models") / "ds2" / model_tag  # CNN weights shared with the non-hybrid run
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = load_processed_arrays(cfg["data_dir"])
    pid = arr.patient_id.astype(str)
    mit = is_mitbih(pid)
    rec = np.char.replace(pid, "mit_", "")
    Xr = build_rr_features(arr.X_rr, arr.patient_id, cfg["features"]["rr_set"])
    tr_pool = (mit & np.isin(rec, DS1)) | (~mit & (not args.mit_only))
    te = mit & np.isin(rec, DS2)
    assert set(rec[mit & np.isin(rec, DS1)]) == set(DS1), "DS1 records missing from data"
    assert set(rec[te]) == set(DS2), "DS2 records missing from data"

    # inner validation: DS1 MIT subjects only (representative of the target), grouped by subject
    idx_pool = np.nonzero(tr_pool)[0]
    ds1_mit = idx_pool[mit[idx_pool]]
    subj = subject_ids(pid)
    itr, iva = inner_split(arr.y[ds1_mit], subj[ds1_mit], 5, int(cfg["cv"]["split_seed"]) + 77)
    va = ds1_mit[iva]
    tr = np.setdiff1d(idx_pool, va)
    assert set(subj[tr]).isdisjoint(set(subj[va])) and set(rec[va]).isdisjoint(set(DS2))
    print(f"[{tag}] train {len(tr)} beats ({len(set(pid[tr]))} records) | inner-val {len(va)} "
          f"({sorted(set(rec[va]))}) | DS2 {te.sum()} beats, SVEB {int((arr.y[te] == 1).sum())}, "
          f"VEB {int((arr.y[te] == 2).sum())}", flush=True)

    probs_te, probs_va = [], []
    for s in cfg["seeds"]:
        ck = out_dir / f"seed{s}.pt"
        if ck.exists():
            m = FusionClassifier(dropout=float(cfg["model"]["dropout"]), use_rr=bool(cfg["model"]["use_rr"]))
            m.load_state_dict(torch.load(ck, map_location="cpu"))
        else:
            m, info = train_model(cfg, arr.X_beats[tr], Xr[tr], arr.y[tr], arr.X_beats[va], Xr[va], arr.y[va],
                                  seed=int(s) + 500, log=lambda x: None)
            torch.save(m.state_dict(), ck)
            print(f"[{tag}] seed {s}: best_epoch {info['best_epoch']}", flush=True)
        probs_te.append(softmax(predict_logits(m, arr.X_beats[te], Xr[te])))
        probs_va.append(softmax(predict_logits(m, arr.X_beats[va], Xr[va])))

    y_te, y_va = arr.y[te], arr.y[va]
    keep = rec[te] != "232"
    rows = []

    def add(name, p_te, p_va):
        t = max(GRID, key=lambda t: class_metrics(y_va, decide(p_va, float(t)))["sveb_f1"])
        for rule, tt in (("argmax", None), (f"t_SVEB={t:.2f} (DS1)", float(t))):
            pred = decide(p_te, tt)
            rows.append((name, rule, "DS2", class_metrics(y_te, pred)))
            rows.append((name, rule, "DS2 sin 232", class_metrics(y_te[keep], pred[keep])))

    for s, pt, pv in zip(cfg["seeds"], probs_te, probs_va):
        add(f"seed{s}", pt, pv)
    add("ensamble", np.mean(probs_te, 0), np.mean(probs_va, 0))
    if args.hybrid:
        from src.models.rr_model import choose_w, make_rr_model

        rr = make_rr_model(args.hybrid, cfg["loss"].get("alpha")).fit(Xr[tr], arr.y[tr])
        r_te, r_va = rr.predict_proba(Xr[te]), rr.predict_proba(Xr[va])
        add(f"solo RR-{args.hybrid}", r_te, r_va)
        for s, pt, pv in zip(cfg["seeds"], probs_te, probs_va):
            w = choose_w(y_va, pv, r_va)
            add(f"seed{s}+RR (w={w:.2f})", (1 - w) * pt + w * r_te, (1 - w) * pv + w * r_va)
        pt, pv = np.mean(probs_te, 0), np.mean(probs_va, 0)
        w = choose_w(y_va, pv, r_va)
        add(f"ensamble+RR (w={w:.2f})", (1 - w) * pt + w * r_te, (1 - w) * pv + w * r_va)

    lines = [f"# DS1 → DS2 (de Chazal) — {tag}", "",
             f"Entrenamiento: {'solo MIT-BIH DS1' if args.mit_only else 'MIT-BIH DS1 + fuentes externas del config'}"
             f" (`{cfg['data_dir']}`). Umbral SVEB elegido en validación interna de DS1. F mapeado a VEB.", "",
             "| modelo | regla | conjunto | " + " | ".join(KEYS) + " |", "|---|---|---|" + "---|" * len(KEYS)]
    for name, rule, subset, m in rows:
        lines.append(f"| {name} | {rule} | {subset} | " + " | ".join(f"{m[k]:.3f}" for k in KEYS) + " |")
    lines += ["", "Referencia literatura (DS2, SVEB Se/PPV/F1 · VEB Se/PPV): de Chazal 2004 75.9/38.5/0.51 · 77.7/81.9; "
              "Llamedo 2011 76.5/41.3/0.53 · 82.9/88.0; Zhang 2014 79.1/36.0/0.495 · 85.5/92.8."]
    out = Path("reports") / "ds2" / f"{tag}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps([(a, b, c, d) for a, b, c, d in rows], default=float))
    np.savez_compressed(out.with_suffix(".npz"), y=y_te, record=rec[te], prob=np.stack(probs_te),
                        val_y=y_va, val_prob=np.stack(probs_va))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
