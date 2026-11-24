"""Ladder of compile targets: which target is best at 10 / 100 / 1,000 (and 30 / 300 / 3,000)
teacher-labelled examples?

For each seed, the training stream is the CLINC150 train split in a seeded random order,
labelled by the teacher (read from the cache, so no API calls). The first N examples train
each target (with temperature scaling fitted on cross-fitted logits). Every compiled
student is evaluated on the full test split (5,500 queries) against GROUND TRUTH, and
also against the teacher's labels (agreement). Gated metrics simulate the confidence gate
with the teacher as fallback, on test and on the val split; the gate threshold tau is chosen
on val (see analyze.py) so test numbers stay unbiased.

One JSON row per (target, seed, N) is appended to runs/ladder.jsonl; finished rows are
skipped on rerun. Confidence/correctness arrays for seed 0 are saved for risk-coverage plots,
and every fitted model is pickled to runs/models/{target}_s{seed}_n{N}.pkl (load with
pickle.load; the result has .predict(texts) and .T).

Run: python -m jit.ladder [--targets logreg,mlp_head] [--seeds 0,1,2] [--sizes 10,100,1000]
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time

import numpy as np

from jit.data import OOS, load_split
from jit.metrics import ece, gated_metrics, risk_coverage
from jit.runtime import TeacherTable
from jit.students import TARGETS, Calibrated
from jit.teacher_cache import RUNS_DIR

OUT = RUNS_DIR / "ladder.jsonl"
CURVES = RUNS_DIR / "ladder_curves"
MODELS = RUNS_DIR / "models"  # one pickled Calibrated model per (target, seed, N)
TAUS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
DEFAULT_SIZES = [10, 30, 100, 300, 1000, 3000]


def make_factory(name: str):
    if name == "lora":
        from jit.lora import LoRA

        return LoRA, 3  # fewer calibration folds: each fold is a full fine-tune
    return TARGETS[name], 5


def stream(seed: int, teacher: TeacherTable) -> tuple[list[str], list[str]]:
    texts = sorted({ex.text.strip() for ex in load_split("train")})
    random.Random(seed).shuffle(texts)
    texts = [t for t in texts if t in teacher]
    return texts, [teacher.table[t] for t in texts]


def done_keys() -> set:
    if not OUT.exists():
        return set()
    return {(r["target"], r["seed"], r["n"]) for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="exact_cache,semantic_cache,logreg,mlp_head")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    ap.add_argument("--out", default=None, help="output jsonl (default runs/ladder.jsonl)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = RUNS_DIR / a.out
    targets = a.targets.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    sizes = [int(s) for s in a.sizes.split(",")]

    teacher = TeacherTable()
    test = [ex for ex in load_split("test") if ex.text.strip() in teacher]
    xt = [ex.text.strip() for ex in test]
    truth = np.array([ex.label for ex in test], dtype=object)
    tlab = np.array([teacher.table[t] for t in xt], dtype=object)
    is_oos = truth == OOS
    print(f"test={len(test)} (teacher acc vs truth {np.mean(tlab == truth):.4f})", flush=True)
    # val split: used only to choose the gate threshold tau (never to train or calibrate)
    val = [ex for ex in load_split("val") if ex.text.strip() in teacher]
    xv = [ex.text.strip() for ex in val]
    vtruth = np.array([ex.label for ex in val], dtype=object)
    vtlab = np.array([teacher.table[t] for t in xv], dtype=object)
    v_oos = vtruth == OOS
    print(f"val={len(val)} (teacher acc vs truth {np.mean(vtlab == vtruth):.4f})", flush=True)
    CURVES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    skip = done_keys()

    for seed in seeds:
        texts, labs = stream(seed, teacher)
        for n in sizes:
            for name in targets:
                if (name, seed, n) in skip:
                    continue
                factory, folds = make_factory(name)
                t0 = time.perf_counter()
                model = Calibrated(factory, n_folds=folds, seed=seed).fit(texts[:n], labs[:n])
                fit_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                pred, conf = model.predict(xt)
                pred_ms = (time.perf_counter() - t1) / len(xt) * 1000
                pred = np.array(pred, dtype=object)
                vpred, vconf = model.predict(xv)
                vpred = np.array(vpred, dtype=object)
                corr = pred == truth
                agree = pred == tlab
                _, _, aurc = risk_coverage(conf, corr)
                row = {
                    "target": name, "seed": seed, "n": n,
                    "classes_covered": len(set(labs[:n])), "class_coverage": len(set(labs[:n])) / 151,
                    "acc": float(corr.mean()), "acc_in_scope": float(corr[~is_oos].mean()),
                    "agree_teacher": float(agree.mean()),
                    "ece": ece(conf, corr), "ece_vs_teacher": ece(conf, agree),
                    "mean_conf": float(conf.mean()), "aurc": aurc,
                    "oos_recall": float((pred[is_oos] == OOS).mean()),
                    "T": model.T, "n_calib": model.n_calib,
                    "fit_s": fit_s, "predict_ms": pred_ms,
                    "gated": [gated_metrics(pred, conf, tlab, truth, tau) for tau in TAUS],
                    # val gate metrics, overall and split by in-scope / out-of-scope, for choosing tau
                    "gated_val": [gated_metrics(vpred, vconf, vtlab, vtruth, tau) for tau in TAUS],
                    "gated_val_ins": [gated_metrics(vpred[~v_oos], vconf[~v_oos], vtlab[~v_oos], vtruth[~v_oos], tau)
                                      for tau in TAUS],
                    "gated_val_oos": [gated_metrics(vpred[v_oos], vconf[v_oos], vtlab[v_oos], vtruth[v_oos], tau)
                                      for tau in TAUS],
                    "gated_ins": [gated_metrics(pred[~is_oos], conf[~is_oos], tlab[~is_oos], truth[~is_oos], tau)
                                  for tau in TAUS],
                    "gated_no_oos_rule": [gated_metrics(pred, conf, tlab, truth, tau, escalate_oos=False) for tau in TAUS],
                }
                with (MODELS / f"{name}_s{seed}_n{n}.pkl").open("wb") as f:
                    pickle.dump(model, f)
                with OUT.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if seed == 0:
                    np.savez(CURVES / f"{name}_n{n}.npz", conf=conf, correct=corr, agree=agree, oos=is_oos)
                g8 = row["gated"][TAUS.index(0.8)]
                print(f"seed={seed} n={n:5d} {name:15s} acc={row['acc']:.3f} ece={row['ece']:.3f} "
                      f"T={model.T:.2f} cls={row['classes_covered']:3d} | tau=.8 cov={g8['coverage']:.2f} "
                      f"sys={g8['system_accuracy']:.3f}  fit={fit_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
