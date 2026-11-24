"""Markdown tables for the report: ladder summary, target-selection rule, gate operating points.

Run `python -m jit.analyze` from the code/ folder; writes runs/tables.md.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from jit.teacher_cache import RUNS_DIR

LADDER_FILES = ("ladder.jsonl", "ladder_lora_v2.jsonl")
OOS_PRIOR = 1000 / 5500  # expected out-of-scope share in deployment (= test); val has only 100 / 3,100
TOL = 0.005  # "matches always-LLM": system accuracy within 0.5 points of it
NAMES = {"exact_cache": "Exact cache", "semantic_cache": "Semantic cache (kNN)", "logreg": "Logistic regression",
         "mlp_head": "MLP head", "lora": "LoRA"}


def load_ladder() -> list[dict]:
    rows = []
    for name in LADDER_FILES:
        p = RUNS_DIR / name
        if p.exists():
            rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    return rows


def teacher_val_acc() -> tuple[float, float, float]:
    """Teacher accuracy on val: overall, in-scope, out-of-scope."""
    from jit.data import OOS, load_split
    from jit.runtime import TeacherTable

    tt = TeacherTable().table
    val = [ex for ex in load_split("val") if ex.text.strip() in tt]
    ok = np.array([tt[ex.text.strip()] == ex.label for ex in val])
    oos = np.array([ex.label == OOS for ex in val])
    return float(ok.mean()), float(ok[~oos].mean()), float(ok[oos].mean())


def choose_tau(r: dict, ref_val: tuple[float, float, float], reweight: bool) -> int | None:
    """Index of the tau with the largest val coverage whose val system accuracy stays within
    TOL of the teacher's val accuracy; None if no tau qualifies (always escalate).
    reweight=True scores val as if OOS_PRIOR of the traffic were out of scope."""
    best = None
    for i in range(len(r["gated_val"])):
        if reweight:
            gi, go = r["gated_val_ins"][i], r["gated_val_oos"][i]
            acc = (1 - OOS_PRIOR) * gi["system_accuracy"] + OOS_PRIOR * go["system_accuracy"]
            cov = (1 - OOS_PRIOR) * gi["coverage"] + OOS_PRIOR * go["coverage"]
            ref = (1 - OOS_PRIOR) * ref_val[1] + OOS_PRIOR * ref_val[2]
        else:
            acc, cov, ref = r["gated_val"][i]["system_accuracy"], r["gated_val"][i]["coverage"], ref_val[0]
        if acc >= ref - TOL and cov > 0 and (best is None or cov > best[0]):
            best = (cov, i)
    return None if best is None else best[1]


def main() -> None:
    rows = load_ladder()
    summ = json.loads((RUNS_DIR / "summary.json").read_text())
    ref = summ["teacher_test_accuracy"]
    by = defaultdict(list)
    for r in rows:
        by[(r["target"], r["n"])].append(r)
    sizes = sorted({r["n"] for r in rows})
    targets = [t for t in NAMES if any(k[0] == t for k in by)]
    out = []

    out.append("### Accuracy vs ground truth (test, 5,500 queries), mean ± std over seeds\n")
    out.append("| N | labels seen | " + " | ".join(NAMES[t] for t in targets) + " |")
    out.append("|---|---|" + "---|" * len(targets))
    for n in sizes:
        cls = np.mean([r["classes_covered"] for t in targets for r in by.get((t, n), [])])
        cells = []
        best = max((np.mean([r["acc"] for r in by[(t, n)]]), t) for t in targets if (t, n) in by)[1]
        for t in targets:
            rs = by.get((t, n))
            if not rs:
                cells.append("–")
                continue
            m, s = np.mean([r["acc"] for r in rs]), np.std([r["acc"] for r in rs])
            cell = f"{m:.1%} ± {s:.1%}"
            cells.append(f"**{cell}**" if t == best else cell)
        out.append(f"| {n} | {cls:.0f} / 151 | " + " | ".join(cells) + " |")

    out.append("\n### ECE (15 bins, after temperature scaling), mean over seeds\n")
    out.append("| N | " + " | ".join(NAMES[t] for t in targets) + " |")
    out.append("|---|" + "---|" * len(targets))
    for n in sizes:
        out.append(f"| {n} | " + " | ".join(
            f"{np.mean([r['ece'] for r in by[(t, n)]]):.3f}" if (t, n) in by else "–" for t in targets) + " |")

    out.append(f"\n### Oracle: LLM calls avoided within {TOL:.1%} of always-LLM accuracy ({ref:.1%}), τ chosen on test\n")
    out.append("Upper bound only: τ is picked per (target, N) by looking at test results. The val-chosen "
               "tables below are the honest numbers.\n")
    out.append("| N | " + " | ".join(NAMES[t] for t in targets) + " | best target |")
    out.append("|---|" + "---|" * len(targets) + "---|")
    rule = []
    for n in sizes:
        cells, best = [], (0.0, "always-LLM", None)
        for t in targets:
            rs = by.get((t, n))
            if not rs:
                cells.append("–")
                continue
            taus = [g["tau"] for g in rs[0]["gated"]]
            ok = []
            for i, tau in enumerate(taus):
                cov = np.mean([r["gated"][i]["coverage"] for r in rs])
                acc = np.mean([r["gated"][i]["system_accuracy"] for r in rs])
                if acc >= ref - TOL:
                    ok.append((cov, tau, acc))
            if ok:
                cov, tau, acc = max(ok)
                cells.append(f"{cov:.0%} (τ={tau})" if cov >= 0.005 else "0%")
                if cov >= 0.005 and cov > best[0]:
                    best = (cov, NAMES[t], tau)
            else:
                cells.append("0%")
        out.append(f"| {n} | " + " | ".join(cells) + f" | {best[1]} |")
        rule.append((n, best))

    # honest version: tau chosen on val, evaluated on test
    if rows and all("gated_val" in r for r in rows):
        ref_val = teacher_val_acc()
        for reweight in (False, True):
            how = (f"val reweighted to {OOS_PRIOR:.0%} out-of-scope" if reweight
                   else f"val as is: teacher {ref_val[0]:.1%}, 3% out-of-scope")
            out.append(f"\n### Gate with τ chosen on val ({how}), evaluated on test\n")
            out.append("Per seed, τ is the threshold with the most val coverage whose val system accuracy stays "
                       "within 0.5 points of the teacher's val accuracy; if none qualifies, always escalate. "
                       "Cells: test LLM calls avoided / test system accuracy minus always-LLM, in points / "
                       "seeds whose test drop exceeds 0.5 points.\n")
            out.append("| N | " + " | ".join(NAMES[t] for t in targets) + " | best target (mean drop ≤ 0.5) |")
            out.append("|---|" + "---|" * len(targets) + "---|")
            for n in sizes:
                cells, best = [], (0.0, "always-LLM")
                for t in targets:
                    rs = by.get((t, n))
                    if not rs:
                        cells.append("–")
                        continue
                    covs, drops = [], []
                    for r in rs:
                        i = choose_tau(r, ref_val, reweight)
                        g = None if i is None else r["gated"][i]
                        covs.append(0.0 if g is None else g["coverage"])
                        drops.append(0.0 if g is None else g["system_accuracy"] - ref)
                    cov, d = float(np.mean(covs)), float(np.mean(drops))
                    bad = sum(x < -TOL for x in drops)
                    cells.append(f"{cov:.0%} / {d * 100:+.1f} / {bad} of {len(rs)}")
                    if cov >= 0.005 and cov > best[0] and d >= -TOL:
                        best = (cov, NAMES[t])
                out.append(f"| {n} | " + " | ".join(cells) + f" | {best[1]} |")

    out.append("\n### OOS recall of the student (predicts `oos` on true out-of-scope queries)\n")
    out.append("| N | " + " | ".join(NAMES[t] for t in targets) + " |")
    out.append("|---|" + "---|" * len(targets))
    for n in sizes:
        out.append(f"| {n} | " + " | ".join(
            f"{np.mean([r['oos_recall'] for r in by[(t, n)]]):.0%}" if (t, n) in by else "–" for t in targets) + " |")
    out.append("\nWith the gate's escalate-oos rule, out-of-scope queries the student labels `oos` go to the "
               "teacher; the column `oos_served_wrong` in runs/ladder.jsonl is the share of true OOS queries "
               "the student answered with an in-scope label.\n")
    # oos served wrong at tau=0.8
    out.append("| N | " + " | ".join(NAMES[t] for t in targets) + " |")
    out.append("|---|" + "---|" * len(targets))
    for n in sizes:
        cells = []
        for t in targets:
            rs = by.get((t, n))
            if not rs:
                cells.append("–")
                continue
            i = [g["tau"] for g in rs[0]["gated"]].index(0.8)
            cells.append(f"{np.mean([r['gated'][i]['oos_served_wrong'] for r in rs]):.1%}")
        out.append(f"| {n} (τ=0.8, OOS served wrong) | " + " | ".join(cells) + " |")

    out.append("\n### Mean compile time incl. calibration (s, CPU)\n")
    out.append("| N | " + " | ".join(NAMES[t] for t in targets) + " |")
    out.append("|---|" + "---|" * len(targets))
    for n in sizes:
        out.append(f"| {n} | " + " | ".join(
            f"{np.mean([r['fit_s'] for r in by[(t, n)]]):.1f}" if (t, n) in by else "–" for t in targets) + " |")

    streams = sorted(RUNS_DIR.glob("stream_*.json"))
    if streams:
        out.append("\n### Stream demo (6,000 calls, recompile every 250 teacher labels, no drift)\n")
        out.append("| target | τ | accuracy | always-LLM | LLM calls | cost (list) | always-LLM cost | N* | compile total |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for f in streams:
            s = json.loads(f.read_text())
            out.append(f"| {NAMES.get(s['target'], s['target'])} | {s['tau']} | {s['overall_accuracy']:.1%} | "
                       f"{s['always_llm_accuracy']:.1%} | {s['teacher_calls']} ({s['teacher_calls'] / s['n']:.0%}) | "
                       f"${s['total_cost_jit']:.2f} | ${s['total_cost_llm']:.2f} | {s['breakeven_step']} | "
                       f"{s['total_compile_s']:.0f} s |")

    (RUNS_DIR / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
