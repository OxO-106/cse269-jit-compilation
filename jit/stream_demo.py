"""Run the JIT runtime on a time-ordered stream and record what happens over time.

The stream is the CLINC150 train split in a seeded random order (no drift: drift
injection is Shuchang's part). The teacher is read from the cache (TeacherTable), so the
run is free and reproducible; each teacher call is charged at the mean list-price cost
per call from runs/teacher_calls.jsonl. Compile time is charged at COMPUTE_USD_PER_HOUR
(an assumed on-demand CPU instance price) so cumulative cost includes "training".

Writes runs/stream_<target>_tau<tau>.json with per-window accuracy (vs ground truth),
escalation rate, cumulative cost of JIT vs always-LLM, and the break-even point N*.

Run: python -m jit.stream_demo [--target logreg] [--tau 0.8] [--n 6000] [--every 250]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics

from jit.data import load_split
from jit.runtime import JITFunction, TeacherTable
from jit.teacher_cache import LOG_PATH, RUNS_DIR

COMPUTE_USD_PER_HOUR = 1.0  # assumption: a 16-32 vCPU on-demand instance
WINDOW = 250


def teacher_cost_per_call() -> float:
    costs = [r["cost_usd"] for r in map(json.loads, LOG_PATH.read_text(encoding="utf-8").splitlines())
             if not r["cache_hit"] and r["cost_usd"] is not None and r["label"] is not None]
    return statistics.mean(costs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="logreg")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    teacher = TeacherTable()
    truth = {ex.text.strip(): ex.label for ex in load_split("train")}
    qs = [q for q in sorted(truth) if q in teacher]
    random.Random(a.seed).shuffle(qs)
    qs = qs[: a.n]
    cpc = teacher_cost_per_call()

    jit = JITFunction(teacher, target=a.target, tau=a.tau, recompile_every=a.every)
    for q in qs:
        jit(q)

    windows = []
    cum_jit = cum_llm = 0.0
    compile_by_step = {}
    for c in jit.compiles:
        compile_by_step[c["step"]] = compile_by_step.get(c["step"], 0) + c["compile_s"]
    breakeven = None
    cum = []
    for d in jit.log:
        cum_llm += cpc
        cum_jit += (cpc if d.source == "teacher" else 0.0) + compile_by_step.get(d.step, 0) / 3600 * COMPUTE_USD_PER_HOUR
        cum.append((cum_jit, cum_llm))
    # N*: first step after which JIT stays cheaper than always-LLM
    for i in range(len(cum) - 1, -1, -1):
        if cum[i][0] >= cum[i][1]:
            breakeven = i + 2 if i + 1 < len(cum) else None
            break
    else:
        breakeven = 1
    for w in range(0, len(jit.log), WINDOW):
        ds = jit.log[w : w + WINDOW]
        windows.append({
            "start": w + 1, "end": w + len(ds),
            "accuracy": sum(d.label == truth[d.query] for d in ds) / len(ds),
            "teacher_rate": sum(d.source == "teacher" for d in ds) / len(ds),
            "student_rate": sum(d.source == "student" for d in ds) / len(ds),
            "student_accuracy": (sum(d.label == truth[d.query] for d in ds if d.source == "student")
                                 / max(1, sum(d.source == "student" for d in ds))),
            "cum_cost_jit": cum[w + len(ds) - 1][0], "cum_cost_llm": cum[w + len(ds) - 1][1],
        })
    summary = {
        "target": a.target, "tau": a.tau, "n": len(qs), "recompile_every": a.every,
        "teacher_cost_per_call_usd": cpc, "compute_usd_per_hour": COMPUTE_USD_PER_HOUR,
        "teacher_calls": sum(d.source == "teacher" for d in jit.log),
        "student_answers": sum(d.source == "student" for d in jit.log),
        "overall_accuracy": sum(d.label == truth[d.query] for d in jit.log) / len(jit.log),
        "always_llm_accuracy": sum(teacher.table[q] == truth[q] for q in qs) / len(qs),
        "total_cost_jit": cum[-1][0], "total_cost_llm": cum[-1][1],
        "total_compile_s": sum(c["compile_s"] for c in jit.compiles),
        "breakeven_step": breakeven,
        "compiles": jit.compiles, "windows": windows,
    }
    out = RUNS_DIR / f"stream_{a.target}_tau{a.tau}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("compiles", "windows")}, indent=2))
    for w in windows[:: max(1, len(windows) // 8)]:
        print(f"  {w['start']:5d}-{w['end']:5d} acc={w['accuracy']:.3f} teacher_rate={w['teacher_rate']:.2f}")


if __name__ == "__main__":
    main()
