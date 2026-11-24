"""Figures for the report, from runs/ladder.jsonl, runs/ladder_curves/, runs/stream_*.json and
runs/decider_test.jsonl. Writes PNGs to code/figures/.

Run `python -m jit.plots` from the code/ folder.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jit.metrics import risk_coverage  # noqa: E402
from jit.teacher_cache import RUNS_DIR  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# validated categorical palette, fixed order; markers give a second, non-color encoding
STYLE = {
    "semantic_cache": ("#2a78d6", "o", "Semantic cache (kNN)"),
    "logreg": ("#eb6834", "s", "Logistic regression"),
    "mlp_head": ("#1baf7a", "^", "MLP decision head"),
    "lora": ("#eda100", "D", "LoRA (bge-small)"),
    "decider": ("#e87ba4", "*", "Strands Decider zero-shot"),
}
TEACHER_COLOR = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def _rows():
    from jit.analyze import load_ladder

    return load_ladder()


def _agg(rows, key):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["target"]][r["n"]].append(r[key] if not callable(key) else key(r))
    return by


def _teacher_acc():
    p = RUNS_DIR / "summary.json"
    return json.loads(p.read_text())["teacher_test_accuracy"] if p.exists() else None


def _decider():
    p = RUNS_DIR / "decider_test.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    return recs or None


def ladder_metric(rows, key, ylabel, title, fname, ref=None, decider_val=None, ylim=None, legend_loc="best"):
    by = _agg(rows, key)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for t, (color, marker, label) in STYLE.items():
        if t not in by:
            continue
        ns = sorted(by[t])
        m = np.array([np.mean(by[t][n]) for n in ns])
        s = np.array([np.std(by[t][n]) for n in ns])
        ax.plot(ns, m, color=color, marker=marker, label=label)
        ax.fill_between(ns, m - s, m + s, color=color, alpha=0.12, linewidth=0)
    if ref is not None:
        ax.axhline(ref, color=TEACHER_COLOR, linestyle="--", linewidth=1.5,
                   label=f"Teacher, always-LLM ({ref:.1%})")
    if decider_val is not None:
        ax.axhline(decider_val, color=STYLE["decider"][0], linestyle=":", linewidth=2,
                   label=STYLE["decider"][2] + f" ({decider_val:.1%}, 350-query subset)")
    ax.set_xscale("log")
    ax.set_xlabel("teacher-labelled training examples (N)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(**(legend_loc if isinstance(legend_loc, dict) else {"loc": legend_loc}), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def class_coverage(rows):
    by = _agg(rows, "classes_covered")
    t = next(iter(by))
    ns = sorted(by[t])
    m = [np.mean(by[t][n]) for n in ns]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(ns, m, color=STYLE["semantic_cache"][0], marker="o")
    for n, v in zip(ns, m):
        ax.annotate(f"{v:.0f}", (n, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, color=INK2)
    ax.axhline(151, color=TEACHER_COLOR, linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_ylim(0, 165)
    ax.set_xlabel("teacher-labelled training examples (N)")
    ax.set_ylabel("labels seen (of 151)")
    ax.set_title("Class coverage of the training stream", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "class_coverage.png", dpi=160)
    plt.close(fig)


def risk_cov(ns=(100, 1000)):
    d = RUNS_DIR / "ladder_curves"
    fig, axes = plt.subplots(1, len(ns), figsize=(11, 4.2), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        for t, (color, marker, label) in STYLE.items():
            f = d / f"{t}_n{n}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            cov, risk, aurc = risk_coverage(z["conf"], z["correct"])
            ax.plot(cov, risk, color=color, label=f"{label} (AURC {aurc:.3f})")
            idx = np.linspace(0, len(cov) - 1, 8).astype(int)
            ax.plot(cov[idx], risk[idx], color=color, marker=marker, linestyle="none")
        ax.set_title(f"N = {n}", loc="left")
        ax.set_xlabel("coverage (fraction answered by the student)")
        ax.legend(fontsize=8, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("risk (error rate vs ground truth)")
    fig.suptitle("Risk–coverage: answering the most confident queries first (seed 0)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "risk_coverage.png", dpi=160)
    plt.close(fig)


def gate_tradeoff(rows, n=1000):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for t, (color, marker, label) in STYLE.items():
        rs = [r for r in rows if r["target"] == t and r["n"] == n]
        if not rs:
            continue
        taus = [g["tau"] for g in rs[0]["gated"]]
        cov = [np.mean([r["gated"][i]["coverage"] for r in rs]) for i in range(len(taus))]
        acc = [np.mean([r["gated"][i]["system_accuracy"] for r in rs]) for i in range(len(taus))]
        ax.plot(cov, acc, color=color, marker=marker, label=label)
        if t == "logreg":  # label thresholds on one series only, below the curve, to avoid collisions
            for c, a, tau in zip(cov, acc, taus):
                if tau in (0.5, 0.7, 0.9):
                    ax.annotate(f"τ={tau}", (c, a), textcoords="offset points", xytext=(-6, -14),
                                fontsize=8, color=INK2, ha="right")
    ref = _teacher_acc()
    if ref:
        ax.axhline(ref, color=TEACHER_COLOR, linestyle="--", linewidth=1.5, label=f"Always-LLM ({ref:.1%})")
    ax.set_xlabel("LLM calls avoided (fraction served by the student)")
    ax.set_ylabel("system accuracy vs ground truth")
    ax.set_title(f"Confidence gate trade-off at N = {n} (mean over seeds)", loc="left")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / f"gate_tradeoff_n{n}.png", dpi=160)
    plt.close(fig)


def stream_plots():
    files = sorted(RUNS_DIR.glob("stream_*.json"))
    if not files:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    seen = set()
    for f in files:
        s = json.loads(f.read_text())
        color, marker, label = STYLE.get(s["target"], ("#6250d6", "o", s["target"]))
        # same target, second config: same color (entity), dashed line + hollow marker
        ls, mfc = ("--", SURFACE) if s["target"] in seen else ("-", color)
        seen.add(s["target"])
        kw = dict(color=color, marker=marker, markevery=4, linestyle=ls, markerfacecolor=mfc,
                  label=f"{label}, τ={s['tau']}")
        x = [w["end"] for w in s["windows"]]
        axes[0].plot(x, [w["accuracy"] for w in s["windows"]], **kw)
        axes[1].plot(x, [w["teacher_rate"] for w in s["windows"]], **kw)
        axes[2].plot(x, [w["cum_cost_jit"] for w in s["windows"]], **kw)
        llm = [w["cum_cost_llm"] for w in s["windows"]]
    for ax in axes[:2]:
        ax.plot([], [], color=TEACHER_COLOR, linestyle=":", label="always-LLM")
    axes[2].plot(x, llm, color=TEACHER_COLOR, linestyle=":", linewidth=2, label="always-LLM")
    axes[0].set_ylabel("accuracy (window of 250)")
    axes[1].set_ylabel("escalation rate (LLM calls)")
    axes[2].set_ylabel("cumulative cost (USD, list price)")
    axes[2].set_xlabel("calls in the stream")
    axes[0].set_title("JIT runtime on a 6,000-call stream (no drift)", loc="left")
    axes[2].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "stream.png", dpi=160)
    plt.close(fig)


def fit_time(rows):
    by = _agg(rows, "fit_s")
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for t, (color, marker, label) in STYLE.items():
        if t not in by:
            continue
        ns = sorted(by[t])
        ax.plot(ns, [np.mean(by[t][n]) for n in ns], color=color, marker=marker, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("teacher-labelled training examples (N)")
    ax.set_ylabel("compile time incl. calibration (s, CPU)")
    ax.set_title("Compile cost per target", loc="left")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(FIG / "compile_time.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    rows = [r for r in _rows() if r["target"] != "exact_cache"]
    dec = _decider()
    dec_acc = np.mean([r["pred"] == r["true"] for r in dec]) if dec else None
    if rows:
        ladder_metric(rows, "acc", "accuracy vs ground truth (test, 5,500)",
                      "Accuracy vs training-set size", "accuracy_vs_n.png", ref=_teacher_acc(),
                      decider_val=dec_acc, ylim=(0, 1), legend_loc=dict(loc="upper left", bbox_to_anchor=(0, 0.89)))
        ladder_metric(rows, "ece", "ECE (15 bins, after temperature scaling)",
                      "Calibration error vs training-set size", "ece_vs_n.png")
        ladder_metric(rows, "oos_recall", "out-of-scope recall (student predicts oos)",
                      "Out-of-scope detection vs training-set size", "oos_recall_vs_n.png", ylim=(0, 1))
        class_coverage(rows)
        risk_cov()
        gate_tradeoff(rows, 1000)
        gate_tradeoff(rows, 100)
        fit_time(rows)
    stream_plots()
    print("figures in", FIG, ":", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
