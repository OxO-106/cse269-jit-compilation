"""Evaluation metrics: accuracy, ECE, risk-coverage, and gated system accuracy."""

from __future__ import annotations

import numpy as np

from jit.data import OOS


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of top-label confidence (equal-width bins)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coverage and selective risk when answering the most confident first; returns AURC too."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    c = np.asarray(correct, float)[order]
    n = np.arange(1, len(c) + 1)
    coverage = n / len(c)
    risk = 1 - np.cumsum(c) / n
    return coverage, risk, float(risk.mean())


def gate(student_pred, conf, teacher_pred, tau: float, escalate_oos: bool = True):
    """Route each query: the student answers if conf >= tau (and, with escalate_oos, it did
    not predict oos); otherwise the teacher's label is used. Returns (final, served_mask)."""
    student_pred = np.asarray(student_pred, dtype=object)
    served = np.asarray(conf) >= tau
    if escalate_oos:
        served &= student_pred != OOS
    final = np.where(served, student_pred, np.asarray(teacher_pred, dtype=object))
    return final, served


def gated_metrics(student_pred, conf, teacher_pred, truth, tau, escalate_oos=True) -> dict:
    truth = np.asarray(truth, dtype=object)
    final, served = gate(student_pred, conf, teacher_pred, tau, escalate_oos)
    is_oos = truth == OOS
    sp = np.asarray(student_pred, dtype=object)
    return {
        "tau": tau,
        "coverage": float(served.mean()),  # fraction answered by the student (no LLM call)
        "system_accuracy": float((final == truth).mean()),
        "served_accuracy": float((sp[served] == truth[served]).mean()) if served.any() else None,
        # out-of-scope queries that end up with the teacher or are labelled oos
        "oos_escalation_recall": float((~served[is_oos] | (sp[is_oos] == OOS)).mean()) if is_oos.any() else None,
        # out-of-scope queries the student answered with an in-scope label (the bad case)
        "oos_served_wrong": float((served[is_oos] & (sp[is_oos] != OOS)).mean()) if is_oos.any() else None,
    }
