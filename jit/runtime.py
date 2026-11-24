"""The JIT AI Function runtime: serve from a compiled student when it is confident,
escalate to the LLM teacher otherwise, and recompile as teacher labels accumulate.

    jit = JITFunction(teacher=CachedTeacherFn(), target="logreg", tau=0.8)
    label = jit("what's my credit score")          # -> Decision(label, source, conf, ...)

Per call:
1. Exact hit in the teacher-label store -> return the stored teacher label (source "cache").
2. Else, if a student is compiled: predict; if calibrated confidence >= tau (and the
   prediction is not "oos" when escalate_oos) -> return it (source "student").
3. Else call the teacher (source "teacher") and store its label. Only teacher labels are
   ever stored; the student never trains on its own predictions.
4. After every `recompile_every` new teacher labels (and at least `min_train`), recompile:
   fit the target on the store and refit its temperature.

`teacher` is any callable query -> label. `CachedTeacherFn` calls the live teacher through
the cache; `TeacherTable` reads only the cache (free and deterministic: for simulations).
Every decision is appended to `self.log` (and optionally a JSONL file) so stream
experiments can compute escalation rate, accuracy and cost over time.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from jit.data import OOS
from jit.students import TARGETS, Calibrated
from jit.teacher_cache import CACHE_PATH, CachedTeacher, prompt_hash


@dataclass
class Decision:
    step: int
    query: str
    label: str
    source: str  # "cache" | "student" | "teacher"
    conf: float | None  # student's calibrated confidence (None if no student yet / cache hit)
    student_label: str | None
    latency_s: float
    n_train: int  # teacher labels in the store at decision time
    compile_version: int


class TeacherTable:
    """Teacher labels read from the SQLite cache (no API calls). Raises KeyError on a miss."""

    def __init__(self, model_id: str | None = None, cache_path: Path = CACHE_PATH):
        from jit import teacher

        model_id = model_id or teacher.TEACHER_MODEL_ID
        db = sqlite3.connect(cache_path)
        rows = db.execute(
            "SELECT query, label FROM labels WHERE model=? AND prompt_hash=?", (model_id, prompt_hash())
        ).fetchall()
        db.close()
        self.table = dict(rows)
        self.model_id = model_id
        self.calls = 0

    def __call__(self, query: str) -> str:
        self.calls += 1
        return self.table[query.strip()]

    def __contains__(self, query: str) -> bool:
        return query.strip() in self.table


class CachedTeacherFn:
    """Synchronous wrapper around CachedTeacher (live LLM on cache misses)."""

    def __init__(self):
        self.ct = CachedTeacher(concurrency=1)
        self.calls = 0

    def __call__(self, query: str) -> str:
        self.calls += 1
        rec = asyncio.run(self.ct.label(query))
        if rec.label is None:
            raise RuntimeError(f"teacher failed: {rec.error}")
        return rec.label


class JITFunction:
    def __init__(
        self,
        teacher: Callable[[str], str],
        target: str = "logreg",
        tau: float = 0.8,
        recompile_every: int = 250,
        min_train: int = 20,
        escalate_oos: bool = True,
        log_path: Path | None = None,
        target_factory: Callable | None = None,
    ):
        self.teacher = teacher
        self.target = target
        self.make_student = target_factory or TARGETS[target]
        self.tau = tau
        self.recompile_every = recompile_every
        self.min_train = min_train
        self.escalate_oos = escalate_oos
        self.log_path = log_path
        self.store: dict[str, str] = {}  # query -> teacher label (insertion-ordered)
        self.model: Calibrated | None = None
        self.version = 0
        self.new_since_compile = 0
        self.step = 0
        self.log: list[Decision] = []
        self.compiles: list[dict] = []

    # -- compilation -----------------------------------------------------------
    def recompile(self) -> None:
        texts, labs = list(self.store), list(self.store.values())
        t0 = time.perf_counter()
        model = Calibrated(self.make_student).fit(texts, labs)
        dt = time.perf_counter() - t0
        self.model = model
        self.version += 1
        self.new_since_compile = 0
        self.compiles.append({"version": self.version, "step": self.step, "n_train": len(texts),
                              "classes": len(set(labs)), "T": model.T, "compile_s": dt})

    def _maybe_recompile(self) -> None:
        if len(self.store) >= self.min_train and (
            self.model is None or self.new_since_compile >= self.recompile_every
        ):
            self.recompile()

    # -- serving -----------------------------------------------------------------
    def __call__(self, query: str) -> Decision:
        self.step += 1
        q = query.strip()
        t0 = time.perf_counter()
        conf = student_label = None
        if q in self.store:
            label, source = self.store[q], "cache"
        else:
            served = False
            if self.model is not None:
                (student_label,), confs = self.model.predict([q])
                conf = float(confs[0])
                served = conf >= self.tau and not (self.escalate_oos and student_label == OOS)
            if served:
                label, source = student_label, "student"
            else:
                label, source = self.teacher(q), "teacher"
                self.store[q] = label
                self.new_since_compile += 1
        d = Decision(self.step, q, label, source, conf, student_label,
                     time.perf_counter() - t0, len(self.store), self.version)
        self.log.append(d)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        self._maybe_recompile()
        return d
