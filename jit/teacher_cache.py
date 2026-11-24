"""Teacher response cache and call log.

Every teacher request goes through `CachedTeacher`, which:

1. Looks the query up in a SQLite cache keyed by (model id, prompt hash, query text).
   Changing the model or the prompt template therefore starts a fresh cache.
2. On a miss, calls the teacher @ai_function, with retries, and stores the label.
3. Appends one JSON line per request to the call log, hit or miss:
   ts, query, label, model, prompt_hash, cache_hit, latency_s, input/output tokens,
   cost_usd (None when the model's price is unknown), attempts, error.

The call log is what later steps use to build the student's training set and to
measure teacher cost; the cache keeps repeated runs from paying for the same query twice.

Run `python -m jit.teacher_cache` from the code/ folder: it labels 10 training queries
twice and prints the hit/miss counts and the last log lines.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai_functions.runtime.usage import subtree_token_usage

from jit import teacher

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
CACHE_PATH = RUNS_DIR / "teacher_cache.sqlite"
LOG_PATH = RUNS_DIR / "teacher_calls.jsonl"

# USD per million tokens (input, output). None = unknown; cost_usd is then logged as None
# and can be filled in later from the token counts.
CNY_PER_USD = 7.1  # approximate conversion used for the Volcengine list price
PRICES_PER_M: dict[str, tuple[float, float] | None] = {
    teacher.BEDROCK_MODEL_ID: (1.10, 5.50),  # Haiku 4.5, us-east-2 regional price
    # DeepSeek-V4.1-Flash economy list price (peak period): 1.6 / 6.4 CNY per M tokens.
    # The account actually pays a flat monthly plan (Agent Plan Large, AFP quota); the
    # list price is what a pay-per-token user would pay, used for cost comparisons.
    teacher.VOLC_MODEL_ID: (1.6 / CNY_PER_USD, 6.4 / CNY_PER_USD),
}

MAX_ATTEMPTS = 3


def prompt_hash() -> str:
    return hashlib.sha256(teacher._PROMPT.encode("utf-8")).hexdigest()[:12]


@dataclass
class CallRecord:
    ts: str
    query: str
    label: str | None
    model: str
    prompt_hash: str
    cache_hit: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    attempts: int
    error: str | None


class CachedTeacher:
    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        log_path: Path = LOG_PATH,
        concurrency: int = 8,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = teacher.TEACHER_MODEL_ID
        self.prompt_hash = prompt_hash()
        self.log_path = log_path
        self._db = sqlite3.connect(cache_path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS labels (
                   model TEXT NOT NULL, prompt_hash TEXT NOT NULL, query TEXT NOT NULL,
                   label TEXT NOT NULL, created TEXT NOT NULL,
                   PRIMARY KEY (model, prompt_hash, query))"""
        )
        self._db.commit()
        self._sem = asyncio.Semaphore(concurrency)
        self.hits = 0
        self.misses = 0
        self.errors = 0

    # -- cache ---------------------------------------------------------------
    def _get(self, query: str) -> str | None:
        row = self._db.execute(
            "SELECT label FROM labels WHERE model=? AND prompt_hash=? AND query=?",
            (self.model_id, self.prompt_hash, query),
        ).fetchone()
        return row[0] if row else None

    def _put(self, query: str, label: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)",
            (self.model_id, self.prompt_hash, query, label, _now()),
        )
        self._db.commit()

    # -- log -----------------------------------------------------------------
    def _log(self, rec: CallRecord) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def _cost(self, inp: int, out: int) -> float | None:
        price = PRICES_PER_M.get(self.model_id)
        if price is None:
            return None
        return (inp * price[0] + out * price[1]) / 1e6

    # -- calls ---------------------------------------------------------------
    async def _call_teacher(self, query: str) -> tuple[str, int, int]:
        handle = await teacher.classify_intent.spawn()
        try:
            pred = await handle.run(query=query)
            usage = await subtree_token_usage(handle.coordinator, handle.id)
        finally:
            await handle.terminate_now()
        return str(getattr(pred, "value", pred)), usage.input_tokens, usage.output_tokens

    async def label(self, query: str) -> CallRecord:
        query = query.strip()
        t0 = time.perf_counter()
        cached = self._get(query)
        if cached is not None:
            self.hits += 1
            rec = CallRecord(_now(), query, cached, self.model_id, self.prompt_hash, True,
                             round(time.perf_counter() - t0, 4), 0, 0, 0.0, 0, None)
            self._log(rec)
            return rec

        async with self._sem:
            t0 = time.perf_counter()  # latency excludes time spent queued behind the semaphore
            label, inp, out, err, attempts = None, 0, 0, None, 0
            for attempts in range(1, MAX_ATTEMPTS + 1):
                try:
                    label, i, o = await self._call_teacher(query)
                    inp, out = inp + i, out + o
                    err = None
                    break
                except Exception as e:  # rate limits, network, unparseable output
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    await asyncio.sleep(2 ** attempts)
        if label is not None:
            self.misses += 1
            self._put(query, label)
        else:
            self.errors += 1
        rec = CallRecord(_now(), query, label, self.model_id, self.prompt_hash, False,
                         round(time.perf_counter() - t0, 4), inp, out, self._cost(inp, out),
                         attempts, err)
        self._log(rec)
        return rec

    async def label_many(self, queries: list[str]) -> list[CallRecord]:
        return await asyncio.gather(*(self.label(q) for q in queries))

    def close(self) -> None:
        self._db.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


if __name__ == "__main__":
    import random

    from jit.data import load_split

    async def main() -> None:
        sample = random.Random(1).sample(load_split("train"), 10)
        queries = [ex.text for ex in sample]
        truth = {ex.text: ex.label for ex in sample}
        ct = CachedTeacher()
        print(f"model={ct.model_id} prompt_hash={ct.prompt_hash}")
        for rnd in (1, 2):
            ct.hits = ct.misses = ct.errors = 0
            t0 = time.perf_counter()
            recs = await ct.label_many(queries)
            acc = sum(r.label == truth[r.query] for r in recs)
            toks = sum(r.input_tokens + r.output_tokens for r in recs)
            print(f"round {rnd}: {time.perf_counter() - t0:5.1f}s  hits={ct.hits} misses={ct.misses} "
                  f"errors={ct.errors}  correct={acc}/{len(recs)}  tokens={toks}")
        ct.close()
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
        print(f"\n{len(lines)} lines in {LOG_PATH}; first miss and last hit:")
        recs = [json.loads(line) for line in lines]
        print(" ", next(r for r in recs if not r["cache_hit"]))
        print(" ", recs[-1])

    asyncio.run(main())
