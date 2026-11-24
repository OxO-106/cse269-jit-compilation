"""Label CLINC150 splits with the teacher, filling the teacher cache.

Every later experiment reads teacher labels from the cache, so runs are reproducible and
never pay for the same query twice. Already-cached queries are skipped (cache hits).

Run `python -m jit.label_pool [split ...]` from the code/ folder (default: train val test).
"""

from __future__ import annotations

import asyncio
import sys
import time

from jit.data import load_split
from jit.teacher_cache import CachedTeacher

CONCURRENCY = 12
CHUNK = 500


async def label_split(ct: CachedTeacher, split: str) -> None:
    queries = sorted({ex.text.strip() for ex in load_split(split)})
    t0 = time.perf_counter()
    done = errs = toks = 0
    for i in range(0, len(queries), CHUNK):
        recs = await ct.label_many(queries[i : i + CHUNK])
        done += len(recs)
        errs += sum(r.label is None for r in recs)
        toks += sum(r.input_tokens + r.output_tokens for r in recs)
        rate = done / max(time.perf_counter() - t0, 1e-9)
        print(f"[{split}] {done}/{len(queries)}  errors={errs}  tokens={toks:,}  {rate:.1f} q/s", flush=True)


async def main(splits: list[str]) -> None:
    ct = CachedTeacher(concurrency=CONCURRENCY)
    print(f"model={ct.model_id} prompt_hash={ct.prompt_hash}", flush=True)
    for split in splits:
        await label_split(ct, split)
    ct.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["train", "val", "test"]))
