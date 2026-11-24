"""Sanity-check the teacher against CLINC150 ground truth.

Draws a stratified sample from the validation split (N_PER_INTENT queries per in-scope
intent plus N_OOS out-of-scope queries), labels it through CachedTeacher, and reports:
overall / in-scope / domain accuracy, OOS precision and recall, errors, token use and
latency, and the most frequent confusions. Results go to runs/sanity_<split>.json.

Run `python -m jit.sanity` from the code/ folder.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from collections import Counter, defaultdict

from jit.data import OOS, intent_to_domain, load_split
from jit.teacher_cache import RUNS_DIR, CachedTeacher

SPLIT = "val"
N_PER_INTENT = 3
N_OOS = 50
SEED = 0


def stratified_sample(split: str = SPLIT) -> list:
    rng = random.Random(SEED)
    by_label = defaultdict(list)
    for ex in load_split(split):
        by_label[ex.label].append(ex)
    sample = []
    for label in sorted(by_label):
        k = N_OOS if label == OOS else N_PER_INTENT
        sample += rng.sample(by_label[label], k)
    return sample


def report(sample, recs) -> dict:
    i2d = intent_to_domain()
    pairs = [(ex, r) for ex, r in zip(sample, recs)]
    ok = [(ex, r) for ex, r in pairs if r.label is not None]
    ins = [(ex, r) for ex, r in ok if ex.label != OOS]
    oos_true = sum(ex.label == OOS for ex, _ in ok)
    oos_pred = sum(r.label == OOS for _, r in ok)
    oos_hit = sum(ex.label == OOS and r.label == OOS for ex, r in ok)
    dom = lambda lab: OOS if lab == OOS else i2d.get(lab, "?")  # noqa: E731
    misses = [r for r in recs if not r.cache_hit and r.label is not None]
    conf = Counter((ex.label, r.label) for ex, r in ok if ex.label != r.label)
    return {
        "model": recs[0].model,
        "prompt_hash": recs[0].prompt_hash,
        "split": SPLIT,
        "n": len(pairs),
        "errors": len(pairs) - len(ok),
        "accuracy": sum(ex.label == r.label for ex, r in ok) / len(ok),
        "in_scope_accuracy": sum(ex.label == r.label for ex, r in ins) / len(ins),
        "in_scope_predicted_oos": sum(r.label == OOS for _, r in ins) / len(ins),
        "domain_accuracy": sum(dom(ex.label) == dom(r.label) for ex, r in ok) / len(ok),
        "oos_recall": oos_hit / oos_true if oos_true else None,
        "oos_precision": oos_hit / oos_pred if oos_pred else None,
        "teacher_calls": len(misses),
        "mean_input_tokens": statistics.mean(r.input_tokens for r in misses) if misses else None,
        "mean_output_tokens": statistics.mean(r.output_tokens for r in misses) if misses else None,
        "median_latency_s": statistics.median(r.latency_s for r in misses) if misses else None,
        "top_confusions": [
            {"true": t, "pred": p, "count": c} for (t, p), c in conf.most_common(15)
        ],
        "wrong": [
            {"query": ex.text, "true": ex.label, "pred": r.label}
            for ex, r in ok if ex.label != r.label
        ],
    }


async def main() -> None:
    sample = stratified_sample()
    ct = CachedTeacher()
    print(f"labelling {len(sample)} {SPLIT} queries with {ct.model_id} ...")
    t0 = time.perf_counter()
    recs = await ct.label_many([ex.text for ex in sample])
    wall = time.perf_counter() - t0
    ct.close()
    rep = report(sample, recs)
    rep["wall_s"] = round(wall, 1)
    out = RUNS_DIR / f"sanity_{SPLIT}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in rep.items():
        if k in ("top_confusions", "wrong"):
            continue
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")
    print("  top confusions (true -> pred):")
    for c in rep["top_confusions"]:
        print(f"    {c['count']}  {c['true']} -> {c['pred']}")
    print(f"written to {out}")


if __name__ == "__main__":
    asyncio.run(main())
