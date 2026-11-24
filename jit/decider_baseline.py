"""Zero-shot Strands Decider baseline (no training data at all).

Asks strands-decider-E2B-gemma4-v2-2611 (the smallest v2 model; runs on CPU in fp32) one
151-way `choice` question per query, options = the CLINC150 labels. Runs on a stratified
test subset (N_PER_INTENT per intent + N_OOS out-of-scope) because CPU inference takes
seconds per query. Results are appended to runs/decider_test.jsonl, so the run resumes
where it stopped.

Run `python -m jit.decider_baseline` from the code/ folder (set HF_HOME=D:\\269\\hf_cache).
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("HF_HOME", r"D:\269\hf_cache")

from jit.data import OOS, intent_to_domain, load_split  # noqa: E402
from jit.students import LABELS  # noqa: E402
from jit.teacher_cache import RUNS_DIR  # noqa: E402

CHECKPOINT = "StrandsAgents/strands-decider-E2B-gemma4-v2-2611"
OUT = RUNS_DIR / "decider_test.jsonl"
N_PER_INTENT, N_OOS, SEED = 2, 50, 0
INSTRUCTIONS = (
    "This is a user query sent to a virtual assistant. Which intent does it express? "
    f"Choose {OOS} if the query does not match any of the listed intents."
)


def subset() -> list:
    rng = random.Random(SEED)
    by = defaultdict(list)
    for ex in load_split("test"):
        by[ex.label].append(ex)
    out = []
    for lab in sorted(by):
        out += rng.sample(by[lab], N_OOS if lab == OOS else N_PER_INTENT)
    return out


def criteria() -> dict[str, str]:
    i2d = intent_to_domain()
    crit = {lab: f"{lab.replace('_', ' ')} ({i2d[lab].replace('_', ' ')})" for lab in LABELS if lab != OOS}
    crit[OOS] = "out of scope: none of the other intents applies"
    return crit


def main() -> None:
    from strands_decider.core import Request
    from strands_decider.decision import load_decision_engine

    done = set()
    if OUT.exists():
        done = {json.loads(line)["query"] for line in OUT.read_text(encoding="utf-8").splitlines()}
    todo = [ex for ex in subset() if ex.text not in done]
    print(f"{len(done)} done, {len(todo)} to go; loading {CHECKPOINT} on cpu ...", flush=True)
    t0 = time.perf_counter()
    # v2 checkpoints (token readout) are served by the DecisionEngine stack, not load_engine
    engine = load_decision_engine(CHECKPOINT, device="cpu")
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    q = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria()}
    with OUT.open("a", encoding="utf-8") as f:
        for k, ex in enumerate(todo, 1):
            t1 = time.perf_counter()
            req = Request.model_validate({"state": ex.text, "questions": {"intent": q}})
            ans = engine.evaluate(req).answers["intent"]
            rec = {"query": ex.text, "true": ex.label, "pred": ans.choice, "confidence": ans.confidence,
                   "p_top": max(ans.probabilities.values()), "latency_s": time.perf_counter() - t1}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} last={rec['latency_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
