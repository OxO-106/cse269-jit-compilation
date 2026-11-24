"""LLM teacher for CLINC150: an @ai_function that maps a query to one closed Intent label.

Two backends, chosen with the JIT_TEACHER environment variable (default "volc"):

- "volc":    DeepSeek V4.1 Flash (economy) on Volcengine Ark, through its OpenAI-compatible
             plan endpoint. Thinking is disabled: the answer is one label. The API key is read
             from ARK_API_KEY, or else from D:/269/api-key.txt (never commit that file).
- "bedrock": Claude Haiku 4.5 on Amazon Bedrock via the `us.` inference profile from the
             project's Region (us-east-2). Needs a fresh `aws login`.

Run `python -m jit.teacher` from the code/ folder for a 5-query smoke test.
"""

from __future__ import annotations

import os
from pathlib import Path

from ai_functions import ai_function

from jit.data import OOS, Intent, domains, intent_to_domain

BACKEND = os.environ.get("JIT_TEACHER", "volc")
MAX_TOKENS = 256  # the answer is one label

# Volcengine Ark
VOLC_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
VOLC_MODEL_ID = "deepseek-v4-1-flash-economy"
VOLC_KEY_FILE = Path(__file__).resolve().parents[2] / "api-key.txt"  # D:/269/api-key.txt

# Amazon Bedrock
BEDROCK_REGION = "us-east-2"
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _volc_key() -> str:
    key = os.environ.get("ARK_API_KEY")
    if key:
        return key.strip()
    return VOLC_KEY_FILE.read_text(encoding="utf-8").strip()


def make_model(backend: str = BACKEND):
    if backend == "volc":
        from strands.models.openai import OpenAIModel

        model = OpenAIModel(
            client_args={"api_key": _volc_key(), "base_url": VOLC_BASE_URL},
            model_id=VOLC_MODEL_ID,
            params={
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        )
        return model, VOLC_MODEL_ID
    if backend == "bedrock":
        from strands.models import BedrockModel

        model = BedrockModel(
            model_id=BEDROCK_MODEL_ID,
            region_name=BEDROCK_REGION,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        return model, BEDROCK_MODEL_ID
    raise ValueError(f"JIT_TEACHER must be 'volc' or 'bedrock', got {backend!r}")


teacher_model, TEACHER_MODEL_ID = make_model()


def _label_catalog() -> str:
    by_domain: dict[str, list[str]] = {d: [] for d in domains()}
    for intent, domain in intent_to_domain().items():
        by_domain[domain].append(intent)
    return "\n".join(f"- {d}: {', '.join(sorted(names))}" for d, names in by_domain.items())


_PROMPT = f"""
You are an intent classifier for a virtual assistant. Classify the user query below
into exactly one intent label.

Intent labels, grouped by domain:
{_label_catalog()}

If the query does not fit any of these intents, answer "{OOS}" (out of scope).
Pick the single best label; do not invent new labels.

<query>
{{query}}
</query>
"""


def _classify_intent(query: str) -> Intent:  # type: ignore[valid-type]
    ...


_classify_intent.__doc__ = _PROMPT
classify_intent = ai_function(model=teacher_model)(_classify_intent)


if __name__ == "__main__":
    import random
    import time

    from jit.data import load_split

    print(f"teacher backend={BACKEND} model={TEACHER_MODEL_ID}")
    rng = random.Random(0)
    sample = rng.sample(load_split("train"), 5)
    correct = 0
    for ex in sample:
        t0 = time.perf_counter()
        pred = classify_intent.run_sync(query=ex.text)
        dt = time.perf_counter() - t0
        label = getattr(pred, "value", pred)
        correct += label == ex.label
        mark = "OK " if label == ex.label else "XX "
        print(f"{mark} {dt:5.1f}s pred={label:25s} true={ex.label:25s} | {ex.text}")
    print(f"{correct}/{len(sample)} correct")
