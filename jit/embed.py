"""Sentence embeddings for every CLINC150 query, computed once on CPU and cached.

Encoder: BAAI/bge-small-en-v1.5 (33M parameters, 384-d, L2-normalized), small enough to
embed all ~24k queries on a laptop CPU in a few minutes. Embeddings are stored in
runs/emb_<name>.npz as a text list plus a float32 matrix; `embed(texts)` looks texts up
and encodes any that are missing (e.g. queries not in CLINC150).

Run `python -m jit.embed` from the code/ folder to build the cache.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from jit.data import SPLITS, load_split
from jit.teacher_cache import RUNS_DIR

ENCODER = "BAAI/bge-small-en-v1.5"
CACHE = RUNS_DIR / f"emb_{ENCODER.split('/')[-1]}.npz"


@lru_cache(maxsize=1)
def encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(ENCODER, device="cpu")


def _encode(texts: list[str]) -> np.ndarray:
    return encoder().encode(texts, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


@lru_cache(maxsize=1)
def _table() -> tuple[dict[str, int], np.ndarray]:
    if not CACHE.exists():
        build()
    z = np.load(CACHE, allow_pickle=False)
    texts = [str(t) for t in z["texts"]]
    return {t: i for i, t in enumerate(texts)}, z["emb"]


def embed(texts: list[str]) -> np.ndarray:
    index, mat = _table()
    out = np.empty((len(texts), mat.shape[1]), dtype=np.float32)
    missing = [i for i, t in enumerate(texts) if t.strip() not in index]
    for i, t in enumerate(texts):
        j = index.get(t.strip())
        if j is not None:
            out[i] = mat[j]
    if missing:
        out[missing] = _encode([texts[i].strip() for i in missing])
    return out


def build() -> None:
    texts = sorted({ex.text.strip() for s in SPLITS for ex in load_split(s)})
    t0 = time.perf_counter()
    emb = _encode(texts)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, texts=np.array(texts), emb=emb)
    print(f"embedded {len(texts)} texts -> {emb.shape} in {time.perf_counter() - t0:.0f}s, saved {CACHE}")


if __name__ == "__main__":
    build()
