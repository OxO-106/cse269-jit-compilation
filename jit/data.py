"""CLINC150 loading: splits, the closed Intent label type, and intent -> domain mapping.

Source: the original release at github.com/clinc/oos-eval ("plus" variant:
150 in-scope intents + out-of-scope, with the larger OOS training set).
Files are downloaded once into code/data/ and read locally afterwards.

Run `python -m jit.data` from the code/ folder to download and print a summary.
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_RAW = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
_FILES = {"data": "data_oos_plus.json", "domains": "domains.json"}

OOS = "oos"
OOS_DOMAIN = "oos"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    text: str
    label: str  # one of the 150 intent names, or "oos"
    domain: str  # one of the 10 domains, or "oos"


def download(force: bool = False) -> None:
    """Fetch the raw JSON files into DATA_DIR if they are not already there."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in _FILES.values():
        path = DATA_DIR / name
        if path.exists() and not force:
            continue
        with urllib.request.urlopen(_RAW + name, timeout=60) as resp:
            path.write_bytes(resp.read())


def _read(key: str) -> dict:
    path = DATA_DIR / _FILES[key]
    if not path.exists():
        download()
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def intent_to_domain() -> dict[str, str]:
    """Map each of the 150 in-scope intents to its domain."""
    mapping = {}
    for domain, intents in _read("domains").items():
        for intent in intents:
            mapping[intent] = domain
    return mapping


@lru_cache(maxsize=1)
def domains() -> tuple[str, ...]:
    return tuple(sorted(_read("domains")))


@lru_cache(maxsize=1)
def intents() -> tuple[str, ...]:
    """The 150 in-scope intent names, sorted."""
    return tuple(sorted(intent_to_domain()))


def labels() -> tuple[str, ...]:
    """All 151 labels: the 150 intents plus "oos"."""
    return intents() + (OOS,)


@lru_cache(maxsize=1)
def make_intent_enum() -> type[Enum]:
    """Closed output type for the teacher: one member per label, value = label name."""
    return Enum("Intent", {name: name for name in labels()}, type=str)


Intent = make_intent_enum()


@lru_cache(maxsize=None)
def load_split(split: str) -> tuple[Example, ...]:
    """Return one split ("train", "val" or "test") with in-scope and OOS examples merged."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    raw = _read("data")
    i2d = intent_to_domain()
    out = [Example(text, label, i2d[label]) for text, label in raw[split]]
    out += [Example(text, OOS, OOS_DOMAIN) for text, _ in raw[f"oos_{split}"]]
    return tuple(out)


def summary() -> str:
    lines = [f"{len(intents())} intents in {len(domains())} domains, {len(labels())} labels total"]
    total_in = total_oos = 0
    for split in SPLITS:
        data = load_split(split)
        n_oos = sum(e.label == OOS for e in data)
        total_in += len(data) - n_oos
        total_oos += n_oos
        lines.append(f"  {split:5s}: {len(data):6d} examples ({len(data) - n_oos} in-scope, {n_oos} oos)")
    lines.append(f"  total: {total_in} in-scope, {total_oos} oos")
    lines.append("Examples per domain (all splits):")
    counts = Counter(e.domain for s in SPLITS for e in load_split(s))
    for domain, n in sorted(counts.items()):
        lines.append(f"  {domain:20s} {n}")
    lines.append("Sample: " + repr(load_split("train")[0]))
    return "\n".join(lines)


if __name__ == "__main__":
    download()
    print(summary())
