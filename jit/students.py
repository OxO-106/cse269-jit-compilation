"""Compile targets ("students") and temperature-scaling calibration.

Every student is trained on teacher-labelled queries and returns logits over the fixed
151-label space (`LABELS`). Labels the student has never seen get a logit of NEG, so the
student can never predict them. Students:

- ExactCache:     text -> teacher label lookup; a miss has uniform probabilities, so
                  confidence 1/151 and the gate always escalates.
- SemanticCache:  kNN on bge-small embeddings; logit per class = max cosine similarity to
                  that class's stored examples (scaled by 1/0.05 before calibration).
- LogReg:         multinomial logistic regression on embeddings.
- MLPHead:        a small 2-layer MLP "decision head" on frozen embeddings.
- LoRA:           see jit/lora.py (fine-tunes the encoder itself).

`Calibrated(student_cls)` wraps a student with temperature scaling: it fits temperature T on
cross-fitted (out-of-fold) logits of the training examples, so no separate calibration
set is needed even at small N, then refits on all examples. See `fit_temperature` for
the two objectives; the default ("top_bce") also uses out-of-fold examples whose label
the fold never saw. With fewer than MIN_CALIB usable examples, T stays 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from jit.data import labels
from jit.embed import embed

LABELS: tuple[str, ...] = labels()
LABEL_IDX = {lab: i for i, lab in enumerate(LABELS)}
C = len(LABELS)
NEG = -1e4
MIN_CALIB = 5


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Student:
    name = "student"
    calibratable = True

    def fit(self, texts: list[str], labs: list[str]) -> "Student":
        raise NotImplementedError

    def logits(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class ExactCache(Student):
    name = "exact_cache"
    calibratable = False

    def fit(self, texts, labs):
        self.table = {t.strip(): lab for t, lab in zip(texts, labs)}
        return self

    def logits(self, texts):
        out = np.zeros((len(texts), C), dtype=np.float32)  # uniform for misses
        for i, t in enumerate(texts):
            lab = self.table.get(t.strip())
            if lab is not None:
                out[i, :] = NEG
                out[i, LABEL_IDX[lab]] = 0.0
        return out


class SemanticCache(Student):
    name = "semantic_cache"
    SCALE = 1 / 0.05

    def fit(self, texts, labs):
        self.X = embed(texts)
        self.y = np.array([LABEL_IDX[lab] for lab in labs])
        self.classes = np.unique(self.y)
        return self

    def logits(self, texts):
        sims = embed(texts) @ self.X.T  # cosine, embeddings are normalized
        out = np.full((len(texts), C), NEG, dtype=np.float32)
        for c in self.classes:
            out[:, c] = sims[:, self.y == c].max(axis=1) * self.SCALE
        return out


class LogReg(Student):
    name = "logreg"

    def __init__(self, Creg: float = 10.0):
        self.Creg = Creg

    def fit(self, texts, labs):
        from sklearn.linear_model import LogisticRegression

        X = embed(texts)
        y = np.array([LABEL_IDX[lab] for lab in labs])
        self.classes = np.unique(y)
        if len(self.classes) == 1:
            self.clf = None
        else:
            self.clf = LogisticRegression(C=self.Creg, max_iter=3000).fit(X, y)
        return self

    def logits(self, texts):
        out = np.full((len(texts), C), NEG, dtype=np.float32)
        if self.clf is None:
            out[:, self.classes[0]] = 0.0
            return out
        d = self.clf.decision_function(embed(texts))
        if d.ndim == 1:  # binary case: sklearn returns one column
            d = np.stack([-d / 2, d / 2], axis=1)
        out[:, self.clf.classes_] = d
        return out


class MLPHead(Student):
    name = "mlp_head"

    def __init__(self, hidden: int = 512, epochs: int = 200, lr: float = 2e-3, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed

    def fit(self, texts, labs):
        import torch

        torch.manual_seed(self.seed)
        X = torch.from_numpy(embed(texts))
        y_full = np.array([LABEL_IDX[lab] for lab in labs])
        self.classes = np.unique(y_full)
        remap = {c: i for i, c in enumerate(self.classes)}
        y = torch.tensor([remap[c] for c in y_full])
        self.net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden, len(self.classes)),
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-2)
        bs = 256
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(y))
            for i in range(0, len(y), bs):
                idx = perm[i : i + bs]
                loss = torch.nn.functional.cross_entropy(self.net(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def logits(self, texts):
        import torch

        with torch.no_grad():
            z = self.net(torch.from_numpy(embed(texts))).numpy()
        out = np.full((len(texts), C), NEG, dtype=np.float32)
        out[:, self.classes] = z
        return out


def fit_temperature(z: np.ndarray, y: np.ndarray, objective: str = "top_bce") -> float:
    """Temperature T for softmax(z / T), fitted on out-of-fold logits z and labels y.

    objective="nll": standard temperature scaling (NLL of the true label). Only usable on
    examples whose label the fold's student has seen.
    objective="top_bce" (default): binary cross-entropy of the top-label confidence
    against "the prediction was correct". It uses every out-of-fold example, including
    those whose label the student has never seen (always wrong), which matters at small N:
    the deployed student meets many unseen labels, and NLL fitted on seen labels only makes
    it overconfident. The gate thresholds exactly this top-label confidence.
    """
    correct = (z.argmax(axis=1) == y).astype(float)

    def loss(logT: float) -> float:
        p = softmax(z / np.exp(logT))
        if objective == "nll":
            return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
        c = np.clip(p.max(axis=1), 1e-6, 1 - 1e-6)
        return float(-(correct * np.log(c) + (1 - correct) * np.log(1 - c)).mean())

    res = minimize_scalar(loss, bounds=(np.log(0.02), np.log(50.0)), method="bounded")
    return float(np.exp(res.x))


class Calibrated:
    """A student plus temperature scaling fitted on cross-fitted logits."""

    def __init__(self, make_student, n_folds: int = 5, seed: int = 0, objective: str = "top_bce"):
        self.make_student = make_student
        self.objective = objective
        self.n_folds = n_folds
        self.seed = seed
        self.T = 1.0
        self.n_calib = 0

    @property
    def name(self) -> str:
        return self.student.name if hasattr(self, "student") else "calibrated"

    def fit(self, texts: list[str], labs: list[str]) -> "Calibrated":
        texts, labs = list(texts), list(labs)
        probe = self.make_student()
        if probe.calibratable and len(texts) >= 2 * self.n_folds:
            rng = np.random.default_rng(self.seed)
            folds = np.array_split(rng.permutation(len(texts)), self.n_folds)
            zs, ys = [], []
            for k in range(self.n_folds):
                hold = set(folds[k].tolist())
                tr = [i for i in range(len(texts)) if i not in hold]
                if self.objective == "nll":
                    te = [i for i in folds[k] if labs[i] in {labs[j] for j in tr}]
                else:
                    te = list(folds[k])
                if not te or len({labs[j] for j in tr}) < 2:
                    continue
                s = self.make_student().fit([texts[i] for i in tr], [labs[i] for i in tr])
                zs.append(s.logits([texts[i] for i in te]))
                ys.extend(LABEL_IDX[labs[i]] for i in te)
            self.n_calib = len(ys)
            if self.n_calib >= MIN_CALIB:
                self.T = fit_temperature(np.concatenate(zs), np.array(ys), self.objective)
        self.student = self.make_student().fit(texts, labs)
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        return softmax(self.student.logits(texts) / self.T)

    def predict(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        p = self.proba(texts)
        idx = p.argmax(axis=1)
        return [LABELS[i] for i in idx], p[np.arange(len(idx)), idx]


TARGETS = {
    "exact_cache": ExactCache,
    "semantic_cache": SemanticCache,
    "logreg": LogReg,
    "mlp_head": MLPHead,
}
