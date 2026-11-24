"""LoRA compile target: fine-tune the bge-small encoder with LoRA adapters plus a linear head.

Unlike the other targets, which train on frozen embeddings, this one adapts the encoder
itself (rank-8 LoRA on the attention query/value projections, ~0.3% of the encoder's
parameters, plus a 384 -> classes head). It runs on CPU: a fixed budget of optimizer
steps (clamped to [MIN_STEPS, MAX_STEPS]) keeps a compile to a few minutes at N <= 3000.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from jit.embed import ENCODER
from jit.students import C, LABEL_IDX, NEG, Student

torch.set_num_threads(16)

BATCH = 16
MIN_STEPS, MAX_STEPS = 100, 1500
EPOCHS = 30
PROTO_SCALE = 20.0  # cosine-classifier temperature for the prototype-initialized head


def _tok():
    from transformers import AutoTokenizer

    if not hasattr(_tok, "t"):
        _tok.t = AutoTokenizer.from_pretrained(ENCODER)
    return _tok.t


class LoRA(Student):
    name = "lora"

    def __init__(self, r: int = 8, lr: float = 1e-3, seed: int = 0):
        self.r, self.lr, self.seed = r, lr, seed

    def _encode(self, texts: list[str]) -> torch.Tensor:
        b = _tok()(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        cls = self.enc(**b).last_hidden_state[:, 0]  # CLS pooling, as bge uses
        return torch.nn.functional.normalize(cls, dim=-1)

    def fit(self, texts, labs):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        torch.manual_seed(self.seed)
        y_full = np.array([LABEL_IDX[lab] for lab in labs])
        self.classes = np.unique(y_full)
        remap = {c: i for i, c in enumerate(self.classes)}
        y = torch.tensor([remap[c] for c in y_full])
        base = AutoModel.from_pretrained(ENCODER)
        cfg = LoraConfig(r=self.r, lora_alpha=2 * self.r, lora_dropout=0.1,
                         target_modules=["query", "value"])
        self.enc = get_peft_model(base, cfg)
        self.head = torch.nn.Linear(base.config.hidden_size, len(self.classes))
        # Prototype init: each class row starts as its mean normalized embedding (scaled),
        # so the untrained model is already a nearest-centroid classifier; fine-tuning
        # then improves on it instead of starting from a random head.
        with torch.no_grad():
            self.enc.eval()
            e = torch.cat([self._encode(texts[i : i + 256]) for i in range(0, len(texts), 256)])
            proto = torch.stack([e[y == k].mean(0) for k in range(len(self.classes))])
            self.head.weight.copy_(torch.nn.functional.normalize(proto, dim=-1) * PROTO_SCALE)
            self.head.bias.zero_()
        params = [p for p in self.enc.parameters() if p.requires_grad] + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        per_epoch = math.ceil(len(texts) / BATCH)
        steps = int(min(MAX_STEPS, max(MIN_STEPS, EPOCHS * per_epoch)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.1)
        self.enc.train()
        g = torch.Generator().manual_seed(self.seed)
        step = 0
        while step < steps:
            perm = torch.randperm(len(texts), generator=g)
            for i in range(0, len(texts), BATCH):
                idx = perm[i : i + BATCH]
                z = self.head(self._encode([texts[j] for j in idx]))
                loss = torch.nn.functional.cross_entropy(z, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                if step >= steps:
                    break
        self.enc.eval()
        self.steps = steps
        return self

    # Pickling keeps only the LoRA adapter weights and the head (~1 MB), not the 130 MB base
    # encoder; unpickling reloads bge-small from the HF cache and re-attaches the adapters.
    def __getstate__(self):
        from peft import get_peft_model_state_dict

        st = {k: v for k, v in self.__dict__.items() if k not in ("enc", "head")}
        if "enc" in self.__dict__:
            st["_lora"] = get_peft_model_state_dict(self.enc)
            st["_head"] = self.head.state_dict()
        return st

    def __setstate__(self, st):
        lora, head = st.pop("_lora", None), st.pop("_head", None)
        self.__dict__.update(st)
        if lora is None:
            return
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from transformers import AutoModel

        base = AutoModel.from_pretrained(ENCODER)
        self.enc = get_peft_model(base, LoraConfig(r=self.r, lora_alpha=2 * self.r, lora_dropout=0.1,
                                                   target_modules=["query", "value"]))
        set_peft_model_state_dict(self.enc, lora)
        self.enc.eval()
        self.head = torch.nn.Linear(base.config.hidden_size, len(self.classes))
        self.head.load_state_dict(head)

    @torch.no_grad()
    def logits(self, texts):
        zs = [self.head(self._encode(texts[i : i + 256])).numpy() for i in range(0, len(texts), 256)]
        out = np.full((len(texts), C), NEG, dtype=np.float32)
        out[:, self.classes] = np.concatenate(zs)
        return out
