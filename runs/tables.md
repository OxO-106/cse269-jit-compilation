### Accuracy vs ground truth (test, 5,500 queries), mean ± std over seeds

| N | labels seen | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA |
|---|---|---|---|---|---|---|
| 10 | 10 / 151 | 0.5% ± 0.0% | 4.7% ± 0.2% | 4.7% ± 0.3% | **4.8% ± 0.2%** | 4.6% ± 0.2% |
| 30 | 27 / 151 | 0.5% ± 0.0% | 12.6% ± 0.6% | 13.2% ± 3.5% | **13.3% ± 1.7%** | 12.6% ± 0.2% |
| 100 | 72 / 151 | 0.5% ± 0.0% | 30.3% ± 1.4% | 29.8% ± 3.9% | **34.0% ± 3.7%** | 32.8% ± 1.7% |
| 300 | 127 / 151 | 0.5% ± 0.0% | 51.6% ± 1.8% | 48.6% ± 3.1% | 57.2% ± 3.1% | **57.5% ± 1.2%** |
| 1000 | 150 / 151 | 0.5% ± 0.0% | 67.9% ± 0.4% | 76.7% ± 0.4% | 77.3% ± 0.6% | **78.7% ± 1.2%** |
| 3000 | 151 / 151 | 0.5% ± 0.0% | 71.5% ± 0.4% | **84.7% ± 0.3%** | 82.4% ± 0.3% | 83.7% ± 0.3% |

### ECE (15 bins, after temperature scaling), mean over seeds

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA |
|---|---|---|---|---|---|
| 10 | 0.001 | 0.096 | 0.088 | 0.096 | 0.125 |
| 30 | 0.001 | 0.093 | 0.102 | 0.095 | 0.103 |
| 100 | 0.001 | 0.070 | 0.079 | 0.089 | 0.045 |
| 300 | 0.001 | 0.040 | 0.038 | 0.028 | 0.039 |
| 1000 | 0.001 | 0.029 | 0.031 | 0.040 | 0.021 |
| 3000 | 0.001 | 0.029 | 0.024 | 0.029 | 0.017 |

### Oracle: LLM calls avoided within 0.5% of always-LLM accuracy (90.5%), τ chosen on test

Upper bound only: τ is picked per (target, N) by looking at test results. The val-chosen tables below are the honest numbers.

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA | best target |
|---|---|---|---|---|---|---|
| 10 | 0% | 1% (τ=0.6) | 0% | 1% (τ=0.6) | 1% (τ=0.6) | LoRA |
| 30 | 0% | 2% (τ=0.5) | 1% (τ=0.6) | 2% (τ=0.5) | 3% (τ=0.6) | LoRA |
| 100 | 0% | 8% (τ=0.7) | 3% (τ=0.8) | 5% (τ=0.8) | 11% (τ=0.7) | LoRA |
| 300 | 0% | 22% (τ=0.85) | 13% (τ=0.85) | 23% (τ=0.85) | 28% (τ=0.8) | LoRA |
| 1000 | 0% | 44% (τ=0.85) | 50% (τ=0.8) | 56% (τ=0.85) | 67% (τ=0.7) | LoRA |
| 3000 | 0% | 50% (τ=0.85) | 70% (τ=0.7) | 66% (τ=0.8) | 76% (τ=0.6) | LoRA |

### Gate with τ chosen on val (val as is: teacher 92.6%, 3% out-of-scope), evaluated on test

Per seed, τ is the threshold with the most val coverage whose val system accuracy stays within 0.5 points of the teacher's val accuracy; if none qualifies, always escalate. Cells: test LLM calls avoided / test system accuracy minus always-LLM, in points / seeds whose test drop exceeds 0.5 points.

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA | best target (mean drop ≤ 0.5) |
|---|---|---|---|---|---|---|
| 10 | 0% / +0.0 / 0 of 5 | 0% / -0.1 / 0 of 5 | 0% / +0.0 / 0 of 5 | 0% / -0.0 / 0 of 5 | 1% / -0.0 / 0 of 3 | LoRA |
| 30 | 0% / +0.0 / 0 of 5 | 3% / -0.1 / 0 of 5 | 1% / -0.1 / 0 of 5 | 2% / -0.1 / 0 of 5 | 3% / -0.2 / 0 of 3 | LoRA |
| 100 | 0% / +0.0 / 0 of 5 | 7% / -0.2 / 1 of 5 | 3% / -0.2 / 0 of 5 | 7% / -0.3 / 0 of 5 | 10% / -0.2 / 0 of 3 | LoRA |
| 300 | 0% / +0.0 / 0 of 5 | 21% / -0.3 / 1 of 5 | 13% / -0.3 / 0 of 5 | 21% / -0.3 / 2 of 5 | 26% / -0.1 / 0 of 3 | LoRA |
| 1000 | 0% / +0.0 / 0 of 5 | 43% / -0.4 / 1 of 5 | 46% / -0.2 / 0 of 5 | 62% / -1.2 / 5 of 5 | 67% / -0.3 / 1 of 3 | LoRA |
| 3000 | 0% / +0.0 / 0 of 5 | 53% / -0.4 / 1 of 5 | 76% / -1.2 / 5 of 5 | 75% / -1.9 / 5 of 5 | 81% / -1.9 / 3 of 3 | Semantic cache (kNN) |

### Gate with τ chosen on val (val reweighted to 18% out-of-scope), evaluated on test

Per seed, τ is the threshold with the most val coverage whose val system accuracy stays within 0.5 points of the teacher's val accuracy; if none qualifies, always escalate. Cells: test LLM calls avoided / test system accuracy minus always-LLM, in points / seeds whose test drop exceeds 0.5 points.

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA | best target (mean drop ≤ 0.5) |
|---|---|---|---|---|---|---|
| 10 | 0% / +0.0 / 0 of 5 | 0% / -0.1 / 0 of 5 | 0% / +0.0 / 0 of 5 | 0% / -0.0 / 0 of 5 | 1% / -0.0 / 0 of 3 | LoRA |
| 30 | 0% / +0.0 / 0 of 5 | 3% / -0.2 / 0 of 5 | 1% / -0.1 / 0 of 5 | 2% / -0.1 / 0 of 5 | 3% / -0.2 / 0 of 3 | LoRA |
| 100 | 0% / +0.0 / 0 of 5 | 8% / -0.3 / 2 of 5 | 4% / -0.4 / 1 of 5 | 7% / -0.3 / 0 of 5 | 10% / -0.2 / 0 of 3 | LoRA |
| 300 | 0% / +0.0 / 0 of 5 | 21% / -0.2 / 1 of 5 | 13% / -0.3 / 0 of 5 | 21% / -0.3 / 2 of 5 | 28% / -0.3 / 1 of 3 | LoRA |
| 1000 | 0% / +0.0 / 0 of 5 | 38% / -0.1 / 0 of 5 | 46% / -0.2 / 0 of 5 | 56% / -0.5 / 2 of 5 | 64% / +0.1 / 1 of 3 | LoRA |
| 3000 | 0% / +0.0 / 0 of 5 | 42% / +0.1 / 0 of 5 | 73% / -0.6 / 3 of 5 | 68% / -0.4 / 2 of 5 | 76% / -0.3 / 1 of 3 | LoRA |

### OOS recall of the student (predicts `oos` on true out-of-scope queries)

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA |
|---|---|---|---|---|---|
| 10 | 0% | 0% | 0% | 0% | 0% |
| 30 | 0% | 1% | 9% | 5% | 0% |
| 100 | 0% | 4% | 44% | 26% | 15% |
| 300 | 0% | 5% | 56% | 36% | 28% |
| 1000 | 0% | 10% | 66% | 47% | 42% |
| 3000 | 0% | 13% | 70% | 58% | 54% |

With the gate's escalate-oos rule, out-of-scope queries the student labels `oos` go to the teacher; the column `oos_served_wrong` in runs/ladder.jsonl is the share of true OOS queries the student answered with an in-scope label.

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA |
|---|---|---|---|---|---|
| 10 (τ=0.8, OOS served wrong) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 30 (τ=0.8, OOS served wrong) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 100 (τ=0.8, OOS served wrong) | 0.0% | 0.1% | 0.2% | 0.1% | 0.1% |
| 300 (τ=0.8, OOS served wrong) | 0.0% | 1.5% | 0.5% | 1.7% | 1.0% |
| 1000 (τ=0.8, OOS served wrong) | 0.0% | 4.9% | 3.2% | 8.1% | 4.1% |
| 3000 (τ=0.8, OOS served wrong) | 0.0% | 5.8% | 5.5% | 7.5% | 5.0% |

### Mean compile time incl. calibration (s, CPU)

| N | Exact cache | Semantic cache (kNN) | Logistic regression | MLP head | LoRA |
|---|---|---|---|---|---|
| 10 | 0.0 | 0.0 | 0.4 | 3.4 | 49.2 |
| 30 | 0.0 | 0.0 | 0.1 | 3.2 | 55.0 |
| 100 | 0.0 | 0.0 | 0.6 | 4.2 | 106.5 |
| 300 | 0.0 | 0.0 | 1.2 | 6.7 | 290.2 |
| 1000 | 0.0 | 0.1 | 6.4 | 25.5 | 758.8 |
| 3000 | 0.0 | 0.2 | 17.8 | 68.8 | 889.4 |

### Stream demo (6,000 calls, recompile every 250 teacher labels, no drift)

| target | τ | accuracy | always-LLM | LLM calls | cost (list) | always-LLM cost | N* | compile total |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.8 | 92.6% | 92.2% | 2283 (38%) | $1.22 | $3.13 | 289 | 103 s |
| Logistic regression | 0.9 | 92.6% | 92.2% | 2665 (44%) | $1.42 | $3.13 | 292 | 119 s |
| Semantic cache (kNN) | 0.8 | 91.7% | 92.2% | 2269 (38%) | $1.18 | $3.13 | 273 | 1 s |
