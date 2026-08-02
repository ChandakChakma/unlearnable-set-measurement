# Same-model Jaccard at N=128, measured

Future Work B.1. Replaces the one analytic value in Section 4.1 with a
measurement. **No existing E0 analysis file was modified** — deltas only.

## Run

| | reference | repeat |
|---|---|---|
| model name | `ckpt1_hi` + `ckpt1_hi_rest` | `ckpt1_n128_rep2` |
| checkpoint | seed 1, global_step_120 | seed 1, global_step_120 |
| eval seed | 0 | 1 |
| prompts | 464 + 559 = 1023 | 1023 (single pass) |
| samples | 128 | 128 |
| pass@1 | 0.4717 | 0.4711 |

Temperature 0.7, max_new_tokens 1024, prompt-type qwen throughout.

**On independence.** `--seed` does not drive sampling here: `gen_utils_sglang.py`
forwards only temperature, top_p, max_new_tokens and stop to the engine, and
`seed_everything()` seeds python/numpy/torch, which the sglang sampler does not
consult. The two evaluations are independent draws because the harness never
seeds sampling — not because the seeds differ. The N=32 pair already demonstrated
this empirically (Jaccard 0.798, not 1.000).

## 1. Measured vs analytic

| | Jaccard | disagreement |
|---|---|---|
| **same model @N=128, MEASURED** | **0.863** | **0.137** |
| same model @N=128, analytic (Section 4.1) | 0.889 | 0.111 |
| difference | **-0.026** | +0.026 |

Flagged sets: reference 289, repeat 296, shared 271, symmetric difference 43.

At N=32 the same model predicted 0.799 against a measured 0.798 — accurate to
0.001. At N=128 the error is **0.026**, 33x larger.
So the binomial model does **not** hold as well at N=128 as it did at N=32.

### Why the model drifts

The prediction takes p̂ as the true rate, but p̂ is itself a 128-sample estimate.
Feeding a noisy p̂ into P(X ≤ ⌊τN⌋) inflates predicted agreement, and the
inflation grows with N because the threshold sits closer to the estimate's own
uncertainty. Re-running the model on a pooled 256-sample rate (reference + repeat)
gives a less noisy input:

| p̂ source | samples | predicted J @N=128 |
|---|---|---|
| reference only (as used in Section 4.1) | 128 | 0.889 |
| reference + repeat pooled | 256 | 0.879 |
| — measured | — | **0.863** |

## 2. Training-variation component, recomputed

| quantity | value |
|---|---|
| cross-seed disagreement @N=128 (seed1 vs seed2) | 0.187 |
| same-model disagreement @N=128, MEASURED | 0.137 |
| **training variation = difference** | **0.050** |
| current estimate in the paper (analytic baseline) | 0.076 |
| **delta** | **-0.026** |

As a share of cross-seed disagreement: **27%** (paper currently says 41%).

## 3. What Section 4.1 needs

**Revise.** The analytic 0.889 should become the measured **0.863**, and
the training-variation component **0.076 → 0.050** (27% rather than 41%).

Direction: the analytic model **over-estimated agreement** (0.889 vs 0.863), so it
**under-estimated same-model disagreement** (0.111 vs 0.137), and that shortfall was
attributed to training instead. The paper therefore **over-states** the
training-variation component — 0.076 against a measured 0.050.

The qualitative claim strengthens rather than weakens: evaluation noise accounts for
an even larger share of cross-seed disagreement than reported (73% at N=128, not
59%), and still 78% at N=32.

### Limitations paragraph

*"The same-model baseline at N=128 is analytic, not measured"* — **can now be
deleted.** It should be replaced by a note that the binomial model was validated
at N=32 (error 0.001) but drifts at N=128 (error 0.026), because p̂ noise propagates into the prediction.
That is a more useful caveat than the one it replaces, and it is now supported.

### One asymmetry worth stating

The reference p̂ is pooled from two passes over different dataset files (464 + 559
prompts); the repeat is a single pass over all 1023. Same checkpoint, same sampling
parameters, same prompts — but not a byte-for-byte protocol replica. If anything
this makes the measured agreement a slight under-estimate, since batch composition
differs between the two.
