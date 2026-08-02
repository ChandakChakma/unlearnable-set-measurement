# E0 — final numbers

Every figure below with the file it came from. Qwen2.5-0.5B, MATH levels 1–4,
1023-prompt stratified subset, GRPO to `global_step_120`, tau=0.1. **E0 is frozen.**

---
## 1. Decay curve — mean over all C(5,j) subsets

*source: `E0_seed_stability/analysis/stability_report.md`*

| seeds j | subsets | mean \|unlearnable\| | min | max | mean \|learnable\| | min | max |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 164.2 | 156 | 171 | 150.2 | 142 | 159 |
| 2 | 10 | 91.2 | 82 | 99 | 114.0 | 107 | 122 |
| 3 | 10 | 61.2 | 54 | 72 | 96.8 | 91 | 104 |
| 4 | 5 | 45.0 | 41 | 50 | 86.8 | 83 | 90 |
| 5 | 1 | 35.0 | 35 | 35 | 81.0 | 81 | 81 |

Both min/max pairs are over the same C(5,j) subsets, so Figure 1(a)'s two shaded
bands are exactly these columns and the caption is verifiable against this table.

Marginal drops: −73.0, −30.0, −16.2, −10.0. Not converged at K=5.
The 3-seed row (**54–72**) is the published protocol's unstated spread.
The learnable 3-seed spread is 91–104 (±7% of the mean) against 54–72 (±15%) for
unlearnable. **Do not read a near-τ density claim into that asymmetry** — it is
confounded by set size (sqrt-scaling alone predicts 1.26x of the observed 2.19x)
and by the differing operators, and the direct per-prompt test in
`distance_to_tau.md` refutes it (candidate-pool flip rates indistinguishable,
p = 0.91). The differential decay is structural: F⁴ vs (1−F)⁴ aggregation,
correlated flags across seeds, and a `no_reward` exclusion applied to one side only.

## 2. Per-seed counts (N=32)

*source: `E0_seed_stability/analysis/per_example.csv`, `results/test_result.jsonl`*

| seed | flagged | no_reward | unlearnable | learnable | pass@1 |
|---|---|---|---|---|---|
| 1 | 301 | 159 | 156 | 159 | 0.473 |
| 2 | 310 | 164 | 165 | 151 | 0.473 |
| 3 | 312 | 166 | 158 | 151 | 0.474 |
| 4 | 317 | 159 | 171 | 142 | 0.474 |
| 5 | 308 | 153 | 171 | 148 | 0.471 |

Initial (untrained) pass@1 = 0.251, flagged 437.

## 3. no_reward series

*source: `checkpoints/*_seed*/prompt_reward_stats.csv`*

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| per seed | 159 | 164 | 166 | 159 | 153 |
| running union | 159 | 211 | 237 | 253 | 264 |

Intersection of all five: **69**. So of 264 prompts ever labelled
no_reward, only 69 are consistently so.

## 4. Jaccard at N=32 — populations named

*source: `E0b_eval_noise/analysis/noise_report.md`, `final_du_report.md` §5*

| comparison | Jaccard | population | n |
|---|---|---|---|
| same model, two independent evals | **0.798** | all prompts both evals cover | 1023 |
| same model, restricted | 0.803 | pass-1 128-sample coverage | 464 |
| across training seeds | **0.741** | all 1023, mean of 10 pairs | 1023 |

**Quote 0.798** as the headline — like-for-like against cross-seed 0.741, both on all 1023.

## 5. Analytic prediction vs observed

*source: `analyze_noise.py`; model E[J] = Σ Pᵢ² / Σ(2Pᵢ − Pᵢ²), Pᵢ = P(X ≤ ⌊τN⌋ | Bin(N, pᵢ))*

| N | predicted same-model J | observed |
|---|---|---|
| 32 | **0.799** | 0.798 (same-model) |
| 64 | 0.847 | — |
| 128 | 0.889 | **0.813** (cross-seed) |
| 256 | 0.921 | — |
| 512 | 0.942 | — |

Validation at N=32: predicted 0.799 vs observed 0.798, **delta +0.001**.

## 6. Measured rollouts per prompt

*source: `prompt_reward_stats.csv`, summed over all 120 steps*

| seed | mean | median | min | max |
|---|---|---|---|---|
| 1 | 53.3 | 56 | 40 | 56 |
| 2 | 54.1 | 56 | 32 | 56 |
| 3 | 53.6 | 56 | 40 | 56 |
| 4 | 53.1 | 56 | 40 | 56 |
| 5 | 53.3 | 56 | 40 | 56 |

Seed 1 total 54,528 rollouts; p25/p75/p90 = 48/56/56.
Histogram is bimodal: 983 prompts in 48–63 (nearly all exactly 56), 40 in 40–47.
**Replaces the draft's derived ~16.** Consistent with 211 gen-batches × 32 ÷ 1023 × 8 ≈ 52.8.

## 7. P(zero correct in 56 rollouts)

| true rate p | P(0 correct) | → mislabelled no_reward |
|---|---|---|
| 0.01 | 0.5696 | 57.0% |
| 0.03 | 0.1816 | 18.2% |
| 0.05 | 0.0566 | 5.7% |
| 0.10 | 0.0027 | 0.3% |

## 8. Exclusion-rule sensitivity

*source: `E0b_eval_noise/analysis/exclusion_sensitivity.md`*; base = 289 sub-tau prompts (seed 1, N=128)

| rule | \|D_u\| | excluded | of those, solvable |
|---|---|---|---|
| (a) union of no_reward, 5 seeds — *published* | **72** | 217 | **119** |
| (b) intersection, 5 seeds | **225** | 64 | 13 |
| (c) single seed | **145.4** (142–148) | 143.6 | 61.0 |
| (d) zero correct in 128 | **184** | 105 | 0 |

Seed choice moves |D_u| by ~4% (142–148); **operator choice moves it 3×** (72 / 145 / 225).
The union has no fixed point — it shrinks monotonically with more seeds.

## 9. D_u_observed vs D_u_solvable — kept separate

| | |
|---|---|
| D_u_observed (sub-tau ∧ positive training reward, union operator) | **72** |
| D_u_solvable (sub-tau ∧ ≥1 correct in 128) | **184** |
| overlap | **65** |
| solvable but not observed | **119** |
| observed but not solvable | **7** |

Do not merge. The 119 are provably solvable yet excluded as verifier failures.

## 10. K=5 intersection vs full-coverage truth

*source: `final_du_report.md` §4*

| | |
|---|---|
| K=5 intersection | 35 |
| D_u at N=128 (seed 1) | 72 |
| TP / FP / FN | 35 / 0 / 37 |
| precision | **1.000** |
| recall | **0.486** (95% CI 0.367–0.607, Clopper–Pearson) |

## 11. Sample size to resolve the threshold

*definition: smallest N with P(flagged) ≥ 0.95 for true rate p, holding for the next 50 values of N*

| true rate p | below tau | N needed |
|---|---|---|
| 0.09 | 0.01 | 2340 |
| 0.08 | 0.02 | 550 |
| 0.07 | 0.03 | 230 |
| 0.06 | 0.04 | 120 |
| 0.05 | 0.05 | 70 |

A prompt at exactly tau is a coin flip at every N; the figure requires a stated offset.

## 12. Cross-seed Jaccard at N=128 — the residual training component

*source: this run, seed-1 and seed-2 checkpoints both at N=128*

| | Jaccard | disagreement |
|---|---|---|
| cross-seed @N=32 | 0.741 | 0.259 |
| same-model @N=32 (pure eval noise) | 0.798 | 0.202 |
| **cross-seed @N=128** | **0.813** | **0.187** |
| same-model @N=128 (analytic) | 0.889 | 0.111 |

Excess over eval noise at N=128: 0.076, i.e. training variation
accounts for ~41% of the N=128 cross-seed disagreement.
At N=32 evaluation noise dominated (0.202 of 0.259); at N=128 it is 0.111 of 0.187.

### Seed-2 comparison (items 4 & 6)

| | seed 1 | seed 2 |
|---|---|---|
| sub-tau before exclusion | 289 | 284 |
| D_u after same no_reward union | 72 | 73 |
| D_u seed-only (not in the other) | 16 | 17 |

D_u overlap **56**; sub-tau overlap 257, union 316. D_u Jaccard 0.629 —
lower than the 0.813 sub-tau Jaccard because the exclusion shrinks both sets and
amplifies relative disagreement.

## 13. GPU hours

| experiment | wall clock |
|---|---|
| E0 training, seeds 2–5 (seed 1 pre-existing) | **11 h 48 m** |
| — per seed | 2 h 56 m / 2 h 57 m / 2 h 58 m / 2 h 56 m |
| E0b repeat eval, 1023×32 | 29 m |
| E0b 128-sample pass 1, 464 prompts | 64 m |
| E0b 128-sample pass 2, 559 prompts | 51 m |
| E0b seed-2, 1023×128 | 114 m |
| **E0b total** | **~4 h 18 m** |
| **grand total** | **~16 h 6 m** on one RTX 3090 Ti |

---

## 14. Inconsistency register

Numbers that appear differently across reports, and which to trust.

| # | figure | where it disagrees | resolution |
|---|---|---|---|
| 1 | "~491 samples" | `noise_report.md`, analytic section | **DO NOT QUOTE.** Came from `32*(1.96/0.5)**2`, a normal-approximation scaling with no criterion attached. Superseded by §11, which states the offset explicitly. |
| 2 | zero-rate count: **98** vs **105** | 98 in an intermediate check; 105 in §8 rule (d) | Both correct, different populations. 98 = never-solved *among the 217 the union excludes*; 105 = never-solved *among all 289 sub-tau*. Difference of 7 = the "observed but not solvable" prompts in §9. |
| 3 | same-model Jaccard **0.798** vs **0.803** | `noise_report.md` headline vs its analytic section | Both correct. 0.798 on all 1023; 0.803 on pass-1's 464. **Quote 0.798**; label 0.803 as restricted. |
| 4 | rule (d) \|D_u\| = **184**, not 191 | expected 289−98 | 191 used the 98 from register #2. Correct arithmetic is 289−105 = **184**. |
| 5 | eval-noise share: **78%** vs **59%** | `noise_report.md` (N=32) vs §12 (N=128) | Not a contradiction — the share is N-dependent. At N=32 noise is 0.202 of 0.259 (78%); at N=128 it is 0.111 of 0.187 (59%). Always state N. |
| 6 | rollouts per prompt: **~16** vs **53.3** | draft estimate vs §6 measurement | The ~16 was derived, never measured. **Use 53.3 mean / 56 median.** |
| 7 | training-variation estimate: **0.057** vs **0.076** | naive N=32 subtraction vs §12 direct measurement | Jaccard disagreement does not decompose additively, so the subtraction is only indicative. **Quote 0.076**, from the direct N=128 measurement against the analytic baseline. |

| 8 | near-τ density of the unlearnable set | an earlier note inferred it from the ±15% vs ±7% set-size spread | **RETRACTED.** Measured directly in `distance_to_tau.md`: candidate-pool flip rates are indistinguishable (Mann-Whitney p = 0.91), and the high-flip share is *lower* for unlearnable (19.9%) than learnable (22.6%). Cite the structural explanation instead. |

## 15. Future work — NOT run, E0 is frozen

1. **Same-model repeat at N=128.** §12's 0.889 baseline is analytic. One more 1023×128 eval of the *same* checkpoint would measure it directly and firm up the 0.076 training component. ~2 h.
2. **Fisher-exact treatment of the tau=0.1 label**, paralleling arXiv:2606.15455 Appendix C's zero-threshold version. Purely analytical, no GPU.
3. **tau sensitivity.** Everything here fixes tau=0.1. The whole analysis re-runs on existing JSONs at tau ∈ {0.05, 0.15, 0.2}. CPU only.
4. **no_reward at N=128.** The 264-prompt union is defined from ~53 training rollouts. Recomputing an equivalent criterion from 128 clean samples would test whether the exclusion survives its own measurement.
5. **Third checkpoint at N=128.** Two seeds give one cross-seed pair; a third gives three, and an error bar on the 0.813.
