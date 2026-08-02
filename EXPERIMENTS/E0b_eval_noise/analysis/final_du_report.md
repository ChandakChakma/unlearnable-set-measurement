# Final D_u at N=128, all 1023 prompts

tau=0.1 | pooled coverage 1023 prompts (pass1 464 + pass2 559, overlap 0)

## 1. D_u

- prompts with 128-sample pass@1 <= 0.1: **289**
- minus no_reward union (264): **|D_u| = 72**

## 2. Prompts no 32-sample seed ever flagged

- **0** of 72 D_u prompts were never flagged in any of the 5 seeds

- binomial expectation for D_u prompts escaping all 5 independent 32-sample evals: **0.05**; observed **0**


## 3. Poolability of the two passes

Pass-1 controls and pass-2 are a random split of the same never-flagged population (619 = 60 + 559), so a difference here would indicate a measurement artifact rather than a real one.

| group | n | mean | median |
|---|---|---|---|
| pass-1 controls | 60 | 0.7174 | 0.8398 |
| pass-2 remainder | 559 | 0.7368 | 0.8438 |

- Kolmogorov-Smirnov: D=0.0815, p=0.833
- Mann-Whitney U: U=16457.5, p=0.812
- verdict: **indistinguishable -- safe to pool**

## 4. K=5 intersection vs full-coverage truth

| | |
|---|---|
| K=5 intersection | 35 |
| D_u at N=128 | 72 |
| true positives | 35 |
| false positives | 0 |
| false negatives | 37 |
| precision | **1.000** |
| recall | **0.486**  (95% CI 0.367-0.607, Clopper-Pearson) |

Clopper-Pearson rather than Wilson: it is exact and does not under-cover near the boundary, and precision here may sit at 1.0.

## 5. The two same-model Jaccard values

| value | population | n |
|---|---|---|
| 0.798 | all prompts both 32-sample evals cover | 1023 |
| 0.803 | restricted to pass-1's 128-sample coverage | 464 |

Both are correct; they differ only in population. The headline table quotes the full-population value because that is what compares like-for-like against the cross-seed 0.741, which is also computed on all 1023. The analytic section quotes the restricted value because the prediction it is compared against can only be formed where 128-sample rates exist.

**Quote the full-population value as the headline**, and label the other explicitly as restricted. With coverage now complete the distinction disappears for future runs.

## 6. What 'resolving the threshold' requires

A prompt whose true rate is exactly tau is a coin flip at every N -- P(flagged) -> 0.5 as N grows, so no sample size fixes it and any figure must be stated at an offset from tau.

Definition used: smallest N such that P(flagged) >= 0.95 for a prompt of true rate p, holding for the next 50 values of N (the floor(tau*N) boundary makes a bare first crossing unreliable).

| true rate p | distance below tau | N needed |
|---|---|---|
| 0.09 | 0.01 | 2340 |
| 0.08 | 0.02 | 550 |
| 0.07 | 0.03 | 230 |
| 0.06 | 0.04 | 120 |
| 0.05 | 0.05 | 70 |

The earlier report's "~491 samples" was **not** derived this way. It came from `32 * (1.96/0.5)**2`, a normal-approximation scaling with no stated criterion attached, and should not be quoted. Use this table instead, with the offset named.
