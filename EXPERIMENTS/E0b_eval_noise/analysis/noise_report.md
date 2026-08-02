# E0b - eval noise vs training-seed instability

Checkpoint: seed 1, global_step_120 | tau=0.1 | 32 samples => flagged iff correct <= 3

## Headline

| comparison | Jaccard on flagged sets |
|---|---|
| **same model, two independent evals** | **0.798** |
| across training seeds (mean of 10 pairs) | 0.741 |

- same-model flagged: 301 vs 305, shared 269

- prompts flagged in exactly one of the two evals: **68**


**Share of the cross-seed disagreement reproduced with no training difference at all: 78%.**


## Analytic check (128-sample estimate, 464 prompts)

| | Jaccard |
|---|---|
| predicted from sampling alone | **0.817** |
| observed, same model | 0.803 |
| observed, across training seeds | 0.741 |

- prompts whose flag probability is between 0.05 and 0.95 (i.e. genuinely uncertain at 32 samples): **178** of 464

- to push a prompt at the threshold below a 5% flip rate you would need roughly 491 samples, not 32


## Reading

- **same-model Jaccard ~= cross-seed Jaccard** -> the instability is measurement noise. The correct fix is more eval samples per prompt, not more training seeds, and E0's decay curve is mostly re-measuring the same coin flip. D_u must be defined at a higher sample count before E1/E2.

- **same-model Jaccard near 1.0 while cross-seed stays ~0.75** -> the instability is genuine training variation and E0's reading stands.

- Anything between: report both, and treat the measurement-noise share as a correction to the decay curve rather than a refutation of it.
