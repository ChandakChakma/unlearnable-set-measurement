# E0 - seed stability of the unlearnable set

Seeds: [1, 2, 3, 4, 5] (K=5) | tau=0.1 | 32 samples/prompt | 1023 prompts

## Headline

- Single seed: **164.2** unlearnable prompts on average.

- Paper's 3-seed protocol: **61.2** (range 54-72 over 10 triples).

- All 5 seeds: **35**.

- So **79%** of the single-seed set does not survive to K=5.

- Of the prompts ever flagged (excluding no-reward), **35** are flagged in all 5 seeds and **122** in some but not all.

- Mean survival of one seed's set into the K=5 intersection: **39%**.


## Decay of |unlearnable| with seed count

| seeds j | subsets | mean \|unlearnable\| | min | max | mean \|learnable\| | min | max |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 164.2 | 156 | 171 | 150.2 | 142 | 159 |
| 2 | 10 | 91.2 | 82 | 99 | 114.0 | 107 | 122 |
| 3 | 10 | 61.2 | 54 | 72 | 96.8 | 91 | 104 |
| 4 | 5 | 45.0 | 41 | 50 | 86.8 | 83 | 90 |
| 5 | 1 | 35.0 | 35 | 35 | 81.0 | 81 | 81 |

Marginal change from the 4th to the 5th seed: **-10.0** prompts. If this is still large in magnitude, the curve has not converged and even K seeds overestimates the stable set.


## Threshold discreteness

pass@1 is a count over 32 samples, so tau=0.1 means flagged iff correct <= 3 (tau*n = 3.2). One extra correct rollout flips the label.

- Ever-flagged, non-no-reward prompts: **157**

- Of those, mean pass@1 within +/-0.06 of tau: **98** (62%) - these are threshold artifacts as much as findings.


## Pairwise seed agreement (Jaccard on flagged sets)

| | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| s1 | 1.00 | 0.72 | 0.72 | 0.76 | 0.77 |
| s2 | 0.72 | 1.00 | 0.74 | 0.75 | 0.73 |
| s3 | 0.72 | 0.74 | 1.00 | 0.74 | 0.75 |
| s4 | 0.76 | 0.75 | 0.74 | 1.00 | 0.75 |
| s5 | 0.77 | 0.73 | 0.75 | 0.75 | 1.00 |

Mean off-diagonal Jaccard: **0.74**. Low values mean the flagged sets are largely seed-specific and the intersection is doing most of the definitional work.


## What to conclude

- **Stable set is large and the curve has flattened** -> unlearnability is a real per-prompt property. Use `class == stable_unlearnable` as D_u for E1/E2 and say explicitly that it is the K-seed set.

- **Stable set is small / curve still falling** -> the published 3-seed number is substantially seed noise. That is a publishable correction on its own, and it means every downstream analysis must be re-run on the stable subset.

- Either way, report D_u as `stable_unlearnable` and carry `boundary` as a separate group. Pooling them is what makes the 3-seed number fragile.


Files: `per_example.csv` (1023 rows), `stability.png`
