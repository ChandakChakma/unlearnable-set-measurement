# Distance to tau, measured per prompt

Tests directly whether learnable prompts sit further from the threshold than
unlearnable ones, rather than inferring it from set-size spread.

**Why the spread argument was not sufficient.** Two confounds:

1. *Size.* Relative spread scales roughly as 1/sqrt(n) under identical per-prompt
   noise. sqrt(96.8/61.2) = **1.26**, against an observed ratio of
   0.294/0.134 = **2.19** at K=3. The observed asymmetry exceeds the
   size prediction, but that comparison alone cannot say by how much or why.
2. *Operator.* `unlearnable` is an intersection, `learnable` a complement-of-union.
   These have different variance properties, so their spreads are not directly
   comparable even at equal size.

**Method.** p̂ is the 128-sample pass@1 for each prompt. q(p̂) is the probability
that a single 32-sample evaluation returns the *minority* label for that prompt:
q = min(F, 1−F) with F = P(X ≤ 3 | X ~ Bin(32, p̂)). q = 0.5 is a coin flip,
q = 0 is a label that never moves.


### Primary: p̂ from seed 1's checkpoint

| group | n | mean \|p̂−τ\| | median | within 0.03 of τ | mean p̂ |
|---|---|---|---|---|---|
| K=5 unlearnable | 35 | 0.0612 | 0.0688 | 17.1% | 0.0388 |
| K=5 learnable | 81 | 0.3455 | 0.2906 | 1.2% | 0.4455 |
| boundary (1-4 seeds) | 122 | 0.0696 | 0.0531 | 26.2% | 0.1420 |

| group | mean flip prob q | median q | q > 0.10 | q > 0.25 |
|---|---|---|---|---|
| K=5 unlearnable | 0.0811 | 0.0171 | 28.6% | 11.4% |
| K=5 learnable | 0.0388 | 0.0002 | 13.6% | 3.7% |
| boundary (1-4 seeds) | 0.2034 | 0.2065 | 63.1% | 37.7% |

### Robustness: p̂ from seed 2's checkpoint

| group | n | mean \|p̂−τ\| | median | within 0.03 of τ | mean p̂ |
|---|---|---|---|---|---|
| K=5 unlearnable | 35 | 0.0606 | 0.0688 | 14.3% | 0.0413 |
| K=5 learnable | 81 | 0.3606 | 0.3531 | 1.2% | 0.4606 |
| boundary (1-4 seeds) | 122 | 0.0821 | 0.0531 | 33.6% | 0.1540 |

| group | mean flip prob q | median q | q > 0.10 | q > 0.25 |
|---|---|---|---|---|
| K=5 unlearnable | 0.0740 | 0.0171 | 25.7% | 5.7% |
| K=5 learnable | 0.0310 | 0.0000 | 11.1% | 2.5% |
| boundary (1-4 seeds) | 0.1945 | 0.1756 | 58.2% | 37.7% |

### Significance

unlearnable vs learnable, |p̂−τ| (seed-1 p̂):
- Mann-Whitney U = 192.0, p = 1.71e-13
- Kolmogorov-Smirnov D = 0.864, p = 1.23e-19

flip probability q: Mann-Whitney U = 1891.0, p = 0.00443
mean q: unlearnable 0.0811 vs learnable 0.0388 (ratio 2.1x)

### Verdict on the Section 3 contradiction

Both statements can hold, and now both are measured rather than one inferred:

- **Both sets decay** because both are thresholded aggregations over noisy
  per-prompt labels. That is a property of the construction, not of unlearnability,
  and Section 3's reading stands.
- **Unlearnable decays faster** because its members genuinely sit closer to τ and
  therefore carry a higher per-prompt flip rate — see the q columns above, which
  state the difference as a flip rate rather than a set-size ratio.

The set-size spread asymmetry should **not** be cited as evidence for the density
claim. Cite the q columns instead: they are per-prompt, size-independent, and
operator-independent.

**Caveat.** p̂ is measured on one checkpoint while the group labels aggregate five,
so a prompt's p̂ and its membership come from different models. The seed-2 block
above repeats the whole analysis on an independent checkpoint as a robustness check.


### The claim does NOT survive the candidate-pool test

The tables above describe the *surviving* K=5 sets. Decay is driven by which
members of the **single-seed candidate pool** drop out, so that pool is the
population the density claim has to be tested on.

| single-seed pool (seed 1) | n | mean \|p̂−τ\| | median | within 0.03 of τ | mean q | median q | q > 0.25 |
|---|---|---|---|---|---|---|---|
| unlearnable (flagged − no_reward) | 156 | 0.0616 | 0.0609 | 19.9% | 0.1183 | 0.0350 | 19.9% |
| learnable (init flagged − flagged) | 159 | 0.2081 | 0.1109 | 14.5% | 0.1273 | 0.0585 | 22.6% |

Mann-Whitney on q: U = 12306.5, **p = 0.906** — no significant
difference. Mean q ratio 0.93x, and the high-flip share is
*lower* for unlearnable (19.9%) than learnable (22.6%).

**So the density claim is refuted for the pool that actually decays.** Near-τ mass
does not explain why unlearnable decays 4.7x while learnable decays 1.9x.


### What explains it: operator asymmetry, plus a third mechanism

The two sets use different operators, which respond oppositely to the same
per-prompt flag probability F = P(flagged at N=32):

- `unlearnable` is an **intersection** — a seed-1 candidate needs F⁴ to survive
- `learnable` is a **complement of union** — it needs (1−F)⁴

| pool | n | mean F | median F | binomial prediction | observed | ratio |
|---|---|---|---|---|---|---|
| unlearnable, flag test only | 156 | 0.806 | 0.965 | 0.641 | 0.577 | 0.90 |
| unlearnable, + no_reward union | 156 | — | — | 0.641 | 0.224 | 0.35 |
| learnable | 159 | 0.212 | 0.071 | 0.611 | 0.509 | 0.83 |

Read the three rows in order — they separate the mechanisms:

1. **Binomial noise alone does not explain it.** For learnable the prediction is
   close (0.611 predicted vs 0.509 observed); for unlearnable on the flag
   test alone it over-predicts (0.641 vs 0.577). The residual on the
   unlearnable side is training variation — flags are correlated across seeds by
   the shared prompt, not independent as F⁴ assumes.
2. **The no_reward union is a separate, large effect.** It drops unlearnable
   survival from 0.577 to 0.224 — a further 61% of the
   pool — and `learnable` is not subject to it at all. This is the same mechanism
   quantified in `E0_final_numbers.md` §8 as ~half the decay.

So the differential decay has **three** contributors, and near-τ density is not
among them: the operator (F⁴ vs (1−F)⁴), correlated flags across seeds, and the
no_reward union which applies to one set only.

### Verdict on the Section 3 contradiction

Section 3 stands; my spread-based note was wrong and should be removed.

- **Both sets decay** because both are thresholded aggregations over noisy labels —
  a property of the construction, not of unlearnability.
- **Unlearnable decays faster** for reasons that are structural, not distributional:
  a different aggregation operator and an exclusion rule applied to one side only.
  The candidate pools' flip rates are statistically indistinguishable (p = 0.91).

**Do not cite the ±15% vs ±7% spread asymmetry as evidence of near-τ density.**
It is confounded by set size (sqrt-scaling alone predicts 1.26x of the observed
2.19x) and by the operator difference, and the direct per-prompt test refutes it.

The one supported density statement is about the boundary group (mean q = 0.203,
37.7% with q > 0.25) versus either surviving set — but that is near-circular, since
that group is *defined* by having disagreed across seeds.
