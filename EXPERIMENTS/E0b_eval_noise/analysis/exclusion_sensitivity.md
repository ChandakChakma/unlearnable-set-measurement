# Exclusion-rule sensitivity of D_u

Base population: **289** prompts with 128-sample pass@1 <= 0.1 over all 1023.
Every row below starts from that same set and differs only in which prompts the
exclusion rule removes. `artifacts` = excluded prompts that are demonstrably
solvable (>=1 correct in 128 samples), so their exclusion cannot be a verifier failure.

## Rules

| rule | \|D_u\| | excluded | of those, solvable (artifacts) |
|---|---|---|---|
| (a) union of no_reward, 5 seeds — *published protocol* | **72** | 217 | **119** |
| (b) intersection of no_reward, 5 seeds | **225** | 64 | 13 |
| (c) single-seed no_reward | **145.4** (range 142–148) | 143.6 | 61.0 |
| (d) zero correct in 128 samples | **184** | 105 | 0 |

Rule (d) confirms the expected arithmetic: 289 - 105 = **184**.
Its `artifacts` count is 0 by construction — a prompt never solved in 128 samples
cannot be shown solvable by that same measurement. It is the only rule here that
excludes nothing demonstrably wrong.

### Single-seed detail (rule c)

| seed | \|D_u\| | excluded | artifacts |
|---|---|---|---|
| 1 | 146 | 143 | 59 |
| 2 | 146 | 143 | 64 |
| 3 | 142 | 147 | 62 |
| 4 | 145 | 144 | 62 |
| 5 | 148 | 141 | 58 |

Spread across seeds: **6 prompts** (142–148) — only ~4% of the single-seed value.
So *which* seed supplies the reward log barely matters. What matters is the
**operator**: single-seed exclusion leaves ~145, the union leaves 72, the
intersection leaves 225. The choice of aggregation moves |D_u| by a factor of
three, while the choice of seed moves it by 4%.

## The two sets, kept separate

- **D_u_observed** = sub-tau AND observed positive reward during training (union across 5 seeds, Chen et al.'s operator): **72**
- **D_u_solvable** = sub-tau AND demonstrably solvable at N=128: **184**
- overlap: **65**
- in D_u_solvable but not D_u_observed: **119**
- in D_u_observed but not D_u_solvable: **7**

These are different objects and should not be merged. D_u_observed asks whether
*this training run happened to sample a correct rollout*; D_u_solvable asks whether
the model *can* solve the prompt at all. The gap between them is the measurement
artifact: 119 prompts are provably solvable yet were excluded as
verifier failures. Reporting a single merged number would assert that the training
log and the 128-sample measurement agree about solvability, which is precisely what
this table shows they do not.

The 7 prompts in D_u_observed but not D_u_solvable are the converse
case: a correct rollout appeared during training, but 128 fresh samples produced
none. At 53 training rollouts vs 128 eval samples that is possible but rare, and it
is a useful sanity check that the number is small.

## Which seed defines "observed positive reward"

The operator is seed-dependent, so D_u_observed is not one set but five.

| seed defining the reward log | \|D_u_observed\| | overlap with D_u_solvable | solvable-but-excluded |
|---|---|---|---|
| 1 | 146 | 125 | 59 |
| 2 | 146 | 120 | 64 |
| 3 | 142 | 122 | 62 |
| 4 | 145 | 122 | 62 |
| 5 | 148 | 126 | 58 |
| union (published) | 72 | 65 | 119 |
| intersection | 225 | 171 | 13 |

Range across single seeds: **142–148**. The published union is
**72**, smaller than every single-seed value, because a union of exclusions
can only grow as seeds are added. Adding a sixth seed would shrink it further, with
no fixed point — the operator has no convergent limit.

Recommended reporting: give D_u_observed with the seed set named explicitly, and
give D_u_solvable alongside it as the measurement-based alternative. Neither alone
is 'the' unlearnable set.
