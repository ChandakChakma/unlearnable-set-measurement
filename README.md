# The Unlearnable Set Has No Fixed Point

Code and per-prompt data for *"The Unlearnable Set Has No Fixed Point: Aggregation
Operators, Not Data, Determine Which RLVR Examples Are Called Unlearnable."*

We examine how the *unlearnable set* of Chen et al. (ICML 2026) is constructed, and find
that the construction is not well posed: both of its aggregation operators shrink the set
monotonically in the number of training seeds, so it has no limit and is undefined without
stating that number. Holding the data fixed, the set ranges from 72 to 225 prompts
depending only on which rule is applied, while individual seeds agree to within four
percent.

All experiments use **Qwen2.5-0.5B** on **MATH levels 1–4** (1023-prompt subset), GRPO to
step 120, on **one RTX 3090 Ti (24 GB)**. Total compute: **~16 GPU-hours**.

---

## Findings

| | |
|---|---|
| `\|D_u\|` by aggregation rule | 72 (published) / 145.4 / 184 / 225 |
| `\|D_u\|` by seed choice | 142–148 (~4%) |
| Excluded prompts that are demonstrably solvable | **119 of 217** |
| Same-model vs cross-seed Jaccard (N=32) | 0.798 vs 0.741 → 78% of disagreement is sampling |
| Same-model vs cross-seed disagreement (N=128) | 0.111 vs 0.187 → 59% |
| K=5 intersection against a 128-sample reference | precision 1.000, recall 0.486 (95% CI 0.367–0.607) |
| Measured training rollouts per prompt | 53.3 mean, 56 median |

---

## Layout

```
EXPERIMENTS/
  E0_seed_stability/          five seeds, decay of the set with seed count
    run_seed.sh               train -> eval -> prune -> record, skip-if-done
    run_all.sh                seeds in sequence; one failure does not abort the rest
    classify.sh               the upstream classification script over N seeds
    analyze_stability.py      decay curve, per-prompt flag counts, boundary mass
    watch_e0.sh               live status dashboard
    make_figures.py           regenerates Figures 1 and 2 from per_example.csv
    manifest.tsv              seed -> checkpoint and result paths, relative to REPO_ROOT
    analysis/
      per_example.csv         per-prompt pass@1 for all 5 seeds, flag counts,
                              Wilson intervals, class assignment
      stability_report.md     decay over all C(5,j) subsets, pairwise Jaccard
      figure1.pdf/.png        Figure 1
      figure2.pdf/.png        Figure 2

  E0b_eval_noise/             the same-model control and the 128-sample reference
    run_rest.sh               completes 128-sample coverage of all 1023 prompts
    make_rest_subset.py       builds the complement of the first high-sample pass
    analyze_noise.py          same-model Jaccard, analytic binomial prediction
    final_du.py               pooled reference, poolability test, precision/recall
    subset_groups.json        the 404 ever-flagged prompts and 60 sampled controls
    analysis/
      noise_report.md         same-model control, analytic prediction
      final_du_report.md      D_u at N=128, poolability, K=5 audit
      exclusion_sensitivity.md   the four aggregation rules
      distance_to_tau.md      per-prompt flip rates by group
      true_rates_all.csv      128-sample pass@1 for all 1023 prompts
      D_u_N128.json           the reference set

E0_final_numbers.md           every number in the paper, with its source file
```

---

## Reproducing

Training and evaluation build on the reference implementation of Chen et al.
(https://github.com/yulinchen99/unlearnability-rlvr); set `REPO_ROOT` to point at it.

```bash
# 1. Five seeds (~12 h). Skip-if-done, so it is safe to interrupt and resume.
cd EXPERIMENTS/E0_seed_stability
bash run_all.sh 2 3 4 5
bash classify.sh && python analyze_stability.py

# 2. Same-model control and the 128-sample reference (~4 h)
cd ../E0b_eval_noise
bash run_rest.sh
python final_du.py
```

Every stage is skip-if-done and writes into `analysis/`.

**Environment.** The runners resolve Python from `${VLLM_ENV_BIN}`, defaulting to
`$HOME/miniconda3/envs/vllm/bin`. Set `VLLM_ENV_BIN` if your environment is elsewhere.
Requires the dependencies of the upstream repository plus `sglang` for evaluation, and
`scipy` for the statistical tests in `final_du.py`.

**Directory layout.** `REPO_ROOT` resolves to `../../unlearnability-rlvr` relative to each
script, so the upstream repository is expected as a sibling directory. Override with the
`REPO_ROOT` environment variable if it lives elsewhere.

**What runs from a fresh clone.** Raw per-sample evaluation outputs (`results/*.json`) and
checkpoints are several hundred megabytes and are not included, so what a clone supports
depends on what you want to do:

| | works from a clone alone |
|---|---|
| Read every number reported in the paper, from `true_rates_all.csv`, `per_example.csv`, and the reports in `analysis/` | yes |
| Regenerate Figures 1 and 2 (`make_figures.py`, from `per_example.csv`) | yes |
| `classify.sh` | no — needs `prompt_reward_stats.csv` from each checkpoint |
| `analyze_stability.py`, `analyze_noise.py`, `final_du.py` | no — need the raw evaluation JSONs |

To reproduce end to end, run `run_all.sh` and `run_rest.sh` first; they regenerate the
training and evaluation artifacts the analysis scripts read (~16 GPU-hours).

`manifest.tsv` maps each seed to its checkpoint and result files, written incrementally by
`run_seed.sh`, with paths stored relative to `REPO_ROOT`. `subset_groups.json` records the
404 ever-flagged prompts and the 60 never-flagged controls sampled for the first
128-sample pass; `final_du.py` needs the control list for the poolability test in §3, and
it cannot be recovered without this file.

`run_seed.sh` prunes each checkpoint to `global_step_120/actor/huggingface` plus
`prompt_reward_stats.csv` after the evaluation JSON exists, reducing 26 GB per seed to
about 1 GB. Set `PRUNE=0` to keep everything.

---

## Data files

`analysis/true_rates_all.csv` is the main artifact. One row per prompt:

| column | meaning |
|---|---|
| `qid` | prompt id, joins across all files |
| `correct`, `n` | correct answers out of 128 samples |
| `p_hat` | 128-sample `pass@1` |
| `in_D_u` | in the reference set |
| `ever_flagged_32` | flagged in at least one of the five 32-sample evaluations |
| `in_K5` | in the five-seed intersection |
| `no_reward` | in the `no_reward` union |

`per_example.csv` gives the same prompts with each seed's 32-sample `pass@1` separately.

---

## A note on verification

The paper's argument is about measurement, so its own numbers were checked against sources
rather than carried forward. `E0_final_numbers.md` §14 is an inconsistency register listing
every figure that appeared differently across intermediate reports, which of them to trust,
and why. Several early numbers were wrong and are recorded there rather than quietly
corrected.

---

## Citation

```bibtex
@misc{TODO,
  title  = {The Unlearnable Set Has No Fixed Point: Aggregation Operators, Not Data,
            Determine Which RLVR Examples Are Called Unlearnable},
  year   = {2026}
}
```

Builds directly on:

```bibtex
@inproceedings{chen2026unlearnability,
  author    = {Chen, Yulin and He, He and Zhao, Chen},
  title     = {The Unlearnability Phenomenon in {RLVR} for Language Models},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  publisher = {PMLR},
  address   = {Seoul, South Korea}
}
```
