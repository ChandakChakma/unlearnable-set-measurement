# Verbatim extract: arXiv:2606.15455

**Understanding Diversity Collapse in RLVR via the Lens of Overtraining**
Suqin Yuan, Jinkun Chen, Jiyang Zheng, Muyang Li, Lei Feng, Dadong Wang, Tao Xiang, Tongliang Liu, Bo An. arXiv:2606.15455v1, 13 June 2026.

Source: https://arxiv.org/html/2606.15455v1 (LaTeXML HTML). Text below is
reproduced verbatim; inline math is rendered as its LaTeX `alttext`, and
figures/tables are omitted with a marker. Nothing is paraphrased.

---



## === Section 5.1 (id=S5.SS1) ===

5.1 Further Analysis

Answer extraction and verifier behavior.
Our Pass@k evaluation uses an any-boxed extraction rule: a response is counted as correct if any boxed expression matches the ground truth after normalization. This relaxes only the answer-placement convention, not the correctness criterion. Empirically, this change improves the base model across all benchmarks and also benefits trained models, especially on MinervaMath, suggesting that some apparent errors are due to answer exposure and formatting rather than missing mathematical content. This also provides a useful lens on recent findings that RLVR can improve performance with extremely small or imperfect training signals (Wang et al., 2025; Shao et al., 2026): a boxed-answer verifier rewards not only reasoning, but also producing answers in the form expected by the evaluator. Conversely, methods that prioritize output diversity may be disadvantaged by such format-sensitive protocols, because they optimize less directly for a single canonical answer presentation. We therefore interpret short-RLVR gains partly as improved instruction following and answer exposure, distinct from genuine reasoning-boundary expansion.

Simple lessons from overtraining.
BBG is derived from the Bayesian boundary utility (Eq. (8)), which estimates how much an update on each problem can still contribute to the reasoning boundary, and weights updates accordingly. As shown in Table 1, BBG attains the highest average Pass@k on every benchmark, and is the only method whose Pass@256 matches or exceeds the base model across all four.

The same overtraining analysis also suggests two simpler approaches (Table 2). The epoch-level overtraining identified in Section 3.1 motivates early stopping, while the boundary saturation identified in Section 3.2 motivates bucket masking, which removes updates from high-success rollout buckets.
Early stopping at 5 epochs already recovers much of the high-k Pass@k lost in later training, e.g. raising MinervaMath Pass@256 from 77.9 under full W-RF back to 79.4, close to the base model’s 80.9, while retaining most of the Pass@1 gain. Bucket masking offers a complementary lever: masking B_{\geq 6/8} preserves nearly all of the full-training Pass@1 while substantially improving Pass@256 on harder benchmarks relative to full W-RF. All three approaches stem from the same overtraining analysis and each balances Pass@1 and high-k Pass@k differently, allowing practitioners to choose according to their priorities.

Efficiency.
BBG spends backward computation only on problems with non-negligible estimated utility for the reference metric. Under the default setting, problems whose boundary utility falls below \tau are gated out, and only active problems are accumulated into an effective batch. This reduces the amount of gradient computation spent on empirically saturated problems. At the same time, dropped problems still require rollouts in order to determine their bucket membership, and BBG may need to sample additional problems before enough active prompts are accumulated. Thus, BBG’s efficiency benefit is primarily a reallocation of the expensive optimization step toward problems with non-negligible estimated contribution to the reference metric.

[FIGURE/TABLE omitted]


## === Appendix C (id=A3) ===

Appendix C Sampling-Noise Analysis

This appendix quantifies how much of the boundary entry and loss observed in Section 3.3 could be explained by finite-sample noise. Since Pass@k is estimated from a finite set of sampled completions (Chen et al., 2021), the induced binary Pass@256 boundary label is noisy at the problem level: a problem with a small but nonzero success probability may receive zero correct samples in one evaluation and at least one correct sample in another. We therefore distinguish weak one-success transitions from stronger multi-success transitions.

For problem i at checkpoint t, let C_{i,t} be the number of correct responses among K=256 independent evaluation samples. If the true single-sample success probability is p_{i,t}, then

\Pr(C_{i,t}=c)=\binom{K}{c}p_{i,t}^{\,c}(1-p_{i,t})^{K-c},\qquad c=0,1,\ldots,K.

(23)

The empirical Pass@256 boundary indicator is

Z_{i,t}=\mathbf{1}\{C_{i,t}>0\}.

(24)

A boundary entry from the base model to checkpoint t is C_{i,0}=0 and C_{i,t}>0, while a boundary loss is C_{i,0}>0 and C_{i,t}=0.

Exact noise test for an observed entry.

Suppose a problem has C_{i,0}=0 under the base model and C_{i,t}=c at a later checkpoint. Under the null hypothesis that the true success probability did not change, H_{0}:p_{i,t}=p_{i,0}, the c successes observed among the combined 2K=512 samples should be exchangeable between the two evaluations. Conditioning on the total number of successes, C_{i,0}+C_{i,t}=c, removes the unknown success probability. Under H_{0}, the c success positions are uniformly distributed among the 2K sample positions. Therefore, the probability that all c successes fall in the later checkpoint is

P_{\mathrm{noise}}(c)=\Pr(C_{i,t}=c,C_{i,0}=0\mid C_{i,0}+C_{i,t}=c,H_{0})=\frac{\binom{K}{c}}{\binom{2K}{c}}.

(25)

Equivalently, for K=256,

P_{\mathrm{noise}}(c)=\frac{\binom{256}{c}}{\binom{512}{c}}=\prod_{j=0}^{c-1}\frac{256-j}{512-j}.

(26)

This is the standard conditional form of Fisher’s exact test for a 2\times 2 table (Fisher, 1922; Agresti, 2013).
This gives the following exact one-sided sampling-noise scores:

\begin{array}[]{c|cccccccc}c&1&2&3&4&5&6&7&10\\
\hline\cr P_{\mathrm{noise}}(c)&0.5000&0.2495&0.1243&0.0618&0.0306&0.0152&0.0075&8.93\times 10^{-4}\end{array}

Thus, a one-success entry is weak evidence and can easily arise from sampling noise. By contrast, entries with c\geq 5 are unlikely under unchanged success probability at the 5\% level, and entries with c\geq 7 are unlikely at the 1\% level.
The same calculation applies to boundary loss. If C_{i,0}=c and C_{i,t}=0, then under the unchanged-policy null,

\Pr(C_{i,0}=c,C_{i,t}=0\mid C_{i,0}+C_{i,t}=c,H_{0})=\frac{\binom{K}{c}}{\binom{2K}{c}}.

Therefore, losing a problem that had only one base correct sample is noisy, while losing a problem with many base correct samples is much less likely to be explained by resampling noise alone.

Observed entry under standard RLVR.

We apply this calculation to the boundary-tracking experiment in Section 3.3. On the held-out benchmarks AIME 2025, MinervaMath, and OlympiadBench, there are 15+57+193=265 problems with C_{i,0}=0 under the base model.

At epoch 2, 70/265=26.4\% of these base-unsolved problems enter the empirical Pass@256 boundary. Of these 70 entries, 27 have at least two correct samples and 3 have at least five correct samples; equivalently, these account for 27/265=10.2\% and 3/265=1.1\% of all base-unsolved problems. For the two intermediate-difficulty benchmarks emphasized in the main text, MinervaMath has 18/57=31.6\% entries, of which 9 have c\geq 2 and 2 have c\geq 5. OlympiadBench has 47/193=24.4\% entries, of which 15 have c\geq 2 and 1 has c\geq 5. Hence, many binary entries are indeed weak one-success events, but a non-negligible subset is supported by repeated correct samples.
At the final checkpoint, epoch 20 (step 140), 38/265=14.3\% of the held-out base-unsolved problems remain inside the empirical boundary. Among these 38 remaining entries, 17 have c\geq 2, 3 have c\geq 5, and one has c\geq 10; as fractions of all base-unsolved problems, these are 6.4\%, 1.1\%, and 0.4\%, respectively.

Across all six evaluated checkpoints, 134/265=50.6\% of the base-unsolved held-out problems enter at least once. Among them, 64/265=24.2\% enter at least once with c\geq 2, 11/265=4.2\% enter at least once with c\geq 5, and 4/265=1.5\% enter at least once with c\geq 7. In addition, 68/265=25.7\% enter in at least two consecutive evaluated checkpoints. These repeated and multi-success transitions are much harder to attribute to a single lucky sample.

Observed loss is also stronger than one-sample noise.

The aggregate Pass@256 decline is driven not only by noisy entry labels but also by boundary loss. At epoch 20, the held-out benchmarks have 152 losses among 15+215+481=711 base-solvable problems, i.e. 152/711=21.4\%. Many of these losses come from problems that had multiple correct samples under the base model: 91 had at least two base correct samples, and 32 had at least five base correct samples. Since P_{\mathrm{noise}}(5)=0.0306, losses from problems with c\geq 5 base successes are unlikely to be explained by resampling noise alone at the 5\% level.

Overall, at epoch 20 the held-out benchmarks have 38 entries but 152 losses, giving 38-152=-114 net boundary transitions. Over the 976 held-out problems, this corresponds to -114/976=-11.7\%. This matches the observed aggregate Pass@256 degradation: RLVR does move some initially unsolved problems into the boundary, but this gain is more than offset by losses on previously solvable problems.
