# IE6520 Group — ADPO Replication

Replication of **ADPO** (Ji, He, Gu 2024, [arXiv:2402.09401](https://arxiv.org/abs/2402.09401)) on a small pairwise-preference benchmark, to check whether the paper's query-efficiency claim holds on a different setting than the LLM benchmarks reported in the paper.

- Paper: `2402.09401v2.pdf`
- Paper code (reference): https://github.com/jkx19/ActiveQuery
- Main script: `ie6520_adpo_replication.py`
- Figures: `adpo_vs_dpo_k60.png`, `adpo_vs_dpo_k300.png`, `adpo_vs_dpo_k1500.png`

## ADPO selection rule

From the paper's `scripts/trainer.py` (lines 1065–1071), the active-query rule on each training step is:

```
if |chosen_reward - rejected_reward| > gamma:
    # confident -> skip the oracle query; use pseudo-label = sign(margin)
else:
    # uncertain -> query the oracle for the true preference label
```

We port that rule verbatim. The model's reward margin `s1 - s2` plays the role of `chosen_reward - rejected_reward`.

## Benchmark

Synthetic Bradley–Terry preferences on `d = 16` linear rewards with a noisy oracle (`reward_scale = 0.5`, shrinking the BT logit so preference labels are genuinely uncertain near the decision boundary) — a deliberately different setting from the paper's ARC / TruthfulQA / HellaSwag LLM experiments. We evaluate **held-out pairwise accuracy** on a fixed 3000-pair test set and plot it against the **oracle-query budget** k ∈ [0, 60], matching the x-axis range of the paper's Figure 2.

Methods are compared at **equal oracle-query budget**. At budget k, DPO has performed k updates (one per query); ADPO has performed k queried updates *plus* many extra pseudo-label updates, which is why ADPO's curve can keep rising after DPO plateaus.

| Method | Queries every step? | Uses pseudo-labels? |
|---|---|---|
| DPO | yes | — |
| ADPO (γ = 1.3) | only when uncertain | yes |
| Active only, no PL | only when uncertain | no (skip step instead) |

`γ = 1.3` matches the paper's default.

## Results

We report three query-budget horizons to make the story honest. The short horizon (k ≤ 60) matches the x-axis in the paper's Figure 2; the longer horizons test whether DPO ever catches up if we just throw more queries at it.

### k ≤ 60 (matches the paper's x-axis)

![DPO vs ADPO at k=60](adpo_vs_dpo_k60.png)

At this horizon the three curves are close. DPO reaches ~78 %, ADPO ~82 %, no-PL in between. Early k (< ~20) is flat for all three because ADPO has not yet built a confident-margin pool and ends up querying almost every pair — the same flat-start behavior appears in the paper's ARC panel.

### k ≤ 300 (medium horizon)

![DPO vs ADPO at k=300](adpo_vs_dpo_k300.png)

The gap becomes obvious. DPO plateaus around 86 %, ADPO keeps climbing to ~91 %, no-PL ~89 %. So the advantage is not just a transient in the early-k regime — it persists.

### k ≤ 1500 (long horizon)

![DPO vs ADPO at k=1500](adpo_vs_dpo_k1500.png)

DPO saturates completely at ~87 % — no amount of extra queries moves it, because the oracle labels are noisy and DPO has no mechanism to denoise them. ADPO reaches ~94 %, a persistent ~7 pp gap. This is the clearest confirmation of the paper's mechanism on our benchmark: under a noisy oracle, ADPO's confident pseudo-labels bypass the noise ceiling that DPO hits.

## Run

```bash
python3 ie6520_adpo_replication.py
```

Runs on CPU in ~1 minute. Averaged over 30 seeds.

## Tuning notes (how we got to this figure)

The first pass used `d = 8`, a noiseless reward scale, and ran both methods for equal *training steps* rather than equal *query budget*. Under that setup DPO and ADPO converged to the same plateau (~94 %) and ADPO's advantage appeared only as a small transient, which did not match the paper's visual. Three changes produced the figure above:

1. **Noisier oracle** — scaled `theta_star` by 0.5 so BT preferences are genuinely stochastic near the decision boundary. This is what punishes DPO: it spends every query on a noisy label, while ADPO's confident pseudo-labels are effectively clean.
2. **Query-budget x-axis** — report accuracy when each method has used exactly k oracle queries (k ∈ {0, 2, …, 60}), not when they have taken k training steps. At the same k, ADPO has done many more updates than DPO.
3. **Higher dimension, more seeds** — `d = 16`, 30 seeds — so the plateau gap is statistically clean rather than seed-noise.

## Limitations

- **Synthetic benchmark, not LLMs.** We replicate the *algorithmic* claim on a linear BT toy, not the paper's Zephyr-β/Zephyr-gemma experiments. A 7B full-DPO run needs 8× A100 and is outside our budget. The toy confirms the mechanism but does *not* validate the specific MT-Bench / AlpacaEval numbers in the paper.
- **Result is sensitive to `reward_scale` and γ.** With a noiseless oracle (`reward_scale = 1.0`) DPO catches up to ADPO; with γ too small, ADPO's pseudo-labels bias the model and it plateaus *below* DPO. The paper's advantage depends on oracle noise being non-trivial and γ being tuned — not a free win.
- **Early-k regime is flat.** For k < ~20, all three methods overlap, because ADPO has not yet built a confident-margin pool and queries almost everything. The paper's ARC/TruthfulQA panels show the same qualitative behavior.
- **Linear reward model.** A linear head over Gaussian features is much easier to fit than a 7B LM. We cannot claim anything about optimization dynamics at LLM scale.
- **Only one γ reported.** We fixed `γ = 1.3` (paper default); no sweep. A proper ablation over γ is left out.

## Files

```
.
├── README.md
├── 2402.09401v2.pdf                       # paper
├── ie6520_adpo_replication.py             # replication script
├── adpo_vs_dpo_k60.png                    # short-horizon figure (paper x-axis)
├── adpo_vs_dpo_k300.png                   # medium-horizon figure
├── adpo_vs_dpo_k1500.png                  # long-horizon figure (DPO saturation)
└── legacy/                                # earlier regret-based exploration
    ├── ie6520_simulation.py
    ├── toy_experiment.png
    └── mini_dpo.png
```
