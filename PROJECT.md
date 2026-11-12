# SFT vs DPO vs GRPO for Short Answer Scoring with ModernBERT (Classification and Regression Heads)

A research comparison of fine-tuning objectives for an automated short answer scorer built on `answerdotai/ModernBERT-base`, crossing two **head types** with three **objectives**:

- **Multiple-target head (classification):** `K` logits, one per score point; the model is a categorical policy over scores.
- **Single-target head (regression):** one scalar output `μ_θ(x)`; treated as the mean of a Gaussian policy over scores when a policy is needed.
- **Objectives:** SFT (cross-entropy for classification, MSE for regression), DPO (gold score preferred over a random alternative score), and GRPO (group-relative policy optimization with a binary exact-match reward). SFT can also be used as a first stage before DPO or GRPO.

A separate model is trained for each `EssaySet` (prompt) in the dataset, so every experiment below is run once per `EssaySet`.

**Extension (§6.6):** four generative-model experiments (G1–G4) score the same test sets with `Qwen/Qwen3.5-4B` served by vLLM, with and without thinking, using the original rubric and then a rubric optimized with DSPy's MIPROv2. They are zero-shot / prompt-optimized comparators for the fine-tuned encoders, not part of the E1–E11 training study. **Extension (§6.7):** G5 (SFT on `Answer: {score}`, no thinking) and G6 (DPO on pairs of right and wrong sampled thinking replies) then train the same model.

---

## 1. Research Questions

- **RQ1 (objectives from scratch, classification).** Starting from the same initialization, how do DPO and GRPO compare with cross-entropy SFT on QWK, SMD, accuracy, and score-distribution fidelity (KL / Jeffreys)?
- **RQ2 (sequential training, classification).** Does following SFT with DPO or GRPO improve on SFT alone, after controlling for extra training (via the continued-SFT controls)?
- **RQ3 (head type).** Under SFT, does a single-target regression head differ from a multiple-target classification head, given that the MSE loss carries ordinal information that cross-entropy does not?
- **RQ4 (preference training for regression).** Do DPO and GRPO work with a regression head, from a cold start and as a second stage after regression SFT, again relative to a continued-SFT control?
- **RQ5 (stability and imbalance).** How do the objectives differ in training stability, sample efficiency, and drift of the predicted score distribution toward or away from common scores?
- **RQ6 (generative scorer).** How does a prompted 4B generative model (Qwen3.5-4B) with the human rubric compare with the fine-tuned ModernBERT arms, does thinking help, and how much does optimizing only the rubric text with MIPROv2 close the gap, with and without thinking?

## 2. Is This a Cogent Idea? (Design Rationale)

Yes, as a study. Each head defines a policy, so all three objectives are well defined. The caveats below shape the design and should be stated plainly in any write-up.

**GRPO (classification)**
- GRPO was built for enormous action spaces (generated sequences) where outcomes cannot be enumerated. Here the action space is small and enumerable, so sampling is deliberately higher-variance than necessary. That is the point of the comparison.
- With a binary exact-match reward, the on-policy expected gradient is closed-form:

  ```
  E[∇J] = ∇ p_θ(y | x) = p_θ(y | x) · ∇ log p_θ(y | x)
  ```

  This is the cross-entropy gradient re-weighted by the model's current probability on the correct class, so in expectation GRPO learns more slowly when the model is wrong (small `p_y`). That is why SFT → GRPO (warm start) is part of the design.

**DPO (classification)**
- For a categorical policy, the softmax normalizer cancels in the DPO log-ratio, so the preference margin between two scores is the **logit difference** `z_w − z_l`, shifted by the reference model's margin. DPO reduces to a pairwise logistic loss on logits.
- Cross-entropy equals `log(1 + Σ_{r≠y} exp(z_r − z_y))`, which pushes against all `K−1` competitors at once. A single-pair DPO loss `log(1 + exp(z_l − z_w))` involves one random competitor, so it is a per-example lower bound on CE with a sparser gradient. Expect slower per-step convergence from a cold start.
- DPO is offline (pairs come from the labeled data) and needs a reference policy. From a random head the reference margin is near zero, so cold-start DPO is close to reference-free. The reference term does real work only after SFT.

**DPO (regression)**
- Model the policy as `π_θ(a | x) = N(μ_θ(x), σ²)` with fixed σ. Then for a pair (chosen `y_w`, rejected `y_l`):

  ```
  log π_θ(y_w|x) − log π_θ(y_l|x) = (y_w − y_l) · (μ − (y_w + y_l)/2) / σ²
  ```

  and, after subtracting the reference's identical expression, the DPO margin becomes

  ```
  margin = β · (y_w − y_l) · (μ_θ − μ_ref) / σ²
  ```

- This margin depends on μ only through its **displacement from the reference** (the pair midpoint cancels). It rewards moving μ up when the gold score is above the rejected one and down otherwise, but it never anchors μ at `y_w`. From a cold start, the net push for a given gold score is roughly toward `y − mean(other scores)`, which preserves the *direction* of the signal (higher gold, higher μ) but not the *scale*. **Prediction:** cold-start DPO on a regression head learns a roughly monotone ranking signal with an unanchored scale, so it needs validation-tuned thresholds to be useful and may underperform regression SFT. It should be most useful as a second stage after SFT, where μ_ref is already well-calibrated. Note that `β` and `σ²` enter only as `β/σ²`, so there is a single effective temperature. This derivation is mine; verify it empirically (it is part of the analysis in §9).
- Like exact-match GRPO, classification DPO as specified uses no ordinal information: rejecting score 0 is as strong against gold 1 as against gold 4. Regression DPO does weight pairs by `|y_w − y_l|`.

## 3. Task and Data

| Item | Value |
|---|---|
| Task | Automated short answer scoring |
| Columns used | `EssayText` (string), `Score1` (integer), `EssaySet` (integer 1..10) |
| Input | The `EssayText` column, used as-is |
| Target | The `Score1` column; `K` is the number of distinct values observed in `Score1` **within each `EssaySet`** |
| Models | One model per `EssaySet` (10 sets), per arm, per seed. Models are never trained on or evaluated across sets |
| Backbone | `answerdotai/ModernBERT-base` |
| Heads | Multiple-target: `K` logits (set-specific `K`). Single-target: 1 scalar output |
| Framework | PyTorch + Hugging Face Transformers (custom DPO and GRPO loops; TRL does not target encoder classifiers/regressors directly) |

### 3.1 Dataset

| Split | Source |
|---|---|
| Train | `data/train_rel_2.tsv` |
| Dev | A 10% random split of train, drawn within each `EssaySet` (stratified by `Score1`; split seed recorded) |
| Test | `data/public_leaderboard_rel_2.tsv`, with scores taken from `data/public_leaderboard_solution.csv` |

- The test scores must be joined to the test file on the row identifier. Verify the identifier and score column names in both files, and keep only rows present in the test file if the solution file contains additional rows.
- The dev split is used for early stopping, threshold fitting (regression), and hyperparameter selection. The test set is touched only for final reporting.

### 3.2 Data handling assumptions

- Within each `EssaySet`, scores are mapped to contiguous class indices `0..K-1` in sorted order for the classification head; the mapping is saved with each run. QWK, SMD, and regression targets use the **original score values**, so ordinal distance is preserved even if scores are not contiguous.
- Regression targets are standardized with the training-set mean and standard deviation of that `EssaySet` for stability; predictions are mapped back to the original scale before evaluation.
- Because each model is trained on a single prompt, the prompt context is implicit and no question or rubric text is added to the input. Generalization to **unseen prompts** is **not** measured.
- Validate on load, per `EssaySet`: no null `EssayText` or `Score1`, integer scores, and print the label distribution and token-length distribution before training. Check that every score present in dev and test also appears in train.

## 4. Shared Setup (Held Constant Across Arms)

- **Initialization:** same pretrained weights per seed and `EssaySet`; heads randomly initialized with a fixed seed.
- **Tokenization:** same tokenizer, max length, and truncation policy for all arms; record the percent of examples truncated per `EssaySet`.
- **Splits:** the fixed train / dev / test splits from §3.1. Check for exact-duplicate `EssayText` across splits within each set and remove or reassign duplicates so they appear in one split only.
- **Optimizer:** AdamW, same LR schedule family, warmup, weight decay, and bf16 precision. Learning rates are tuned per arm (see §6).
- **Seeds:** minimum 5 seeds per configuration. Sequential arms and controls use the **same seed's** stage-1 checkpoint, so paired comparisons are valid.
- **Budget accounting:** report examples seen and wall-clock for every arm. For sequential arms and controls, report the stage-1 and stage-2 budgets separately and in total.

### Mapping predictions to scores

- **Multiple-target head:** predicted score `ŷ` = argmax over logits.
- **Single-target head:** predicted score `ŷ` = the continuous output `μ_θ(x)` (mapped back to the original scale), converted to a score by **monotone thresholds fit on the dev set** to maximize QWK (the same procedure for every regression arm, fit separately for each `EssaySet`). Rounding to the nearest observed score is reported as a secondary result. Thresholds are essential for fairness to DPO-regression, whose scale is unanchored (§2).

## 5. Training Objectives

### 5.1 SFT, multiple targets (cross-entropy)

```
L_SFT-MC(θ) = - (1/B) Σ_i log p_θ(y_i | x_i)
```

Early stopping on dev QWK.

### 5.2 SFT, single target (regression)

```
L_SFT-Reg(θ) = (1/B) Σ_i ( μ_θ(x_i) − ỹ_i )²       (ỹ = standardized score)
```

Equivalent to maximum likelihood under the Gaussian policy. Early stopping on dev QWK (after threshold fitting).

### 5.3 DPO, multiple targets

For each example `(x_i, y_i)`: chosen `y_w = y_i`; rejected `y_l` drawn uniformly from the other `K−1` scores, resampled each time the example is seen.

```
L_DPO(θ) = - (1/B) Σ_i log σ( β · [ (z_w − z_l)_θ − (z_w − z_l)_ref ] )
```

Reference `π_ref`: frozen initial model (cold start) or the SFT checkpoint (sequential). Instrument preference accuracy (`z_w > z_l`), mean margin, and the gap between preference accuracy and top-1 accuracy. If `K = 2`, the rejected score is deterministic.

### 5.4 DPO, single target (regression)

Same pairs as 5.3, with the Gaussian-policy margin from §2:

```
L_DPO-Reg(θ) = - (1/B) Σ_i log σ( β' · (y_w − y_l) · (μ_θ(x_i) − μ_ref(x_i)) ),    β' = β / σ²
```

(`y` values on the standardized scale.) Reference `μ_ref`: frozen initial model (cold start) or the regression-SFT checkpoint (sequential). Instrument preference accuracy (`sign(μ_θ − μ_ref)` agrees with `sign(y_w − y_l)`), the drift `μ_θ − μ_ref`, and the correlation between μ and gold score.

### 5.5 GRPO, multiple targets (binary exact-match reward)

For each input `x_i`:
1. Sample a group of `G` scores from the policy: `a_{i,1..G} ~ Categorical(softmax(logits_θ(x_i)))`.
2. Reward `r_{i,j} = 1` if `a_{i,j} == y_i`, else `0`.
3. Group-relative advantage: `A_{i,j} = (r_{i,j} − mean_j(r_{i,·})) / (std_j(r_{i,·}) + ε)`.
4. Loss (single on-policy update per batch, so the importance ratio is 1 and clipping is inactive; add PPO-style clipping only if multiple epochs per batch are used):

```
L_GRPO(θ) = - (1/(B·G)) Σ_{i,j} A_{i,j} · log p_θ(a_{i,j} | x_i)  +  β_KL · KL(p_θ ‖ p_ref)
```

`p_ref` is the frozen initial policy (cold start) or the SFT checkpoint (sequential). `β_KL` may be 0 in the base configuration.

**Behaviors to instrument, not hide**
- **Degenerate groups:** if all `G` samples are right or all wrong, std = 0 and the advantage is 0. Log the fraction of groups with non-zero advantage.
- **Cold start:** a random head gives near-uniform samples, so the signal is weak early. This is why SFT → GRPO is included.
- **Compute:** the `G` draws come from one forward pass per input, so GRPO here is cheap but noisy.

### 5.6 GRPO, single target (regression; Gaussian policy)

The regression head is the mean of a Gaussian policy over the standardized score scale, `π_θ(a | x) = N(μ_θ(x), σ_g²)`, with a **fixed** exploration scale `σ_g` (config `grpo_sigma`, a hyperparameter searched like `G` and `β_KL`). It plays the role of σ in §5.4 but is a real sampling scale here, not folded into β'.

For each input `x_i`:
1. Sample `G` actions `a_{i,j} ~ N(μ_θ(x_i), σ_g²)` (continuous, standardized scale).
2. Map each action to the **nearest observed score** on the standardized scale (a fixed rule, no dev-set information) and set `r_{i,j} = 1` if that score equals the gold score, else 0. This is the same binary exact-match reward as §5.5. The dev-fit thresholds of §4 are used only to score predictions at evaluation, never in the reward.
3. Advantages as in §5.5: `A_{i,j} = (r_{i,j} − mean_j(r_{i,·})) / (std_j(r_{i,·}) + ε)`.
4. Loss, single on-policy update per batch:

```
L_GRPO-Reg(θ) = - (1/(B·G)) Σ_{i,j} A_{i,j} · log N(a_{i,j}; μ_θ(x_i), σ_g²)  +  β_KL · (1/B) Σ_i (μ_θ(x_i) − μ_ref(x_i))² / (2 σ_g²)
```

The second term is the exact `KL(N(μ_θ, σ_g²) ‖ N(μ_ref, σ_g²))`. `μ_ref` is the frozen initial model (cold start, E12) or the regression-SFT checkpoint (sequential, E13).

**Behaviors to instrument, not hide**
- **Degenerate groups:** if all `G` samples fall in the gold score's bin, or none do, the advantage is 0. With a cold-start head `μ ≈ 0` and a small `σ_g`, most groups for gold scores far from the mean are all-wrong, so the signal is weaker than in the classification case. This is why SFT → GRPO (E13) is included.
- **Scale:** the gradient on μ is proportional to `A · (a − μ) / σ_g²`, so `σ_g` sets both exploration and effective step size (gradient clipping applies). It is the one hyperparameter with no classification counterpart.
- **Logged diagnostics:** mean reward, fraction of groups with non-zero advantage, drift `μ_θ − μ_ref` (signed and absolute), KL to the reference, and the correlation between μ and the gold score.
- **Sampling:** a Gaussian action can land outside the observed score range; it is mapped to the nearest end score, so tail samples are not discarded.

## 6. Experiments

Thirteen experiments are run for every `EssaySet`: ten core experiments (E1–E8, E12, E13) and three continued-SFT controls (E9–E11). E12 and E13 were added to complete the grid (regression GRPO, cold and after SFT); the numbering keeps E1–E11 unchanged. "MC" = multiple-target (classification) head; "Reg" = single-target (regression) head.

### 6.1 Core experiments

| ID | Head | Training | Init / reference | Role |
|---|---|---|---|---|
| E1 | MC | SFT (cross-entropy) | pretrained + random head | **Baseline (MC)** |
| E2 | MC | DPO | cold start; `π_ref` = initial model | Objective from scratch |
| E3 | MC | GRPO | cold start; `p_ref` = initial model | Objective from scratch |
| E4 | MC | SFT → DPO | init from E1; `π_ref` = E1 | Sequential |
| E5 | MC | SFT → GRPO | init from E1; `p_ref` = E1 | Sequential |
| E6 | Reg | SFT (MSE) | pretrained + random head | **Baseline (Reg)** |
| E7 | Reg | DPO | cold start; `μ_ref` = initial model | Objective from scratch |
| E8 | Reg | SFT → DPO | init from E6; `μ_ref` = E6 | Sequential |
| E12 | Reg | GRPO (Gaussian policy, §5.6) | cold start; `μ_ref` = initial model | Objective from scratch |
| E13 | Reg | SFT → GRPO (Gaussian policy, §5.6) | init from E6; `μ_ref` = E6 | Sequential |

### 6.2 Controls: continued SFT

Without these, any gain from "SFT → X" cannot be separated from simply training longer. Each control starts from the same stage-1 checkpoint as the arm it controls, then continues the same SFT objective for the same stage-2 budget (same number of additional examples seen and the same LR schedule family and warmup), with checkpoint selection by dev QWK as in the controlled arm.

| ID | Head | Training | Init | Matched to | Role |
|---|---|---|---|---|---|
| E9 | MC | SFT → SFT (continued cross-entropy) | init from E1 | Stage 2 of E4 (DPO) | Extra-training control |
| E10 | MC | SFT → SFT (continued cross-entropy) | init from E1 | Stage 2 of E5 (GRPO) | Extra-training control |
| E11 | Reg | SFT → SFT (continued MSE) | init from E6 | Stage 2 of E8 (DPO) and E13 (GRPO) | Extra-training control |

### 6.3 Planned comparisons

- **Objectives from scratch (RQ1):** E1 vs E2 vs E3.
- **Sequential vs SFT alone (RQ2, RQ4):** E1 vs E4, E1 vs E5, E6 vs E8, E6 vs E13.
- **Sequential vs matched extra training (RQ2, RQ4):** E4 vs E9, E5 vs E10, E8 vs E11, E13 vs E11. These are the decisive comparisons for whether DPO or GRPO adds anything beyond more SFT.
- **Extra training alone:** E1 vs E9, E1 vs E10, E6 vs E11.
- **Head type (RQ3):** E1 vs E6; also E2 vs E7 and E4 vs E8 for DPO, and E3 vs E12 and E5 vs E13 for GRPO.
- **Cold start vs warm start:** E2 vs E4, E3 vs E5, E7 vs E8, E12 vs E13.
- **Regression objectives (RQ4):** E6 vs E7 vs E12 from scratch; E8 vs E13 after SFT (DPO vs GRPO on the same head).

### 6.4 Tuning protocol and compute

- Every arm gets the same tuning budget (number of configurations and seeds). Searched hyperparameters: learning rate (all arms); `β_dpo` (E2, E4, E7, E8); group size `G` and `β_KL` (E3, E5, E12, E13); Gaussian policy scale `σ_g` (E12, E13; log-uniform on [0.1, 1] in standardized units); σ or equivalently `β'` (E7, E8). Controls reuse the learning rate selected for their stage-1 SFT arm. Report the search space and the selected values. Selection is by dev QWK.
- **Assumption:** to keep the budget tractable across 10 `EssaySet`s, one hyperparameter configuration is selected per arm using dev QWK averaged across sets, rather than tuning separately for each set. Revisit if the sets differ greatly in size or `K`.
- Scale of the study: 11 experiments × 10 `EssaySet`s × 5 seeds is 550 final runs, before tuning. Sequential arms and controls reuse stage-1 checkpoints, so stage 1 is trained once per set and seed.

### 6.5 Optional ablations (out of scope for the headline study)

DPO with all `K−1` rejected scores averaged vs one random rejected score; fixed vs resampled rejected pairs; low-data regimes (5%, 10%, 25% of training data); on-policy DPO for regression; ordinal partial-credit rewards or ordinal-aware pairing; ordinal losses (EMD/CORAL) for classification.

### 6.6 Generative-model experiments (G1–G4)

A prompted decoder model replaces the fine-tuned encoder. Nothing is trained, so there are no seeds and no checkpoints. The model is `Qwen/Qwen3.5-4B`, served by vLLM at `http://10.151.66.132:8000/v1` (OpenAI-compatible; override with `VLLM_API_BASE`). Compute is external, so these runs are independent of the GPU jobs above. Code: `src/gen_common.py`, `src/gen_zeroshot.py`, `src/gen_mipro.py`, `src/gen_retry.py`.

**Servers.** The first runs used `10.151.66.132`, which has since been switched to another model (it now reports `google/gemma-4-26B-A4B-it`). Later runs use `10.151.66.140` and `10.151.66.182` (both `Qwen/Qwen3.5-4B`), selected with `VLLM_API_BASE`; the default is `.140`. `check_server()` aborts a run unless the server reports `Qwen/Qwen3.5-4B`, so a wrong host cannot silently score every reply as the lowest score point, and each `metrics.json` records `api_base` for the runs made after this change. G2 and G4 are split by `EssaySet` across the two servers (odd sets on `.140`, even sets on `.182`).

| ID | Thinking | Rubric | Role |
|---|---|---|---|
| G1 | off | original (`data/rubrics_md/rubric_<k>.md`, as-is) | Zero-shot, no thinking |
| G2 | on | original | Zero-shot, thinking |
| G3 | off | MIPROv2-optimized | Rubric optimization, no thinking |
| G4 | on | MIPROv2-optimized | Rubric optimization, thinking |

**Fixed prompt structure (held constant across G1–G4).** DSPy with a minimal custom adapter (`MinimalAdapter`):
- system message = the rubric + a fixed line that also states the valid scores of that set, e.g. `Reply with only the integer score: one of 0, 1, 2, 3.` The line is not part of the optimized text, so the optimizer cannot drop score points (a first G3 run, with a suffix that did not list the scores, rewrote set 6's rubric into a binary 0/1 verdict and never predicted 2 or 3)
- user message = the student response, unchanged
- output = one integer; the first integer after the last `</think>` is taken (or in the whole reply when thinking is off).
- no few-shot demos (`max_bootstrapped_demos = max_labeled_demos = 0`), no chain-of-thought module, no other fields.

The rubric is the signature's instructions, so it is the **only** thing MIPROv2 can rewrite; the suffix, the user message, the decoding settings and the adapter are not touched. G1 and G2 are exactly MIPROv2's starting program, so they must use the same suffix, so G3 vs G1 and G4 vs G2 isolate the effect of the rubric optimization.

**Thinking** is switched with the vLLM chat-template argument `enable_thinking` (the server's default is on). The server has no reasoning parser, so the reasoning arrives inline in the message content ending in `</think>`.

**Decoding (fixed per mode).** Thinking off: temperature 0, 64 max tokens. Thinking on: temperature 0.6, top-p 0.95, 16384 max tokens (Qwen's recommended sampling for thinking). The thinking runs are therefore stochastic; a reply truncated before `</think>` has no answer.

**MIPROv2 setup.** `auto="light"`, seed 0, 100 randomly drawn training examples as the trainset and the full dev split as the valset, one optimization per `EssaySet` and mode (10 sets × 2 modes). The instruction proposer is the same served model with thinking off. The optimization metric is a per-example quadratic agreement, `1 − (ŷ − y)² / (max − min)²`, with 0 for an unparseable reply: QWK needs the whole set, so this surrogate keeps the same preference for near misses. The test split is read once, after the optimized rubric is fixed. The optimized rubric is saved to `results/<G>/default/seed0/set<k>/rubric.md`.

**Score-point constraint (G3, G4).** The suffix lists the valid scores, but an optimized rubric can still contradict it (in the second G3 run, set 8's rubric said "assign a score from 1 to 2" and the model never predicted 0, giving test QWK 0.185 against 0.440 for G1). After optimization, a rubric is accepted only if every score point of the set appears next to the word "score" or "point" on the same line (`covers_all_scores` in `src/gen_common.py`; all ten original rubrics pass). A rejected rubric is saved as `rubric_rejected.md`, the original rubric is used for that set, and `rubric_fallback: true` is recorded in `metrics.json`; such a set then equals G1 (or G2) up to sampling noise. The check is a simple text match, so a rubric can pass it and still bias the scores; it is not a guarantee. The number of fallback sets is reported with every G3/G4 mean.

**Scoring and reporting.** The same metrics, splits and bootstrap CIs as §7, written to the same `metrics.json` layout. A reply that is unparseable, truncated, or outside the set's score points is a parse failure. A parse failure is scored as the lowest score point, and the rate is reported (`dev_fail_rate`, `test_fail_rate`) beside every result so that it cannot be hidden in QWK.

**Retry rule for parse failures (applied to G2; apply the same rule to G4).** After the main run, each parse failure is re-sampled, up to 3 further attempts (`src/gen_retry.py`), and replaced by the first parseable answer. A retry is triggered only by a parse failure, never by the correctness of the answer, so the gold scores play no part. Metrics are recomputed after the retries; the pre-retry metrics are kept in `metrics.json` under `before_retry`, and the number of replies retried is recorded under `retry`. A reply still unparseable after 3 attempts is scored as the lowest score point. Because the thinking runs sample at temperature 0.6, a retried reply can score differently from an ordinary one. The non-thinking runs (G1, G3) are greedy: they had no parse failures in G1, and re-sampling a greedy reply would return the same text, so the rule does not apply to them. G2 result on all 10 sets: mean test QWK 0.511 before retries and 0.524 after (G1: 0.499); 226 test and 87 dev replies were retried and 3 test replies still had no answer.

**Planned comparisons.** G1 vs G2 (thinking, original rubric); G3 vs G4 (thinking, optimized rubric); G1 vs G3 and G2 vs G4 (effect of rubric optimization); G1–G4 vs E1 and E6 (generative vs fine-tuned encoders). The same paired Wilcoxon test across the 10 `EssaySet`s applies; there are no seeds, so per-set seed tests do not.

### 6.7 Generative fine-tuning experiments (G5–G8)

The same `Qwen/Qwen3.5-4B` is now trained, so these arms are the decoder analogues of E1 (SFT) and E2/E4 (DPO). Both start from the optimized rubrics of §6.6 and use the same fixed prompt structure, split, parse rule and metrics, so G5 vs G3 and G6 vs G4 isolate what training adds to prompt optimization. Code: `src/gen_local.py` (shared), `src/gen_sft.py` (G5), `src/gen_collect.py` (G6 collection), `src/gen_dpo.py` (G6 training) and `src/gen_dpo_eval.py` (G6 evaluation). All were smoke-tested on `EssaySet` 1 with `--limit` (nothing written): G5 trains and selects on dev; G6 collection, DPO training (peak 12.9 GB GPU memory for a 5,828-token completion) and local vLLM evaluation with the LoRA adapter run end to end.

| ID | Thinking | Rubric | Training | Compute |
|---|---|---|---|---|
| G5 | off | G3's optimized rubric (original where G3 fell back) | SFT on `Answer: {score}` | local GPU (this machine) |
| G6 | on | G4's optimized rubric (original where G4 fell back) | DPO on pairs of sampled thinking + answer | data from the vLLM servers; training local |
| G7 | on | as G6 | SFT on right thinking replies, then DPO on the G6 pairs | as G6 |
| G8 | on | as G6 | SFT on right thinking replies only (stage 1 of G7) | as G6 |

**G5 (SFT, no thinking).** One model per `EssaySet`. Each training example becomes one chat: system = the set's optimized rubric (`results/G3/default/seed0/set<k>/rubric.md`) + the fixed suffix, user = the student response, assistant = `Answer: {score}` with the gold score (the standard reply). Loss is on the assistant tokens only. The suffix is changed for G5 to ask for that form (`Reply in the form "Answer: <score>", where <score> is one of 0, 1, 2, 3.`, scores per set), so that training and inference prompts agree; the parser (first integer in the reply) is unchanged. Thinking is off in training and at inference. Inference: greedy, 64 max tokens, as G3. Training is local, in bf16 with LoRA (see *LoRA settings* below); checkpoint selection uses dev QWK only, with the same rule as E1.

**G6 (DPO, thinking).** Two stages.
1. *Collection (vLLM instances at `10.151.66.140` and `10.151.66.182`, split by `EssaySet` as in G2/G4).* For every **training**-split example of a set, sample 10 independent replies with thinking on, using G4's prompt (the set's optimized rubric + suffix) and G4's decoding (temperature 0.6, top-p 0.95, 16384 max tokens). Keep each reply whole (thinking and answer) with its parsed score. A reply is *right* if the parsed score equals the gold score (exact match, as the GRPO reward in §5.5) and *wrong* if it parses to a different score. Replies that do not parse (truncated, no integer) are kept in the collection file but never used in a pair. The 10 replies are stored per example in `results/G6/collect/set<k>.jsonl` (the test split is never sampled).
2. *Pair construction.* For every training example with at least one right and at least one wrong reply among its 10, form exactly one pair: chosen = a right reply and rejected = a wrong reply, each drawn uniformly at random (seeded) from its group. Only examples with at least one parsable right reply and at least one parsable wrong reply are used; examples whose parsable replies are all right, all wrong, or absent give no pair. The pairs form the DPO training set for that `EssaySet`.
3. *DPO (local; `src/gen_dpo.py`, then `src/gen_dpo_eval.py`).* The start point is the base model (no generative model has been fine-tuned before this step). Policy and reference are the base model with LoRA (reference = adapter disabled), the standard DPO loss of §5.3 over the whole reply (thinking and answer). Log-probabilities are summed over the whole completion. A pair whose chosen or rejected completion is longer than `--max_completion_tokens` (default 4096, GPU memory) is dropped; the number dropped and the length quantiles are recorded in `train.json`, so the effect of the cap on the data is visible. Inference is G4's: thinking on, temperature 0.6, top-p 0.95, 16384 max tokens, with the retry rule of §6.6 for parse failures, run with local vLLM serving the adapter (`gen_dpo_eval.py`).

**LoRA settings (G5 and G6, standard values, fixed, not tuned).** bf16 weights and compute; LoRA rank 16, alpha 32, dropout 0.05, on all linear layers of the decoder (PEFT `all-linear`, i.e. everything except the output head); AdamW, cosine schedule, 3% warmup, gradient clipping 1.0, gradient checkpointing. G5: learning rate 2e-4, effective batch 16, 2 epochs. G6: learning rate 5e-5, DPO β = 0.1, effective batch 8, **3 epochs** (config c3 of `src/gen_dpo_sweep.py`, selected on dev QWK from 6 configurations on `EssaySet` 1, one tuning seed, original 487-pair pool; dev QWK 0.782 vs 0.736 for the 1-epoch default and 0.702 for G4; learning rates of 1e-4 and above broke generation, 28–64% parse failures; written to `configs/tuned_G6.json`). The G6 pairs are drawn from `results/G6/collect_merged` (original replies pooled with the temperature-1 resample of `src/gen_resample.py`, see `src/gen_merge_collect.py`), with a 16384-token cap as in the sweep. G5 values are common defaults, not tuned.

**G7 (SFT then DPO, thinking).** Same collected replies, prompt and decoding as G6 (`src/gen_sft_dpo.py`, evaluated with `src/gen_dpo_eval.py --exp G7`). *Stage 1:* every training example with at least one parsable right reply, i.e. the examples that give a DPO pair and the examples whose parsable replies are all right, contributes one SFT target, a right reply drawn at random (seeded), whole (thinking and answer); loss is the mean token cross-entropy over the completion. On `collect_merged` that is 12,267 of 15,335 examples (6,801 pair + 5,466 only-right); examples with no right reply are not used. *Stage 2:* DPO on exactly G6's pairs (same seed, so the same pairs), continuing the same LoRA adapter, with the **SFT model as the DPO reference** (its log-probs are computed once, before stage 2). Stage 2 uses G6's tuned lr, β and epochs (`configs/tuned_G6.json`) so that G7 vs G6 isolates the SFT stage. The stage-1 hyperparameters (lr 5e-5, 1 epoch, batch 8) are **not tuned**. The SFT target and the DPO chosen reply of a pair example may be the same reply. Self-generated right replies already have a low token loss (about 0.06 per token in a smoke test), so stage 1 mostly sharpens the model toward its own correct reasoning (rejection-sampling fine-tuning) rather than teaching new content. **G8** is stage 1 alone (`python src/gen_sft_dpo.py --sft_only`; evaluated with `src/gen_dpo_eval.py --exp G8`; results in `results/G8`), the SFT-only arm and the matched control for G7: G8 vs G4 is the effect of SFT, G7 vs G8 the effect of DPO after SFT, G7 vs G6 the effect of SFT before DPO. G7 also saves its stage-1 adapter as `seed<s>_sft`, but G8 is the reportable run, since it is the one with its own `train.json` and `metrics.json`. When a G8 adapter exists for the same set and seed with the same SFT settings and data, G7 loads it instead of repeating stage 1 (`--no_reuse` forces retraining), so G7 starts from exactly the model G8 evaluates; run G8 first.

**Reporting.** As G1–G4 (§7 metrics, same layout in `results/G5`, `results/G6`, bootstrap CIs, parse-failure rate beside every result), plus the number of examples, the share of examples that give a pair, and the right-reply rate of the 10 samples (the pass rate) for G6. G5 and G6 are trained, so they have seeds: start with seed 0, tuning seeds are ≥100 and never reported, as for E1–E11. Hyperparameters are chosen on dev only.

**Planned comparisons.** G5 vs G3 (SFT vs prompt only, no thinking); G6 vs G4 (DPO vs prompt only, thinking); G5 vs G6 (SFT without thinking vs DPO with thinking: two things differ at once, so read it as a system comparison, not an objective comparison); G5, G6 and G7 vs E1, E2 and E6 (generative vs encoder); G7 vs G6 (does an SFT stage on right replies help DPO), G7 vs G8 (does DPO add to SFT), G8 vs G4 (SFT alone vs prompt only), G7 vs G4 (system effect of SFT + DPO).

## 7. Metrics

All metrics are computed per `EssaySet` on the test set (and on dev for model selection). Let `y` be the gold score and `ŷ` the predicted score on the original score scale, obtained as described in §4. `K` below is the number of score points in that `EssaySet`.

**1. Quadratic Weighted Kappa (QWK)** (primary)

```
QWK = 1 − Σ_ij w_ij · O_ij / Σ_ij w_ij · E_ij,    w_ij = (i − j)² / (K − 1)²
```

`O` is the observed confusion matrix of gold vs. predicted scores and `E` is the expected matrix under independence of the two marginals.

**2. Standardized Mean Difference (SMD)**

```
SMD = (mean(ŷ) − mean(y)) / sqrt( (var(ŷ) + var(y)) / 2 )
```

Signed, so the direction of bias is visible (positive = over-scoring on average). Computed on original score values, using the discretized predicted scores. A common operational threshold in automated scoring is |SMD| ≤ 0.15.

**3. Accuracy (exact match)**

```
Accuracy = (1/N) Σ_i 1[ŷ_i = y_i]
```

**4. KL divergence and Jeffreys divergence between the true and predicted score distributions**

Let `P` be the empirical distribution of true scores over the test set of an `EssaySet` and `Q` the empirical distribution of predicted scores over the same set, each over the `K` score points (`P_k = count(y = k)/N`, `Q_k = count(ŷ = k)/N`).

```
KL(P ‖ Q)     = Σ_k P_k · log(P_k / Q_k)
Jeffreys(P,Q) = KL(P ‖ Q) + KL(Q ‖ P)
```

- Natural log (nats). `KL(P ‖ Q)` is directional with the true distribution as the reference; Jeffreys is its symmetric version.
- **Smoothing:** a model that never predicts some score gives an infinite divergence. Apply the same additive smoothing to the counts of both distributions (default `α = 0.5` per class, then renormalize), report `α`, and keep it fixed across all arms and sets.
- **What this measures:** distribution-level fidelity, not per-example correctness. A model can have zero divergence and still be wrong on every example, so read these alongside QWK and accuracy. Per-example KL against a one-hot true label reduces to negative log-likelihood and is deliberately not used.

**Aggregation, uncertainty, and reporting**
- **Per set:** mean ± std over seeds for every metric, for every `EssaySet` and arm.
- **Headline:** the unweighted mean over the 10 `EssaySet`s of each metric (each set's value being its mean over seeds). Because `K` and difficulty differ by set, also report per-set tables and do not pool examples across sets.
- **Significance:** for each planned comparison in §6.3, a paired Wilcoxon signed-rank test across the 10 `EssaySet`s (using per-set seed means), plus per-set paired bootstrap or Wilcoxon tests across seeds.
- Bootstrap confidence intervals over test examples for single runs.
- Learning curves (dev QWK vs. examples seen), a table of true vs. predicted score counts for the best run per arm, and confusion matrices as descriptive figures only (not metrics).

**Training diagnostics (not evaluation metrics)**
- GRPO: mean reward, fraction of non-degenerate groups, policy entropy, KL to the reference policy
- DPO (MC): preference accuracy, mean margin, KL to the reference policy
- DPO (Reg): preference accuracy, drift `μ_θ − μ_ref`, correlation of μ with gold score
- Gradient norm statistics per arm, used to check the GRPO `p_y · ∇log p_y` relationship and the single-pair-DPO lower-bound-of-CE argument from §2

## 8. Risks and Threats to Validity

- **Hyperparameter fairness:** GRPO has `G`, `β_KL`, and advantage normalization; DPO has `β_dpo` and pairing strategy; regression DPO adds σ. Give all arms equal tuning budgets. Sharing one configuration across `EssaySet`s (§6.4) may favor some sets over others.
- **Threshold advantage for regression:** regression arms fit `K−1` thresholds on dev, which classification arms do not. Report the rounding-based results as well, and keep the procedure identical across regression arms.
- **Extra-training confound:** SFT → X arms see more total training than SFT alone. Controls E9–E11 address this; the comparison is valid only if the stage-2 budgets are truly matched, so log and verify them.
- **Small sets and rare scores:** per-set training data may be small and some score points rare, which makes DPO's random rejection, GRPO's exact-match reward, and the KL / Jeffreys estimates noisy. Smoothing (§7) keeps divergences finite but does not remove this noise.
- **Label noise/rater disagreement:** `Score1` is a single rater's score. All objectives treat it as ground truth; if human agreement is low, every arm is bounded by it. Note which score the test file's solution provides, and confirm it is the same scoring convention.
- **Class imbalance:** exact-match reward, CE, and uniform random rejection favor majority classes differently (with DPO, rare classes appear more often as rejected than as chosen). Interpret KL / Jeffreys divergence between true and predicted distributions with this in mind.
- **Per-prompt models only:** each model sees one prompt, so results say nothing about generalization to unseen prompts.
- **Single backbone/dataset:** conclusions apply to ModernBERT-base on this dataset.
- **Framing risk:** because the action space is finite (and, for regression, a scalar), avoid implying GRPO or DPO is the natural tool. The study asks whether they are competitive and what they change.
- **Generative arms (G1–G4):** (a) The rubric files are used as-is, including their header table and prompt text, so G1–G4 see more task context than the fine-tuned encoders, which see only the response. (b) Thinking runs are sampled (temperature 0.6) and use one run per set, so their sampling noise is not measured; the 16384-token cap truncates some replies, which are retried under the rule in §6.6. (c) MIPROv2 optimizes a surrogate metric on 100 training examples and the dev split, so its gains can be noise on small sets; the dev split is also used to pick the rubric, so dev QWK is optimistic for G3/G4. (d) The proposer and the scorer are the same served model. (e) A shared, externally hosted server means latency and throughput vary, and the model weights are not pinned by this repo; record the server's reported model id and date with each run. (f) Student responses are sent to the vLLM server; confirm it is an approved internal host before running at scale.
- **Generative training arms (G5–G6):** (a) The optimized rubrics were chosen with the dev split and 100 training examples, so G5 and G6 start from a rubric tuned on data they also select checkpoints on; dev QWK is optimistic. (b) G6's pairs come from one example's own 10 samples, so DPO sees only examples the model sometimes gets right and sometimes wrong; examples it always or never gets right are dropped, which skews the pairs toward medium-difficulty items and the pair count shrinks if the model is mostly right or wrong. (c) A thinking reply can be thousands of tokens, so a pair of two replies may not fit 24 GB for a 4B model with LoRA; a length cap on pairs would then drop long replies and bias the data towards short reasoning (see §10). (d) Choosing the chosen and rejected reply uniformly at random ignores reply length, so a length difference between right and wrong replies can be learned by DPO instead of correctness. (e) G5 changes the suffix and G6 trains on the answer inside the same reply, so neither is a pure change of one factor against G3/G4 except the training itself. (f) Student responses go to the vLLM servers during collection; confirm they are approved internal hosts (as for G1–G4).
- **Gaussian-policy assumption:** the regression DPO analysis rests on a fixed-σ Gaussian policy. It is a modeling choice, not a property of the regression head.

## 9. Milestones

1. Data loading and validation per `EssaySet`: join test scores, label mapping, dev split, duplicate checks; E1 and E6 baselines on one pilot set.
2. DPO (MC and Reg) and GRPO implementations with unit tests (reward, advantage, degenerate groups; margin identities: logit difference for MC, `β'(y_w − y_l)(μ − μ_ref)` for Reg).
3. Threshold-fitting and evaluation pipeline; verify metrics on known inputs.
4. Pilot all of E1–E13 on one `EssaySet` and a small seed count to confirm that budgets match for the controls and to estimate total compute.
5. Run E1–E13 across all 10 `EssaySet`s and 5 seeds.
6. Generative arms: G1/G2 on all 10 `EssaySet`s, then MIPROv2 (G3/G4), after a small smoke test of each script.
7. Generative training: G5 (local SFT) and G6 (collection from the vLLM servers, then local DPO), each smoke-tested on one `EssaySet` first.
8. Analysis: planned comparisons, gradient diagnostics, check of the §2 predictions (including the regression-DPO scale-anchoring prediction), and write-up.

## 10. Open Items

- Confirm the row identifier and score column names for joining `public_leaderboard_solution.csv` to `public_leaderboard_rel_2.tsv`.
- Confirm the vLLM server (`10.151.66.132:8000`) is an approved host for student text and will stay up for the full G1–G4 runs; MIPROv2 with thinking is the most expensive generative arm.
- **G5/G6 decisions settled:** G5 uses G3's rubrics and G6 uses G4's; G6's DPO starts from the base model; unparsable samples are excluded and only examples with both a parsable right and a parsable wrong reply give a pair; both arms use bf16 + LoRA with the standard values in §6.7. **G5/G6 still open:** (3) **Pair length:** with chunked log-probabilities a 5,828-token completion fits in 12.9 GB, so the 4096-token default cap is probably conservative; in a 3-example smoke sample 17% of replies exceeded 4096 tokens (median 1,228, max 6,088), so the cap would drop a visible share of pairs. Decide the cap after collecting one full set. (4) **How G6 is evaluated:** local vLLM 0.28 serves the LoRA-adapted model (smoke-tested: it loads the adapter and generates); the 16384-token thinking runs are still slow on one laptop GPU, and the full-run cost is unmeasured. (5) Collection size: 15,335 training examples × 10 samples = 153,350 thinking replies in total (1,150–1,627 examples per set); the cost has not been measured, so measure tokens per second on one set first and consider a per-set sample of training examples if it is too slow.
- Compute available (GPU type, hours), which determines seed count and tuning breadth given the 10-set scale (§6.4).
- **Superseded generative runs.** The first G1 and G3 runs used the suffix `Reply with only the integer score.` without the score list (G1 mean test QWK 0.499, G3 0.496). They are kept in `results/_superseded_v1/` and replaced by reruns with the suffix in §6.6. A second G3 run (suffix with scores, no score-point constraint; mean test QWK 0.526 against 0.456 for G1) is in `results/_superseded_v2/`; it was replaced by a run with the score-point constraint. The first G2 run (with its retry; mean test QWK 0.524) used the old suffix and is kept in `results/_superseded_v1/G2/`; G2 is being rerun with the current suffix, and G4 runs first.
- **G3 with the score-point constraint (current).** Mean test QWK 0.508 over 10 sets against 0.456 for G1 (+0.052). Sets 3 and 6 fell back to the original rubric (their optimized rubrics dropped a score point), so they repeat G1 up to sampling noise (0.364 vs 0.360 and 0.575 vs 0.581). Set 8, whose rubric broke the score range in the unconstrained run, passed the check this time, so the rule was not exercised on it. Rubrics differ between runs because the proposer is sampled.
