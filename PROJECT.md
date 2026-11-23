# SFT vs DPO vs GRPO for Short Answer Scoring with ModernBERT (Classification and Regression Heads)

A research comparison of fine-tuning objectives for an automated short answer scorer built on `answerdotai/ModernBERT-base`, crossing two **head types** with three **objectives**:

- **Multiple-target head (classification):** `K` logits, one per score point; the model is a categorical policy over scores.
- **Single-target head (regression):** one scalar output `μ_θ(x)`; treated as the mean of a Gaussian policy over scores when a policy is needed.
- **Objectives:** SFT (cross-entropy for classification, MSE for regression), DPO (gold score preferred over a random alternative score), and GRPO (group-relative policy optimization with a binary exact-match reward). Every objective is trained directly from the pretrained backbone; there is no SFT-then-DPO/GRPO stage.

A separate model is trained for each `EssaySet` (prompt) in the dataset, so every experiment below is run once per `EssaySet`.

**Extension (§6.5):** five decision-model experiments (D1–D5) score the same test sets with `vllm-sr/d3-mini`, a Qwen3.5-4B-based classifier, zero-shot, with an optimized rubric, and fine-tuned with LoRA (SFT, DPO, GRPO). They are comparators for the fine-tuned encoders, not part of the E-arm training study.

---

## 1. Research Questions

- **RQ1 (objectives from scratch, classification).** Starting from the same initialization, how do DPO and GRPO compare with cross-entropy SFT on QWK, SMD, accuracy, and score-distribution fidelity (KL / Jeffreys)?
- **RQ3 (head type).** Under SFT, does a single-target regression head differ from a multiple-target classification head, given that the MSE loss carries ordinal information that cross-entropy does not?
- **RQ4 (preference training for regression).** Do DPO and GRPO work with a regression head from a cold start, relative to regression SFT?
- **RQ5 (stability and imbalance).** How do the objectives differ in training stability, sample efficiency, and drift of the predicted score distribution toward or away from common scores?
- **RQ6 (decision model).** How does `d3-mini` with the human rubric compare with the fine-tuned ModernBERT arms, how much does optimizing only the rubric text with MIPROv2 close the gap, and do LoRA SFT, DPO and GRPO on the same model behave as they do on the encoder?

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
- DPO is offline (pairs come from the labeled data) and needs a reference policy. From a random head the reference margin is near zero, so cold-start DPO is close to reference-free. With no SFT stage, the reference term stays near zero throughout, so DPO here is effectively reference-free.

**DPO (regression)**
- Model the policy as `π_θ(a | x) = N(μ_θ(x), σ²)` with fixed σ. Then for a pair (chosen `y_w`, rejected `y_l`):

  ```
  log π_θ(y_w|x) − log π_θ(y_l|x) = (y_w − y_l) · (μ − (y_w + y_l)/2) / σ²
  ```

  and, after subtracting the reference's identical expression, the DPO margin becomes

  ```
  margin = β · (y_w − y_l) · (μ_θ − μ_ref) / σ²
  ```

- This margin depends on μ only through its **displacement from the reference** (the pair midpoint cancels). It rewards moving μ up when the gold score is above the rejected one and down otherwise, but it never anchors μ at `y_w`. From a cold start, the net push for a given gold score is roughly toward `y − mean(other scores)`, which preserves the *direction* of the signal (higher gold, higher μ) but not the *scale*. **Prediction:** cold-start DPO on a regression head learns a roughly monotone ranking signal with an unanchored scale, so it needs validation-tuned thresholds to be useful and may underperform regression SFT. Note that `β` and `σ²` enter only as `β/σ²`, so there is a single effective temperature. This derivation is mine; verify it empirically (it is part of the analysis in §9).
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
- **Seeds:** minimum 5 seeds per configuration. The same seed gives the same backbone initialization and head initialization across arms, so paired comparisons are valid.
- **Budget accounting:** report examples seen and wall-clock for every arm.

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

Reference `π_ref`: frozen initial model. Instrument preference accuracy (`z_w > z_l`), mean margin, and the gap between preference accuracy and top-1 accuracy. If `K = 2`, the rejected score is deterministic.

### 5.4 DPO, single target (regression)

Same pairs as 5.3, with the Gaussian-policy margin from §2:

```
L_DPO-Reg(θ) = - (1/B) Σ_i log σ( β' · (y_w − y_l) · (μ_θ(x_i) − μ_ref(x_i)) ),    β' = β / σ²
```

(`y` values on the standardized scale.) Reference `μ_ref`: frozen initial model. Instrument preference accuracy (`sign(μ_θ − μ_ref)` agrees with `sign(y_w − y_l)`), the drift `μ_θ − μ_ref`, and the correlation between μ and gold score.

### 5.5 GRPO, multiple targets (binary exact-match reward)

For each input `x_i`:
1. Sample a group of `G` scores from the policy: `a_{i,1..G} ~ Categorical(softmax(logits_θ(x_i)))`.
2. Reward `r_{i,j} = 1` if `a_{i,j} == y_i`, else `0`.
3. Group-relative advantage: `A_{i,j} = (r_{i,j} − mean_j(r_{i,·})) / (std_j(r_{i,·}) + ε)`.
4. Loss (single on-policy update per batch, so the importance ratio is 1 and clipping is inactive; add PPO-style clipping only if multiple epochs per batch are used):

```
L_GRPO(θ) = - (1/(B·G)) Σ_{i,j} A_{i,j} · log p_θ(a_{i,j} | x_i)  +  β_KL · KL(p_θ ‖ p_ref)
```

`p_ref` is the frozen initial policy. `β_KL` may be 0 in the base configuration.

**Behaviors to instrument, not hide**
- **Degenerate groups:** if all `G` samples are right or all wrong, std = 0 and the advantage is 0. Log the fraction of groups with non-zero advantage.
- **Cold start:** a random head gives near-uniform samples, so the signal is weak early. Expect GRPO to learn slowly and to be the weakest arm.
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

The second term is the exact `KL(N(μ_θ, σ_g²) ‖ N(μ_ref, σ_g²))`. `μ_ref` is the frozen initial model (E12).

**Behaviors to instrument, not hide**
- **Degenerate groups:** if all `G` samples fall in the gold score's bin, or none do, the advantage is 0. With a cold-start head `μ ≈ 0` and a small `σ_g`, most groups for gold scores far from the mean are all-wrong, so the signal is weaker than in the classification case. Expect E12 to learn slowly.
- **Scale:** the gradient on μ is proportional to `A · (a − μ) / σ_g²`, so `σ_g` sets both exploration and effective step size (gradient clipping applies). It is the one hyperparameter with no classification counterpart.
- **Logged diagnostics:** mean reward, fraction of groups with non-zero advantage, drift `μ_θ − μ_ref` (signed and absolute), KL to the reference, and the correlation between μ and the gold score.
- **Sampling:** a Gaussian action can land outside the observed score range; it is mapped to the nearest end score, so tail samples are not discarded.

## 6. Experiments

Six experiments are run for every `EssaySet`: SFT, DPO and GRPO for each head type, all trained from the pretrained backbone with a random head (cold start). The IDs keep their original numbering (E4, E5, E8–E11 and E13, the SFT → DPO / GRPO arms and their continued-SFT controls, were dropped and archived in `results/_retired_sequential/`). "MC" = multiple-target (classification) head; "Reg" = single-target (regression) head.

### 6.1 Experiments

| ID | Head | Training | Init / reference |
|---|---|---|---|
| E1 | MC | SFT (cross-entropy) | pretrained + random head |
| E2 | MC | DPO | `π_ref` = initial model |
| E3 | MC | GRPO | `p_ref` = initial model |
| E6 | Reg | SFT (MSE) | pretrained + random head |
| E7 | Reg | DPO | `μ_ref` = initial model |
| E12 | Reg | GRPO (Gaussian policy, §5.6) | `μ_ref` = initial model |

### 6.2 Planned comparisons

- **Objectives, classification (RQ1):** E1 vs E2 vs E3.
- **Objectives, regression (RQ4):** E6 vs E7 vs E12.
- **Head type (RQ3):** E1 vs E6 for SFT, E2 vs E7 for DPO, E3 vs E12 for GRPO.

### 6.3 Tuning protocol and compute

- Every arm gets the same tuning budget (number of configurations and seeds). Searched hyperparameters: learning rate (all arms); `β_dpo` (E2, E7); group size `G` and `β_KL` (E3, E12); Gaussian policy scale `σ_g` (E12; log-uniform on [0.1, 1] in standardized units); σ or equivalently `β'` (E7). Report the search space and the selected values. Selection is by dev QWK.
- **Assumption:** to keep the budget tractable across 10 `EssaySet`s, one hyperparameter configuration is selected per arm using dev QWK averaged across sets, rather than tuning separately for each set. Revisit if the sets differ greatly in size or `K`.
- Scale of the study: 6 experiments × 10 `EssaySet`s × 5 seeds is 300 final runs, before tuning.

### 6.4 Optional ablations (out of scope for the headline study)

DPO with all `K−1` rejected scores averaged vs one random rejected score; fixed vs resampled rejected pairs; low-data regimes (5%, 10%, 25% of training data); on-policy DPO for regression; ordinal partial-credit rewards or ordinal-aware pairing; ordinal losses (EMD/CORAL) for classification.

### 6.5 Decision-model experiments (D1-D5)

`vllm-sr/d3-mini` (Decision 3.0, vLLM Semantic Router; built on Qwen3.5-4B, 4.54B parameters incl. a vision encoder) is a classifier, not a chat model: it generates no text and returns a probability for every option. It is served on `/v1/systemone` (vLLM >= 0.33.0, loaded with `trust_remote_code`; transformers 5.17.0 pinned by the model card), not on `/v1/chat/completions`, so it cannot use a Qwen3.5-4B chat server (which D2 uses only as the MIPROv2 instruction proposer). A request has `state` (the text to judge) and `questions`; a `score` question has 2-10 levels with `instructions` and per-level `criteria`; the prompt is fixed by the runtime (a top-level `instructions` field is rejected). Not stated by the model card or recipe (to be verified by a smoke test before D3): the response schema, whether the logits are differentiable through the HF class, and any fine-tuning support.

Mapping: `state` = the student response, one `score` question per essay set, `criteria` = the rubric's score levels, prediction = argmax over the level probabilities. Same split, test set, metrics (section 7) and failure accounting as E (a failed call counts as the lowest level).

| ID | Training | What changes |
|---|---|---|
| D1 | none | the original rubric as `instructions`/`criteria`; also report expected score and calibration from the probabilities |
| D2 | none | DSPy MIPROv2 optimizes the `instructions` and `criteria` text only (a thin DSPy module calls `/v1/systemone`; the proposer LM is a normal Qwen3.5-4B chat server ; no few-shot demos, the interface has no slot for them; candidates that drop a score level are rejected; selection on dev) |
| D3 | SFT, **LoRA** | D2's optimized text fixed in the request; cross-entropy on the gold level's probability (5.1) |
| D4 | DPO, **LoRA** | chosen = gold level, rejected = a level exactly one score point off (adjacent value, direction drawn at random with a fixed seed; one neighbour at the ends); preference loss on the level log-probabilities (5.3) |
| D5 | GRPO, **LoRA** | group of levels sampled from the output distribution, binary exact-match reward (5.5), KL to the reference |

**LoRA (D3-D5, decided by the user).** Standard settings (`LORA` in `src/d3_model.py`: rank 16, alpha 32, dropout 0.05, `all-linear`), bf16, gradient checkpointing, no full fine-tuning. Evaluation serves merged weights (earlier vLLM versions here did not apply adapters; a merged folder must also contain `decision_config.json`). Learning rate, beta and epochs are tuned on dev with tuning seeds >= 100, never reported.

**Decided.** D4 and D5 start from scratch, i.e. from the base d3-mini weights with a fresh LoRA adapter (no D3 adapter, no SFT stage), with D2's optimized text in the request; the reference for the DPO loss and the GRPO KL term is the base model with the adapter disabled. D3, D4 and D5 are therefore three parallel arms off the same starting point. Same data and splits as E. d3-mini is served on this machine (own venv `~/vllm034` with vLLM 0.34.0, because 0.34.0 pins torch 2.14 and `~/ml` keeps vLLM 0.28.0 with torch 2.13 for the training stack; transformers there is whatever 0.34.0 allows, 5.16.1-5.22, not the model card's 5.17.0 pin, because 5.17.0 needs huggingface-hub<2 and vLLM 0.34.0 needs >=2). The DSPy proposer LM for D2 is the Qwen3.5-4B server.

**Smoke test (2026-11-20).** (1) vLLM 0.34.0 serves d3-mini as a pooling model (`D3ForSequenceClassification`); it needs `--max-model-len` capped (4096 works; the default 262144 left no KV cache at 0.6 GPU utilization). (2) `/v1/systemone` request: `questions` is a **dict** `{name: {type: "score", instructions, criteria: [ordered list of level texts]}}` (a list of questions or dict criteria gives a 400). Response: `answers[name] = {score (expected value), legend, probabilities {"0": p0, ...}, confidence}`; levels are keyed by list index, so a set whose scores do not start at 0 needs an index mapping. (3) The model loads in `~/ml` (transformers 5.17.0, `trust_remote_code`) as `D3Model` (4.54B), but **the packaged class refuses training**: `train(True)` raises "d3 runs in inference mode only", `save_pretrained` raises NotImplementedError, dtype casts are refused, the backbone is frozen and eval, and logits are computed under `torch.inference_mode()`. The underlying computation is differentiable in principle: left-padded prompt tokens -> backbone (or the bidirectional `PromptEncoder`) -> last-position hidden state -> `hidden.float() @ readout.T` (a frozen 255-code readout from `readout.safetensors`) -> mask to the option count -> softmax(logits / temperature). **D3-D5 therefore need a custom trainer** that rebuilds this path with gradients and a LoRA adapter on the backbone, and a custom export (merge LoRA into the backbone shards in a copy of the package so vLLM can serve it; `MODEL_MANIFEST.json` hashes will no longer match). Not yet checked: that a re-implemented forward reproduces the served probabilities, and that vLLM serves the merged copy. Order-averaging (forward and reversed option order) is part of the runtime and must be decided for training.

**Custom trainer (built and checked 2026-11-20).** `src/d3_common.py` (request mapping: `data/rubrics_md` split into instructions + per-level criteria; `/v1/systemone` client), `src/d3_model.py` (`D3Policy`: the runtime's own tokenization and backbone, readout and temperature, with a gradient-enabled copy of its bidirectional encoder, per-layer checkpointing, and LoRA r16/alpha32/dropout 0.05 on every linear layer of the text decoder, 32.5M trainable parameters; fp32 adapter weights over a bf16 base), `src/train_d3.py --mode sft|dpo|grpo` (D3/D4/D5; losses as described in the table and in 5.1/5.3/5.5; D4's rejected level is one score point off, fixed per example with a seeded draw, instead of uniform over the other K-1; D4 and D5 reference = base model with the adapter off), `src/d3_export.py` (merges the adapter into a symlinked copy of the package, rewriting only the backbone shards), `src/d3_eval.py` (starts the vLLM 0.34.0 server of `~/vllm034` itself; D1/D2 on the base weights, D3-D5 on the merged copy, deleted afterwards unless `--keep`; metrics as every arm plus test NLL and the QWK of the rounded expected score). Checks: the forward matches the runtime's plain logits path to 6e-8 and the public `system_one` output to 0.010 (fast-path numerics); gradients reach all 496 adapter tensors; the vLLM-served merged weights reproduce the trainer's probabilities with the adapter on to 0.029 (bf16 rounding of the merge), same argmax on all 8 test essays; all three modes and the evaluator run end to end. Prompts are about 2,300 tokens for set 3 (the rubric files include the reading passage), peak 14 GB, about 14 s per step of 8 examples on the 24 GB card. Defaults are **not tuned** (SFT lr 1e-4, 2 epochs; DPO lr 5e-5, beta 0.1, 1 epoch; GRPO lr 2e-5, group 8, KL 0, 1 pass; batch 16). D3-D5 take `--text original` until D2 has written `results/D2/default/seed0/set<k>/prompt.json` (`instructions`, `criteria`), then `--text D2`. The merge changes the shards listed in `MODEL_MANIFEST.json`, so a runtime that verifies the package would flag them (vLLM serving does not). Check the package LICENSE before sharing any adapter or merged weights.

**Planned comparisons.** D1 vs D2 (prompt optimization), D3, D4 and D5 vs D2 and vs each other (SFT, preference and reward training from the same starting point), D1-D5 vs E1-E6.

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
- **Significance:** for each planned comparison in §6.2, a paired Wilcoxon signed-rank test across the 10 `EssaySet`s (using per-set seed means), plus per-set paired bootstrap or Wilcoxon tests across seeds.
- Bootstrap confidence intervals over test examples for single runs.
- Learning curves (dev QWK vs. examples seen), a table of true vs. predicted score counts for the best run per arm, and confusion matrices as descriptive figures only (not metrics).

**Training diagnostics (not evaluation metrics)**
- GRPO: mean reward, fraction of non-degenerate groups, policy entropy, KL to the reference policy
- DPO (MC): preference accuracy, mean margin, KL to the reference policy
- DPO (Reg): preference accuracy, drift `μ_θ − μ_ref`, correlation of μ with gold score
- Gradient norm statistics per arm, used to check the GRPO `p_y · ∇log p_y` relationship and the single-pair-DPO lower-bound-of-CE argument from §2


## 8. Risks and Threats to Validity

- **Hyperparameter fairness:** GRPO has `G`, `β_KL`, and advantage normalization; DPO has `β_dpo` and pairing strategy; regression DPO adds σ. Give all arms equal tuning budgets. Sharing one configuration across `EssaySet`s (§6.2) may favor some sets over others.
- **Threshold advantage for regression:** regression arms fit `K−1` thresholds on dev, which classification arms do not. Report the rounding-based results as well, and keep the procedure identical across regression arms.
- **Small sets and rare scores:** per-set training data may be small and some score points rare, which makes DPO's random rejection, GRPO's exact-match reward, and the KL / Jeffreys estimates noisy. Smoothing (§7) keeps divergences finite but does not remove this noise.
- **Label noise/rater disagreement:** `Score1` is a single rater's score. All objectives treat it as ground truth; if human agreement is low, every arm is bounded by it. Note which score the test file's solution provides, and confirm it is the same scoring convention.
- **Class imbalance:** exact-match reward, CE, and uniform random rejection favor majority classes differently (with DPO, rare classes appear more often as rejected than as chosen). Interpret KL / Jeffreys divergence between true and predicted distributions with this in mind.
- **Per-prompt models only:** each model sees one prompt, so results say nothing about generalization to unseen prompts.
- **Single backbone/dataset:** conclusions apply to ModernBERT-base on this dataset.
- **Framing risk:** because the action space is finite (and, for regression, a scalar), avoid implying GRPO or DPO is the natural tool. The study asks whether they are competitive and what they change.
- **Gaussian-policy assumption:** the regression DPO analysis rests on a fixed-σ Gaussian policy. It is a modeling choice, not a property of the regression head.

## 9. Milestones

1. Data loading and validation per `EssaySet`: join test scores, label mapping, dev split, duplicate checks; E1 and E6 baselines on one pilot set.
2. DPO (MC and Reg) and GRPO implementations with unit tests (reward, advantage, degenerate groups; margin identities: logit difference for MC, `β'(y_w − y_l)(μ − μ_ref)` for Reg).
3. Threshold-fitting and evaluation pipeline; verify metrics on known inputs.
4. Pilot all six E arms on one `EssaySet` and a small seed count to estimate total compute.
5. Run E1, E2, E3, E6, E7, E12 across all 10 `EssaySet`s and 5 seeds.
6. Decision-model arms: D1/D2 (inference, MIPROv2), then D3-D5 (LoRA) after a smoke test of each script.
7. Analysis: planned comparisons, gradient diagnostics, check of the §2 predictions (including the regression-DPO scale-anchoring prediction), and write-up.

## 10. Open Items

- Confirm the row identifier and score column names for joining `public_leaderboard_solution.csv` to `public_leaderboard_rel_2.tsv`.
- Compute available (GPU type, hours), which determines seed count and tuning breadth given the 10-set scale (§6.3).
