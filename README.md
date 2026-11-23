# RF-Scoring

Short-answer scoring with ModernBERT (E1–E13: SFT / DPO / GRPO, classification and regression heads) and the decision model `vllm-sr/d3-mini` (D1–D5). Full design, results notes and metrics: [PROJECT.md](PROJECT.md).

```
data/        prep.py builds data/splits/{train,dev,test}.csv from the raw TSVs; rubrics_md/ = per-set rubrics used by D
src/         E: model, train_{sft,dpo,grpo}, eval, launch, tune_optuna, utils, test_losses
             D: d3_common, d3_model, train_d3, d3_export, d3_eval, d3_mipro
             common.py: ROOT, scores_of, load_split (shared)
configs/     E1–E13 YAML, tuned_optuna.json
analysis/    aggregate.py -> results/summary*.md and CSVs
results/     E1–E11, D1–D3 runs, optuna_*, logs, summaries
```

The generative-model experiments (G1–G9) live in `/mnt/c/GitHub/GenerativeRF`.
