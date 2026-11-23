# RF-Scoring

Short-answer scoring with ModernBERT (SFT, DPO and GRPO, each with a classification and a regression head, all from a cold start: E1 E2 E3 / E6 E7 E12) and the decision models `vllm-sr/d3-mini` and `vllm-sr/d3-flash` (D1–D5, run per model with `--model d3-mini|d3-flash`). Full design, results notes and metrics: [PROJECT.md](PROJECT.md).

```
data/        prep.py builds data/splits/{train,dev,test}.csv from the raw TSVs; rubrics_md/ = per-set rubrics used by D
src/         E: model, train_{sft,dpo,grpo}, eval, launch (tune / select / final), tune_optuna (search / report), utils, test_losses
             D: d3_common, d3_model, train_d3, d3_export, d3_eval, d3_mipro
             common.py: ROOT, scores_of, load_split (shared)
configs/     E1 E2 E3 E6 E7 E12 YAML, tuned_optuna.json
analysis/    aggregate.py -> results/summary*.md and CSVs
results/     E runs, D runs as results/D#/<model>/ (retired SFT→DPO/GRPO arms in _retired_sequential/), optuna_*, logs, summaries
```

The generative-model experiments (G1–G9) live in `/mnt/c/GitHub/GenerativeRF`.
