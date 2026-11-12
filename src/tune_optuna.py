"""Bayesian (TPE) hyperparameter search with Optuna, one study per arm, identical budget for every arm.
usage: python src/tune_optuna.py stage1 [--essay_set 1] [--n_trials 30] [--arms E1 E2 ...]
       python src/tune_optuna.py ckpt   [--essay_set 1]        # stage-1 checkpoints at the selected params (tuning seeds)
       python src/tune_optuna.py stage2 [--essay_set 1] [--n_trials 30] [--arms E4 E5 E8]
       python src/tune_optuna.py report [--essay_set 1]

Fairness: every arm gets the same sampler (TPE, same seed), pruner, trial count, tuning seeds and objective (mean best dev QWK
over the tuning seeds). The test set is never read. Seeds >= 100 are tuning seeds and are never reported.
Order mirrors launch.py: stage1 -> ckpt (E1, E6 at their selected params) -> stage2. Controls E9-E11 reuse the stage-1 LR.
Studies are SQLite-backed (results/optuna/), so an interrupted search resumes where it stopped.
Each trial trains the first seed with pruning on running-best dev QWK (read from the trainer's log), then the remaining seeds."""
import argparse, json, re, subprocess, sys
from pathlib import Path
import optuna, yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "optuna"
TUNED = ROOT / "configs" / "tuned_optuna.json"
TUNE_SEEDS = (100, 101)
STAGE1, STAGE2 = ["E1", "E2", "E3", "E6", "E7", "E12"], ["E4", "E5", "E8", "E13"]
STAGE1_OF = {"E4": "E1", "E5": "E1", "E8": "E6", "E13": "E6"}
STEP_RE = re.compile(r"step (\d+) dev_qwk (-?[\d.]+)")

cfg_of = lambda base: yaml.safe_load(open(ROOT / "configs" / f"{base}.yaml"))

def suggest(trial, base):
    """Search space per arm. lr is shared by every arm; arm-specific parameters follow launch.py's ranges (G extended to 64)."""
    c = cfg_of(base)
    p = {"lr": trial.suggest_float("lr", 5e-6, 2e-4, log=True)}
    if c["arm"] == "dpo":
        p["dpo_beta"] = trial.suggest_float("dpo_beta", 0.01, 1.0, log=True) if c["head"] == "mc" else trial.suggest_float("dpo_beta", 0.1, 10.0, log=True)
    if c["arm"] == "grpo":
        p["G"] = trial.suggest_categorical("G", [4, 8, 16, 32, 64])
        p["kl_beta"] = trial.suggest_categorical("kl_beta", [0.0, 0.01, 0.1, 1.0])
        if c["head"] == "reg": p["grpo_sigma"] = trial.suggest_float("grpo_sigma", 0.1, 1.0, log=True)   # Gaussian policy std (standardized scale)
    return p

def command(base, seed, essay_set, tag, params, extra=()):
    c = cfg_of(base)
    ov = [f"{k}={v}" for k, v in params.items()] + list(extra)
    return [sys.executable, str(ROOT / "src" / f"train_{c['arm']}.py"), "--config", str(ROOT / "configs" / f"{base}.yaml"), "--seed", str(seed),
            "--exp", f"optuna_{base}", "--tag", tag, "--essay_sets", str(essay_set), "--set", *ov]

def train(cmd, trial=None):
    """Run one training job, streaming its log; report running-best dev QWK to the pruner and kill the job if pruned."""
    best, log = -2.0, []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.append(line)
        m = STEP_RE.search(line)
        if m and trial is not None:
            step, q = int(m.group(1)), float(m.group(2)); best = max(best, q)
            trial.report(best, step)
            if trial.should_prune():
                proc.kill(); proc.wait(); raise optuna.TrialPruned()
    if proc.wait() != 0:
        raise RuntimeError("training failed:\n" + "".join(log[-15:]))

def dev_qwk(base, seed, essay_set, tag):
    f = ROOT / "results" / f"optuna_{base}" / tag / f"seed{seed}" / f"set{essay_set}" / "metrics.json"
    return json.load(open(f))["dev"]["qwk"]   # dev only; the test entry is never read

def objective_for(base, essay_set):
    def objective(trial):
        params, tag, scores = suggest(trial, base), f"t{trial.number}", []
        for i, seed in enumerate(TUNE_SEEDS):
            cmd = command(base, seed, essay_set, tag, params, ["save_ckpt=false"])
            for attempt in (0, 1):   # one retry: GPU faults were seen with several concurrent processes
                try: train(cmd, trial if i == 0 else None); break
                except RuntimeError:
                    if attempt: raise
            scores.append(dev_qwk(base, seed, essay_set, tag))
        trial.set_user_attr("seed_qwk", scores)
        return sum(scores) / len(scores)
    return objective

def study_for(base, essay_set):
    OUT.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=f"{base}_set{essay_set}", storage=f"sqlite:///{OUT}/{base}_set{essay_set}.db", load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(base[1:]), multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=200, interval_steps=50))

def search(arms, essay_set, n_trials):
    for base in arms:
        study = study_for(base, essay_set)
        done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED) for t in study.trials)
        if STAGE1_OF.get(base):   # stage-2 arms need stage-1 checkpoints at the tuning seeds
            for seed in TUNE_SEEDS:
                s1 = STAGE1_OF[base]
                assert (ROOT / "results" / s1 / "default" / f"seed{seed}" / f"set{essay_set}" / "metrics.json").exists(), f"run `ckpt` first ({s1} seed {seed})"
        study.optimize(objective_for(base, essay_set), n_trials=max(0, n_trials - done), catch=(RuntimeError,))
        best = study.best_trial
        tuned = json.load(open(TUNED)) if TUNED.exists() else {}
        tuned.setdefault(f"set{essay_set}", {})[base] = dict(best.params, dev_qwk=best.value)
        json.dump(tuned, open(TUNED, "w"), indent=1)
        print(base, "best", best.params, "dev QWK %.4f" % best.value, "| trials", len(study.trials))

def stage1_ckpts(essay_set):
    """E1 and E6 at their selected params on the tuning seeds, keeping checkpoints for the stage-2 searches."""
    tuned = json.load(open(TUNED))[f"set{essay_set}"]
    for base in ("E1", "E6"):
        params = {k: v for k, v in tuned[base].items() if k != "dev_qwk"}
        for seed in TUNE_SEEDS:
            cmd = [sys.executable, str(ROOT / "src" / "train_sft.py"), "--config", str(ROOT / "configs" / f"{base}.yaml"), "--seed", str(seed),
                   "--essay_sets", str(essay_set), "--set", *[f"{k}={v}" for k, v in params.items()], "save_ckpt=true"]
            train(cmd)

def report(essay_set):
    print(f"{'arm':<5}{'trials':>7}{'pruned':>7}{'fail':>6}{'best dev QWK':>14}  params")
    for base in STAGE1 + STAGE2:
        db = OUT / f"{base}_set{essay_set}.db"
        if not db.exists(): continue   # load_study, not create_study: reporting must not create empty studies
        try: s = optuna.load_study(study_name=f"{base}_set{essay_set}", storage=f"sqlite:///{db}")
        except KeyError: continue
        st = lambda k: sum(t.state == k for t in s.trials)
        S = optuna.trial.TrialState
        if st(S.COMPLETE):
            print(f"{base:<5}{len(s.trials):>7}{st(S.PRUNED):>7}{st(S.FAIL):>6}{s.best_value:>14.4f}  {s.best_params}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["stage1", "ckpt", "stage2", "report"])
    ap.add_argument("--essay_set", type=int, default=1); ap.add_argument("--n_trials", type=int, default=30); ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if a.phase == "stage1": search(a.arms or STAGE1, a.essay_set, a.n_trials)
    elif a.phase == "stage2": search(a.arms or STAGE2, a.essay_set, a.n_trials)
    elif a.phase == "ckpt": stage1_ckpts(a.essay_set)
    else: report(a.essay_set)
