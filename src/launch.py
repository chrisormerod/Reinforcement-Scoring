"""Run experiment phases with a few concurrent worker processes. Jobs whose metrics.json exists are skipped by the trainer.
usage: python src/launch.py PHASE [--workers 3] [--essay_sets 1 2] [--seeds 0 1]

Order (stage-2 arms need stage-1 checkpoints of the same seed):
  tune1      random search for E1,E2,E3,E6,E7,E12 on the tuning seeds -> select1  (writes configs/tuned.json)
  tuneseeds  E1,E6 at their selected LR on the tuning seeds (stage-1 checkpoints for tuning E4,E5,E8,E13)
  tune2      random search for E4,E5,E8,E13                          -> select2
  final1     E1,E2,E3,E6,E7,E12 on the final seeds
  final2     E4,E5,E8,E13 and the controls E9,E10,E11 on the final seeds (controls reuse the stage-1 arm's LR; E11 also controls E13)
Every arm gets N_CONFIGS configurations x len(TUNE_SEEDS) seeds x all essay sets; one configuration per arm is chosen by
mean dev QWK across sets and seeds. Tuning runs live in results/tune_<E>/<cfg>; seeds >= 100 are never reported."""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
TUNED = ROOT / "configs" / "tuned.json"
N_CONFIGS, TUNE_SEEDS, FINAL_SEEDS = 6, (100, 101), tuple(range(5))
TUNE1, TUNE2 = ["E1", "E2", "E3", "E6", "E7", "E12"], ["E4", "E5", "E8", "E13"]
CONTROL_LR_FROM = {"E9": "E1", "E10": "E1", "E11": "E6"}   # controls reuse the stage-1 arm's selected LR
ESSAY_SETS = []  # empty = all sets

cfg_of = lambda base: yaml.safe_load(open(ROOT / "configs" / f"{base}.yaml"))

def sample_configs(base, n=N_CONFIGS):
    """Same number of random configs for every arm; reproducible per arm. lr is searched for all arms;
    dpo_beta (beta for mc, beta' = beta/sigma^2 for reg) for DPO arms; G and kl_beta for GRPO arms."""
    c = cfg_of(base); rng = np.random.default_rng(int(base[1:]))
    log = lambda lo, hi: float(f"{10 ** rng.uniform(np.log10(lo), np.log10(hi)):.2g}")
    out = []
    for _ in range(n):
        p = {"lr": log(5e-6, 2e-4)}
        if c["arm"] == "dpo": p["dpo_beta"] = log(0.01, 1.0) if c["head"] == "mc" else log(0.1, 10.0)
        if c["arm"] == "grpo": p.update(G=int(rng.choice([4, 8, 16, 32])), kl_beta=float(rng.choice([0.0, 0.01, 0.1])))
        if c["arm"] == "grpo" and c["head"] == "reg": p["grpo_sigma"] = log(0.1, 1.0)   # drawn last, so the other arms' configs are unchanged
        out.append(p)
    return out

def tuned(base):
    return json.load(open(TUNED))[CONTROL_LR_FROM.get(base, base)]

def job(base, seed, exp=None, tag=None, params=None, extra=()):
    c = cfg_of(base)
    cmd = [sys.executable, str(ROOT / "src" / f"train_{c['arm']}.py"), "--config", str(ROOT / "configs" / f"{base}.yaml"), "--seed", str(seed)]
    if exp: cmd += ["--exp", exp]
    if tag: cmd += ["--tag", tag]
    ov = [f"{k}={v}" for k, v in (params or {}).items()] + list(extra)
    if ov: cmd += ["--set", *ov]
    if ESSAY_SETS: cmd += ["--essay_sets", *map(str, ESSAY_SETS)]
    return cmd

def tune_jobs(bases, seeds):   # no checkpoints are kept for tuning runs
    return [job(b, s, f"tune_{b}", f"c{i}", p, ["save_ckpt=false"]) for b in bases for i, p in enumerate(sample_configs(b)) for s in seeds]

def select(bases):
    """per-arm configuration with the best mean dev QWK over all sets and tuning seeds (test set never used)"""
    tuned_all = json.load(open(TUNED)) if TUNED.exists() else {}
    summary = {}
    for b in bases:
        sc = {}
        for i, p in enumerate(sample_configs(b)):
            ms = [json.load(open(f)) for f in (ROOT / "results" / f"tune_{b}" / f"c{i}").glob("seed*/set*/metrics.json")]
            if ms: sc[i] = (float(np.mean([m["dev"]["qwk"] for m in ms])), len(ms))
        assert sc, f"no tuning runs found for {b}"
        i = max(sc, key=lambda k: sc[k][0])
        tuned_all[b] = sample_configs(b)[i]
        summary[b] = {"search": {f"c{k}": dict(params=sample_configs(b)[k], dev_qwk=v[0], n_runs=v[1]) for k, v in sc.items()}, "selected": f"c{i}"}
        print(b, "selected", f"c{i}", tuned_all[b], "dev QWK %.4f" % sc[i][0], "| runs per config:", sorted({v[1] for v in sc.values()}))
    json.dump(tuned_all, open(TUNED, "w"), indent=1)
    out = ROOT / "results" / "tuning_summary.json"
    old = json.load(open(out)) if out.exists() else {}
    json.dump({**old, **summary}, open(out, "w"), indent=1)

def final_jobs(bases, seeds):
    return [job(b, s, params=tuned(b)) for b in bases for s in seeds]

def phases(seeds):
    return {
        "tune1": lambda: tune_jobs(TUNE1, TUNE_SEEDS),
        "tuneseeds": lambda: final_jobs(["E1", "E6"], TUNE_SEEDS),
        "tune2": lambda: tune_jobs(TUNE2, TUNE_SEEDS),
        "final1": lambda: final_jobs(TUNE1, seeds),
        "final2": lambda: final_jobs(TUNE2 + list(CONTROL_LR_FROM), seeds),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--essay_sets", nargs="*", type=int, default=[]); ap.add_argument("--seeds", nargs="*", type=int, default=list(FINAL_SEEDS))
    a = ap.parse_args(); ESSAY_SETS[:] = a.essay_sets
    if a.phase in ("select1", "select2"):
        select(TUNE1 if a.phase == "select1" else TUNE2); sys.exit()
    jobs = phases(tuple(a.seeds))[a.phase]()
    log = ROOT / "results" / "logs"; log.mkdir(parents=True, exist_ok=True)
    def go(i_c):
        i, c = i_c
        with open(log / f"{a.phase}_{i}.log", "w") as f:
            return subprocess.call(c, stdout=f, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(a.workers) as ex:
        rcs = list(ex.map(go, enumerate(jobs)))
    print(a.phase, "finished; failures:", sum(rc != 0 for rc in rcs))
