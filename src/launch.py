"""Run experiment phases with a few concurrent worker processes. Jobs whose metrics.json exists are skipped by the trainer.
usage: python src/launch.py PHASE [--workers 3] [--essay_sets 1 2] [--seeds 0 1]

Six arms, all trained from the pretrained backbone (cold start): E1 E6 (SFT), E2 E7 (DPO), E3 E12 (GRPO).
  tune    random search for every arm on the tuning seeds
  select  pick one configuration per arm by mean dev QWK (writes configs/tuned.json)
  final   every arm on the final seeds at its selected configuration
Every arm gets N_CONFIGS configurations x len(TUNE_SEEDS) seeds x all essay sets. Tuning runs live in results/tune_<E>/<cfg>;
seeds >= 100 are never reported."""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
TUNED = ROOT / "configs" / "tuned.json"
N_CONFIGS, TUNE_SEEDS, FINAL_SEEDS = 6, (100, 101), tuple(range(5))
ARMS = ["E1", "E2", "E3", "E6", "E7", "E12"]
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
    return json.load(open(TUNED))[base]

def job(base, seed, exp=None, tag=None, params=None, extra=()):
    c = cfg_of(base)
    cmd = [sys.executable, str(ROOT / "src" / f"train_{c['arm']}.py"), "--config", str(ROOT / "configs" / f"{base}.yaml"), "--seed", str(seed)]
    if exp: cmd += ["--exp", exp]
    if tag: cmd += ["--tag", tag]
    ov = [f"{k}={v}" for k, v in (params or {}).items()] + list(extra)
    if ov: cmd += ["--set", *ov]
    if ESSAY_SETS: cmd += ["--essay_sets", *map(str, ESSAY_SETS)]
    return cmd

def tune_jobs(bases, seeds):
    return [job(b, s, f"tune_{b}", f"c{i}", p) for b in bases for i, p in enumerate(sample_configs(b)) for s in seeds]

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
    return {"tune": lambda: tune_jobs(ARMS, TUNE_SEEDS), "final": lambda: final_jobs(ARMS, seeds)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase"); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--essay_sets", nargs="*", type=int, default=[]); ap.add_argument("--seeds", nargs="*", type=int, default=list(FINAL_SEEDS))
    a = ap.parse_args(); ESSAY_SETS[:] = a.essay_sets
    if a.phase == "select":
        select(ARMS); sys.exit()
    jobs = phases(tuple(a.seeds))[a.phase]()
    log = ROOT / "results" / "logs"; log.mkdir(parents=True, exist_ok=True)
    def go(i_c):
        i, c = i_c
        with open(log / f"{a.phase}_{i}.log", "w") as f:
            return subprocess.call(c, stdout=f, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(a.workers) as ex:
        rcs = list(ex.map(go, enumerate(jobs)))
    print(a.phase, "finished; failures:", sum(rc != 0 for rc in rcs))
