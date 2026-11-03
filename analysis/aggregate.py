"""Aggregate results/E*/<tag>/seed*/set*/metrics.json (seeds >= 100 are tuning seeds and are excluded).
Tag `default`: writes results/summary.md plus results/summary_all_runs.csv, learning_curves.csv, counts_best_run.csv.
Tag `tuned` (Optuna parameters from set 1): the same files with a `_tuned` suffix, and a tuned-vs-default table in summary_tuned.md.
usage: python -I analysis/aggregate.py [default] [tuned]    (no argument = both tags that have results)"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
EXPS = [f"E{i}" for i in range(1, 12)]
MET = ["qwk", "acc", "smd", "abs_smd", "kl_true_pred", "jeffreys"]
HIGHER_BETTER = {"qwk", "acc"}
REG = {"E6", "E7", "E8", "E11"}
# section 6.3
COMPARISONS = [("Objectives from scratch (RQ1)", [("E1", "E2"), ("E1", "E3"), ("E2", "E3")]),
               ("Sequential vs SFT alone (RQ2, RQ4)", [("E1", "E4"), ("E1", "E5"), ("E6", "E8")]),
               ("Sequential vs matched extra training (RQ2, RQ4)", [("E4", "E9"), ("E5", "E10"), ("E8", "E11")]),
               ("Extra training alone", [("E1", "E9"), ("E1", "E10"), ("E6", "E11")]),
               ("Head type (RQ3)", [("E1", "E6"), ("E2", "E7"), ("E4", "E8")]),
               ("Cold vs warm start", [("E2", "E4"), ("E3", "E5"), ("E7", "E8")])]
# stage-2 budget pairs that must match (section 6.2)
MATCHED = [("E4", "E9"), ("E5", "E10"), ("E8", "E11")]

def load(tag="default"):
    rows, curves, counts = [], [], []
    for f in sorted(glob.glob(str(ROOT / f"results/E*/{tag}/seed*/set*/metrics.json"))):
        m = json.load(open(f))
        if m["seed"] >= 100 or m["exp"] not in EXPS: continue
        s1 = m["stage1"] or {}
        row = dict(exp=m["exp"], seed=m["seed"], set=m["essay_set"], dev_qwk=m["dev"]["qwk"], lr=m["cfg"]["lr"], steps=m["steps"],
                   examples_stage1=s1.get("examples_seen", 0), examples_stage2=m["examples_seen"] if s1 else 0, examples_total=m["examples_seen"] + s1.get("examples_seen", 0),
                   time_stage1=s1.get("train_time", 0.0), time_stage2=m["total_train_time"] if s1 else 0.0, time_total=m["total_train_time"] + s1.get("train_time", 0.0),
                   best_step=m["best_step"])
        for k in MET:
            row[k] = abs(m["test"]["smd"]) if k == "abs_smd" else m["test"][k]
        if "test_round" in m:
            for k in MET: row[k + "_round"] = abs(m["test_round"]["smd"]) if k == "abs_smd" else m["test_round"][k]
        rows.append(row)
        counts.append(dict(exp=m["exp"], seed=m["seed"], set=m["essay_set"], dev_qwk=m["dev"]["qwk"], true_counts=m["test"]["true_counts"], pred_counts=m["test"]["pred_counts"]))
        curves += [dict(exp=m["exp"], seed=m["seed"], set=m["essay_set"], **{k: v for k, v in c.items()}) for c in m["curve"]]
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(counts)

def ms(x): return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}" if len(x) > 1 else f"{np.mean(x):.3f}"

def headline(df, suffix=""):
    """per-seed macro mean over sets, then mean ± std over seeds; sets with missing seeds are kept as they are"""
    rows = {}
    for e, d in df.groupby("exp"):
        if suffix and e not in REG: continue
        per_seed = d.groupby("seed")[[m + suffix for m in MET]].mean()
        rows[e] = {m: ms(per_seed[m + suffix]) for m in MET} | {"n_sets": d["set"].nunique(), "n_seeds": d.seed.nunique()}
    return pd.DataFrame(rows).T.reindex([e for e in EXPS if e in rows])

def per_set(df, metric):
    g = df.groupby(["set", "exp"])[metric].apply(ms).unstack("exp")
    return g[[e for e in EXPS if e in g.columns]]

def compare(df, a, b, metric):
    """paired Wilcoxon across sets (per-set seed means) + count of sets where each wins; per-set seed-paired tests counted at p<0.05"""
    pa, pb = (df[df.exp == e].pivot_table(index="set", columns="seed", values=metric) for e in (a, b))
    sets = pa.index.intersection(pb.index)
    if len(sets) < 2: return None
    ma, mb = pa.loc[sets].mean(axis=1), pb.loc[sets].mean(axis=1)
    diff = ma - mb
    p = wilcoxon(ma, mb).pvalue if (diff != 0).any() else 1.0
    seeds = pa.columns.intersection(pb.columns)
    sig = 0
    for s in sets:
        x, y = pa.loc[s, seeds], pb.loc[s, seeds]
        if len(seeds) >= 2 and (x != y).any() and wilcoxon(x, y).pvalue < 0.05: sig += 1
    wins_a = int((diff > 0).sum() if metric in HIGHER_BETTER else (diff < 0).sum())
    return f"{diff.mean():+.4f}", f"{p:.3f}", f"{wins_a}/{len(sets)}", f"{sig}/{len(sets)}"

def budget_checks(df):
    out = []
    for a, b in MATCHED:
        da, db = df[df.exp == a].set_index(["seed", "set"]), df[df.exp == b].set_index(["seed", "set"])
        common = da.index.intersection(db.index)
        ok = len(common) > 0 and (da.loc[common, "examples_stage2"] == db.loc[common, "examples_stage2"]).all() and (da.loc[common, "steps"] == db.loc[common, "steps"]).all()
        out.append(f"- {a} vs {b}: stage-2 examples {'MATCH' if ok else 'MISMATCH (or no common runs)'} over {len(common)} paired runs")
    return out

def step0_table(df, counts_best=None):
    """per arm: sets whose selected checkpoint is step 0 (stage-2 arms: the stage-1 model is returned unchanged)"""
    g = df.assign(step0=df.best_step == 0).groupby("exp").agg(runs=("step0", "size"), at_step0=("step0", "sum"))
    return g.reindex([e for e in EXPS if e in g.index]).astype(int)

def tuned_vs_default(dd, dt):
    """mean over sets of seed 0 (the only seed the tuned runs have), default vs tuned, for every metric; paired Wilcoxon across sets on QWK"""
    rows = []
    for e in EXPS:
        a, b = dd[(dd.exp == e) & (dd.seed == 0)].set_index("set"), dt[(dt.exp == e) & (dt.seed == 0)].set_index("set")
        sets = a.index.intersection(b.index)
        if not len(sets): continue
        diff = b.loc[sets, "qwk"] - a.loc[sets, "qwk"]
        p = wilcoxon(b.loc[sets, "qwk"], a.loc[sets, "qwk"]).pvalue if (diff != 0).any() and len(sets) > 1 else float("nan")
        rows.append(dict(exp=e, n_sets=len(sets), qwk_default=a.loc[sets, "qwk"].mean(), qwk_tuned=b.loc[sets, "qwk"].mean(), diff=diff.mean(), p_wilcoxon=p,
                         sets_tuned_better=int((diff > 0).sum()), acc_default=a.loc[sets, "acc"].mean(), acc_tuned=b.loc[sets, "acc"].mean(),
                         jeffreys_default=a.loc[sets, "jeffreys"].mean(), jeffreys_tuned=b.loc[sets, "jeffreys"].mean()))
    return pd.DataFrame(rows).set_index("exp").round(4)

def main(tag="default"):
    sfx = "" if tag == "default" else f"_{tag}"
    df, curves, counts = load(tag)
    assert len(df), f"no results found for tag {tag}"
    df.to_csv(ROOT / "results" / f"summary_all_runs{sfx}.csv", index=False)
    curves.to_csv(ROOT / "results" / f"learning_curves{sfx}.csv", index=False)
    best = counts.loc[counts.groupby(["exp", "set"]).dev_qwk.idxmax()]
    best.to_csv(ROOT / "results" / f"counts_best_run{sfx}.csv", index=False)

    md = [f"# Results summary (test set), tag: {tag}\n",
          "Headline: per-seed mean over EssaySets, then mean ± std over seeds. Regression arms (E6, E7, E8, E11) use dev-fit thresholds; ",
          "rounding to the nearest observed score is the secondary table. `abs_smd` is |SMD|; the signed SMD is in the per-set tables.\n",
          "## Headline (primary decoding)\n", headline(df).to_markdown(), "\n",
          "## Selected checkpoint at step 0 (stage-2 arms return their stage-1 model unchanged)\n", step0_table(df).to_markdown(), "\n",
          "## Headline, regression arms with rounding (secondary)\n", headline(df, "_round").to_markdown(), "\n",
          "## Budgets (examples seen, mean over runs; stage 1 + stage 2)\n",
          df.groupby("exp")[["steps", "examples_stage1", "examples_stage2", "examples_total", "time_stage1", "time_stage2", "time_total"]].mean().reindex(
              [e for e in EXPS if e in set(df.exp)]).round(0).to_markdown(), "\n", "Matched stage-2 budgets (controls):", *budget_checks(df), "\n",
          "## Planned comparisons (section 6.3)\n",
          "Columns: mean difference of per-set seed means (A - B), paired Wilcoxon across the EssaySets (p), sets where A is better, sets with per-set seed-paired Wilcoxon p<0.05 ",
          "(with 5 seeds the smallest attainable per-set p is 0.0625, so that count is 0 at n=5). Regression arms use thresholded predictions.\n"]
    for title, pairs in COMPARISONS:
        md.append(f"### {title}\n")
        rows = []
        for a, b in pairs:
            for m in MET:
                r = compare(df, a, b, m)
                if r: rows.append([f"{a} vs {b}", m, *r])
        md.append(pd.DataFrame(rows, columns=["A vs B", "metric", "mean diff", "p (sets)", "A better in", "sig sets"]).to_markdown(index=False) if rows else "no paired runs yet")
        md.append("")
    if tag != "default":
        dd = load("default")[0]
        md += ["## Tuned vs default (seed 0, mean over sets; Wilcoxon on per-set QWK)\n", tuned_vs_default(dd, df).to_markdown(), ""]
    md.append("## Per-set tables (mean ± std over seeds)\n")
    for m in ["qwk", "acc", "smd", "kl_true_pred", "jeffreys"]:
        md += [f"### {m}\n", per_set(df, m).to_markdown(), ""]
    (ROOT / "results" / f"summary{sfx}.md").write_text("\n".join(map(str, md)) + "\n")
    print("\n".join(map(str, md)))

if __name__ == "__main__":
    for t in sys.argv[1:] or ["default", "tuned"]:
        if glob.glob(str(ROOT / f"results/E*/{t}/seed*/set*/metrics.json")): main(t)
