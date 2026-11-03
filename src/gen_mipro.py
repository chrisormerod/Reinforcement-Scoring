"""G3 (no thinking) and G4 (thinking): MIPROv2 optimizes the rubric, the signature's instructions, and nothing else.
Prompt structure, fixed suffix, decoding and the (zero) demos are held fixed; the starting point is the original rubric, i.e. G1 / G2.
usage: python src/gen_mipro.py G3|G4 [--essay_sets 1 ...] [--auto light] [--n_train 100] [--limit N]
Optimization uses a random subset of the training split as trainset and the dev split as valset; the test split is read once, after the rubric is chosen.
Rule: an optimized rubric that does not mention every score point of the set is rejected (saved as rubric_rejected.md) and the original rubric is used.
Writes results/<G>/default/seed0/set<k>/{metrics.json, rubric.md}."""
import argparse, json
import dspy, numpy as np
from dspy.teleprompt import MIPROv2
from gen_common import *
from gen_zeroshot import score_split
from eval import bootstrap_ci

EXPS = {"G3": False, "G4": True}   # value = thinking
SEED = 0

def examples(split, es, n=None):
    texts, y = load_split(split, es)
    ex = [dspy.Example(response=t, score=s).with_inputs("response") for t, s in zip(texts, y)]
    if n and n < len(ex): ex = [ex[i] for i in np.random.default_rng(SEED).choice(len(ex), n, replace=False)]
    return ex

def run(exp, es, auto, n_train, limit, threads):
    out = ROOT / "results" / exp / "default" / "seed0" / f"set{es}"
    if (out / "metrics.json").exists() and not limit: print("done already:", out); return
    thinking, scores = EXPS[exp], scores_of(es)
    task_lm = make_lm(thinking)
    prompt_lm = make_lm(False, temperature=0.7, max_tokens=2048)    # instruction proposer: same served model, thinking off
    dspy.configure(lm=task_lm, adapter=MinimalAdapter(scores_of(es)))
    program = make_program(rubric_of(es))
    opt = MIPROv2(metric=agreement_metric(scores), prompt_model=prompt_lm, task_model=task_lm, auto=auto, num_threads=threads,
                  max_bootstrapped_demos=0, max_labeled_demos=0, seed=SEED, max_errors=10 ** 6)
    trainset, valset = examples("train", es, n_train), examples("dev", es)
    if limit: trainset, valset = trainset[:limit], valset[:limit]
    best = opt.compile(program, trainset=trainset, valset=valset, requires_permission_to_run=False)
    rubric = best.signature.instructions
    # constraint: an optimized rubric that no longer mentions every score point is rejected and the original rubric is used for this set
    fallback = not covers_all_scores(rubric, scores)
    rejected = rubric if fallback else None
    if fallback:
        print(f"[{exp} set{es}] optimized rubric drops a score point; falling back to the original rubric", flush=True)
        best, rubric = make_program(rubric_of(es)), rubric_of(es)
    dev, test = score_split(best, "dev", es, limit, threads), score_split(best, "test", es, limit, threads)
    res = dict(exp=exp, seed=0, essay_set=es, head="gen", thinking=thinking, model=MODEL, api_base=API_BASE, scores=scores, decoding=DECODING[thinking], auto=auto,
               n_train_examples=len(trainset), n_dev_examples=len(valset), rubric_changed=rubric != rubric_of(es), rubric_fallback=fallback, rubric_chars=len(rubric),
               dev=dev["metrics"], test=test["metrics"], test_ci=bootstrap_ci(test["y_idx"], test["p_idx"], scores, n=300, seed=0),
               dev_fail_rate=dev["fail_rate"], test_fail_rate=test["fail_rate"], n_dev=dev["n"], n_test=test["n"], eval_seconds=dev["seconds"] + test["seconds"])
    print(f"[{exp} set{es}] dev_qwk {dev['metrics']['qwk']:.4f} test_qwk {test['metrics']['qwk']:.4f} acc {test['metrics']['acc']:.3f} "
          f"fail {test['fail_rate']:.3f} rubric_changed {res['rubric_changed']}", flush=True)
    if limit: return
    out.mkdir(parents=True, exist_ok=True)
    (out / "rubric.md").write_text(rubric)
    if rejected: (out / "rubric_rejected.md").write_text(rejected)
    np.savez_compressed(out / "preds.npz", dev_pred=np.array([-1 if p is None else p for p in dev["raw"]]), test_pred=np.array([-1 if p is None else p for p in test["raw"]]))
    json.dump(res, open(out / "metrics.json", "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exp", choices=list(EXPS)); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"]); ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    for es in a.essay_sets: run(a.exp, es, a.auto, a.n_train, a.limit, a.threads)
