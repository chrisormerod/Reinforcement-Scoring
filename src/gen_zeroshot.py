"""G1 (no thinking) and G2 (thinking): Qwen3.5-4B scores with the original rubric as the system prompt. No training.
usage: python src/gen_zeroshot.py G1|G2 [--essay_sets 1 2 ...] [--limit N] [--threads 32]
Writes results/<G>/default/seed0/set<k>/metrics.json (same layout and metric keys as the trained arms; there is one deterministic-config run, no seeds)."""
import argparse, json, time
import dspy, numpy as np
from gen_common import *
from eval import all_metrics, bootstrap_ci

EXPS = {"G1": False, "G2": True}   # value = thinking

def score_split(program, split, es, limit, threads):
    texts, y = load_split(split, es)
    if limit: texts, y = texts[:limit], y[:limit]
    scores = scores_of(es)
    t0 = time.time()
    pred = predict_all(program, texts, threads)
    idx = [to_index(p, scores) for p in pred]
    failed = [i is None for i in idx]
    # an unparseable or out-of-range answer is scored as the lowest score point; the rate is reported alongside
    p_idx = np.array([0 if f else i for f, i in zip(failed, idx)])
    y_idx = np.array([scores.index(v) for v in y])
    return dict(metrics=all_metrics(y_idx, p_idx, scores, with_per_class=(split == "test")), fail_rate=float(np.mean(failed)),
                n=len(y), seconds=time.time() - t0, y_idx=y_idx, p_idx=p_idx, raw=pred)

def run(exp, es, limit, threads):
    out = ROOT / "results" / exp / "default" / "seed0" / f"set{es}"
    if (out / "metrics.json").exists() and not limit: print("done already:", out); return
    thinking = EXPS[exp]
    dspy.configure(lm=make_lm(thinking), adapter=MinimalAdapter(scores_of(es)))
    program = make_program(rubric_of(es))
    dev, test = score_split(program, "dev", es, limit, threads), score_split(program, "test", es, limit, threads)
    res = dict(exp=exp, seed=0, essay_set=es, head="gen", thinking=thinking, model=MODEL, api_base=API_BASE, scores=scores_of(es), decoding=DECODING[thinking],
               dev=dev["metrics"], test=test["metrics"], test_ci=bootstrap_ci(test["y_idx"], test["p_idx"], scores_of(es), n=300, seed=0),
               dev_fail_rate=dev["fail_rate"], test_fail_rate=test["fail_rate"], n_dev=dev["n"], n_test=test["n"],
               seconds=dev["seconds"] + test["seconds"])
    print(f"[{exp} set{es}] dev_qwk {dev['metrics']['qwk']:.4f} test_qwk {test['metrics']['qwk']:.4f} acc {test['metrics']['acc']:.3f} "
          f"fail {test['fail_rate']:.3f} {res['seconds']:.0f}s", flush=True)
    if limit: return
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "preds.npz", dev_pred=np.array([-1 if p is None else p for p in dev["raw"]]), test_pred=np.array([-1 if p is None else p for p in test["raw"]]))
    json.dump(res, open(out / "metrics.json", "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exp", choices=list(EXPS)); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    for es in a.essay_sets: run(a.exp, es, a.limit, a.threads)
