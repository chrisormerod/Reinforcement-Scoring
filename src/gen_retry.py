"""Re-sample the replies that had no parseable answer (e.g. thinking truncated at the token cap) for a zero-shot run, up to --attempts times,
then recompute the metrics. Retries depend only on parse failure, never on correctness. The pre-retry metrics are kept under `before_retry`.
usage: python src/gen_retry.py G2 [--essay_sets 1 ...] [--attempts 3] [--threads 32]"""
import argparse, json
import dspy, numpy as np
from gen_common import *
THINKING = {"G1": False, "G2": True, "G3": False, "G4": True}
from eval import all_metrics, bootstrap_ci

def retry(program, texts, pred, attempts, threads):
    pred = list(pred)
    for _ in range(attempts):
        todo = [i for i, p in enumerate(pred) if p < 0 or p not in SCORES]
        if not todo: break
        for i, p in zip(todo, predict_all(program, [texts[i] for i in todo], threads)):
            if p is not None: pred[i] = p
    return pred

def run(exp, es, attempts, threads):
    global SCORES
    out = ROOT / "results" / exp / "default" / "seed0" / f"set{es}"
    res = json.load(open(out / "metrics.json"))
    if "before_retry" in res: print("already retried:", out); return
    SCORES = scores = scores_of(es)
    dspy.configure(lm=make_lm(THINKING[exp]), adapter=MinimalAdapter(scores_of(es)))
    saved = out / "rubric.md"   # G3/G4: the optimized (or fallback) rubric that produced the stored predictions
    program = make_program(saved.read_text() if saved.exists() else rubric_of(es))
    z = np.load(out / "preds.npz"); new, n_retried = {}, {}
    for split in ("dev", "test"):
        texts, y = load_split(split, es)
        old = z[f"{split}_pred"].tolist()
        n_retried[split] = sum(p not in scores for p in old)
        new[split] = retry(program, texts, old, attempts, threads)
        fail = [p not in scores for p in new[split]]
        p_idx = np.array([0 if f else scores.index(p) for f, p in zip(fail, new[split])]); y_idx = np.array([scores.index(v) for v in y])
        new[split + "_m"] = all_metrics(y_idx, p_idx, scores, with_per_class=(split == "test")); new[split + "_fail"] = float(np.mean(fail))
        if split == "test": new["ci"] = bootstrap_ci(y_idx, p_idx, scores, n=300, seed=0)
    res["before_retry"] = dict(dev=res["dev"], test=res["test"], test_ci=res["test_ci"], dev_fail_rate=res["dev_fail_rate"], test_fail_rate=res["test_fail_rate"])
    res.update(dev=new["dev_m"], test=new["test_m"], test_ci=new["ci"], dev_fail_rate=new["dev_fail"], test_fail_rate=new["test_fail"],
               retry=dict(attempts=attempts, n_retried=n_retried, api_base=API_BASE))
    np.savez_compressed(out / "preds.npz", dev_pred=np.array(new["dev"]), test_pred=np.array(new["test"]))
    json.dump(res, open(out / "metrics.json", "w"))
    b = res["before_retry"]
    print(f"[{exp} set{es}] retried {n_retried} | test_qwk {b['test']['qwk']:.4f} -> {res['test']['qwk']:.4f} | fail {b['test_fail_rate']:.3f} -> {res['test_fail_rate']:.3f}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exp", choices=list(THINKING)); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--attempts", type=int, default=3); ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    for es in a.essay_sets: run(a.exp, es, a.attempts, a.threads)
