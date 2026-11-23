"""Evaluate a decision-model arm through vLLM (/v1/systemone): D1 (base, original rubric), D2 (base, optimized text), D3/D4/D5 (merged LoRA weights, optimized text or original per train.json).
usage: python src/d3_eval.py --exp D1|D3|D4|D5 [--essay_sets 3 ...] [--seed 0] [--text original|D2] [--port 8000] [--limit N] [--keep]    (run in ~/ml; it starts and stops the vLLM server of ~/vllm034 itself)
Metrics as every other arm (src/eval.py all_metrics on level indices: QWK, SMD, accuracy, KL/Jeffreys, per-class on test; 300-draw bootstrap CIs), prediction = argmax of the level probabilities; a failed call
(after 3 tries) counts as the lowest level and in fail_rate. Also reports the mean test negative log-likelihood of the gold level and the QWK of the rounded expected score.
Writes results/<exp>/<model>/seed<s>/set<k>/metrics.json and preds.npz (dev/test predicted score VALUES, -1 for a failure, and the test level probabilities). D1/D2 have no seed (written under seed0); --limit runs write nothing.
The merged folder of a trained arm is deleted afterwards unless --keep (it is 9 GB)."""
import argparse, json, shutil, subprocess, time, os, signal, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from common import ROOT, scores_of, load_split
from eval import all_metrics, bootstrap_ci
from d3_common import CKPT_DIR, MODELS, question, load_text, systemone, probs_of
from d3_export import merge

VLLM = os.path.expanduser("~/vllm034/bin/vllm")

class Server:
    def __init__(self, path, name, port): self.model, self.name, self.port, self.p = path, name, port, None
    def __enter__(self):
        self.log = open(os.path.expanduser(f"~/d3_eval_server_{self.name}_{self.port}.log"), "w")
        self.p = subprocess.Popen([VLLM, "serve", str(self.model), "--served-model-name", MODELS[self.name], "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.8", "--max-model-len", "4096", "--max-num-seqs", "32", "--port", str(self.port)],
                                  stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 1500:
            if self.p.poll() is not None: raise RuntimeError(f"vLLM exited early, see {self.log.name}")
            try:
                urllib.request.urlopen(f"http://localhost:{self.port}/v1/models", timeout=5); return self
            except Exception: time.sleep(5)
        raise RuntimeError("vLLM did not start in 25 minutes")
    def __exit__(self, *e):
        os.killpg(self.p.pid, signal.SIGTERM); self.p.wait(timeout=120); time.sleep(5)

def score_split(texts, q, K, api, model, threads=16):
    def one(t):
        for _ in range(3):
            try:
                p = probs_of(systemone(t, q, model, api=api), K)
                if p is not None: return p
            except Exception: time.sleep(3)
        return None
    with ThreadPoolExecutor(threads) as ex: return list(ex.map(one, texts))

def metrics_of(probs, y, scores, per_class):
    K = len(scores); y_idx = np.array([scores.index(v) for v in y]); fail = np.array([p is None for p in probs])
    P = np.array([[1.0 / K] * K if p is None else p for p in probs]); p_idx = np.where(fail, 0, P.argmax(1))
    exp_idx = np.clip(np.rint(P @ np.arange(K)), 0, K - 1).astype(int)
    return dict(metrics=all_metrics(y_idx, p_idx, scores, with_per_class=per_class), fail_rate=float(fail.mean()), y_idx=y_idx, p_idx=p_idx, P=P, fail=fail,
                nll=float(-np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, None))[~fail].mean()), qwk_expected=all_metrics(y_idx, np.where(fail, 0, exp_idx), scores)["qwk"])

def run(es, a, api, text_from):
    exp = a.exp; out = ROOT / "results" / exp / a.model / f"seed{a.seed if exp in ('D3', 'D4', 'D5') else 0}" / f"set{es}"
    scores = scores_of(es); K = len(scores); ins, crit = load_text(es, text_from, a.model); q = question(ins, crit); res = {}
    for split in ("dev", "test"):
        texts, y = load_split(split, es)
        if a.limit: texts, y = texts[:a.limit], y[:a.limit]
        t0 = time.time(); res[split] = metrics_of(score_split(texts, q, K, api, MODELS[a.model]), y, scores, split == "test"); res[split]["seconds"] = time.time() - t0
    d, t = res["dev"], res["test"]
    m = dict(exp=exp, head="d3", model=MODELS[a.model], text=text_from, scores=scores, essay_set=es, seed=a.seed, dev=d["metrics"], test=t["metrics"], test_ci=bootstrap_ci(t["y_idx"], t["p_idx"], scores, n=300, seed=0),
             dev_fail_rate=d["fail_rate"], test_fail_rate=t["fail_rate"], n_dev=len(d["y_idx"]), n_test=len(t["y_idx"]), test_nll=t["nll"], dev_nll=d["nll"], test_qwk_expected_score=t["qwk_expected"], dev_qwk_expected_score=d["qwk_expected"],
             eval_seconds=d["seconds"] + t["seconds"])
    if exp in ("D3", "D4", "D5") and (out / "train.json").exists():
        tr = json.load(open(out / "train.json")); m["train"] = {k: tr[k] for k in ("mode", "lr", "beta", "kl", "group", "batch", "epochs", "steps", "n_train", "lora", "train_seconds")}
    print(f"[{exp} set{es}] dev_qwk {d['metrics']['qwk']:.4f} test_qwk {t['metrics']['qwk']:.4f} acc {t['metrics']['acc']:.3f} nll {t['nll']:.3f} fail {t['fail_rate']:.3f}", flush=True)
    if a.limit: return
    out.mkdir(parents=True, exist_ok=True); json.dump(m, open(out / "metrics.json", "w"))
    sv = lambda r: np.array([-1 if f else scores[i] for f, i in zip(r["fail"], r["p_idx"])])
    np.savez_compressed(out / "preds.npz", dev_pred=sv(d), test_pred=sv(t), test_probs=t["P"], dev_probs=d["P"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="d3-mini", choices=list(MODELS)); ap.add_argument("--exp", required=True, choices=["D1", "D2", "D3", "D4", "D5"]); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--text", default=None); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", action="store_true"); ap.add_argument("--adapter_suffix", default="", help="e.g. _limit to evaluate a --limit adapter (testing)")
    a = ap.parse_args(); api = f"http://localhost:{a.port}/v1"
    text_from = a.text or ("original" if a.exp == "D1" else "D2")
    if a.exp in ("D1", "D2"):   # base weights: one server for all sets
        todo = [es for es in a.essay_sets if a.limit or not (ROOT / "results" / a.exp / a.model / "seed0" / f"set{es}" / "metrics.json").exists()]
        if todo:
            with Server(MODELS[a.model], a.model, a.port):
                for es in todo: run(es, a, api, text_from)
        return
    for es in a.essay_sets:
        if not a.limit and (ROOT / "results" / a.exp / a.model / f"seed{a.seed}" / f"set{es}" / "metrics.json").exists(): print("done already:", a.exp, es); continue
        ad = CKPT_DIR / a.exp / a.model / f"set{es}" / f"seed{a.seed}{a.adapter_suffix}"; assert (ad / "adapter.pt").exists(), f"train {a.exp} set {es} first (src/train_d3.py)"
        tj = ROOT / "results" / a.exp / a.model / f"seed{a.seed}" / f"set{es}" / "train.json"; tf = json.load(open(tj))["text"] if tj.exists() else text_from
        merged = merge(ad, a.model)
        with Server(merged, a.model, a.port): run(es, a, api, tf)
        if not a.keep: shutil.rmtree(merged, ignore_errors=True)

if __name__ == "__main__": main()
