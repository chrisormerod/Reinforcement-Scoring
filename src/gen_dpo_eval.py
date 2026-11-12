"""G6 evaluation: the DPO-trained adapter (src/gen_dpo.py) scored with local vLLM, thinking on, G4's prompt and decoding (temperature 0.6, top-p 0.95, 16384 tokens).
usage: python src/gen_dpo_eval.py [--exp G6|G7|G8] [--essay_sets 1 2 ...] [--seed 0] [--limit N]   (G7 = src/gen_sft_dpo.py, G8 = the same with --sft_only; same evaluation)
Parse failures are re-sampled up to 3 times (the retry rule of PROJECT.md section 6.6); metrics before the retries are kept under `before_retry`.
Needs the whole GPU: do not run it while src/gen_sft.py or src/gen_dpo.py is training. Writes results/G6/default/seed<s>/set<k>/metrics.json. --limit runs write nothing.
The adapter is MERGED into the base weights (gen_local.merge_adapter, saved next to the adapter as <adapter>_merged) and vLLM serves the merged model: vLLM 0.28 loads this
architecture's LoRA adapter without applying it (identical greedy outputs and log-probs with and without it), so LoRARequest must not be used. One vLLM instance per set, so
several sets run as separate processes. Check a merge with src/gen_merge_check.py."""
import argparse, json, subprocess, sys, time
import numpy as np
from gen_common import ROOT, MODEL, DECODING, format_suffix, scores_of, load_split, to_index
from gen_local import *
from eval import bootstrap_ci

def reply_texts(llm, sp, lora, rubric, suffix, texts):
    msgs = [[{"role": "system", "content": f"{rubric}\n\n{suffix}"}, {"role": "user", "content": t}] for t in texts]
    return [o.outputs[0].text for o in llm.chat(msgs, sp, lora_request=lora, chat_template_kwargs={"enable_thinking": True}, use_tqdm=False)]

def run(es, a):
    from vllm import LLM, SamplingParams
    out = ROOT / "results" / a.exp / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "metrics.json").exists() and not a.limit: print("done already:", out); return
    adapter = CKPT_DIR / a.exp / "default" / f"set{es}" / f"seed{a.seed}"
    assert adapter.exists() and (out / "train.json").exists(), f"train set {es} first (src/gen_dpo.py or src/gen_sft_dpo.py)"
    llm = LLM(str(merge_adapter(adapter)), dtype="bfloat16", max_model_len=24576, gpu_memory_utilization=0.85, seed=a.seed)   # merged weights, no LoRA at serve time
    sp = SamplingParams(temperature=DECODING[True]["temperature"], top_p=DECODING[True]["top_p"], max_tokens=DECODING[True]["max_tokens"])   # no per-request seed: a retry must draw a new sample
    lora = None   # the adapter is already merged into the served weights
    scores, rubric = scores_of(es), saved_rubric("G4", es); suffix = format_suffix(scores)
    res = {}
    for split in ("dev", "test"):
        texts, y = load_split(split, es)
        if a.limit: texts, y = texts[:a.limit], y[:a.limit]
        t0 = time.time(); reps = reply_texts(llm, sp, lora, rubric, suffix, texts)
        first = score_replies(reps, y, scores, split == "test"); n_retried = 0
        for attempt in range(3):   # retry rule: only parse failures are re-sampled
            bad = [i for i, r in enumerate(reps) if to_index(parse_reply(r), scores) is None]
            if not bad: break
            n_retried += len(bad) if attempt == 0 else 0
            for i, r in zip(bad, reply_texts(llm, sp, lora, rubric, suffix, [texts[i] for i in bad])): reps[i] = r
        res[split] = dict(final=score_replies(reps, y, scores, split == "test"), first=first, n_retried=n_retried, seconds=time.time() - t0)
    tr = json.load(open(out / "train.json"))
    d, t = res["dev"]["final"], res["test"]["final"]
    m = dict(exp=a.exp, served="merged adapter weights", seed=a.seed, essay_set=es, head="gen", thinking=True, model=MODEL, scores=scores, decoding=DECODING[True], rubric_from="G4", lora=LORA,
             train=dict(lr=tr["lr"], beta=tr["beta"], batch=tr["batch"], steps=tr["steps"], pairs=tr["pairs"], train_seconds=tr["train_seconds"]),
             dev=d["metrics"], test=t["metrics"], test_ci=bootstrap_ci(t["y_idx"], t["p_idx"], scores, n=300, seed=0),
             dev_fail_rate=d["fail_rate"], test_fail_rate=t["fail_rate"], n_dev=len(d["y_idx"]), n_test=len(t["y_idx"]),
             before_retry=dict(dev=res["dev"]["first"]["metrics"], test=res["test"]["first"]["metrics"], dev_fail_rate=res["dev"]["first"]["fail_rate"],
                               test_fail_rate=res["test"]["first"]["fail_rate"]),
             retry=dict(attempts=3, dev_replies_retried=res["dev"]["n_retried"], test_replies_retried=res["test"]["n_retried"]),
             eval_seconds=res["dev"]["seconds"] + res["test"]["seconds"])
    print(f"[{a.exp} set{es} s{a.seed}] dev_qwk {d['metrics']['qwk']:.4f} test_qwk {t['metrics']['qwk']:.4f} acc {t['metrics']['acc']:.3f} fail {t['fail_rate']:.3f} "
          f"(before retry {res['test']['first']['fail_rate']:.3f})", flush=True)
    if a.limit: return
    json.dump(m, open(out / "metrics.json", "w"))
    np.savez_compressed(out / "preds.npz", dev_pred=np.array([-1 if p is None else p for p in d["parsed"]]), test_pred=np.array([-1 if p is None else p for p in t["parsed"]]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11))); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--exp", default="G6", choices=["G6", "G7", "G8", "G9"])
    a = ap.parse_args()
    if len(a.essay_sets) == 1: run(a.essay_sets[0], a)
    else:   # a fresh process per set, so that each merged model gets a clean GPU
        for es in a.essay_sets:
            subprocess.run([sys.executable, __file__, "--essay_sets", str(es), "--seed", str(a.seed)] + (["--limit", str(a.limit)] if a.limit else []))
