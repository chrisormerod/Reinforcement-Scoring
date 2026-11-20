"""Native PyTorch evaluation (no vLLM) of a LoRA adapter trained with thinking on: G4's prompt (the set's optimized rubric + suffix) and decoding (temperature 0.6, top-p 0.95, 16384 tokens),
with the retry rule of PROJECT.md section 6.6 (parse failures re-sampled up to 3 times). Same metrics.json layout as src/gen_dpo_eval.py, plus engine="native-pytorch".
usage: python src/gen_native_eval.py [--exp G9] [--essay_sets 1] [--seed 0] [--adapter DIR] [--gen_batch 16] [--max_tokens 16384] [--limit N]   (--limit N: N examples per split, nothing written)
The adapter (default $RF_CKPT_DIR/<exp>/default/set<k>/seed<s>) is merged into the bf16 base weights in GPU memory. Replies are generated in length-sorted, left-padded batches with HF generate;
every finished batch is appended to <out>/replies_cache.jsonl, so an interrupted run resumes where it stopped. Writes results/<exp>/default/seed<s>/set<k>/metrics.json and preds.npz."""
import argparse, json, time
import numpy as np
from gen_common import ROOT, MODEL, DECODING, format_suffix, scores_of, load_split, to_index
from gen_local import *
from eval import bootstrap_ci

def generate(model, tok, prompts, a, eos, desc, cache=None):
    """one reply per prompt (list of str), sampled at G4's decoding. cache: dict prompt-index -> reply, filled in place and flushed by the caller's callback."""
    import torch
    from tqdm.auto import tqdm
    d = DECODING[True]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))   # similar lengths per batch -> little padding
    todo = [i for i in order if i not in cache]
    bar = tqdm(total=len(order), initial=len(order) - len(todo), desc=desc, dynamic_ncols=True)
    for s in range(0, len(todo), a.gen_batch):
        ids = todo[s:s + a.gen_batch]
        enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=d["temperature"], top_p=d["top_p"], top_k=0, max_new_tokens=a.max_tokens, use_cache=True,
                               pad_token_id=tok.pad_token_id, eos_token_id=list(eos))
        for i, row in zip(ids, g[:, enc["input_ids"].shape[1]:]):
            cache[i] = tok.decode(row, skip_special_tokens=True)
        bar.update(len(ids)); cache_flush(cache)
        torch.cuda.empty_cache()
    bar.close()
    return [cache[i] for i in range(len(prompts))]

def cache_flush(cache): pass   # replaced per split in run()

def run(es, a):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    out = ROOT / "results" / a.exp / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "metrics.json").exists() and not a.limit: print("done already:", out); return
    adapter = a.adapter or CKPT_DIR / a.exp / "default" / f"set{es}" / f"seed{a.seed}"
    assert (adapter / "adapter_config.json").exists(), f"no adapter in {adapter}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    tok = load_tokenizer(); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().eval()
    e = model.generation_config.eos_token_id
    eos = set(([e] if isinstance(e, int) else list(e or [])) + [tok.eos_token_id])
    scores, rubric = scores_of(es), saved_rubric("G4", es); suffix = format_suffix(scores)
    res = {}
    for split in ("dev", "test"):
        texts, y = load_split(split, es)
        if a.limit: texts, y = texts[:a.limit], y[:a.limit]
        prompts = [prompt_text(tok, rubric, suffix, t, True) for t in texts]
        cache_file = out / f"replies_cache_{split}.jsonl"
        cache = {}
        if cache_file.exists() and not a.limit:
            for l in open(cache_file): r = json.loads(l); cache[r["i"]] = r["reply"]
        saved = set(cache)
        def flush(c, f=cache_file):
            if a.limit: return
            with open(f, "a") as fh:
                for i in sorted(set(c) - saved): fh.write(json.dumps(dict(i=i, reply=c[i])) + "\n"); saved.add(i)
        globals()["cache_flush"] = flush
        t0 = time.time(); reps = generate(model, tok, prompts, a, eos, f"{a.exp} set{es} {split}", cache)
        first = score_replies(reps, y, scores, split == "test"); n_retried = 0
        for attempt in range(3):   # retry rule: only parse failures are re-sampled
            bad = [i for i, r in enumerate(reps) if to_index(parse_reply(r), scores) is None]
            if not bad: break
            n_retried += len(bad) if attempt == 0 else 0
            retry = generate(model, tok, [prompts[i] for i in bad], a, eos, f"{split} retry {attempt + 1} ({len(bad)})", {})
            for i, r in zip(bad, retry): reps[i] = r
        res[split] = dict(final=score_replies(reps, y, scores, split == "test"), first=first, n_retried=n_retried, seconds=time.time() - t0, replies=reps)
    tf = out / "train.json"
    if tf.exists(): tr = json.load(open(tf)); train = dict(lr=tr["lr"], beta=tr["beta"], batch=tr["batch"], steps=tr["steps"], pairs=tr["pairs"], train_seconds=tr["train_seconds"])
    else:   # no train.json on this machine: summarize the training log kept next to the adapter, if any
        bf = adapter / "batches.jsonl"; b = [json.loads(l) for l in open(bf)] if bf.exists() else []
        train = dict(steps=len(b), note="train.json not available; summary from batches.jsonl", beta=0.0,
                     first10_mean_reward=float(np.mean([x["mean_reward"] for x in b[:10]])) if b else None, last10_mean_reward=float(np.mean([x["mean_reward"] for x in b[-10:]])) if b else None,
                     first10_fail_rate=float(np.mean([x["fail_rate"] for x in b[:10]])) if b else None, last10_fail_rate=float(np.mean([x["fail_rate"] for x in b[-10:]])) if b else None)
    d, t = res["dev"]["final"], res["test"]["final"]
    m = dict(exp=a.exp, engine="native-pytorch", served="merged adapter weights", adapter=str(adapter), seed=a.seed, essay_set=es, head="gen", thinking=True, model=MODEL, scores=scores,
             decoding=DECODING[True], rubric_from="G4", lora=LORA, train=train, dev=d["metrics"], test=t["metrics"], test_ci=bootstrap_ci(t["y_idx"], t["p_idx"], scores, n=300, seed=0),
             dev_fail_rate=d["fail_rate"], test_fail_rate=t["fail_rate"], n_dev=len(d["y_idx"]), n_test=len(t["y_idx"]),
             before_retry=dict(dev=res["dev"]["first"]["metrics"], test=res["test"]["first"]["metrics"], dev_fail_rate=res["dev"]["first"]["fail_rate"], test_fail_rate=res["test"]["first"]["fail_rate"]),
             retry=dict(attempts=3, dev_replies_retried=res["dev"]["n_retried"], test_replies_retried=res["test"]["n_retried"]),
             eval_seconds=res["dev"]["seconds"] + res["test"]["seconds"])
    print(f"[{a.exp} set{es} s{a.seed}] dev_qwk {d['metrics']['qwk']:.4f} test_qwk {t['metrics']['qwk']:.4f} acc {t['metrics']['acc']:.3f} fail {t['fail_rate']:.3f} "
          f"(before retry {res['test']['first']['fail_rate']:.3f})", flush=True)
    if a.limit: return
    json.dump(m, open(out / "metrics.json", "w"))
    np.savez_compressed(out / "preds.npz", dev_pred=np.array([-1 if p is None else p for p in d["parsed"]]), test_pred=np.array([-1 if p is None else p for p in t["parsed"]]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", default="G9"); ap.add_argument("--essay_sets", nargs="*", type=int, default=[1]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter", type=Path, default=None, help="adapter directory (default: the final adapter under $RF_CKPT_DIR)")
    ap.add_argument("--gen_batch", type=int, default=16); ap.add_argument("--max_tokens", type=int, default=DECODING[True]["max_tokens"]); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
