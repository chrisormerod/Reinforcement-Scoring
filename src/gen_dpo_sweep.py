"""G6 hyperparameter sweep (DPO stage), selected on DEV QWK only. The test split is never scored here.
usage:
  python src/gen_dpo_sweep.py run    [--essay_sets 1] [--seeds 100] [--configs c0 c1 ...] [--limit N]   # train then dev-eval every config, skipping finished ones
  python src/gen_dpo_sweep.py train  --cfg c1 --essay_set 1 --seed 100 [--limit N]                        # one config, training only
  python src/gen_dpo_sweep.py eval   --cfg c1 --essay_set 1 --seed 100 [--limit N]                        # one config, dev evaluation only
  python src/gen_dpo_sweep.py select [--essay_sets 1]                                                     # table of dev QWK per config; writes configs/tuned_G6.json
Needs the whole GPU (training peaks near 19 GB; vLLM evaluation uses 85% of GPU memory), so stop any vLLM server on the machine first. Training and evaluation run
as separate processes so that the GPU is free for vLLM. Needs results/G6/collect/set<k>.jsonl complete for the set.
Tuning seeds are >= 100 (as in launch.py) and are never reported. Outputs: results/tune_G6/<cfg>/seed<s>/set<k>/{train.json,dev.json}; adapters in $RF_CKPT_DIR/tune_G6/<cfg>/set<k>/seed<s>.
Everything else (pairs, prompt, LoRA, decoding, retry rule, batch 8, cosine schedule with 3% warmup) is as in gen_dpo.py / gen_dpo_eval.py; only lr, beta and epochs vary.
DEV NOISE: sampling at temperature 0.6 on ~170 dev responses makes dev QWK noisy (differences below ~0.03 are probably noise); c0 repeats the default so that the noise is visible."""
import argparse, json, random, subprocess, sys, time
from pathlib import Path
import numpy as np
from gen_common import ROOT, MODEL, DECODING, format_suffix, scores_of, load_split, to_index
from gen_local import *

OUT = ROOT / "results" / "tune_G6"
# name: (learning rate, DPO beta, epochs). c0 is the default used for the reported G6 run, repeated on a tuning seed as a noise reference.
CONFIGS = {"c0": (5e-5, 0.1, 1), "c1": (2e-4, 0.1, 1), "c2": (2e-4, 0.3, 1), "c3": (5e-5, 0.1, 3), "c4": (2e-4, 0.1, 3), "c5": (1e-4, 0.1, 2)}
BATCH, MAX_COMPLETION_TOKENS = 8, 16384
paths = lambda cfg, es, seed: (OUT / cfg / f"seed{seed}" / f"set{es}", CKPT_DIR / "tune_G6" / cfg / f"set{es}" / f"seed{seed}")

def train(cfg, es, seed, limit=0):
    import torch, torch.nn.functional as F
    from transformers import get_cosine_schedule_with_warmup
    from gen_dpo import build_pairs, completion_logp
    lr, beta, epochs = CONFIGS[cfg]; out, adapter = paths(cfg, es, seed)
    if (out / "train.json").exists() and not limit: print("train done already:", out); return
    rows = [json.loads(l) for l in open(ROOT / "results" / "G6" / "collect" / f"set{es}.jsonl")]
    texts, _ = load_split("train", es)
    assert len(rows) == len(texts) or limit, f"set {es}: {len(rows)} of {len(texts)} examples collected; finish gen_collect.py first"
    pairs, stats = build_pairs(rows, random.Random(seed))
    scores, rubric = scores_of(es), saved_rubric("G4", es); tok = load_tokenizer(); enc = []
    for c, r, i in pairs:
        p = tok(prompt_text(tok, rubric, format_suffix(scores), texts[i], True), add_special_tokens=False).input_ids
        cc, rr = (tok(t + "<|im_end|>", add_special_tokens=False).input_ids for t in (c, r))
        if max(len(cc), len(rr)) <= MAX_COMPLETION_TOKENS: enc.append((p, cc, rr))
    if limit: enc = enc[:limit]
    assert enc, "no pairs"
    print(f"[sweep {cfg} set{es} s{seed}] lr {lr} beta {beta} epochs {epochs}: {len(enc)} pairs of {len(pairs)}", flush=True)
    model = load_lora_model(seed); model.eval(); ref = []
    with torch.no_grad(), model.disable_adapter():
        for p, cc, rr in enc: ref.append((completion_logp(model, p, cc).item(), completion_logp(model, p, rr).item()))
    model.train(); params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    per_epoch = max(1, len(enc) // BATCH); T = per_epoch * epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * T)), T); curve, t0, step = [], time.time(), 0
    for ep in range(epochs):
        order = np.random.default_rng(seed + 1000 * ep).permutation(len(enc))
        for k in range(per_epoch):
            step += 1; acc, loss_sum, margin = 0, 0.0, 0.0
            for j in order[k * BATCH:(k + 1) * BATCH]:
                p, cc, rr = enc[j]
                z = beta * ((completion_logp(model, p, cc) - ref[j][0]) - (completion_logp(model, p, rr) - ref[j][1]))
                loss = -F.logsigmoid(z) / BATCH
                loss.backward(); loss_sum += loss.item(); acc += int(z.item() > 0); margin += z.item() / BATCH
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            curve.append(dict(step=step, epoch=ep + 1, loss=loss_sum, pref_acc=acc / BATCH, margin=margin))
            if step % 5 == 0 or step == T: print(f"[sweep {cfg} set{es} s{seed}] step {step}/{T} loss {loss_sum:.4f} pref_acc {acc / BATCH:.2f} margin {margin:.3f} {time.time() - t0:.0f}s", flush=True)
    if limit: return
    out.mkdir(parents=True, exist_ok=True); adapter.parent.mkdir(parents=True, exist_ok=True); model.save_pretrained(adapter)
    json.dump(dict(cfg=cfg, lr=lr, beta=beta, epochs=epochs, batch=BATCH, steps=T, pairs=stats, curve=curve, train_seconds=time.time() - t0), open(out / "train.json", "w"))

def evaluate(cfg, es, seed, limit=0):
    """dev split only; parse failures are re-sampled up to 3 times (the retry rule of PROJECT.md 6.6)"""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from gen_dpo_eval import reply_texts
    out, adapter = paths(cfg, es, seed)
    if (out / "dev.json").exists() and not limit: print("eval done already:", out); return
    assert adapter.exists() and (out / "train.json").exists(), f"train {cfg} set {es} seed {seed} first"
    llm = LLM(MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=LORA["r"], max_model_len=24576, gpu_memory_utilization=0.85, seed=seed)
    sp = SamplingParams(temperature=DECODING[True]["temperature"], top_p=DECODING[True]["top_p"], max_tokens=DECODING[True]["max_tokens"])
    lora = LoRARequest(f"{cfg}_set{es}", 1, str(adapter))
    scores, rubric = scores_of(es), saved_rubric("G4", es); suffix = format_suffix(scores)
    texts, y = load_split("dev", es)
    if limit: texts, y = texts[:limit], y[:limit]
    t0 = time.time(); reps = reply_texts(llm, sp, lora, rubric, suffix, texts); first = score_replies(reps, y, scores)
    for _ in range(3):
        bad = [i for i, r in enumerate(reps) if to_index(parse_reply(r), scores) is None]
        if not bad: break
        for i, r in zip(bad, reply_texts(llm, sp, lora, rubric, suffix, [texts[i] for i in bad])): reps[i] = r
    fin = score_replies(reps, y, scores)
    print(f"[sweep {cfg} set{es} s{seed}] dev_qwk {fin['metrics']['qwk']:.4f} (before retry {first['metrics']['qwk']:.4f}) fail {fin['fail_rate']:.3f}", flush=True)
    if limit: return
    tr = json.load(open(out / "train.json"))
    json.dump(dict(cfg=cfg, lr=tr["lr"], beta=tr["beta"], epochs=tr["epochs"], dev=fin["metrics"], dev_fail_rate=fin["fail_rate"], n_dev=len(y),
                   before_retry=dict(dev=first["metrics"], dev_fail_rate=first["fail_rate"]), eval_seconds=time.time() - t0), open(out / "dev.json", "w"))

def run(a):
    me = str(Path(__file__).resolve())
    for seed in a.seeds:
        assert seed >= 100, "tuning seeds are >= 100"
        for es in a.essay_sets:
            for cfg in a.configs:
                out, _ = paths(cfg, es, seed)
                if (out / "dev.json").exists() and not a.limit: print("skip (done):", cfg, es, seed); continue
                for phase in ("train", "eval"):   # separate processes: training frees the GPU before vLLM takes 85% of it
                    cmd = [sys.executable, me, phase, "--cfg", cfg, "--essay_set", str(es), "--seed", str(seed)] + (["--limit", str(a.limit)] if a.limit else [])
                    if subprocess.run(cmd).returncode != 0:
                        print(f"FAILED: {' '.join(cmd)}", flush=True); break

def select(a):
    rows = []
    for cfg, (lr, beta, ep) in CONFIGS.items():
        q = [json.load(open(f))["dev"]["qwk"] for es in a.essay_sets for f in sorted(OUT.glob(f"{cfg}/seed*/set{es}/dev.json"))]
        if q: rows.append((float(np.mean(q)), cfg, lr, beta, ep, len(q)))
    assert rows, "no finished configs"
    ref = [json.load(open(ROOT / "results" / "G4" / "default" / "seed0" / f"set{es}" / "metrics.json"))["dev"]["qwk"] for es in a.essay_sets]
    print(f"reference: G4 (prompt only) dev QWK {np.mean(ref):.3f}\n\n| cfg | lr | beta | epochs | n runs | mean dev QWK |\n|---|---|---|---|---|---|")
    for m, cfg, lr, beta, ep, n in sorted(rows, reverse=True): print(f"| {cfg} | {lr:g} | {beta:g} | {ep} | {n} | {m:.3f} |")
    m, cfg, lr, beta, ep, n = max(rows)
    json.dump(dict(cfg=cfg, lr=lr, beta=beta, epochs=ep, mean_dev_qwk=m, essay_sets=a.essay_sets), open(ROOT / "configs" / "tuned_G6.json", "w"), indent=1)
    print(f"\nbest: {cfg} (lr {lr:g}, beta {beta:g}, epochs {ep}), written to configs/tuned_G6.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["run", "train", "eval", "select"])
    ap.add_argument("--essay_sets", nargs="*", type=int, default=[1]); ap.add_argument("--seeds", nargs="*", type=int, default=[100])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS)); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cfg"); ap.add_argument("--essay_set", type=int); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.phase == "run": run(a)
    elif a.phase == "select": select(a)
    else: (train if a.phase == "train" else evaluate)(a.cfg, a.essay_set, a.seed, a.limit)
