"""G6 stage 2: DPO of Qwen3.5-4B (bf16 + LoRA, local GPU) on pairs built from the collected thinking replies (src/gen_collect.py).
usage: python src/gen_dpo.py [--essay_sets 1 2 ...] [--seed 0] [--lr L] [--beta B] [--epochs E] [--batch 8] [--max_completion_tokens 16384] [--collect_dir collect_merged] [--limit N]
Pairs: every training example with at least one parsable right reply and at least one parsable wrong reply gives ONE pair, chosen = a right reply and
rejected = a wrong reply, each drawn uniformly at random (seeded) from its replies. Unparsable replies are never used.
Policy = base + LoRA; reference = the same model with the adapter disabled (its log-probs are computed once, before training). Loss: -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))),
log-probs summed over the whole completion (thinking and answer). The start point is the base model. lr, beta and epochs default to the values in configs/tuned_G6.json (written by gen_dpo_sweep.py select); max_completion_tokens (16384), batch and schedule match the sweep.
The pairs come from results/G6/<collect_dir> (default collect_merged: original replies pooled with the resampled ones, see gen_merge_collect.py).
A pair whose chosen or rejected completion exceeds --max_completion_tokens is dropped (GPU memory); the number dropped and the token-length quantiles are printed and saved.
Writes the adapter to $RF_CKPT_DIR/G6/default/set<k>/seed<s> and results/G6/default/seed<s>/set<k>/train.json. Evaluation is src/gen_dpo_eval.py."""
import argparse, json, random, time
import numpy as np, torch, torch.nn.functional as F
from gen_common import ROOT, format_suffix, scores_of, load_split
from gen_local import *
from transformers import get_cosine_schedule_with_warmup

def build_pairs(rows, rng):
    """one (chosen text, rejected text, example id) per example that has both a right and a wrong parsable reply; also the pass-rate statistics"""
    pairs, n_all, n_right = [], 0, 0
    for x in rows:
        right = [r for r in x["replies"] if r["score"] is not None and r["score"] == x["gold"]]
        wrong = [r for r in x["replies"] if r["score"] is not None and r["score"] != x["gold"]]
        n_all += len(x["replies"]); n_right += sum(r["score"] == x["gold"] for r in x["replies"])
        if right and wrong: pairs.append((right[rng.randrange(len(right))]["text"], wrong[rng.randrange(len(wrong))]["text"], x["i"]))
    return pairs, dict(n_examples=len(rows), n_pairs=len(pairs), pass_rate=n_right / max(1, n_all),
                       n_all_right=sum(all(r["score"] == x["gold"] for r in x["replies"]) for x in rows),
                       n_no_wrong_parsable=sum(not any(r["score"] is not None and r["score"] != x["gold"] for r in x["replies"]) for x in rows),
                       n_no_right_parsable=sum(not any(r["score"] is not None and r["score"] == x["gold"] for r in x["replies"]) for x in rows))

def completion_logp(model, prompt_ids, comp_ids, chunk=512):
    """summed log-probability of the completion tokens given the prompt. The output head and the loss run in checkpointed chunks, so the
    (completion length x 248k vocabulary) logits are never held at once: that tensor alone is several GB for a 5k-token reply"""
    from torch.utils.checkpoint import checkpoint
    ids = torch.tensor([prompt_ids + comp_ids], device="cuda"); n = len(comp_ids)
    inner = model.get_base_model()   # the adapter layers are modules inside it, so this still uses (or, when disabled, skips) the LoRA weights
    h = inner.model(input_ids=ids).last_hidden_state[0, -n - 1:-1]   # hidden states that predict the n completion tokens
    tgt = ids[0, -n:]
    nll = lambda hh, tt: F.cross_entropy(inner.lm_head(hh).float(), tt, reduction="sum")
    return -sum(checkpoint(nll, h[i:i + chunk], tgt[i:i + chunk], use_reentrant=False) for i in range(0, n, chunk))

def run(es, a):
    out = ROOT / "results" / "G6" / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "train.json").exists() and not a.limit: print("done already:", out); return
    f = ROOT / "results" / "G6" / a.collect_dir / (f"set{es}.jsonl" if not a.limit else f"set{es}.limit.jsonl")
    rows = [json.loads(l) for l in open(f)]
    assert len(rows) == len(load_split("train", es)[0]) or a.limit, f"{f} has {len(rows)} examples; finish gen_collect.py first"
    rng = random.Random(a.seed)
    pairs, stats = build_pairs(rows, rng)
    scores, rubric = scores_of(es), saved_rubric("G4", es)
    tok = load_tokenizer(); texts, _ = load_split("train", es)
    enc, lens = [], []
    for c, r, i in pairs:
        p = tok(prompt_text(tok, rubric, format_suffix(scores), texts[i], True), add_special_tokens=False).input_ids
        cc, rr = (tok(t + "<|im_end|>", add_special_tokens=False).input_ids for t in (c, r))
        lens.append(max(len(cc), len(rr)))
        if max(len(cc), len(rr)) <= a.max_completion_tokens: enc.append((p, cc, rr))
    q = {f"p{k}": float(np.percentile(lens, k)) for k in (50, 90, 99)} if lens else {}
    stats.update(n_pairs_used=len(enc), n_pairs_dropped_too_long=len(pairs) - len(enc), max_completion_tokens=a.max_completion_tokens, completion_token_quantiles=q)
    print(f"[G6 set{es}] {stats}", flush=True)
    assert enc, "no pairs to train on"
    if a.limit: enc = enc[:a.limit]
    model = load_lora_model(a.seed)
    model.eval(); ref = []
    with torch.no_grad(), model.disable_adapter():   # reference log-probs, once
        for p, cc, rr in enc: ref.append((completion_logp(model, p, cc).item(), completion_logp(model, p, rr).item()))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    per_epoch = max(1, len(enc) // a.batch); T = per_epoch * a.epochs   # as in gen_dpo_sweep.py: a fresh permutation per epoch, one cosine schedule over all steps
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * T)), T); curve, t0, step = [], time.time(), 0
    for ep in range(a.epochs):
        order = np.random.default_rng(a.seed + 1000 * ep).permutation(len(enc))
        for k in range(per_epoch):
            step += 1; acc, loss_sum, margin = 0, 0.0, 0.0
            for j in order[k * a.batch: (k + 1) * a.batch]:
                p, cc, rr = enc[j]
                pc, pr = completion_logp(model, p, cc), completion_logp(model, p, rr)
                z = a.beta * ((pc - ref[j][0]) - (pr - ref[j][1]))
                loss = -F.logsigmoid(z) / a.batch
                loss.backward(); loss_sum += loss.item(); acc += int(z.item() > 0); margin += z.item() / a.batch
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            curve.append(dict(step=step, epoch=ep + 1, loss=loss_sum, pref_acc=acc / a.batch, margin=margin))
            if step % 5 == 0 or step == T: print(f"[G6 set{es} s{a.seed}] step {step}/{T} loss {loss_sum:.4f} pref_acc {acc / a.batch:.2f} margin {margin:.3f} peak_gpu_gb {torch.cuda.max_memory_allocated() / 2**30:.1f} {time.time() - t0:.0f}s", flush=True)
    if a.limit: return
    model.save_pretrained(CKPT_DIR / "G6" / "default" / f"set{es}" / f"seed{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(exp="G6", seed=a.seed, essay_set=es, lora=LORA, lr=a.lr, beta=a.beta, batch=a.batch, epochs=a.epochs, steps=T, pairs=stats, curve=curve,
                   train_seconds=time.time() - t0), open(out / "train.json", "w"))
    del model, opt; torch.cuda.empty_cache()

if __name__ == "__main__":
    tuned = json.load(open(ROOT / "configs" / "tuned_G6.json"))   # selected on dev QWK by gen_dpo_sweep.py select
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11))); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=tuned["lr"]); ap.add_argument("--beta", type=float, default=tuned["beta"]); ap.add_argument("--epochs", type=int, default=tuned["epochs"]); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_completion_tokens", type=int, default=16384); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--collect_dir", default="collect_merged")
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
