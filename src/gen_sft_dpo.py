"""G7: SFT then DPO of Qwen3.5-4B (bf16 + LoRA, local GPU) on the collected thinking replies (src/gen_collect.py, pooled by src/gen_merge_collect.py). Thinking on, G4's prompt.
usage: python src/gen_sft_dpo.py [--essay_sets 1 2 ...] [--seed 0] [--sft_lr 5e-5] [--sft_epochs 1] [--lr L] [--beta B] [--epochs E] [--batch 8] [--max_completion_tokens 16384] [--collect_dir collect_merged] [--limit N]
Stage 1 (SFT): every training example with at least one parsable RIGHT reply (examples that give a DPO pair AND examples whose parsable replies are all right) gives one SFT target = a right reply drawn
uniformly at random (seeded), whole (thinking and answer). Loss: mean token cross-entropy over the completion (prompt masked), averaged over the examples of a batch. Examples with no right reply are not used.
G7 REUSES the G8 adapter (results/G8 + $RF_CKPT_DIR/G8) for stage 1 when it exists for the set and seed with the same --sft_lr/--sft_epochs/--batch/--collect_dir/--max_completion_tokens and the same SFT data; otherwise (or with --no_reuse) it trains stage 1 itself.
Stage 2 (DPO): exactly the pairs of G6 (build_pairs with the same seed, so the same pairs as src/gen_dpo.py), continuing the SAME LoRA adapter. The reference is the stage-1 (SFT) model: its log-probs
are computed once, with the adapter active, before stage 2 starts. Loss, schedule, batch and the lr/beta/epochs defaults (configs/tuned_G6.json) are those of G6, so G7 vs G6 isolates the SFT stage.
The SFT hyperparameters (--sft_lr, --sft_epochs) are NOT tuned. Cosine schedule with 3% warmup per stage, gradient clipping 1.0, a fresh optimizer per stage.
With --sft_only only stage 1 runs and the result is experiment G8 (SFT alone; results/G8, adapter $RF_CKPT_DIR/G8/default/set<k>/seed<s>); evaluate with src/gen_dpo_eval.py --exp G8.
Without it, writes the SFT adapter to $RF_CKPT_DIR/G7/default/set<k>/seed<s>_sft, the final adapter to .../seed<s>, and results/G7/default/seed<s>/set<k>/train.json. Evaluation: python src/gen_dpo_eval.py --exp G7."""
import argparse, json, random, time
import numpy as np, torch, torch.nn.functional as F
from gen_common import ROOT, format_suffix, scores_of, load_split
from pathlib import Path
from gen_local import *
from gen_dpo import build_pairs, completion_logp
from transformers import get_cosine_schedule_with_warmup

def sft_targets(rows, rng):
    """(right reply text, example id) for every example with a parsable right reply"""
    out = []
    for x in rows:
        right = [r for r in x["replies"] if r["score"] is not None and r["score"] == x["gold"]]
        if right: out.append((right[rng.randrange(len(right))]["text"], x["i"]))
    return out

def load_adapter(model, path):
    """copy a saved LoRA adapter's weights into the (fresh) adapter of `model`"""
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    res = set_peft_model_state_dict(model, load_file(str(Path(path) / "adapter_model.safetensors")))
    assert not getattr(res, "unexpected_keys", None), res.unexpected_keys

def g8_match(es, a, stats):
    """(adapter dir, G8 train.json) if G8 trained this set and seed with the same SFT settings and data, else None (stage 1 is then trained here)"""
    ad = CKPT_DIR / "G8" / "default" / f"set{es}" / f"seed{a.seed}"; tj = ROOT / "results" / "G8" / "default" / f"seed{a.seed}" / f"set{es}" / "train.json"
    if not (ad / "adapter_model.safetensors").exists() or not tj.exists(): return None
    g = json.load(open(tj))
    same = dict(lr=a.sft_lr, epochs=a.sft_epochs, batch=a.batch, collect_dir=a.collect_dir) == {k: g.get(k) for k in ("lr", "epochs", "batch", "collect_dir")} \
        and g["pairs"].get("n_sft_used") == stats["n_sft_used"] and g["pairs"].get("max_completion_tokens") == a.max_completion_tokens
    if not same: print(f"[G7 set{es}] a G8 adapter exists but its settings differ ({tj}); training stage 1 here", flush=True); return None
    return ad, g

def run(es, a):
    exp = "G8" if a.sft_only else "G7"   # G8 = stage 1 alone
    out = ROOT / "results" / exp / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "train.json").exists() and not a.limit: print("done already:", out); return
    f = ROOT / "results" / "G6" / a.collect_dir / f"set{es}.jsonl"
    rows = [json.loads(l) for l in open(f)]
    texts, _ = load_split("train", es)
    assert len(rows) == len(texts), f"{f} has {len(rows)} of {len(texts)} examples"
    pairs, stats = build_pairs(rows, random.Random(a.seed))
    sft = sft_targets(rows, random.Random(a.seed + 7))
    scores, rubric = scores_of(es), saved_rubric("G4", es); tok = load_tokenizer()
    prompt = lambda i: tok(prompt_text(tok, rubric, format_suffix(scores), texts[i], True), add_special_tokens=False).input_ids
    comp = lambda t: tok(t + "<|im_end|>", add_special_tokens=False).input_ids
    ok = lambda *c: max(map(len, c)) <= a.max_completion_tokens
    enc_sft = [(prompt(i), c) for t, i in sft if ok(c := comp(t))]
    enc = [(prompt(i), c, r) for ct, rt, i in pairs if ok(c := comp(ct), r := comp(rt))]
    stats.update(n_sft_examples=len(sft), n_sft_used=len(enc_sft), n_pairs_used=len(enc), n_pairs_dropped_too_long=len(pairs) - len(enc), max_completion_tokens=a.max_completion_tokens)
    print(f"[G7 set{es}] {stats}", flush=True)
    assert enc_sft and (enc or a.sft_only), "no SFT targets or no pairs"
    if a.limit: enc_sft, enc = enc_sft[:a.limit], enc[:a.limit]
    model = load_lora_model(a.seed); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    t0 = time.time(); curve_sft, curve = [], []
    # ---- stage 1: SFT on right replies (G7 reuses the G8 adapter when a matching one exists)
    reuse = None if a.sft_only or a.no_reuse else g8_match(es, a, stats)
    if reuse:
        load_adapter(model, reuse[0]); g8 = reuse[1]; T1, curve_sft, sft_seconds = g8["steps"], g8["curve"], g8["train_seconds"]
        print(f"[G7 set{es} s{a.seed}] stage 1 skipped: loaded the G8 adapter {reuse[0]}", flush=True)
    else:
        opt = torch.optim.AdamW(params, lr=a.sft_lr, weight_decay=0.0)
        per_epoch = max(1, len(enc_sft) // a.batch); T1 = per_epoch * a.sft_epochs
        sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * T1)), T1); step = 0
        for ep in range(a.sft_epochs):
            order = np.random.default_rng(a.seed + 2000 * (ep + 1)).permutation(len(enc_sft))
            for k in range(per_epoch):
                step += 1; loss_sum = 0.0
                for j in order[k * a.batch:(k + 1) * a.batch]:
                    p, c = enc_sft[j]
                    loss = -completion_logp(model, p, c) / len(c) / a.batch
                    loss.backward(); loss_sum += loss.item()
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                curve_sft.append(dict(step=step, epoch=ep + 1, loss=loss_sum))
                if step % 5 == 0 or step == T1: print(f"[G7 set{es} s{a.seed}] SFT step {step}/{T1} loss {loss_sum:.4f} {time.time() - t0:.0f}s", flush=True)
        sft_seconds = time.time() - t0
    if a.sft_only:
        if a.limit: return
        model.save_pretrained(CKPT_DIR / "G8" / "default" / f"set{es}" / f"seed{a.seed}"); out.mkdir(parents=True, exist_ok=True)
        json.dump(dict(exp="G8", seed=a.seed, essay_set=es, lora=LORA, lr=a.sft_lr, beta=None, batch=a.batch, epochs=a.sft_epochs, steps=T1, pairs=stats, curve=curve_sft, collect_dir=a.collect_dir,
                       train_seconds=sft_seconds), open(out / "train.json", "w"))
        del model, opt; torch.cuda.empty_cache(); return
    if not a.limit and not reuse: model.save_pretrained(CKPT_DIR / "G7" / "default" / f"set{es}" / f"seed{a.seed}_sft")
    # ---- stage 2: DPO, reference = the SFT model (same adapter, log-probs computed once now)
    model.eval(); ref = []
    with torch.no_grad():
        for p, cc, rr in enc: ref.append((completion_logp(model, p, cc).item(), completion_logp(model, p, rr).item()))
    model.train(); opt = None
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    per_epoch = max(1, len(enc) // a.batch); T = per_epoch * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * T)), T); step = 0
    for ep in range(a.epochs):
        order = np.random.default_rng(a.seed + 1000 * ep).permutation(len(enc))
        for k in range(per_epoch):
            step += 1; acc, loss_sum, margin = 0, 0.0, 0.0
            for j in order[k * a.batch:(k + 1) * a.batch]:
                p, cc, rr = enc[j]
                z = a.beta * ((completion_logp(model, p, cc) - ref[j][0]) - (completion_logp(model, p, rr) - ref[j][1]))
                loss = -F.logsigmoid(z) / a.batch
                loss.backward(); loss_sum += loss.item(); acc += int(z.item() > 0); margin += z.item() / a.batch
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            curve.append(dict(step=step, epoch=ep + 1, loss=loss_sum, pref_acc=acc / a.batch, margin=margin))
            if step % 5 == 0 or step == T: print(f"[G7 set{es} s{a.seed}] DPO step {step}/{T} loss {loss_sum:.4f} pref_acc {acc / a.batch:.2f} margin {margin:.3f} peak_gpu_gb {torch.cuda.max_memory_allocated() / 2**30:.1f} {time.time() - t0:.0f}s", flush=True)
    if a.limit: return
    model.save_pretrained(CKPT_DIR / "G7" / "default" / f"set{es}" / f"seed{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(exp="G7", seed=a.seed, essay_set=es, lora=LORA, lr=a.lr, beta=a.beta, batch=a.batch, epochs=a.epochs, steps=T, pairs=stats, curve=curve,
                   sft=dict(lr=a.sft_lr, epochs=a.sft_epochs, steps=T1, n_examples=len(enc_sft), curve=curve_sft, seconds=sft_seconds, reused_from="G8" if reuse else None), train_seconds=time.time() - t0), open(out / "train.json", "w"))
    del model, opt; torch.cuda.empty_cache()

if __name__ == "__main__":
    tuned = json.load(open(ROOT / "configs" / "tuned_G6.json"))   # DPO stage: the G6 selection, so that G7 differs from G6 only by the SFT stage
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11))); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_reuse", action="store_true", help="G7: always train stage 1 even if a matching G8 adapter exists")
    ap.add_argument("--sft_only", action="store_true", help="stage 1 only (experiment G8)")
    ap.add_argument("--sft_lr", type=float, default=5e-5); ap.add_argument("--sft_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=tuned["lr"]); ap.add_argument("--beta", type=float, default=tuned["beta"]); ap.add_argument("--epochs", type=int, default=tuned["epochs"])
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max_completion_tokens", type=int, default=16384)
    ap.add_argument("--collect_dir", default="collect_merged"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
