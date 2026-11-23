"""Custom LoRA trainer for vllm-sr/d3-mini and d3-flash (--model) (D3 SFT, D4 DPO, D5 GRPO). Run in ~/ml. The packaged model refuses training, so src/d3_model.py rebuilds its logits path with gradients (see d3_common.py).
usage: python src/train_d3.py --mode sft|dpo|grpo [--essay_sets 3 ...] [--seed 0] [--text original|D2] [--lr L] [--epochs E] [--batch 16] [--mb 4] [--beta B] [--group 8] [--kl 0] [--limit N]
All three start from the BASE d3-mini weights with a fresh LoRA adapter (r16, alpha 32, dropout 0.05, every linear layer of the text decoder) and the same request (`--text`: the original rubric split into
instructions + per-level criteria, or D2's optimized text from results/D2/<model>/seed0/set<k>/prompt.json). The model's output is a distribution over the set's K score levels, p = softmax(logits / T).
  sft  (D3): loss = -log p(gold), mean over the batch (PROJECT.md 5.1).
  dpo  (D4): one pair per example, chosen = gold level, rejected = a level exactly ONE score point off (adjacent value, direction drawn once with a fixed seed, one neighbour at the ends; the same draw as the other preference arms);
             loss = -log sigmoid(beta * ((log p(c) - log p_ref(c)) - (log p(r) - log p_ref(r)))), reference = the base model (adapter off), its log-probs computed once before training (5.3).
  grpo (D5): per prompt sample --group levels from p, reward 1 if the sampled level is gold else 0, advantage = (r - mean) / (std + 1e-4) per group, loss = -(1/(B G)) sum A log p(a) + kl * KL(p || p_ref),
             one on-policy update per batch so no clipping (5.5); the reference is the base model (adapter off) and is only evaluated when --kl > 0.
AdamW, cosine schedule with 3% warmup, gradient clipping 1.0, bf16 base with fp32 adapter weights. Defaults are NOT tuned. The adapter is saved as $RF_CKPT_DIR/D<n>/<model>/set<k>/seed<s>/adapter.pt and the log to
results/D<n>/<model>/seed<s>/set<k>/train.json. Export for serving: src/d3_export.py; evaluation: src/d3_eval.py."""
import argparse, json, random, time
import numpy as np, torch, torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from common import ROOT, scores_of, load_split
from d3_common import CKPT_DIR, MODELS, question, original_rubric, load_text
from d3_model import D3Policy, LORA

EXP = dict(sft="D3", dpo="D4", grpo="D5")

def wrong_level(seed, es, i, k, K):
    """index of a level exactly one score point off gold index k (seeded draw per example)"""
    nb = [j for j in (k - 1, k + 1) if 0 <= j < K]
    return random.Random(f"{seed}-{es}-{i}").choice(nb)

def run(es, a):
    exp = EXP[a.mode]; out = ROOT / "results" / exp / a.model / f"seed{a.seed}" / f"set{es}"
    if (out / "train.json").exists() and not a.limit: print("done already:", out); return
    texts, y = load_split("train", es); scores = scores_of(es); K = len(scores)
    gold = np.array([scores.index(v) for v in y])
    ins, crit = load_text(es, a.text, a.model); q = question(ins, crit)
    pol = D3Policy(a.model, seed=a.seed); params = pol.params
    enc = [pol.prompt_ids(t, q) for t in texts]; seqs = [e[0] for e in enc]; counts = [e[1] for e in enc]
    assert set(counts) == {K}, (set(counts), K)
    N = len(texts) if not a.limit else min(a.limit, len(texts)); idx_all = np.arange(N)
    lens = [len(s) for s in seqs]; print(f"[{exp} set{es}] {N} examples, {K} levels, prompt tokens mean {np.mean(lens):.0f} max {max(lens)}, trainable {sum(p.numel() for p in params)}", flush=True)

    def fwd(idx, grad):
        """log-probs [len(idx), K] in micro-batches of --mb, with a backward closure when `grad` (called once per micro-batch by the loss functions below)"""
        for s in range(0, len(idx), a.mb):
            sub = idx[s:s + a.mb]
            with torch.set_grad_enabled(grad):
                yield sub, pol.log_probs([seqs[j] for j in sub], [K] * len(sub))[:, :K]

    ref = None
    if a.mode == "dpo" or (a.mode == "grpo" and a.kl > 0):   # reference log-probs of the base model, computed once
        t0 = time.time(); pol.set_adapter_enabled(False); pol.train_mode(False); ref = np.zeros((N, K), dtype=np.float32)
        for sub, lp in fwd(idx_all, False): ref[sub] = lp.float().cpu().numpy()
        pol.set_adapter_enabled(True); print(f"[{exp} set{es}] reference log-probs {time.time() - t0:.0f}s (base top-1 acc on train {np.mean(ref.argmax(1) == gold[:N]):.3f})", flush=True)
    rej = np.array([wrong_level(a.seed, es, i, gold[i], K) for i in range(N)]) if a.mode == "dpo" else None

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    per_epoch = max(1, N // a.batch); T = per_epoch * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * T)), T)
    pol.train_mode(True); curve, step, t0 = [], 0, time.time(); gen = torch.Generator(device="cuda").manual_seed(a.seed)
    for ep in range(a.epochs):
        order = np.random.default_rng(a.seed + 1000 * ep).permutation(N)
        for k in range(per_epoch):
            step += 1; batch = order[k * a.batch:(k + 1) * a.batch]; B = len(batch); rec = dict(step=step, epoch=ep + 1, loss=0.0)
            acc = margin = reward = nonzero = klv = top1 = 0.0
            for sub, lp in fwd(batch, True):
                g = torch.as_tensor(gold[sub], device=lp.device)
                if a.mode == "sft":
                    loss = -lp.gather(1, g[:, None]).sum() / B; top1 += (lp.argmax(1) == g).sum().item()
                elif a.mode == "dpo":
                    r = torch.as_tensor(rej[sub], device=lp.device); rf = torch.as_tensor(ref[sub], device=lp.device)
                    z = a.beta * ((lp.gather(1, g[:, None])[:, 0] - rf.gather(1, g[:, None])[:, 0]) - (lp.gather(1, r[:, None])[:, 0] - rf.gather(1, r[:, None])[:, 0]))
                    loss = -F.logsigmoid(z).sum() / B; acc += (z > 0).sum().item(); margin += z.sum().item() / B; top1 += (lp.argmax(1) == g).sum().item()
                else:
                    p = lp.detach().exp(); act = torch.multinomial(p, a.group, replacement=True, generator=gen)   # [n, G]
                    rew = (act == g[:, None]).float(); adv = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True, unbiased=False) + 1e-4)
                    loss = -(adv * lp.gather(1, act)).sum() / (B * a.group)
                    if a.kl > 0:
                        rf = torch.as_tensor(ref[sub], device=lp.device); kl = (lp.exp() * (lp - rf)).sum(1); loss = loss + a.kl * kl.sum() / B; klv += kl.sum().item() / B
                    reward += rew.mean(1).sum().item() / B; nonzero += (rew.std(1, unbiased=False) > 0).sum().item() / B; top1 += (lp.argmax(1) == g).sum().item()
                loss.backward(); rec["loss"] += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            rec.update(top1=top1 / B);
            if a.mode == "dpo": rec.update(pref_acc=acc / B, margin=margin)
            if a.mode == "grpo": rec.update(reward=reward, frac_nonzero_adv=nonzero, kl=klv)
            curve.append(rec)
            if step % 5 == 0 or step == T: print(f"[{exp} set{es} s{a.seed}] step {step}/{T} " + " ".join(f"{k} {v:.4f}" for k, v in rec.items() if k not in ("step", "epoch")) + f" peak_gb {torch.cuda.max_memory_allocated() / 2**30:.1f} {time.time() - t0:.0f}s", flush=True)
    ad = CKPT_DIR / exp / a.model / f"set{es}" / (f"seed{a.seed}" if not a.limit else f"seed{a.seed}_limit"); ad.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in pol.peft.named_parameters() if "lora_" in n}, ad / "adapter.pt")
    if a.limit: return   # a --limit run only leaves a *_limit adapter (for testing the export), no results
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(exp=exp, model=a.model, mode=a.mode, seed=a.seed, essay_set=es, lora=LORA, lr=a.lr, beta=a.beta if a.mode == "dpo" else None, kl=a.kl if a.mode == "grpo" else None, group=a.group if a.mode == "grpo" else None,
                   batch=a.batch, mb=a.mb, epochs=a.epochs, steps=T, n_train=N, text=a.text, curve=curve, train_seconds=time.time() - t0), open(out / "train.json", "w"))
    del pol, opt; torch.cuda.empty_cache()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True, choices=list(EXP)); ap.add_argument("--model", default="d3-mini", choices=list(MODELS)); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11))); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", default="original", help="original, or an experiment id (D2) whose optimized text is read from results/<id>/<model>/seed0/set<k>/prompt.json")
    ap.add_argument("--lr", type=float, default=None); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--group", type=int, default=8); ap.add_argument("--kl", type=float, default=0.0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.lr = a.lr if a.lr is not None else dict(sft=1e-4, dpo=5e-5, grpo=2e-5)[a.mode]; a.epochs = a.epochs if a.epochs is not None else dict(sft=2, dpo=1, grpo=1)[a.mode]
    for es in a.essay_sets: run(es, a)
