"""G9: GRPO of Qwen3.5-4B (bf16 + LoRA, local GPU, thinking on) with a reward function only: a per-completion quadratic score-distance reward.
usage: python src/gen_grpo.py [--essay_sets 1 ...] [--seed 0] [--lr 2e-5] [--steps 100] [--prompts_per_step 8] [--num_generations 8]
                              [--max_completion 8192] [--fail_reward -0.5] [--gen_batch 8] [--limit N]   (--limit N: N training prompts, 2 steps, nothing written)
WHY THIS REWARD. QWK needs the whole sample (observed vs expected disagreement), so it cannot be a per-completion reward and a group-sized batch is a poor estimate of it.
QWK = 1 - sum(w*O)/sum(w*E) with quadratic weights w = (i-j)^2/(K-1)^2, so its numerator is the summed squared score error over the sample. The reward is that error for ONE
completion: r = 1 - ((pred - gold)/(max - min))^2, in [0, 1] (1 = exact, 0 = the largest possible miss). It equals the agreement_metric that MIPROv2 used for G3/G4.
A reply that never closes its thinking, has no integer after </think>, or gives a score outside the set gets --fail_reward (< 0, so worse than the largest miss).
Strictness differs from gen_local.parse_reply on purpose: a truncated reply without </think> must NOT be scored on an integer taken from inside its thinking.
Group-relative advantages (scale_rewards=group) make the reward's scale irrelevant; only the ordering within a group matters.
Monitoring only (never rewarded): every generation batch appends mean reward, exact-match rate, fail rate, mean length and the batch QWK to <out>/batches.jsonl.
Setup = G4's: the set's optimized rubric + suffix, thinking on, temperature 0.6, top-p 0.95. Start point = the base model (as G6). No KL term (beta=0), TRL's DAPO loss, one update per generation (on-policy).
Native PyTorch: no TRL, no vLLM. Rollouts come from the policy itself (HF generate on the LoRA model, so they are always on the current weights), then one
optimizer step per generation. Loss = DAPO's: sum over all completion tokens of -A * ratio (ratio == 1 in value, on-policy), divided by the batch's total completion tokens.
Each sequence is its own forward/backward pass (8k-token replies); the lm_head + log-softmax is chunked and checkpointed so full-vocab logits are never held.
Logged per step in log_history: loss-free stats (reward, reward_std, frac_zero_std_groups, completion length, grad_norm, lr).
Writes the adapter to $RF_CKPT_DIR/G9/default/set<k>/seed<s> and results/G9/default/seed<s>/set<k>/train.json; evaluate with src/gen_dpo_eval.py --exp G9."""
import argparse, json, math, re, time
import numpy as np
from gen_common import ROOT, format_suffix, scores_of, load_split
from gen_local import *

def completion_text(c):
    return c

def make_reward(scores, fail_reward, log_file):
    r2 = float(max(scores) - min(scores)) ** 2
    def parse_strict(text):
        if "</think>" not in text: return None   # truncated (or never closed) thinking
        m = INT_RE.search(text.rsplit("</think>", 1)[1])
        return int(m.group()) if m and int(m.group()) in scores else None
    def reward(prompts, completions, lengths, gold):
        """one float per completion; called once per generation batch (prompts_per_step x num_generations). prompts: user texts."""
        pred = [parse_strict(c) for c in completions]
        out = [fail_reward if p is None else 1.0 - (p - g) ** 2 / r2 for p, g in zip(pred, gold)]
        ok = [i for i, p in enumerate(pred) if p is not None]
        qwk = None
        if len(ok) > 1 and len({gold[i] for i in ok}) > 1:   # monitor only; undefined when the batch has a single gold class
            from sklearn.metrics import cohen_kappa_score
            qwk = float(cohen_kappa_score([gold[i] for i in ok], [pred[i] for i in ok], labels=scores, weights="quadratic"))
        rec = dict(n=len(out), mean_reward=float(np.mean(out)), exact=float(np.mean([p == g for p, g in zip(pred, gold)])), fail_rate=float(np.mean([p is None for p in pred])),
                   mean_len=float(np.mean(lengths)), batch_qwk=qwk)
        with open(log_file, "a") as f: f.write(json.dumps(rec) + "\n")
        if not log_file.with_name("samples_first.jsonl").exists():   # debug aid: the first batch's raw (prompt, gold, parsed, completion) triples, to check alignment and that thinking is on
            with open(log_file.with_name("samples_first.jsonl"), "w") as f:
                for pr, t, g, pd_ in zip(prompts, completions, gold, pred):
                    f.write(json.dumps(dict(user_tail=pr[-80:], gold=g, pred=pd_, n_chars=len(t), head=t[:150], tail=t[-150:])) + "\n")
        return out
    return reward

def eos_ids(model, tok):
    e = model.generation_config.eos_token_id
    e = [e] if isinstance(e, int) else list(e or [])
    return set(e + [tok.eos_token_id])

@__import__("torch").no_grad()
def rollout(model, tok, prompt_ids, n, a, eos):
    """sample n completions for one prompt (list of ids). Returns a list of completion id lists, each cut after its first EOS."""
    import torch
    model.eval()
    outs = []
    for i in range(0, n, a.gen_batch):
        k = min(a.gen_batch, n - i)
        x = torch.tensor([prompt_ids] * k, device="cuda")
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=DECODING_T, top_p=DECODING_P, top_k=0,
                           max_new_tokens=a.max_completion, use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=list(eos))
        for row in g[:, len(prompt_ids):].tolist():
            cut = next((j + 1 for j, t in enumerate(row) if t in eos), len(row))
            outs.append(row[:cut])
    model.train()
    return outs

def completion_logps(model, prompt_ids, comp_ids):
    """log pi(comp | prompt) per completion token at temperature DECODING_T; the lm_head is applied in checkpointed chunks so the full-vocab logits never coexist"""
    import torch
    from torch.utils.checkpoint import checkpoint
    inner = model.get_base_model()
    ids = torch.tensor([prompt_ids + comp_ids], device="cuda")
    h = inner.model(input_ids=ids).last_hidden_state[0, len(prompt_ids) - 1:-1]   # positions that predict the completion tokens
    tgt = torch.tensor(comp_ids, device="cuda")
    def chunk(hc, tc):
        lp = torch.log_softmax(inner.lm_head(hc).float() / DECODING_T, dim=-1)
        return lp.gather(-1, tc[:, None])[:, 0]
    return torch.cat([checkpoint(chunk, h[i:i + 1024], tgt[i:i + 1024], use_reentrant=False) for i in range(0, len(comp_ids), 1024)])

def run(es, a):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    out = ROOT / "results" / "G9" / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "train.json").exists() and not a.limit: print("done already:", out); return
    scores, rubric = scores_of(es), saved_rubric("G4", es)
    texts, y = load_split("train", es)
    if a.limit: texts, y = texts[:a.limit], y[:a.limit]
    work = CKPT_DIR / "G9" / "default" / f"set{es}" / f"seed{a.seed}"; work.mkdir(parents=True, exist_ok=True)
    out_log = work / "batches.jsonl"
    if out_log.exists(): out_log.unlink()
    gen_batch = a.prompts_per_step * a.num_generations
    steps = 2 if a.limit else a.steps
    tok = load_tokenizer()
    model = load_lora_model(a.seed); model.train()
    eos = eos_ids(model, tok)
    suffix = format_suffix(scores)
    prompts = [prompt_text(tok, rubric, suffix, t, True) for t in texts]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.03 * steps), steps)
    reward_fn = make_reward(scores, a.fail_reward, out_log)
    rng = np.random.default_rng(a.seed); order = []
    log_history = []; t0 = time.time()
    for step in range(1, steps + 1):
        if len(order) < a.prompts_per_step: order += rng.permutation(len(prompts)).tolist()
        idx, order = order[:a.prompts_per_step], order[a.prompts_per_step:]
        P, C, G = [], [], []   # per completion: prompt ids, completion ids, gold
        for i in idx:
            pid = tok(prompts[i], add_special_tokens=False)["input_ids"]
            for c in rollout(model, tok, pid, a.num_generations, a, eos): P.append(pid); C.append(c); G.append(int(y[i]))
        comps = [tok.decode(c, skip_special_tokens=True) for c in C]
        r = np.array(reward_fn([prompts[i][-80:] for i in idx for _ in range(a.num_generations)], comps, [len(c) for c in C], G), dtype=np.float64)
        rg = r.reshape(len(idx), a.num_generations)
        adv = ((rg - rg.mean(1, keepdims=True)) / (rg.std(1, keepdims=True) + 1e-4)).reshape(-1)   # scale_rewards="group"
        total_tokens = sum(len(c) for c in C)
        opt.zero_grad(set_to_none=True)
        for pid, c, A in zip(P, C, adv):
            if abs(A) < 1e-8 or not c: continue   # zero advantage -> zero gradient
            lp = completion_logps(model, pid, c)
            loss = -(float(A) * torch.exp(lp - lp.detach())).sum() / total_tokens   # on-policy: ratio is 1, gradient is -A * grad logp
            loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step(); sched.step()
        rec = dict(step=step, reward=float(r.mean()), reward_std=float(rg.std(1).mean()), frac_zero_std_groups=float((rg.std(1) < 1e-8).mean()),
                   mean_len=float(np.mean([len(c) for c in C])), grad_norm=gn, lr=sched.get_last_lr()[0], elapsed=time.time() - t0)
        log_history.append(rec); print(json.dumps(rec), flush=True)
        if not a.limit and a.save_steps and step % a.save_steps == 0 and step < steps: model.save_pretrained(work / f"checkpoint-{step}")
        torch.cuda.empty_cache()
    secs = time.time() - t0
    print(f"[G9 set{es} s{a.seed}] trained {steps} steps in {secs:.0f}s", flush=True)
    if a.limit: return
    model.save_pretrained(work)   # the adapter, in the layout merge_adapter / gen_dpo_eval expect
    batches = [json.loads(l) for l in open(out_log)]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(exp="G9", seed=a.seed, essay_set=es, lora=LORA, lr=a.lr, beta=0.0, batch=gen_batch, steps=steps, train_seconds=secs, engine="native-pytorch",
                   pairs=dict(note="GRPO: no pairs; reward statistics instead", n_train_prompts=len(texts), prompts_per_step=a.prompts_per_step, num_generations=a.num_generations,
                              max_completion=a.max_completion, fail_reward=a.fail_reward, reward="1-((pred-gold)/(max-min))^2",
                              first10_mean_reward=float(np.mean([b["mean_reward"] for b in batches[:10]])), last10_mean_reward=float(np.mean([b["mean_reward"] for b in batches[-10:]])),
                              first10_fail_rate=float(np.mean([b["fail_rate"] for b in batches[:10]])), last10_fail_rate=float(np.mean([b["fail_rate"] for b in batches[-10:]]))),
                   batches=batches, log_history=log_history), open(out / "train.json", "w"))

DECODING_T, DECODING_P = 0.6, 0.95   # G4's decoding

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=[1]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--prompts_per_step", type=int, default=8); ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion", type=int, default=8192); ap.add_argument("--fail_reward", type=float, default=-0.5)
    ap.add_argument("--gen_batch", type=int, default=8, help="completions sampled in parallel per generate call"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
