"""G5: SFT of Qwen3.5-4B (bf16 + LoRA, local GPU) on the optimized no-thinking rubric (G3) with the standard reply `Answer: {score}`.
usage: python src/gen_sft.py [--essay_sets 1 2 ...] [--seed 0] [--lr 2e-4] [--epochs 2] [--batch 16] [--micro 4] [--limit N]
One model per EssaySet. Training chats: system = G3's rubric + answer_suffix, user = the response, assistant = `Answer: {score}` (loss on the assistant tokens only).
Thinking is off in training and inference; inference is greedy with 64 new tokens (as G3). The checkpoint with the best dev QWK is kept; the test split is read once, at the end.
Writes results/G5/default/seed<s>/set<k>/metrics.json and the adapter under $RF_CKPT_DIR/G5. --limit runs write nothing."""
import argparse, json, math, random, time
import numpy as np, torch
from gen_common import ROOT, MODEL, scores_of, load_split
from gen_local import *
from eval import bootstrap_ci
from transformers import get_cosine_schedule_with_warmup

DECODING = dict(temperature=0.0, max_new_tokens=64)

def encode(tok, rubric, suffix, texts, y, max_len):
    """(input ids, labels) with the prompt masked; chats longer than max_len are dropped and counted"""
    out, dropped = [], 0
    for t, s in zip(texts, y):
        p = tok(prompt_text(tok, rubric, suffix, t, False), add_special_tokens=False).input_ids
        a = tok(f"Answer: {s}<|im_end|>", add_special_tokens=False).input_ids
        if len(p) + len(a) > max_len: dropped += 1; continue
        out.append((p + a, [-100] * len(p) + a))
    return out, dropped

def collate(batch, pad_id):
    L = max(len(x) for x, _ in batch)
    ids = torch.tensor([x + [pad_id] * (L - len(x)) for x, _ in batch]); lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in batch])
    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch])   # not ids != pad_id: the pad token may also be a real token (eos)
    return ids.cuda(), att.cuda(), lab.cuda()

def answer_loss(model, ids, att, lab):
    """mean cross-entropy over the labelled (answer) tokens. The output head is applied to those positions only: the library loss builds
    (batch x length x 248k vocabulary) logits for every position, which ran out of memory on long rubric prompts"""
    h = model.get_base_model().model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]; keep = tgt != -100
    return torch.nn.functional.cross_entropy(model.get_base_model().lm_head(h[keep]).float(), tgt[keep])

@torch.no_grad()
def generate(model, tok, prompts, bs=16):
    """greedy replies in input order (left-padded batches, longest prompts first)"""
    model.eval(); model.config.use_cache = True; tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i])); out = [None] * len(prompts)
    for b in range(0, len(order), bs):
        ix = order[b:b + bs]
        enc = tok([prompts[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=DECODING["max_new_tokens"], do_sample=False, pad_token_id=tok.pad_token_id)
        for i, g in zip(ix, gen[:, enc.input_ids.shape[1]:]): out[i] = tok.decode(g, skip_special_tokens=True)
    model.config.use_cache = False; model.train()
    return out

def evaluate(model, tok, rubric, suffix, split, es, scores, limit, with_per_class=False):
    texts, y = load_split(split, es)
    if limit: texts, y = texts[:limit], y[:limit]
    return score_replies(generate(model, tok, [prompt_text(tok, rubric, suffix, t, False) for t in texts]), y, scores, with_per_class)

def run(es, a):
    out = ROOT / "results" / "G5" / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "metrics.json").exists() and not a.limit: print("done already:", out); return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    scores, rubric = scores_of(es), saved_rubric("G3", es)
    suffix = answer_suffix(scores)
    tok = load_tokenizer()
    texts, y = load_split("train", es)
    if a.limit: texts, y = texts[:a.limit], y[:a.limit]
    data, dropped = encode(tok, rubric, suffix, texts, y, a.max_len)
    model = load_lora_model(a.seed); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    spe = len(data) // a.batch; T = a.max_steps or a.epochs * spe; every = a.eval_every or max(1, spe // 2)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * T)), T)
    accum = a.batch // a.micro
    gen = np.random.default_rng(a.seed); perm = []
    best, best_state, curve, t0 = (-2.0, 0), None, [], time.time()
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    for step in range(1, T + 1):
        if len(perm) < a.batch: perm = list(gen.permutation(len(data))) + perm   # next epoch's order
        ix, perm = perm[-a.batch:], perm[:-a.batch]
        loss_sum = 0.0
        for m in range(accum):
            ids, att, lab = collate([data[i] for i in ix[m * a.micro:(m + 1) * a.micro]], tok.pad_token_id)
            loss = answer_loss(model, ids, att, lab) / accum
            loss.backward(); loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % every == 0 or step == T:
            d = evaluate(model, tok, rubric, suffix, "dev", es, scores, a.limit)
            q = d["metrics"]["qwk"]; curve.append(dict(step=step, loss=loss_sum, dev_qwk=q, dev_fail_rate=d["fail_rate"]))
            print(f"[G5 set{es} s{a.seed}] step {step}/{T} loss {loss_sum:.4f} dev_qwk {q:.4f} fail {d['fail_rate']:.3f} peak_gpu_gb {torch.cuda.max_memory_allocated() / 2**30:.1f} {time.time() - t0:.0f}s", flush=True)
            if q > best[0]: best = (q, step); best_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    set_peft_model_state_dict(model, best_state)   # the dev-selected checkpoint; the test split is read only now
    dev, test = (evaluate(model, tok, rubric, suffix, s, es, scores, a.limit, s == "test") for s in ("dev", "test"))
    res = dict(exp="G5", seed=a.seed, essay_set=es, head="gen", thinking=False, model=MODEL, scores=scores, decoding=DECODING, rubric_from="G3",
               lora=LORA, lr=a.lr, epochs=a.epochs, batch=a.batch, micro_batch=a.micro, steps=T, best_step=best[1], n_train=len(data), n_train_dropped_too_long=dropped,
               dev=dev["metrics"], test=test["metrics"], test_ci=bootstrap_ci(test["y_idx"], test["p_idx"], scores, n=300, seed=0),
               dev_fail_rate=dev["fail_rate"], test_fail_rate=test["fail_rate"], n_dev=len(dev["y_idx"]), n_test=len(test["y_idx"]), curve=curve,
               train_seconds=time.time() - t0)
    print(f"[G5 set{es} s{a.seed}] DONE best step {best[1]} dev_qwk {dev['metrics']['qwk']:.4f} test_qwk {test['metrics']['qwk']:.4f} acc {test['metrics']['acc']:.3f} "
          f"fail {test['fail_rate']:.3f}", flush=True)
    if a.limit: return
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR / "G5" / "default" / f"set{es}" / f"seed{a.seed}")
    np.savez_compressed(out / "preds.npz", dev_pred=np.array([-1 if p is None else p for p in dev["parsed"]]), test_pred=np.array([-1 if p is None else p for p in test["parsed"]]))
    json.dump(res, open(out / "metrics.json", "w"))
    del model, opt; torch.cuda.empty_cache()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11))); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4); ap.add_argument("--max_len", type=int, default=4096); ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); assert a.batch % a.micro == 0
    for es in a.essay_sets: run(es, a)
