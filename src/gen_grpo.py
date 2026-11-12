"""G9: GRPO of Qwen3.5-4B (bf16 + LoRA, local GPU, thinking on) with a reward function only: a per-completion quadratic score-distance reward.
usage: python src/gen_grpo.py [--essay_sets 1 ...] [--seed 0] [--lr 2e-5] [--steps 100] [--prompts_per_step 8] [--num_generations 8]
                              [--max_completion 8192] [--fail_reward -0.5] [--limit N]   (--limit N: N training prompts, 2 steps, nothing written)
WHY THIS REWARD. QWK needs the whole sample (observed vs expected disagreement), so it cannot be a per-completion reward and a group-sized batch is a poor estimate of it.
QWK = 1 - sum(w*O)/sum(w*E) with quadratic weights w = (i-j)^2/(K-1)^2, so its numerator is the summed squared score error over the sample. The reward is that error for ONE
completion: r = 1 - ((pred - gold)/(max - min))^2, in [0, 1] (1 = exact, 0 = the largest possible miss). It equals the agreement_metric that MIPROv2 used for G3/G4.
A reply that never closes its thinking, has no integer after </think>, or gives a score outside the set gets --fail_reward (< 0, so worse than the largest miss).
Strictness differs from gen_local.parse_reply on purpose: a truncated reply without </think> must NOT be scored on an integer taken from inside its thinking.
Group-relative advantages (scale_rewards=group) make the reward's scale irrelevant; only the ordering within a group matters.
Monitoring only (never rewarded): every generation batch appends mean reward, exact-match rate, fail rate, mean length and the batch QWK to <out>/batches.jsonl.
Setup = G4's: the set's optimized rubric + suffix, thinking on, temperature 0.6, top-p 0.95. Start point = the base model (as G6). No KL term (beta=0), TRL's DAPO loss, one update per generation (on-policy).
Generation is vLLM in colocated mode with sleep mode (the 24 GB card cannot hold the policy, a vLLM copy and a 16k KV cache at once); TRL pushes the MERGED LoRA weights into vLLM
at every step, so vLLM's inability to apply LoRARequest for this architecture (see gen_local.merge_adapter) does not matter here - but check `sampling/sampling_logp_difference`
in the log: it should stay small (~0.01-0.05). A large, growing value means vLLM is sampling from stale weights.
Writes the adapter to $RF_CKPT_DIR/G9/default/set<k>/seed<s> and results/G9/default/seed<s>/set<k>/train.json; evaluate with src/gen_dpo_eval.py --exp G9."""
import argparse, json, re, time
import numpy as np
from gen_common import ROOT, format_suffix, scores_of, load_split
from gen_local import *

def completion_text(c):
    return c[-1]["content"] if isinstance(c, list) else c

def make_reward(scores, fail_reward, log_file):
    r2 = float(max(scores) - min(scores)) ** 2
    def parse_strict(text):
        if "</think>" not in text: return None   # truncated (or never closed) thinking
        m = INT_RE.search(text.rsplit("</think>", 1)[1])
        return int(m.group()) if m and int(m.group()) in scores else None
    def reward(prompts, completions, completion_ids=None, gold=None, **kw):
        """one float per completion; TRL passes the whole generation batch (prompts_per_step x num_generations) in one call"""
        pred = [parse_strict(completion_text(c)) for c in completions]
        out = [fail_reward if p is None else 1.0 - (p - g) ** 2 / r2 for p, g in zip(pred, gold)]
        ok = [i for i, p in enumerate(pred) if p is not None]
        qwk = None
        if len(ok) > 1 and len({gold[i] for i in ok}) > 1:   # monitor only; undefined when the batch has a single gold class
            from sklearn.metrics import cohen_kappa_score
            qwk = float(cohen_kappa_score([gold[i] for i in ok], [pred[i] for i in ok], labels=scores, weights="quadratic"))
        rec = dict(n=len(out), mean_reward=float(np.mean(out)), exact=float(np.mean([p == g for p, g in zip(pred, gold)])), fail_rate=float(np.mean([p is None for p in pred])),
                   mean_len=float(np.mean([len(x) for x in completion_ids])) if completion_ids else None, batch_qwk=qwk)
        with open(log_file, "a") as f: f.write(json.dumps(rec) + "\n")
        return out
    return reward

def run(es, a):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    out = ROOT / "results" / "G9" / "default" / f"seed{a.seed}" / f"set{es}"
    if (out / "train.json").exists() and not a.limit: print("done already:", out); return
    scores, rubric = scores_of(es), saved_rubric("G4", es)
    texts, y = load_split("train", es)
    if a.limit: texts, y = texts[:a.limit], y[:a.limit]
    ds = Dataset.from_list([dict(prompt=[{"role": "system", "content": f"{rubric}\n\n{format_suffix(scores)}"}, {"role": "user", "content": t}], gold=int(g)) for t, g in zip(texts, y)])
    work = CKPT_DIR / "G9" / "default" / f"set{es}" / f"seed{a.seed}"; work.mkdir(parents=True, exist_ok=True)
    out_log = work / "batches.jsonl"
    if out_log.exists(): out_log.unlink()
    gen_batch = a.prompts_per_step * a.num_generations
    steps = 2 if a.limit else a.steps
    cfg = GRPOConfig(
        output_dir=str(work / "trainer"), seed=a.seed, bf16=True, report_to="none", logging_steps=1, save_strategy="steps" if not a.limit else "no", save_steps=a.save_steps, save_only_model=True,
        max_steps=steps, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03, max_grad_norm=1.0, weight_decay=0.0, gradient_checkpointing=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=gen_batch,   # one sequence per backward pass: 8k-token replies; generation batch = per_device * accum
        num_generations=a.num_generations, max_completion_length=a.max_completion, temperature=DECODING_T, top_p=DECODING_P, top_k=0,
        chat_template_kwargs={"enable_thinking": True}, beta=0.0, loss_type="dapo", scale_rewards="group", epsilon=0.2, mask_truncated_completions=False,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem, vllm_enable_sleep_mode=True, vllm_max_model_length=a.max_completion + 4096,
        model_init_kwargs=dict(dtype=torch.bfloat16), log_completions=False)
    trainer = GRPOTrainer(model=MODEL, reward_funcs=[make_reward(scores, a.fail_reward, out_log)], args=cfg, train_dataset=ds,
                          peft_config=LoraConfig(**LORA, task_type="CAUSAL_LM"))
    t0 = time.time(); trainer.train(); secs = time.time() - t0
    print(f"[G9 set{es} s{a.seed}] trained {steps} steps in {secs:.0f}s", flush=True)
    if a.limit: return
    trainer.model.save_pretrained(work)   # the adapter, in the layout merge_adapter / gen_dpo_eval expect
    batches = [json.loads(l) for l in open(out_log)]
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(exp="G9", seed=a.seed, essay_set=es, lora=LORA, lr=a.lr, beta=0.0, batch=gen_batch, steps=steps, train_seconds=secs,
                   pairs=dict(note="GRPO: no pairs; reward statistics instead", n_train_prompts=len(texts), prompts_per_step=a.prompts_per_step, num_generations=a.num_generations,
                              max_completion=a.max_completion, fail_reward=a.fail_reward, reward="1-((pred-gold)/(max-min))^2",
                              first10_mean_reward=float(np.mean([b["mean_reward"] for b in batches[:10]])), last10_mean_reward=float(np.mean([b["mean_reward"] for b in batches[-10:]])),
                              first10_fail_rate=float(np.mean([b["fail_rate"] for b in batches[:10]])), last10_fail_rate=float(np.mean([b["fail_rate"] for b in batches[-10:]]))),
                   batches=batches, log_history=trainer.state.log_history), open(out / "train.json", "w"))

DECODING_T, DECODING_P = 0.6, 0.95   # G4's decoding

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=[1]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--prompts_per_step", type=int, default=8); ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion", type=int, default=8192); ap.add_argument("--fail_reward", type=float, default=-0.5)
    ap.add_argument("--vllm_mem", type=float, default=0.45); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
