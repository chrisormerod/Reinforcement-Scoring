"""Shared pieces for the locally trained generative arms (G5 SFT, G6 DPO): prompt text, model loading, LoRA, reply parsing and scoring.
Kept apart from gen_common.py (which the served-model runs G1-G4 import) so that nothing here can affect those runs."""
import os, re
from pathlib import Path
import numpy as np
from gen_common import ROOT, MODEL, scores_of, load_split, to_index
from eval import all_metrics

CKPT_DIR = Path(os.environ.get("RF_CKPT_DIR", Path.home() / "rf_ckpt"))
INT_RE = re.compile(r"-?\d+")
# standard LoRA values (PROJECT.md section 6.7); "all-linear" = every linear layer of the decoder except the output head
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")

def answer_suffix(scores):   # G5 only: the reply format the model is trained to produce
    return f'Reply in the form "Answer: <score>", where <score> is one of {", ".join(map(str, scores))}.'

def saved_rubric(exp, es):
    """the optimized rubric stored by gen_mipro.py (the original rubric on fallback sets)"""
    f = ROOT / "results" / exp / "default" / "seed0" / f"set{es}" / "rubric.md"
    assert f.exists(), f"{f} is missing: run {exp} on set {es} first"
    return f.read_text()

def parse_reply(text):
    """same rule as MinimalAdapter.parse: first integer after the last </think>; None for a reply that never closed its thinking"""
    if "</think>" in text: text = text.rsplit("</think>", 1)[1]
    elif text.lstrip().startswith("Thinking"): return None
    m = INT_RE.search(text)
    return int(m.group()) if m else None

def prompt_text(tok, rubric, suffix, response, thinking):
    """chat prompt up to the start of the assistant reply: system = rubric + suffix, user = response (as MinimalAdapter)"""
    msgs = [{"role": "system", "content": f"{rubric}\n\n{suffix}"}, {"role": "user", "content": response}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)

def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return tok

def load_lora_model(seed):
    """base model in bf16 on the GPU with a fresh LoRA adapter and gradient checkpointing"""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    return get_peft_model(model, LoraConfig(**LORA, task_type="CAUSAL_LM"))

def score_replies(replies, y, scores, with_per_class=False):
    """metrics for raw reply texts (or None for a failed call); an unparseable or out-of-range answer is scored as the lowest score point
    and counted in fail_rate, exactly as gen_zeroshot.score_split"""
    parsed = [None if r is None else parse_reply(r) for r in replies]
    idx = [to_index(p, scores) if p is not None else None for p in parsed]
    failed = [i is None for i in idx]
    p_idx = np.array([0 if f else i for f, i in zip(failed, idx)])
    y_idx = np.array([scores.index(v) for v in y])
    return dict(metrics=all_metrics(y_idx, p_idx, scores, with_per_class=with_per_class), fail_rate=float(np.mean(failed)), y_idx=y_idx, p_idx=p_idx, parsed=parsed)
