"""Verify in plain PyTorch (no vLLM) that a merged LoRA adapter reproduces the adapter's behaviour and differs from the base model.
usage: python src/gen_merge_check.py ADAPTER_DIR [--essay_set 1] [--n 4] [--tokens 300]
For n collected replies (results/G6/collect/set<k>.jsonl) it prints the summed log-probability of the first --tokens completion tokens under
(1) the base model, (2) base + adapter (PEFT), (3) the merged model reloaded from disk (merge_adapter). Expected: (2) ~ (3) (bf16 rounding only) and both differ from (1).
Merges the adapter if <ADAPTER_DIR>_merged does not exist. Uses the GPU for the forward passes, one model at a time."""
import argparse, json, gc, torch
from gen_common import ROOT, MODEL, format_suffix, scores_of, load_split
from gen_local import saved_rubric, prompt_text, load_tokenizer, merge_adapter
from transformers import AutoModelForCausalLM
from peft import PeftModel

@torch.no_grad()
def logps(model, items):
    out = []
    for p, c in items:
        ids = torch.tensor([p + c], device="cuda")
        lg = model(input_ids=ids).logits[0, len(p) - 1:-1].float().log_softmax(-1)
        out.append(lg.gather(1, torch.tensor(c, device="cuda")[:, None]).sum().item())
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("adapter"); ap.add_argument("--essay_set", type=int, default=1)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--tokens", type=int, default=300)
    a = ap.parse_args()
    tok, scores, rubric = load_tokenizer(), scores_of(a.essay_set), saved_rubric("G4", a.essay_set)
    texts, _ = load_split("train", a.essay_set)
    rows = [json.loads(l) for _, l in zip(range(a.n), open(ROOT / "results" / "G6" / "collect" / f"set{a.essay_set}.jsonl"))]
    items = []
    for x in rows:
        p = tok(prompt_text(tok, rubric, format_suffix(scores), texts[x["i"]], True), add_special_tokens=False).input_ids
        c = tok(x["replies"][0]["text"], add_special_tokens=False).input_ids[:a.tokens]
        items.append((p, c))
    merged = merge_adapter(a.adapter); print("merged model:", merged, flush=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    lp_base = logps(base, items)
    peft = PeftModel.from_pretrained(base, a.adapter).eval()
    lp_peft = logps(peft, items)
    del peft, base; gc.collect(); torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16).to("cuda").eval()
    lp_merged = logps(m, items)
    print("| reply | base | base + adapter (PEFT) | merged (from disk) | PEFT - base | merged - PEFT |\n|---|---|---|---|---|---|")
    for i, (b, p, g) in enumerate(zip(lp_base, lp_peft, lp_merged)): print(f"| {i} | {b:.3f} | {p:.3f} | {g:.3f} | {p - b:+.3f} | {g - p:+.3f} |")
