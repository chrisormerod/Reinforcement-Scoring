"""Merge a train_d3.py LoRA adapter into a servable copy of a vllm-sr d3 model (--model d3-mini|d3-flash).
usage: python src/d3_export.py --exp D3|D4|D5 --essay_set K [--seed 0] [--limit_adapter]   (run in ~/ml; CPU/GPU light)
Writes <adapter dir>_merged/: every file of the package is symlinked EXCEPT the backbone weight shards and their index, which are rewritten with W' = W + (alpha/r) * B @ A for every adapted linear layer
(computed in fp32, stored bf16, same tensor names and shard layout as the original; readout.safetensors and decision_config.json are untouched). MODEL_MANIFEST.json still lists the ORIGINAL hashes, so a
runtime that verifies the package will report the modified shards; vLLM serving is checked by src/d3_eval.py. Delete the merged folder to force a re-merge."""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from d3_common import CKPT_DIR, MODELS
from d3_model import LORA

def snapshot(model):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODELS[model]))

def merge(adapter_dir, model):
    adapter_dir = Path(adapter_dir); out = Path(str(adapter_dir) + "_merged")
    if (out / "model.safetensors.index.json").exists(): return out
    src = snapshot(model); out.mkdir(parents=True, exist_ok=True)
    ad = torch.load(adapter_dir / "adapter.pt"); scale = LORA["lora_alpha"] / LORA["r"]
    pairs = {}
    for n, t in ad.items():   # base_model.model.<module path>.lora_A.default.weight
        base, ab = n.removeprefix("base_model.model.").split(".lora_")[0], n.split(".lora_")[1][0]
        pairs.setdefault(base, {})[ab] = t.float()
    index = json.load(open(src / "model.safetensors.index.json")); used = 0
    for shard in sorted(set(index["weight_map"].values())):
        w = load_file(str(src / shard))
        for name in list(w):
            mod = name.removesuffix(".weight")
            if name.endswith(".weight") and mod in pairs:
                d = scale * (pairs[mod]["B"] @ pairs[mod]["A"])
                assert d.shape == w[name].shape, (name, d.shape, w[name].shape)
                w[name] = (w[name].float() + d).to(w[name].dtype); used += 1
        save_file(w, str(out / shard), metadata={"format": "pt"})
    assert used == len(pairs), f"{used} of {len(pairs)} adapted modules found in the checkpoint"
    shutil.copy(src / "model.safetensors.index.json", out / "model.safetensors.index.json")
    for f in src.iterdir():
        if f.name.startswith("model-") or f.name == "model.safetensors.index.json" or (out / f.name).exists(): continue
        (out / f.name).symlink_to(f.resolve())
    print(f"merged {used} modules into {out}"); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--model", default="d3-mini", choices=list(MODELS)); ap.add_argument("--essay_set", type=int, required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit_adapter", action="store_true")
    a = ap.parse_args(); merge(CKPT_DIR / a.exp / a.model / f"set{a.essay_set}" / (f"seed{a.seed}" + ("_limit" if a.limit_adapter else "")), a.model)
