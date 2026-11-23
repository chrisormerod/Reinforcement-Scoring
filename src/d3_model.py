"""Trainable view of a vllm-sr d3 model (d3-mini or d3-flash) (see d3_common.py for why it has to be rebuilt). Uses the runtime's own objects: its tokenizer/prompt code (`prepare`), its backbone, readout and temperature;
only the logits step is re-implemented (with gradients, optional per-layer checkpointing, and a LoRA adapter on the text decoder's linear layers).
Run in ~/ml (transformers 5.17.0, peft); the packaged class itself refuses train mode, so nothing here calls D3Model.train."""
import os
os.environ.setdefault("D3_FUSED", "0")   # no fused inference layers in the backbone we train
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from d3_common import MODELS
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05)   # standard LoRA values; target = every linear layer of the text decoder

class D3Policy:
    def __init__(self, model, seed=0, lora=True, device="cuda", grad_ckpt=True):
        from transformers import AutoModel
        torch.manual_seed(seed)
        self.model = AutoModel.from_pretrained(MODELS[model], trust_remote_code=True)
        self.rt = self.model._require()
        assert self.rt.fast is None, f"the fast inference path is active ({self.rt.fast}); set D3_FUSED=0"
        assert self.rt.encoder is not None, "this checkpoint is causal; D3Policy implements the noncausal encoder path"
        self.bb, self.device, self.grad_ckpt, self.T = self.rt.backbone, torch.device(device), grad_ckpt, self.rt.temperature
        self.peft = None
        if lora: self.add_lora()

    def add_lora(self):
        from peft import LoraConfig, get_peft_model
        names = [n for n, m in self.bb.language_model.named_modules() if isinstance(m, nn.Linear)]
        names = [f"language_model.{n}" for n in names]
        # in-place: the adapter layers replace the Linear modules inside the same decoder layers, so the runtime's PromptEncoder (which holds those layers) uses them
        self.peft = get_peft_model(self.bb, LoraConfig(**LORA, target_modules=names))
        for p in self.peft.parameters(): p.requires_grad_(False)
        for n, p in self.peft.named_parameters():
            if "lora_" in n: p.requires_grad_(True); p.data = p.data.float()   # fp32 adapter weights (stable updates); the base stays bf16
        self.params = [p for p in self.peft.parameters() if p.requires_grad]

    def load_adapter(self, path):
        """load adapter weights saved by train_d3.py (adapter.pt)"""
        sd = torch.load(path); own = dict(self.peft.named_parameters())
        assert set(sd) == {n for n in own if "lora_" in n}, "adapter keys do not match this model"
        with torch.no_grad():
            for n, t in sd.items(): own[n].copy_(t)

    def set_adapter_enabled(self, on):
        """the reference model of DPO/GRPO is the base with the adapter off"""
        (self.peft.base_model.enable_adapter_layers if on else self.peft.base_model.disable_adapter_layers)()

    def train_mode(self, on=True): self.bb.train(on); return self

    def prompt_ids(self, state, questions):
        """(token ids, option count) of the served prompt for one request with one score question"""
        p = self.rt.prepare(state, questions); assert not p.errors, p.errors
        key = p.keys[0]; return p.sequences[key], len(p.questions[key].keys)

    def _encode(self, ids, mask):
        enc, bb = self.rt.encoder, self.bb
        embeds = bb.get_input_embeddings()(ids)
        rows, width = mask.shape
        ramp = torch.arange(width, device=ids.device).view(1, 1, -1).expand(4, rows, -1)
        text_positions, mrope = ramp[0], ramp[1:]
        lm = enc.language_model
        rope = lm.rotary_emb(embeds, mrope); keys = mask.bool()[:, None, None, :]
        scan = enc.recurrent_mask(config=lm.config, inputs_embeds=embeds, attention_mask=mask, past_key_values=None, position_ids=text_positions)
        h = embeds
        for layer, softmax in enc.stack:
            f = lambda x, layer=layer, softmax=softmax: layer(x, position_embeddings=rope, attention_mask=keys if softmax else scan, position_ids=text_positions, past_key_values=None, use_cache=False)
            h = checkpoint(f, h, use_reentrant=False) if (self.grad_ckpt and torch.is_grad_enabled()) else f(h)
        return lm.norm(h)

    def log_probs(self, seqs, counts):
        """[B, 255] log-probabilities over the option codes (-inf beyond each row's option count), with gradients when grad mode is on"""
        rt, dev = self.rt, self.device
        width = max(map(len, seqs)); ids = torch.full((len(seqs), width), rt.pad_id, dtype=torch.long); mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs): ids[i, width - len(s):] = torch.as_tensor(s); mask[i, width - len(s):] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        hidden = self._encode(ids, mask)[:, -1]
        logits = hidden.float() @ rt.readout.T
        limit = torch.as_tensor(list(counts), device=dev)[:, None]
        logits = logits.masked_fill(torch.arange(logits.shape[1], device=dev)[None] >= limit, float("-inf"))
        return torch.log_softmax(logits / self.T, dim=-1)
