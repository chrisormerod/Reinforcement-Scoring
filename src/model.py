"""ModernBERT + head. mc: K logits (classification). reg: one scalar output (regression).
Seeding immediately before construction fixes the head init per seed."""
import torch
from transformers import AutoModelForSequenceClassification

def build_model(name, K, seed, head="mc"):
    torch.manual_seed(seed)
    n_out = K if head == "mc" else 1
    return AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_out, attn_implementation="sdpa")
