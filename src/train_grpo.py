"""GRPO over the K score points (multiple-target head), binary exact-match reward, one on-policy update per batch (ratio = 1, no clipping)."""
import torch
import torch.nn.functional as F
from utils import main

EPS = 1e-4

def grpo_terms(logits, y, G, gen):
    logp = F.log_softmax(logits, -1)
    a = torch.multinomial(logp.detach().exp(), G, replacement=True, generator=gen)       # [B, G]
    r = (a == y[:, None]).float()
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + EPS)
    return logp, a, r, adv

def make_loss(cfg, env):
    G, beta = cfg["G"], cfg["kl_beta"]
    def loss_fn(logits, y, idx):
        logp, a, r, adv = grpo_terms(logits, y, G, env.gen)
        pg = -(adv.detach() * logp.gather(1, a)).mean()
        p = logp.exp()
        kl = (p * (logp - F.log_softmax(env.ref_out[idx], -1))).sum(-1).mean()          # exact KL(p_theta || p_ref)
        ent = -(p * logp).sum(-1).mean()
        d = dict(d_reward=r.mean(), d_nondeg=(r.std(1) > 0).float().mean(), d_entropy=ent, d_kl_ref=kl)
        return pg + beta * kl, d
    return loss_fn

if __name__ == "__main__":
    main(make_loss, "grpo")
