"""GRPO with a binary exact-match reward, one on-policy update per batch (ratio = 1, no clipping).
mc:  policy = Categorical(softmax(logits)) over the K score points; reward 1 if the sampled score equals the gold score.
reg: policy = Gaussian N(mu_theta(x), sigma^2) on the standardized scale with FIXED sigma (cfg grpo_sigma, an exploration scale that is searched like G and kl_beta).
     A sampled action a is mapped to the nearest observed score (on the standardized scale, no dev-set information) and the reward is 1 if that score equals the gold score.
     log pi(a|x) = -(a - mu)^2 / (2 sigma^2) + const, so the policy-gradient term pushes mu toward sampled actions with positive advantage;
     the KL penalty to the frozen reference is exact, KL(N(mu, s^2) || N(mu_ref, s^2)) = (mu - mu_ref)^2 / (2 s^2)."""
import torch
import torch.nn.functional as F
from utils import main

EPS = 1e-4

def group_advantage(r):
    return (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + EPS)

def grpo_terms(logits, y, G, gen):
    logp = F.log_softmax(logits, -1)
    a = torch.multinomial(logp.detach().exp(), G, replacement=True, generator=gen)       # [B, G]
    r = (a == y[:, None]).float()
    return logp, a, r, group_advantage(r)

def grpo_terms_reg(mu, y, zvals, sigma, G, gen):
    """mu [B] (standardized scale), y [B] score indices, zvals [K] standardized score values. Returns actions a [B, G], their score indices, reward, advantage."""
    a = mu.detach()[:, None] + sigma * torch.randn(mu.shape[0], G, device=mu.device, generator=gen)
    cls = (a[..., None] - zvals).abs().argmin(-1)       # nearest observed score
    r = (cls == y[:, None]).float()
    return a, cls, r, group_advantage(r)

def make_loss(cfg, env):
    G, beta = cfg["G"], cfg["kl_beta"]
    if env.head == "reg":
        sigma = cfg["grpo_sigma"]
        def loss_fn(out, y, idx):
            mu = out[:, 0]
            a, cls, r, adv = grpo_terms_reg(mu, y, env.zvals, sigma, G, env.gen)
            logp = -(a - mu[:, None]) ** 2 / (2 * sigma ** 2)
            pg = -(adv.detach() * logp).mean()
            drift = mu - env.ref_out[idx][:, 0]
            kl = (drift ** 2 / (2 * sigma ** 2)).mean()
            corr = torch.nan_to_num(torch.corrcoef(torch.stack([mu.detach(), env.zvals[y]]))[0, 1])
            d = dict(d_reward=r.mean(), d_nondeg=(r.std(1) > 0).float().mean(), d_drift=drift.mean(), d_abs_drift=drift.abs().mean(), d_kl_ref=kl, d_corr_gold=corr)
            return pg + beta * kl, d
        return loss_fn
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
