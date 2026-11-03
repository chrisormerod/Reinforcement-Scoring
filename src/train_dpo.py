"""DPO: gold score chosen, a uniformly random other score rejected (redrawn every time the example is seen).
mc:  for a softmax policy log pi(w) - log pi(l) = z_w - z_l, so the loss is a logistic loss on logit margins.
reg: Gaussian policy N(mu, sigma^2): margin = beta' (y_w - y_l)(mu - mu_ref) on the standardized scale, beta' = beta / sigma^2."""
import torch
import torch.nn.functional as F
from utils import main

def draw_rejected(y, K, g):  # uniform over the K-1 other scores (deterministic if K = 2)
    return (y + torch.randint(1, K, y.shape, device=y.device, generator=g)) % K

def make_loss(cfg, env):
    beta, K = cfg["dpo_beta"], env.K
    def loss_fn(out, y, idx):
        rej = draw_rejected(y, K, env.gen)
        ref = env.ref_out[idx]
        if env.head == "mc":
            z, zr = F.log_softmax(out, -1), F.log_softmax(ref, -1)
            g = lambda t, a: t.gather(1, a[:, None])[:, 0]
            margin = (g(z, y) - g(z, rej)) - (g(zr, y) - g(zr, rej))
            loss = -F.logsigmoid(beta * margin).mean()
            kl = (z.exp() * (z - zr)).sum(-1).mean()
            return loss, dict(d_pref_acc=(g(out, y) > g(out, rej)).float().mean(), d_margin=(beta * margin).mean(), d_kl_ref=kl,
                              d_top1_acc=(out.argmax(1) == y).float().mean())
        yw, yl = env.zvals[y], env.zvals[rej]
        mu = out[:, 0]; drift = mu - ref[:, 0]
        margin = beta * (yw - yl) * drift
        loss = -F.logsigmoid(margin).mean()
        corr = torch.nan_to_num(torch.corrcoef(torch.stack([mu.detach(), yw]))[0, 1])
        return loss, dict(d_pref_acc=(torch.sign(drift) == torch.sign(yw - yl)).float().mean(), d_margin=margin.mean(),
                          d_drift=drift.mean(), d_abs_drift=drift.abs().mean(), d_corr_gold=corr)
    return loss_fn

if __name__ == "__main__":
    main(make_loss, "dpo")
