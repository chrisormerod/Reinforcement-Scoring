"""Unit tests: GRPO reward/advantage/degenerate groups, DPO pair construction and margin identities (mc and reg),
SFT losses, metrics and threshold fitting. Run: python -I src/test_losses.py (or pytest src)"""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from types import SimpleNamespace
import numpy as np, torch, torch.nn.functional as F
import eval as ev, train_dpo, train_grpo, train_sft

K = 4
def gen(s=0):
    g = torch.Generator(); g.manual_seed(s); return g

def env(head, ref, N, zvals=None, seed=5):
    return SimpleNamespace(K=K, head=head, ref_out=ref, N=N, gen=gen(seed), zvals=torch.tensor([-1.2, -0.4, 0.4, 1.2]) if zvals is None else zvals)

# ---------------------------------------------------------------- GRPO
def test_grpo_degenerate_groups_have_zero_advantage():
    logits = torch.tensor([[50., 0, 0, 0]] * 3)           # near-deterministic on class 0
    _, a, r, adv = train_grpo.grpo_terms(logits, torch.tensor([0, 0, 1]), 8, gen())
    assert (r[:2] == 1).all() and (r[2] == 0).all()        # all right / all right / all wrong
    assert adv.abs().max() < 1e-3

def test_grpo_advantage_zero_mean_and_reward_binary():
    logits = torch.zeros(5, K)
    _, a, r, adv = train_grpo.grpo_terms(logits, torch.zeros(5, dtype=torch.long), 16, gen(1))
    assert set(r.unique().tolist()) <= {0., 1.}
    assert adv.mean(1).abs().max() < 1e-5

def test_grpo_expected_gradient_matches_p_times_grad_logp():
    # E[grad J] = grad p_y = p_y * grad log p_y (checked by exact enumeration with unnormalised advantage r)
    z = torch.randn(1, K, requires_grad=True); y = 2
    p = F.softmax(z, -1)
    p[0, y].backward(); g1 = z.grad.clone(); z.grad = None
    (p[0, y].detach() * F.log_softmax(z, -1)[0, y]).backward()
    assert torch.allclose(g1, z.grad, atol=1e-6)

def test_grpo_loss_diagnostics_and_zero_kl_at_reference():
    z = torch.randn(16, K); y = torch.randint(0, K, (16,))
    loss, d = train_grpo.make_loss(dict(G=8, kl_beta=0.1), env("mc", z, 16))(z, y, torch.arange(16))
    assert abs(d["d_kl_ref"].item()) < 1e-6 and 0 <= d["d_nondeg"].item() <= 1

# ---------------------------------------------------------------- GRPO, single target (Gaussian policy)
def test_grpo_reg_reward_is_exact_match_of_nearest_score():
    zv = torch.tensor([-1.2, -0.4, 0.4, 1.2]); y = torch.tensor([0, 3, 1])
    mu = torch.tensor([-1.2, 1.2, 1.2])                    # on the gold score / on the gold score / on a wrong score
    a, cls, r, adv = train_grpo.grpo_terms_reg(mu, y, zv, 1e-3, 8, gen())
    assert (r[0] == 1).all() and (r[1] == 1).all() and (r[2] == 0).all() and adv.abs().max() < 1e-3   # tiny sigma: every group is degenerate
    assert (cls[2] == 3).all() and set(r.unique().tolist()) <= {0., 1.}

def test_grpo_reg_zero_kl_at_reference_and_kl_formula():
    N = 16; mu = torch.randn(N, 1); y = torch.randint(0, K, (N,))
    cfg = dict(G=8, kl_beta=0.1, grpo_sigma=0.5)
    _, d = train_grpo.make_loss(cfg, env("reg", mu.clone(), N))(mu, y, torch.arange(N))
    assert abs(d["d_kl_ref"].item()) < 1e-7 and abs(d["d_drift"].item()) < 1e-7
    _, d = train_grpo.make_loss(cfg, env("reg", mu.clone(), N))(mu + 0.3, y, torch.arange(N))
    assert abs(d["d_kl_ref"].item() - 0.3 ** 2 / (2 * 0.5 ** 2)) < 1e-6      # KL(N(m, s^2) || N(m0, s^2)) = (m - m0)^2 / (2 s^2)

def test_grpo_reg_gradient_raises_mu_when_gold_is_above_and_lowers_it_when_below():
    # the group is only informative when some samples are right: sigma wide enough to reach the gold bin from mu
    for gold, sign in ((3, -1), (0, +1)):                  # loss gradient < 0 means descent raises mu
        mu = torch.zeros(64, 1, requires_grad=True); y = torch.full((64,), gold)
        train_grpo.make_loss(dict(G=64, kl_beta=0.0, grpo_sigma=0.8), env("reg", torch.zeros(64, 1), 64))(mu, y, torch.arange(64))[0].backward()
        assert sign * mu.grad.sum().item() > 0

# ---------------------------------------------------------------- DPO, multiple targets
def test_dpo_rejected_never_gold_and_uniform():
    y = torch.zeros(20000, dtype=torch.long)
    rej = train_dpo.draw_rejected(y, K, gen(3))
    assert (rej != y).all()
    freq = torch.bincount(rej, minlength=K)[1:].float() / len(y)
    assert (freq - 1 / (K - 1)).abs().max() < 0.02

def test_dpo_k2_rejected_is_deterministic():
    y = torch.randint(0, 2, (64,))
    assert (train_dpo.draw_rejected(y, 2, gen()) == 1 - y).all()

def test_dpo_mc_margin_is_logit_difference():
    z = torch.randn(64, K); y = torch.randint(0, K, (64,)); l = (y + 1) % K
    lp = F.log_softmax(z, -1)
    assert torch.allclose(lp.gather(1, y[:, None]) - lp.gather(1, l[:, None]), z.gather(1, y[:, None]) - z.gather(1, l[:, None]), atol=1e-5)

def test_dpo_mc_zero_start_is_log2_and_pref_acc_in_range():
    z = torch.randn(32, K); y = torch.randint(0, K, (32,))
    loss, d = train_dpo.make_loss(dict(dpo_beta=0.1), env("mc", z, 32))(z, y, torch.arange(32))   # reference == policy
    assert abs(loss.item() - np.log(2)) < 1e-5 and 0 <= d["d_pref_acc"].item() <= 1

def test_ce_dominates_single_pair_loss():
    # CE = log(1 + sum_r exp(z_r - z_y)) >= log(1 + exp(z_r - z_y)) for any single r
    z = torch.randn(64, K); y = torch.randint(0, K, (64,))
    ce = F.cross_entropy(z, y, reduction="none")
    r = train_dpo.draw_rejected(y, K, gen(2))
    pair = -F.logsigmoid(z.gather(1, y[:, None])[:, 0] - z.gather(1, r[:, None])[:, 0])
    assert (ce >= pair - 1e-6).all()

# ---------------------------------------------------------------- DPO, single target
def test_dpo_reg_margin_matches_gaussian_policy_log_ratio():
    # log N(y_w; mu, s^2) - log N(y_l; mu, s^2) - (same at mu_ref) == (y_w - y_l)(mu - mu_ref) / s^2
    yw, yl, mu, mr, s = torch.randn(5).double(), torch.randn(5).double(), torch.randn(5).double(), torch.randn(5).double(), 0.7
    lp = lambda y, m: torch.distributions.Normal(m, s).log_prob(y)
    lhs = (lp(yw, mu) - lp(yl, mu)) - (lp(yw, mr) - lp(yl, mr))
    assert torch.allclose(lhs, (yw - yl) * (mu - mr) / s ** 2)

def test_dpo_reg_loss_matches_formula_zero_start_log2_and_diagnostics():
    N = 32; y = torch.randint(0, K, (N,)); mu = torch.randn(N, 1)
    e = env("reg", mu.clone(), N)
    loss, d = train_dpo.make_loss(dict(dpo_beta=2.0), e)(mu, y, torch.arange(N))
    assert abs(loss.item() - np.log(2)) < 1e-5 and abs(d["d_drift"].item()) < 1e-7        # mu == mu_ref -> margin 0
    mu2 = mu + 0.3
    e = env("reg", mu.clone(), N, seed=9)
    loss, d = train_dpo.make_loss(dict(dpo_beta=2.0), e)(mu2, y, torch.arange(N))
    rej = train_dpo.draw_rejected(y, K, gen(9))
    expect = -F.logsigmoid(2.0 * (e.zvals[y] - e.zvals[rej]) * 0.3).mean()
    assert torch.allclose(loss, expect, atol=1e-5) and abs(d["d_drift"].item() - 0.3) < 1e-6

def test_dpo_reg_gradient_pushes_mu_up_for_higher_gold_and_down_for_lower():
    mu = torch.zeros(2, 1, requires_grad=True); y = torch.tensor([3, 0])        # gold highest / lowest -> rejected is always lower / higher
    e = env("reg", torch.zeros(2, 1), 2)
    train_dpo.make_loss(dict(dpo_beta=1.0), e)(mu, y, torch.arange(2))[0].backward()
    assert mu.grad[0, 0] < 0 and mu.grad[1, 0] > 0                                # descent raises mu for gold=3, lowers it for gold=0

# ---------------------------------------------------------------- SFT
def test_sft_losses():
    z = torch.randn(8, K); y = torch.randint(0, K, (8,))
    assert torch.allclose(train_sft.make_loss({}, env("mc", z, 8))(z, y, None)[0], F.cross_entropy(z, y))
    mu = torch.randn(8, 1); e = env("reg", mu, 8)
    assert torch.allclose(train_sft.make_loss({}, e)(mu, y, None)[0], ((mu[:, 0] - e.zvals[y]) ** 2).mean())

# ---------------------------------------------------------------- metrics
def test_metrics():
    y = [0, 1, 2, 3, 3]; assert ev.qwk(y, y, [0, 1, 2, 3]) == 1.0
    assert ev.all_metrics(y, y, [0, 1, 2, 3])["jeffreys"] == 0
    m = ev.all_metrics([0, 0, 1, 1], [1, 1, 1, 1], [0, 1])
    assert m["smd"] > 0 and m["jeffreys"] > 0 and m["acc"] == 0.5

def test_qwk_matches_sklearn_and_uses_original_values():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0); y = rng.integers(0, 4, 300); p = np.clip(y + rng.integers(-1, 2, 300), 0, 3)
    assert abs(ev.qwk(y, p, [0, 1, 2, 3]) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9
    # non-contiguous scores 0,1,5: a 1<->5 confusion costs more than a 0<->1 confusion
    yy = np.array([0, 1, 2, 0, 1, 2])
    assert ev.qwk(yy, np.array([1, 0, 2, 0, 1, 2]), [0, 1, 5]) > ev.qwk(yy, np.array([0, 2, 1, 0, 1, 2]), [0, 1, 5])

def test_smd_sign():
    assert ev.smd(np.array([0, 1, 2]), np.array([1, 2, 3])) > 0 and ev.smd(np.array([1, 2, 3]), np.array([0, 1, 2])) < 0

def test_kl_finite_when_class_never_predicted():
    assert np.isfinite(ev.all_metrics([0, 1, 2, 2], [0, 0, 0, 0], [0, 1, 2])["jeffreys"])

def test_threshold_fit_recovers_known_cuts_and_is_monotone():
    rng = np.random.default_rng(1); scores = [0, 1, 2]; y = rng.integers(0, 3, 600)
    mu = np.array([0.0, 2.0, 3.5])[y] + rng.normal(0, 0.2, 600)           # unanchored scale: cuts nowhere near midpoints of 0,1,2
    th = ev.fit_thresholds(mu, y, scores)
    assert (np.diff(th) >= 0).all() and len(th) == 2
    q_fit = ev.qwk(y, ev.apply_thresholds(mu, th), scores)
    q_round = ev.qwk(y, ev.round_to_scores(mu, scores), scores)
    assert q_fit > 0.9 and q_fit > q_round

def test_round_to_scores():
    assert ev.round_to_scores(np.array([-3., 0.4, 0.6, 2.4, 9.]), [0, 1, 5]).tolist() == [0, 0, 1, 1, 2]

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
