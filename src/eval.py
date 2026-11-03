"""Metrics: QWK, SMD, accuracy, KL / Jeffreys divergence between true and predicted score distributions.
Regression decoding: monotone thresholds fit on dev to maximise QWK, and nearest-observed-score rounding."""
import numpy as np

ALPHA = 0.5  # additive smoothing per class, fixed across all arms and sets

def qwk(y, p, scores):
    """y, p are class indices; scores maps index -> original score value, so ordinal distance uses the original scale."""
    y, p, s = np.asarray(y), np.asarray(p), np.asarray(scores, dtype=float)
    K = len(s)
    O = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = (s[:, None] - s[None, :]) ** 2
    num, den = (W * O).sum(), (W * E).sum()
    if den == 0:                       # one marginal is constant: perfect only if both are the same constant
        return 1.0 if num == 0 else 0.0
    return float(1 - num / den)

def smd(y_vals, p_vals):
    d = np.sqrt((np.var(p_vals) + np.var(y_vals)) / 2)
    return float((np.mean(p_vals) - np.mean(y_vals)) / d) if d > 0 else 0.0

def kl_jeffreys(y, p, K, alpha=ALPHA):
    P = np.bincount(y, minlength=K) + alpha
    Q = np.bincount(p, minlength=K) + alpha
    P, Q = P / P.sum(), Q / Q.sum()
    kl_pq, kl_qp = float(np.sum(P * np.log(P / Q))), float(np.sum(Q * np.log(Q / P)))
    return kl_pq, kl_qp, kl_pq + kl_qp

def all_metrics(y_idx, p_idx, scores, with_per_class=False):
    """y_idx / p_idx are class indices; `scores` maps index -> original score value."""
    y_idx, p_idx, scores = np.asarray(y_idx), np.asarray(p_idx), np.asarray(scores)
    K = len(scores)
    kl_pq, kl_qp, jeff = kl_jeffreys(y_idx, p_idx, K)
    m = {"qwk": qwk(y_idx, p_idx, scores), "smd": smd(scores[y_idx], scores[p_idx]), "acc": float((y_idx == p_idx).mean()),
         "kl_true_pred": kl_pq, "kl_pred_true": kl_qp, "jeffreys": jeff}
    if with_per_class:
        m["true_counts"] = np.bincount(y_idx, minlength=K).tolist()
        m["pred_counts"] = np.bincount(p_idx, minlength=K).tolist()
        m["confusion"] = np.bincount(y_idx * K + p_idx, minlength=K * K).reshape(K, K).tolist()
    return m

def bootstrap_ci(y_idx, p_idx, scores, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_idx, p_idx = np.asarray(y_idx), np.asarray(p_idx)
    runs = [all_metrics(y_idx[i], p_idx[i], scores) for i in (rng.integers(0, len(y_idx), len(y_idx)) for _ in range(n))]
    return {k: [float(np.percentile([r[k] for r in runs], q)) for q in (2.5, 97.5)] for k in runs[0]}

# ---------------------------------------------------------------- regression decoding
def apply_thresholds(mu, th):
    """mu on the original score scale -> class index; th are the K-1 ascending cut points."""
    return np.searchsorted(th, mu, side="right")

def round_to_scores(mu, scores):
    return np.abs(np.asarray(mu)[:, None] - np.asarray(scores, dtype=float)[None, :]).argmin(1)

def fit_thresholds(mu, y_idx, scores, passes=3, grid=100):
    """Monotone thresholds maximising QWK on (mu, y): coordinate ascent over a quantile grid, starting from midpoints
    between observed scores. Identical procedure for every regression arm, fit separately for each EssaySet."""
    s = np.asarray(scores, dtype=float)
    th = (s[:-1] + s[1:]) / 2
    cand = np.unique(np.quantile(mu, np.linspace(0, 1, grid)))
    best = qwk(y_idx, apply_thresholds(mu, th), s)
    for _ in range(passes):
        changed = False
        for j in range(len(th)):
            lo = th[j - 1] if j > 0 else -np.inf
            hi = th[j + 1] if j < len(th) - 1 else np.inf
            for c in cand[(cand >= lo) & (cand <= hi)]:
                t = th.copy(); t[j] = c
                q = qwk(y_idx, apply_thresholds(mu, t), s)
                if q > best + 1e-12: best, th, changed = q, t, True
        if not changed: break
    return th
