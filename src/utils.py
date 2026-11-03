"""Shared pieces: config, data, the single training loop used by all arms, evaluation, logging.
One model per EssaySet. head = mc (K logits) or reg (one scalar, standardized targets)."""
import argparse, json, os, random, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd, torch, yaml
from transformers import AutoTokenizer

from eval import all_metrics, bootstrap_ci, fit_thresholds, apply_thresholds, round_to_scores
from model import build_model

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = Path(os.environ.get("RF_CKPT_DIR", Path.home() / "rf_ckpt"))
DEFAULTS = dict(arm="sft", head="mc", lr=2e-5, batch_size=8, epochs=10, max_steps=None, warmup_frac=0.06, weight_decay=0.01,
                eval_every=50, patience=None, grad_clip=1.0, warm_from=None, warm_tag=None, save_ckpt=False,
                G=8, kl_beta=0.0, dpo_beta=0.1)   # dpo_beta is beta for mc and beta' = beta / sigma^2 for reg

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_cfg(path, overrides=()):
    cfg = {**DEFAULTS, **yaml.safe_load(open(path))}
    for o in overrides:
        k, v = o.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):  # YAML 1.1 reads "5e-06" (no dot) as a string
            try: val = float(val)
            except ValueError: pass
        cfg[k] = val
    return cfg

# ---------------------------------------------------------------- data
ESSAY_SETS = list(range(1, 11))

def load_data(essay_set):
    """One essay set. K and the label mapping are the distinct scores observed in that set's training split.
    Regression targets are standardized with the training mean / sd of this set (`zvals[k]` = standardized score of class k)."""
    d = ROOT / "data" / "splits"
    meta = json.load(open(d / "meta.json"))
    frames = {n: pd.read_csv(d / f"{n}.csv", keep_default_na=False) for n in ("train", "dev", "test")}
    frames = {n: df[df.essay_set == essay_set] for n, df in frames.items()}
    scores = sorted(frames["train"].score.unique().tolist())
    mapping = {s: i for i, s in enumerate(scores)}
    assert all(set(df.score) <= set(scores) for df in frames.values()), f"set {essay_set}: dev/test score not seen in train"
    mean, sd = float(frames["train"].score.mean()), float(frames["train"].score.std(ddof=0))
    tok = AutoTokenizer.from_pretrained(meta["model"])
    out = {}
    for n, df in frames.items():
        enc = tok(df.text.tolist(), truncation=True, max_length=meta["max_len"])["input_ids"]
        out[n] = dict(ids=enc, y=np.array([mapping[s] for s in df.score]))
    st = SimpleNamespace(scores=np.array(scores, dtype=float), mean=mean, sd=sd, zvals=torch.tensor((np.array(scores) - mean) / sd, dtype=torch.float32))
    return out, st, meta, tok.pad_token_id

def collate(data, idx, pad_id, device):
    seqs = [data["ids"][i] for i in idx]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s); mask[j, :len(s)] = 1
    return ids.to(device), mask.to(device), torch.tensor(data["y"][idx]).to(device), torch.tensor(idx).to(device)

@torch.no_grad()
def predict_out(model, data, pad_id, device, bs=128):
    """Raw head outputs, [N, K] logits (mc) or [N, 1] standardized mu (reg)."""
    was = model.training; model.eval()
    order = np.argsort([len(x) for x in data["ids"]])
    out = np.zeros((len(order), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        ids, mask, _, _ = collate(data, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[idx] = model(input_ids=ids, attention_mask=mask).logits.float().cpu().numpy()
    model.train(was)
    return out

def decode(head, out, st, y=None, th=None):
    """Head outputs -> class indices. mc: argmax. reg: mu on the original scale, cut by thresholds (fit on `y` if th is None).
    Returns (pred_idx, thresholds, rounded_idx); the last two are None for mc."""
    if head == "mc":
        return out.argmax(1), None, None
    mu = out[:, 0] * st.sd + st.mean
    if th is None: th = fit_thresholds(mu, y, st.scores)
    return apply_thresholds(mu, th), th, round_to_scores(mu, st.scores)

# ---------------------------------------------------------------- training
def ckpt_path(exp, tag, essay_set, seed):
    return CKPT_DIR / exp / (tag or "default") / f"set{essay_set}" / f"seed{seed}.pt"

def run(cfg, make_loss, exp, seed, tag=None, essay_set=1):
    """make_loss(cfg, env) -> loss_fn(out, y, idx) -> (loss, diag dict). env has K, head, ref_out, N, gen, zvals."""
    device = "cuda"
    head = cfg["head"]
    out_dir = ROOT / "results" / exp / (tag or "default") / f"seed{seed}" / f"set{essay_set}"
    if (out_dir / "metrics.json").exists():
        print("done already:", out_dir); return
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    data, st, meta, pad_id = load_data(essay_set)
    scores = st.scores.astype(int).tolist()
    K, N = len(scores), len(data["train"]["y"])
    spe = N // cfg["batch_size"]   # steps per epoch (drop-last batches)
    T = cfg["max_steps"] or cfg["epochs"] * spe
    name = f"{exp}/{tag} s{seed} set{essay_set}"
    model = build_model(meta["model"], K, seed, head).to(device)
    stage1 = None
    if cfg["warm_from"]:
        model.load_state_dict(torch.load(ckpt_path(cfg["warm_from"], cfg["warm_tag"], essay_set, seed), map_location=device))
        s1 = json.load(open(ROOT / "results" / cfg["warm_from"] / (cfg["warm_tag"] or "default") / f"seed{seed}" / f"set{essay_set}" / "metrics.json"))
        stage1 = dict(exp=cfg["warm_from"], steps=s1["steps"], examples_seen=s1["examples_seen"], train_time=s1["total_train_time"])
    # frozen reference = starting policy; its outputs are precomputed once per training example
    ref_out = torch.tensor(predict_out(model, data["train"], pad_id, device)).to(device)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    env = SimpleNamespace(K=K, head=head, ref_out=ref_out, N=N, gen=gen, zvals=st.zvals.to(device))
    loss_fn = make_loss(cfg, env)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)
    W = int(cfg["warmup_frac"] * T)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / W if s < W else max(0.0, (T - s) / (T - W)))

    rng = np.random.default_rng(seed)
    def batches():
        while True:
            perm = rng.permutation(N)
            for i in range(0, N - cfg["batch_size"] + 1, cfg["batch_size"]):
                yield perm[i:i + cfg["batch_size"]]
    stream = batches()

    def evaluate(step, train_time):
        out = predict_out(model, data["dev"], pad_id, device)
        p, _, _ = decode(head, out, st, data["dev"]["y"])   # reg: thresholds fit on dev, then QWK of the thresholded predictions
        m = all_metrics(data["dev"]["y"], p, scores)
        return dict(step=step, examples=step * cfg["batch_size"], train_time=train_time, **{f"dev_{k}": v for k, v in m.items()})

    curve, diag_acc = [], {}
    best = dict(qwk=-2, step=0, state=None); bad = 0; train_time = 0.0
    model.train()
    for step in range(0, T + 1):
        if step % cfg["eval_every"] == 0 or step == T:
            ev = evaluate(step, train_time)
            for k, v in diag_acc.items():
                ev[k] = float(np.mean(v))
                if k == "d_gradnorm": ev["d_gradnorm_std"] = float(np.std(v)); ev["d_gradnorm_max"] = float(np.max(v))
            diag_acc = {}
            curve.append(ev)
            print(f"[{name}] step {step} dev_qwk {ev['dev_qwk']:.4f} acc {ev['dev_acc']:.3f}"
                  + "".join(f" {k} {ev[k]:.3f}" for k in ev if k.startswith("d_") and not k.startswith("d_gradnorm_")), flush=True)
            if ev["dev_qwk"] > best["qwk"]:
                best = dict(qwk=ev["dev_qwk"], step=step, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); bad = 0
            else:
                bad += 1
                if cfg["patience"] and bad >= cfg["patience"]:
                    print("early stop at", step); break
        if step == T: break
        torch.cuda.synchronize(); t0 = time.time()
        idx = next(stream)
        ids, mask, y, gidx = collate(data["train"], idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=mask).logits
        loss, d = loss_fn(out.float(), y, gidx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step(); sched.step()
        torch.cuda.synchronize(); train_time += time.time() - t0
        for k, v in {"d_loss": loss, "d_gradnorm": gn, **d}.items():
            diag_acc.setdefault(k, []).append(float(v))
    steps_run = step

    model.load_state_dict(best["state"])
    if cfg["save_ckpt"]:
        p = ckpt_path(exp, tag, essay_set, seed); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best["state"], p)
    o_dev, o_test = predict_out(model, data["dev"], pad_id, device), predict_out(model, data["test"], pad_id, device)
    yd, yt = data["dev"]["y"], data["test"]["y"]
    pd_, th, rd = decode(head, o_dev, st, yd)
    pt, _, rt = decode(head, o_test, st, th=th)
    res = dict(exp=exp, tag=tag, seed=seed, essay_set=essay_set, head=head, n_train=N, steps=T, steps_run=steps_run, cfg=cfg, scores=scores,
               best_step=best["step"], best_examples=best["step"] * cfg["batch_size"],
               examples_seen=steps_run * cfg["batch_size"], stage1=stage1,
               train_time_to_best=next(c["train_time"] for c in curve if c["step"] == best["step"]), total_train_time=train_time,
               dev=all_metrics(yd, pd_, scores), test=all_metrics(yt, pt, scores, with_per_class=True),
               test_ci=bootstrap_ci(yt, pt, scores, n=300, seed=seed), curve=curve)
    if head == "reg":   # secondary decoding: round to the nearest observed score
        res.update(thresholds=th.tolist(), dev_round=all_metrics(yd, rd, scores), test_round=all_metrics(yt, rt, scores, with_per_class=True))
    np.savez_compressed(out_dir / "preds.npz", dev_out=o_dev, test_out=o_test)
    json.dump(res, open(out_dir / "metrics.json", "w"))
    print(f"[{name}] DONE best step {best['step']} dev_qwk {best['qwk']:.4f} test_qwk {res['test']['qwk']:.4f} acc {res['test']['acc']:.4f}")

def main(make_loss, arm):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None); ap.add_argument("--exp", default=None); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--essay_sets", nargs="*", type=int, default=ESSAY_SETS)
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)
    assert cfg["arm"] == arm, f"config arm {cfg['arm']} != {arm}"
    if arm == "grpo": assert cfg["head"] == "mc", "GRPO is defined for the multiple-target head only"
    for es in a.essay_sets:   # one model per essay set; finished sets are skipped, so a crashed job resumes
        run(cfg, make_loss, a.exp or Path(a.config).stem, a.seed, a.tag, es)
