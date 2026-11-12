"""G6 stage 1c: pool the resampled replies (results/G6/collect_t<T>/) into the original replies (results/G6/collect/), per training example.
usage: python src/gen_merge_collect.py [--essay_sets 1 2 ...] [--resample_dir collect_t1] [--out collect_merged]
Writes results/G6/<out>/set<k>.jsonl, one line per training example in the original order, same format as gen_collect.py: replies = original replies followed by the resampled ones
(each reply from the resample is tagged with its temperature). The input files are not modified. Prints, per set, how many examples give a DPO pair before and after pooling.
Then train with: python src/gen_dpo.py --collect_dir collect_merged"""
import argparse, json
from gen_common import ROOT
from gen_resample import gives_pair

def run(es, a):
    base = ROOT / "results" / "G6"
    rows = [json.loads(l) for l in open(base / "collect" / f"set{es}.jsonl")]
    new = {}
    for l in open(base / a.resample_dir / f"set{es}.jsonl"):
        y = json.loads(l); new.setdefault(y["i"], []).extend(dict(r, temperature=y.get("temperature")) for r in y["replies"])
    before = sum(gives_pair(x) for x in rows)
    merged = [dict(x, replies=x["replies"] + new.get(x["i"], [])) for x in rows]
    after = sum(gives_pair(x) for x in merged)
    out = base / a.out; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"set{es}.jsonl", "w") as f:
        for x in merged: f.write(json.dumps(x) + "\n")
    print(f"[merge set{es}] {len(rows)} examples, {len(new)} resampled; with a pair: {before} -> {after} ({after / len(rows):.1%}); still without: {len(rows) - after}", flush=True)
    return len(rows), before, after

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--resample_dir", default="collect_t1"); ap.add_argument("--out", default="collect_merged")
    a = ap.parse_args()
    t = [run(es, a) for es in a.essay_sets]
    print(f"[merge] total {sum(x[0] for x in t)} examples; with a pair {sum(x[1] for x in t)} -> {sum(x[2] for x in t)}")
