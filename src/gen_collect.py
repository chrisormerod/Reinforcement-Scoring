"""G6 stage 1: sample 10 thinking replies per TRAINING example from a vLLM server, with G4's prompt (G4's optimized rubric + the fixed suffix) and decoding.
usage: VLLM_API_BASE=http://10.151.66.140:8000/v1 python src/gen_collect.py --essay_sets 1 3 5 7 9 [--n 10] [--threads 16] [--limit N]
Run one process per server, with the sets split between the servers (odd sets on .140, even sets on .182 as for G2/G4).
Writes results/G6/collect/set<k>.jsonl, one line per example: {i, gold, replies: [{text, finish, score}]}. `score` is the parsed integer, or null when the reply
is truncated, has no integer, or is outside the set's score points. Examples already in the file are skipped, so an interrupted run resumes. The test split is never sampled."""
import argparse, json, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from gen_common import ROOT, MODEL, DECODING, API_BASE, format_suffix, scores_of, load_split, check_server
from gen_local import saved_rubric, parse_reply

def sample(messages, n):
    body = dict(model=MODEL, messages=messages, n=n, chat_template_kwargs={"enable_thinking": True}, **DECODING[True])
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{API_BASE}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r: return json.load(r)["choices"]
        except Exception as e:
            if attempt == 2: raise
            time.sleep(10 * (attempt + 1))

def run(es, a):
    check_server()
    out = ROOT / "results" / "G6" / "collect"; out.mkdir(parents=True, exist_ok=True)
    f = out / (f"set{es}.jsonl" if not a.limit else f"set{es}.limit.jsonl")
    scores, rubric = scores_of(es), saved_rubric("G4", es)
    texts, y = load_split("train", es)
    if a.limit: texts, y = texts[:a.limit], y[:a.limit]
    done = {json.loads(l)["i"] for l in open(f)} if f.exists() else set()
    todo = [i for i in range(len(texts)) if i not in done]
    print(f"[G6 collect set{es}] {len(done)} done, {len(todo)} to do, server {API_BASE}", flush=True)
    lock, fh, t0, n_done = threading.Lock(), open(f, "a"), time.time(), [0]
    def one(i):
        msgs = [{"role": "system", "content": f"{rubric}\n\n{format_suffix(scores)}"}, {"role": "user", "content": texts[i]}]
        reps = []
        for c in sample(msgs, a.n):
            text, finish = c["message"]["content"] or "", c["finish_reason"]
            s = None if finish == "length" else parse_reply(text)   # a truncated reply has no answer
            reps.append(dict(text=text, finish=finish, score=s if s in scores else None))
        with lock:
            fh.write(json.dumps(dict(i=i, gold=y[i], replies=reps, api_base=API_BASE)) + "\n"); fh.flush(); n_done[0] += 1
            if n_done[0] % 25 == 0: print(f"[G6 collect set{es}] {n_done[0]}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(one, todo))
    rows = [json.loads(l) for l in open(f)]
    parsable = [r["score"] for x in rows for r in x["replies"] if r["score"] is not None]
    right = sum(r["score"] == x["gold"] for x in rows for r in x["replies"])
    n_all = sum(len(x["replies"]) for x in rows)
    print(f"[G6 collect set{es}] DONE {len(rows)} examples, {n_all} replies: {right / n_all:.3f} right, {len(parsable) / n_all:.3f} parsable", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
