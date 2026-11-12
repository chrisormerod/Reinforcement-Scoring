"""G6 stage 1b: resample ONLY the training examples that gave no DPO pair in results/G6/collect/set<k>.jsonl, at a different temperature.
An example gives a pair when it has at least one parsable right reply and one parsable wrong reply (src/gen_dpo.py build_pairs). Examples with only right, only wrong or only
unparsable replies are resampled here with --n new replies each, with G4's prompt and decoding except for the temperature (top-p 0.95 and 16384 max tokens unchanged).
usage: VLLM_API_BASE=http://10.151.66.140:8000/v1 python src/gen_resample.py --essay_sets 1 2 ... [--temperature 1.0] [--n 10] [--threads 16] [--limit N]
Writes results/G6/collect_t<temperature>/set<k>.jsonl, only the resampled examples, same line format as gen_collect.py plus `temperature`; the original files are not modified.
Examples already in the output are skipped, so an interrupted run resumes. The training split only; the test split is never sampled.
This does NOT change DPO training: src/gen_dpo.py still reads only results/G6/collect/. Merging the new replies into the pair pool is a separate step."""
import argparse, json, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from gen_common import ROOT, MODEL, DECODING, API_BASE, format_suffix, scores_of, load_split, check_server
from gen_local import saved_rubric, parse_reply

def sample(messages, n, temperature):
    body = dict(model=MODEL, messages=messages, n=n, chat_template_kwargs={"enable_thinking": True}, **{**DECODING[True], "temperature": temperature})
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{API_BASE}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r: return json.load(r)["choices"]
        except Exception as e:
            if attempt == 2: raise
            time.sleep(10 * (attempt + 1))

def gives_pair(x):
    return any(r["score"] is not None and r["score"] == x["gold"] for r in x["replies"]) and any(r["score"] is not None and r["score"] != x["gold"] for r in x["replies"])

def run(es, a):
    check_server()
    src = ROOT / "results" / "G6" / "collect" / f"set{es}.jsonl"
    out = ROOT / "results" / "G6" / f"collect_t{a.temperature:g}"; out.mkdir(parents=True, exist_ok=True)
    f = out / (f"set{es}.jsonl" if not a.limit else f"set{es}.limit.jsonl")
    scores, rubric = scores_of(es), saved_rubric("G4", es)
    texts, _ = load_split("train", es)
    rows = [json.loads(l) for l in open(src)]
    need = [x for x in rows if not gives_pair(x)]
    done = {json.loads(l)["i"] for l in open(f)} if f.exists() else set()
    todo = [x for x in need if x["i"] not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"[G6 resample set{es}] {len(rows)} collected, {len(need)} without a pair, {len(done)} resampled already, {len(todo)} to do, T={a.temperature:g}, server {API_BASE}", flush=True)
    lock, fh, t0, n_done = threading.Lock(), open(f, "a"), time.time(), [0]
    def one(x):
        i = x["i"]
        msgs = [{"role": "system", "content": f"{rubric}\n\n{format_suffix(scores)}"}, {"role": "user", "content": texts[i]}]
        reps = []
        for c in sample(msgs, a.n, a.temperature):
            text, finish = c["message"]["content"] or "", c["finish_reason"]
            s = None if finish == "length" else parse_reply(text)
            reps.append(dict(text=text, finish=finish, score=s if s in scores else None))
        with lock:
            fh.write(json.dumps(dict(i=i, gold=x["gold"], replies=reps, temperature=a.temperature, api_base=API_BASE)) + "\n"); fh.flush(); n_done[0] += 1
            if n_done[0] % 25 == 0: print(f"[G6 resample set{es}] {n_done[0]}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(one, todo))
    new = {}
    for l in open(f): y = json.loads(l); new[y["i"]] = y
    n_pairs = sum(gives_pair(dict(gold=y["gold"], replies=y["replies"] + next(x["replies"] for x in rows if x["i"] == y["i"]))) for y in new.values()) if new else 0
    n_alone = sum(gives_pair(y) for y in new.values())
    allr = [r for y in new.values() for r in y["replies"]]
    print(f"[G6 resample set{es}] DONE {len(new)} examples, {len(allr)} replies: {sum(r['score'] is not None for r in allr) / max(1, len(allr)):.3f} parsable; "
          f"{n_alone} give a pair from the new replies alone, {n_pairs} give one when pooled with the original replies", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
