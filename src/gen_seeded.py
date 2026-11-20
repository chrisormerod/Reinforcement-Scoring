"""G7/G8 data: score-seeded rationales for every TRAINING example, written by Qwen3.5-4B served by vLLM (use the .182 server).
usage: VLLM_API_BASE=http://10.151.66.182:8000/v1 python src/gen_seeded.py --essay_sets 1 2 ... [--threads 32] [--limit N] [--seed 0]
For each training example the generator sees G4's rubric, the student response AND the answer, and writes the reasoning of a scorer who arrives at that answer (thinking mode off; the text it writes is
used as the THINKING block of a thinking-mode reply, so SFT/DPO can teach a model that sees only rubric + response to produce it). Two variants per example:
  right: the gold score.          wrong: a score exactly one score point off (an adjacent value of the set's score list; the direction is drawn at random (seeded), only one neighbour exists at the ends).
A rationale is accepted only if the generation finished, has at least 40 words, ends by concluding the seeded score (last "score N" statement) and does not leak the seeding (told, given, provided, hint, ...).
Up to --tries rounds of --n samples are drawn for a variant; the first accepted one is kept (failures are recorded with null text).
Writes results/G7/seeded/set<k>.jsonl: {i, gold, wrong_score, right: {text, tries}, wrong: {text, tries}}, where text is the full reply "<rationale>\\n</think>\\n\\n<score>" in the format of the collected replies
(the prompt ends with the opened <think>). Examples already in the file are skipped, so an interrupted run resumes. The test split is never touched."""
import argparse, json, random, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from gen_common import ROOT, MODEL, API_BASE, scores_of, load_split, check_server
from gen_local import saved_rubric

LEAK = re.compile(r"(?i)\b(was|been|being|were|is|are)\s+(told|given|provided|stated|specified|determined|known|assigned|seeded)\b|\bgiven (score|answer)\b|\bprovided (score|answer)\b|\b(target|correct|true|gold|expected|pre-?determined) (score|answer)\b"
                  r"|\bhint\b|\binstruction|\bsystem (prompt|message)|\bthe prompt (says|states|tells)|\bI am (told|asked to (justify|explain|arrive))|\bjustif(y|ies|ication of) (the|this) (score|answer)\b|\bwork(ing)? backwards?|\bin advance\b")
CONCLUDES = re.compile(r"(?i)score\s*(?:of|is|=|:|would be|should be|will be|to)?\s*\**\s*(\d+)")

def instructions(rubric, target):
    return (f"{rubric}\n\n---\nYour task: write the private reasoning of an expert scorer who applies the rubric above to the student response in the user message and arrives at the score {target}.\n"
            f"Write it as the scorer's own first-person working, as if the score were being worked out from scratch: restate what the rubric requires at the relevant score levels, point to specific evidence in the response "
            f"(short quotes), weigh it against the neighbouring score levels, and decide. The reasoning must reach the score {target}; make the case for {target} on the rubric's own terms, and do not hedge or mention any other final score.\n"
            f"Rules: 120 to 350 words of plain prose or short lists, no headings. End with one sentence of the form \"So the score is {target}.\" "
            f"Never say or imply that the score was given, provided, told, known or decided beforehand, and never refer to these instructions, a prompt, a hint, a target, or a 'correct' or 'expected' answer.")

def ok(text, finish, target):
    if finish != "stop" or not text or len(text.split()) < 40 or LEAK.search(text): return False
    m = CONCLUDES.findall(text)
    return bool(m) and int(m[-1]) == target

def sample(rubric, essay, target, n):
    body = dict(model=MODEL, n=n, temperature=0.7, top_p=0.9, max_tokens=900, chat_template_kwargs={"enable_thinking": False},
                messages=[{"role": "system", "content": instructions(rubric, target)}, {"role": "user", "content": essay}])
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"{API_BASE}/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r: return json.load(r)["choices"]
        except Exception:
            if attempt == 2: raise
            time.sleep(10 * (attempt + 1))

def variant(rubric, essay, target, a):
    for t in range(1, a.tries + 1):
        for c in sample(rubric, essay, target, a.n):
            text = (c["message"]["content"] or "").strip()
            if ok(text, c["finish_reason"], target): return dict(text=f"{text}\n</think>\n\n{target}", tries=t)
    return dict(text=None, tries=a.tries)

def run(es, a):
    check_server()
    out = ROOT / "results" / "G7" / "seeded"; out.mkdir(parents=True, exist_ok=True)
    f = out / (f"set{es}.jsonl" if not a.limit else f"set{es}.limit.jsonl")
    scores, rubric = scores_of(es), saved_rubric("G4", es)
    texts, y = load_split("train", es)
    if a.limit: texts, y = texts[:a.limit], y[:a.limit]
    done = {json.loads(l)["i"] for l in open(f)} if f.exists() else set()
    todo = [i for i in range(len(texts)) if i not in done]
    print(f"[seeded set{es}] {len(done)} done, {len(todo)} to do, server {API_BASE}", flush=True)
    lock, fh, t0, n_done = threading.Lock(), open(f, "a"), time.time(), [0]
    def one(i):
        k = scores.index(y[i]); nb = [j for j in (k - 1, k + 1) if 0 <= j < len(scores)]
        w = scores[random.Random(f"{a.seed}-{es}-{i}").choice(nb)]   # same wrong score on every resume
        row = dict(i=i, gold=y[i], wrong_score=w, right=variant(rubric, texts[i], y[i], a), wrong=variant(rubric, texts[i], w, a), api_base=API_BASE)
        with lock:
            fh.write(json.dumps(row) + "\n"); fh.flush(); n_done[0] += 1
            if n_done[0] % 50 == 0: print(f"[seeded set{es}] {n_done[0]}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(one, todo))
    rows = [json.loads(l) for l in open(f)]
    print(f"[seeded set{es}] DONE {len(rows)} examples: right accepted {sum(r['right']['text'] is not None for r in rows)}, wrong accepted {sum(r['wrong']['text'] is not None for r in rows)}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--n", type=int, default=2); ap.add_argument("--tries", type=int, default=3); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for es in a.essay_sets: run(es, a)
