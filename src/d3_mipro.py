"""D2: MIPROv2 optimizes the text that d3-mini is shown, and nothing else.
usage: VLLM_API_BASE=http://10.151.66.182:8000/v1 python src/d3_mipro.py [--essay_sets 1 ...] [--auto light] [--n_train 100] [--limit N] [--port 8000] [--threads 16] ; then python src/d3_eval.py --exp D2
Task model = d3-mini through /v1/systemone (a thin dspy.BaseLM, `D3LM`: DSPy sends it a system message = the candidate text and a user message = the student response; it parses the candidate text back into
instructions + per-level criteria with split_rubric, calls the server and answers with the argmax score value). Instruction proposer = the Qwen3.5-4B chat server at $VLLM_API_BASE (thinking off), as in G4.
The program has ONE predictor whose instruction is the whole rubric text in the original files' layout (prompt/passage/preamble, then one '##### Score N' section per score point), so MIPROv2 rewrites instructions
and criteria together. No demos (the interface has no slot for them). Metric = agreement_metric (1 - squared score distance / squared range) of the argmax prediction, as G3/G4. Trainset = a seeded random subset of
--n_train training examples, valset = the dev split; the test split is untouched here (d3_eval.py reads it once, afterwards).
Parsing rule: a candidate whose text splits into a non-empty section for every score point (split_rubric) gives the model its own instructions AND criteria; any other candidate (the proposer often drops the
'##### Score N' headings) is used as the INSTRUCTIONS with the original per-level criteria kept, so every candidate is a valid request and none is wasted. The saved prompt.json is the same function of the final text.
Writes results/D2/default/seed0/set<k>/{prompt.json (instructions, criteria: what d3_eval.py and train_d3.py --text D2 read), rubric.md, optimize.json}."""
import argparse, json, time
import dspy, numpy as np
from dspy.teleprompt import MIPROv2
from gen_common import ROOT, scores_of, MinimalAdapter, make_lm, make_program, agreement_metric, API_BASE
from gen_mipro import examples
from d3_common import D3_ID, split_rubric, original_rubric, question, systemone, probs_of
from d3_eval import Server
SEED = 0

def parse_candidate(text, scores, crit0):
    """(instructions, criteria, criteria_rewritten) of a candidate text"""
    try: ins, crit = split_rubric(text, scores); return ins, crit, True
    except AssertionError: return text.strip(), list(crit0), False

def render(instructions, criteria, scores):
    """the text MIPROv2 starts from and rewrites: the original layout, level sections as '##### Score N'"""
    return instructions.rstrip() + "\n\n" + "\n\n".join(f"##### Score {s}\n\n{c}" for s, c in zip(scores, criteria)) + "\n"

class D3LM(dspy.BaseLM):
    def __init__(self, scores, crit0, api):
        super().__init__(model="d3-mini", cache=False); self.scores, self.crit0, self.api = scores, crit0, api; self.n_calls = 0; self.n_criteria_kept = 0
    def forward(self, prompt=None, messages=None, **kw):
        text, essay = messages[0]["content"], messages[-1]["content"]; self.n_calls += 1
        ins, crit, rewritten = parse_candidate(text, self.scores, self.crit0); self.n_criteria_kept += not rewritten
        p = probs_of(systemone(essay, question(ins, crit), api=self.api), len(self.scores))
        return {"choices": [{"message": {"role": "assistant", "content": str(self.scores[int(np.argmax(p))])}, "finish_reason": "stop"}], "model": "d3-mini", "usage": {}}

class D3Adapter(MinimalAdapter):
    """system = the candidate text only (no format suffix: the model's output is fixed); user = the response"""
    def format(self, signature, demos, inputs):
        if not (set(signature.input_fields) == {"response"} and set(signature.output_fields) == {"score"}): return dspy.ChatAdapter.format(self, signature, demos, inputs)
        return [{"role": "system", "content": signature.instructions}, {"role": "user", "content": inputs["response"]}]

def run(es, a, api):
    out = ROOT / "results" / "D2" / "default" / "seed0" / f"set{es}"
    if (out / "prompt.json").exists() and not a.limit: print("done already:", out); return
    scores = scores_of(es); ins0, crit0 = original_rubric(es); start = render(ins0, crit0, scores)
    assert split_rubric(start, scores) == (ins0, crit0), "the rendered starting text does not parse back to the original"
    task_lm = D3LM(scores, crit0, api); prompt_lm = make_lm(False, temperature=0.7, max_tokens=a.proposer_tokens)   # the proposer rewrites the whole text, so it needs room
    dspy.configure(lm=task_lm, adapter=D3Adapter(scores))
    opt = MIPROv2(metric=agreement_metric(scores), prompt_model=prompt_lm, task_model=task_lm, auto=a.auto, num_threads=a.threads, max_bootstrapped_demos=0, max_labeled_demos=0, seed=SEED, max_errors=10 ** 6)
    trainset, valset = examples("train", es, a.n_train), examples("dev", es)
    if a.limit: trainset, valset = trainset[:a.limit], valset[:a.limit]
    t0 = time.time(); best = opt.compile(make_program(start), trainset=trainset, valset=valset, requires_permission_to_run=False)
    text = best.signature.instructions; ins, crit, rewritten = parse_candidate(text, scores, crit0)
    info = dict(exp="D2", essay_set=es, auto=a.auto, n_train_examples=len(trainset), n_dev_examples=len(valset), changed=text != start, criteria_rewritten=rewritten, criteria_changed=crit != crit0, instructions_changed=ins != ins0,
                chars=len(text), mipro_val_score=getattr(best, "score", None), task_calls=task_lm.n_calls, calls_with_original_criteria=task_lm.n_criteria_kept, proposer=API_BASE, proposer_tokens=a.proposer_tokens, optimize_seconds=time.time() - t0)
    print(f"[D2 set{es}] changed {info['changed']} instructions_changed {info['instructions_changed']} criteria_changed {info['criteria_changed']} val_score {info['mipro_val_score']} calls with original criteria {task_lm.n_criteria_kept}/{task_lm.n_calls} {info['optimize_seconds']:.0f}s", flush=True)
    if a.limit: return
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(instructions=ins, criteria=crit), open(out / "prompt.json", "w")); (out / "rubric.md").write_text(text)
    json.dump(info, open(out / "optimize.json", "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--essay_sets", nargs="*", type=int, default=list(range(1, 11))); ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--n_train", type=int, default=100); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--proposer_tokens", type=int, default=6144); ap.add_argument("--no_server", action="store_true", help="use the base-model d3-mini server already running on --port")
    a = ap.parse_args(); api = f"http://localhost:{a.port}/v1"
    if a.no_server:
        for es in a.essay_sets: run(es, a, api)
    else:
        with Server(D3_ID, a.port):
            for es in a.essay_sets: run(es, a, api)
