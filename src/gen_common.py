"""Shared pieces for the generative-model arms (G1-G4): a Qwen3.5-4B scorer served by vLLM, driven through DSPy.
The prompt structure is fixed and minimal: system = <rubric> + one fixed instruction, user = the student response, output = one integer.
The rubric is the only thing MIPROv2 is allowed to change (it is the signature's instructions)."""
import json, os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import dspy, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
API_BASE = os.environ.get("VLLM_API_BASE", "http://10.151.66.140:8000/v1")   # one of the Qwen3.5-4B servers; 10.151.66.132 now serves another model
MODEL = "Qwen/Qwen3.5-4B"
def format_suffix(scores):   # fixed, never optimized: also fixes the score range, which the optimizer must not be able to rewrite
    return f"Reply with only the integer score: one of {', '.join(map(str, scores))}."
# Decoding is fixed per mode (Qwen's recommended settings for thinking; greedy when not thinking).
DECODING = {False: dict(temperature=0.0, max_tokens=64), True: dict(temperature=0.6, top_p=0.95, max_tokens=16384)}
INT_RE = re.compile(r"-?\d+")

meta = json.load(open(ROOT / "data" / "splits" / "meta.json"))

def scores_of(essay_set):
    return meta["sets"][str(essay_set)]["scores"]

def rubric_of(essay_set):
    return (ROOT / "data" / "rubrics_md" / f"rubric_{essay_set}.md").read_text()   # used as-is, including its header table

def load_split(split, essay_set):
    d = pd.read_csv(ROOT / "data" / "splits" / f"{split}.csv")
    d = d[d.essay_set == essay_set].reset_index(drop=True)
    return d["text"].astype(str).tolist(), d["score"].astype(int).tolist()

class ScoreSig(dspy.Signature):
    response: str = dspy.InputField()
    score: int = dspy.OutputField()

def is_scoring(signature):
    return set(signature.input_fields) == {"response"} and set(signature.output_fields) == {"score"}

class MinimalAdapter(dspy.ChatAdapter):
    """Scoring calls: system = signature instructions (the rubric) + fixed suffix; user = the response; parse = first integer after the thinking block.
    Any other signature (MIPROv2's own instruction-proposal calls) uses DSPy's default chat format."""
    def __init__(self, scores, **kw):
        super().__init__(**kw); self.suffix = format_suffix(scores)

    def format(self, signature, demos, inputs):
        if not is_scoring(signature): return super().format(signature, demos, inputs)
        return [{"role": "system", "content": f"{signature.instructions}\n\n{self.suffix}"},
                {"role": "user", "content": inputs["response"]}]

    def parse(self, signature, completion):
        if not is_scoring(signature): return super().parse(signature, completion)
        if "</think>" in completion: completion = completion.rsplit("</think>", 1)[1]
        elif completion.lstrip().startswith("Thinking"): return {"score": None}   # reasoning truncated before an answer
        m = INT_RE.search(completion)
        return {"score": int(m.group()) if m else None}

def check_server():
    """abort unless the server really serves MODEL: a wrong model gives 404s that would otherwise be scored as the lowest score point"""
    with urllib.request.urlopen(f"{API_BASE}/models", timeout=15) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    assert MODEL in ids, f"{API_BASE} serves {ids}, not {MODEL}"

def make_lm(thinking, **kw):
    check_server()
    return dspy.LM(f"openai/{MODEL}", api_base=API_BASE, api_key="EMPTY", cache=False, num_retries=3,
                   extra_body={"chat_template_kwargs": {"enable_thinking": thinking}}, **{**DECODING[thinking], **kw})

def make_program(rubric):
    sig = ScoreSig.with_instructions(rubric)
    return dspy.Predict(sig)

def covers_all_scores(rubric, scores):
    """True if every score point of the set is mentioned next to the word 'score' or 'point' (same line, within a few words).
    Used to reject optimized rubrics that dropped a score point; the check is deliberately simple and conservative."""
    return all(re.search(rf"(?i)(score|point)[^\n]{{0,25}}(?<![\d.]){s}(?![\d.])|(?<![\d.]){s}(?![\d.])[^\n]{{0,15}}(score|point)", rubric) for s in scores)

def to_index(score, scores):
    """map a predicted score to a class index; missing or out-of-range predictions become None (counted as failures)"""
    return scores.index(score) if score in scores else None

def predict_all(program, texts, threads=32):
    """predicted scores in input order; None when the answer is unparseable or the call failed"""
    def one(t):
        try: return program(response=t).score
        except Exception: return None
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(one, texts))

def agreement_metric(scores):
    """per-example metric for MIPROv2: 1 - squared score distance / squared range; unparseable = 0.
    A per-example surrogate for QWK (which needs the whole set), so optimizing it favours near misses over far misses."""
    r2 = float(max(scores) - min(scores)) ** 2
    def metric(example, pred, trace=None):
        p = getattr(pred, "score", None)
        return 0.0 if p is None else max(0.0, 1.0 - (p - example.score) ** 2 / r2)
    return metric
