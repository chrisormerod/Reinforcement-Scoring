"""Shared pieces for the decision-model arms (D1-D5): vllm-sr/d3-mini, a Qwen3.5-4B classifier that returns a probability per option instead of text.
Request mapping: state = the student response; one `score` question per essay set: instructions = everything in the rubric file before the per-level text (prompt, passage,
preamble), criteria = the ordered per-level texts (one per score point of the set). Level i of the model is score point scores[i].
Training path: the packaged d3 class refuses to train (train(True) raises, inference_mode, frozen backbone, no save_pretrained), so this module rebuilds its plain logits path WITH gradients:
left-padded prompt tokens (made by the runtime's own `prepare`, so the prompt is exactly the served one) -> backbone (bidirectional PromptEncoder for noncausal checkpoints) -> last-position hidden
-> hidden.float() @ readout.T (frozen, FP32) -> mask to the option count -> softmax(logits / temperature). A LoRA adapter on the text decoder's linear layers is the only trainable part."""
import json, os, re, urllib.request
from pathlib import Path
import numpy as np
from common import ROOT, scores_of, load_split

D3_ID = "vllm-sr/d3-mini"
D3_API = os.environ.get("D3_API_BASE", "http://localhost:8000/v1")
CKPT_DIR = Path(os.environ.get("RF_CKPT_DIR", Path.home() / "rf_ckpt"))
LEVEL = re.compile(r"(?i)^(?:#+\s*|\*\*\s*)?(?:score(?:\s+point)?\s*)?(\d+)(?:\s*-?\s*points?)?\s*[:.]?\s*\**\s*:?\s*$")
ANY_HEADING = re.compile(r"^#+\s")

def level_of(line):
    """the score point a heading line names ('##### Score 2', '2 points', '**Score Point 1:**', ...), else None; a bare number only counts after '#' or when the line says score/point"""
    l = line.strip(); m = LEVEL.match(l)
    return int(m.group(1)) if m and (l.startswith("#") or re.search(r"(?i)score|point", l)) else None

def split_rubric(text, scores):
    """(instructions, [level text per score point]) from a rubric text: the LAST heading for each score point starts its level text, which runs to the next heading (of any kind, or the next level
    heading) after it; the instructions are everything before the first of those level headings. Raises AssertionError if a score point has no level heading or an empty text."""
    lines = text.split("\n"); heads = {}
    for n, l in enumerate(lines):
        s = level_of(l)
        if s in scores: heads.setdefault(s, []).append(n)
    assert all(s in heads for s in scores), f"no level heading for score points {[s for s in scores if s not in heads]}"
    start = {s: heads[s][-1] for s in scores}
    first = min(start.values()); levels = []
    for s in scores:
        n = start[s] + 1; end = next((j for j in range(n, len(lines)) if ANY_HEADING.match(lines[j]) or level_of(lines[j]) in scores), len(lines))
        body = "\n".join(lines[n:end]).strip()
        assert body, f"empty level text for score point {s}"
        levels.append(body)
    return "\n".join(lines[:first]).strip(), levels

def original_rubric(es):
    """(instructions, criteria) of set es from data/rubrics_md (D1's input)"""
    return split_rubric((ROOT / "data" / "rubrics_md" / f"rubric_{es}.md").read_text(), scores_of(es))

def question(instructions, criteria):
    return {"score": dict(type="score", instructions=instructions, criteria=list(criteria))}

def systemone(state, questions, model=D3_ID, api=None, timeout=300):
    """one /v1/systemone call; returns the parsed JSON (raises on HTTP errors)"""
    body = json.dumps(dict(model=model, state=state, questions=questions)).encode()
    req = urllib.request.Request(f"{api or D3_API}/systemone", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def probs_of(resp, n):
    """level probabilities (list of n floats) from a systemone response; None if the answer is missing"""
    a = resp.get("answers", {}).get("score", {}).get("probabilities")
    return None if a is None else [float(a[str(i)]) for i in range(n)]

def load_text(es, source):
    """(instructions, criteria) for set es: the original rubric, or the optimized text of experiment `source` (D2) saved as results/<source>/default/seed0/set<es>/prompt.json"""
    if source == "original": return original_rubric(es)
    f = ROOT / "results" / source / "default" / "seed0" / f"set{es}" / "prompt.json"
    assert f.exists(), f"{f} is missing: run {source} on set {es} first"
    d = json.load(open(f)); return d["instructions"], d["criteria"]
