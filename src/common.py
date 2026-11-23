"""Data access shared by every script: repo root, per-set score lists, and the train/dev/test splits written by data/prep.py."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
meta = json.load(open(ROOT / "data" / "splits" / "meta.json"))

def scores_of(essay_set):
    return meta["sets"][str(essay_set)]["scores"]

def load_split(split, essay_set):
    d = pd.read_csv(ROOT / "data" / "splits" / f"{split}.csv")
    d = d[d.essay_set == essay_set].reset_index(drop=True)
    return d["text"].astype(str).tolist(), d["score"].astype(int).tolist()
