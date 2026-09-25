"""Measure how often the assignment's BERT max_length=64 truncates NYT test text."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

root = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--model", default="google-bert/bert-base-uncased")
a = p.parse_args()
with open(root/"data/nyt.csv", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
split = json.loads((root/"results/splits.json").read_text())
tok = AutoTokenizer.from_pretrained(a.model)
ids = tok([rows[i]["text"] for i in split["test"]],add_special_tokens=True,truncation=False,verbose=False)["input_ids"]
lengths = np.array([len(x) for x in ids])
result = {"model": "google-bert/bert-base-uncased", "max_length": 64,
          "test_n": int(len(lengths)), "test_truncated_n": int(np.sum(lengths>64)),
          "test_truncated_fraction": float(np.mean(lengths>64)),
          "test_median_tokens": float(np.median(lengths)),
          "test_p90_tokens": float(np.percentile(lengths,90)),
          "test_min_tokens": int(np.min(lengths)), "test_max_tokens": int(np.max(lengths))}
(root/"results/truncation_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
