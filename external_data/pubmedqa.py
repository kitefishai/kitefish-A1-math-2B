import os

from datasets import load_dataset
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = Path(os.getenv("VOLUME"))
out = Path(BASE_DIR) / "external/pubmedqa_sciq.jsonl"


with out.open("w", encoding="utf-8") as f:

    pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    for ex in pubmed:
        q = ex["question"]
        a = ex["long_answer"]
        f.write(json.dumps({"text": f"Q: {q}\nA: {a}"}) + "\n")

    sciq = load_dataset("sciq", split="train")
    for ex in sciq:
        q = ex["question"]
        a = ex["correct_answer"]
        f.write(json.dumps({"text": f"Q: {q}\nA: {a}"}) + "\n")
