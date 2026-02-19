import os

from datasets import load_dataset
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = Path(os.getenv("VOLUME"))
out = Path(BASE_DIR) / "external/ultrachat.jsonl"


ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

with out.open("w", encoding="utf-8") as f:
    for ex in ds:
        conv = ex.get("messages", [])
        text = []
        for m in conv:
            role = m.get("role", "")
            content = m.get("content", "")
            text.append(f"{role.upper()}: {content}")
        joined = "\n".join(text).strip()
        if joined:
            f.write(json.dumps({"text": joined}, ensure_ascii=False) + "\n")
