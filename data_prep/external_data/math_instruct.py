import os
import json
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = Path(os.getenv("VOLUME"))
out = Path(BASE_DIR) / "external/mathinstruct.jsonl"

ds = load_dataset("TIGER-Lab/MathInstruct", split="train")

with out.open("w", encoding="utf-8") as f:
    for ex in ds:
        inst = ex.get("instruction", "").strip()
        outp = ex.get("output", "").strip()
        if inst and outp:
            f.write(
                json.dumps(
                    {"instruction": inst, "output": outp},
                    ensure_ascii=False,
                )
                + "\n"
            )
