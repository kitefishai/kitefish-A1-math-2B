import os
import json

from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset


load_dotenv()

BASE_DIR    = Path(os.getenv("VOLUME"))
out = Path(BASE_DIR) / "external/openwebmath.jsonl"


ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

with out.open("w", encoding="utf-8") as f:
    for ex in tqdm(ds):
        text = ex.get("text", "").strip()
        if text:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
