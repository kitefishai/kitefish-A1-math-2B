import os

from datasets import load_dataset
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_DIR    = Path(os.getenv("VOLUME"))
out = Path(BASE_DIR) / "external/stackexchange.jsonl"


ds = load_dataset("lvwerra/stack-exchange-paired", split="test", streaming=True)

KEEP_SITES = {
    "https://math.stackexchange.com",
    "https://mathoverflow.net",
    "https://stats.stackexchange.com",
    "https://cs.stackexchange.com",
}

with out.open("w", encoding="utf-8") as f:
    for ex in tqdm(ds):
        for site in ex.get("metadata"):
            if site.lower() not in KEEP_SITES:
                continue

        q = ex.get("question", "").strip()
        a = ex.get("response_j", "").strip()
        if q and a:
            text = f"Question:\n{q}\n\nAnswer:\n{a}"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
