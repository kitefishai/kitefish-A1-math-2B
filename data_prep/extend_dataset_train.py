import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Use /mnt disk for all heavy data
# ============================================================
BASE_DIR    = Path(os.getenv("VOLUME"))
IN_DIR   = BASE_DIR / "processed_15GB"
EXTERNAL_DIR   = BASE_DIR / "external" / "train"
OUT_DIR = BASE_DIR / "dataset_train_val"


OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_TRAIN = Path(IN_DIR) / "train.jsonl"
date_str = datetime.now().strftime("%Y%m%d")
OUTPUT_FILE = Path(OUT_DIR) / f"train_merged_{date_str}.jsonl"

# Sampling / repeat factors (based on your plan)
DATASETS = {
    "mathinstruct_train.jsonl": {
        "source": "mathinstruct",
        "repeat": 5,
        "text_key": "instruction",  # common format
        "answer_key": "output",
    },
    "pubmedqa_sciq_train.jsonl": {
        "source": "pubmedqa_sciq",
        "repeat": 1,
        "text_key": "text",
    },
    "ultrachat_train.jsonl": {
            "source": "ultrachat",
            "repeat": 1,
            "text_key": "content",
        },
    "openwebmath_5_train.jsonl": {
            "source": "openwebmath",
            "repeat": 1,
            "text_key": "text",
        },
    "stackexchange_1_train.jsonl": {
            "source": "stackexchange",
            "repeat": 1,
            "text_key": "text",
        },

}

# -------------------------
# Helpers
# -------------------------
async def read_jsonl(path: Path) -> AsyncGenerator[Any, None]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


async def normalize_record(
    record: Dict,
    *,
    source: str,
    text_key: str,
    answer_key: str | None = None,
) -> dict[str, str] | None:
    """
    Normalize different dataset schemas into one.
    """
    if answer_key:
        text = (
            f"Problem:\n{record.get(text_key, '')}\n\n"
            f"Solution:\n{record.get(answer_key, '')}"
        )
    else:
        text = record.get(text_key, "")

    text = text.strip()
    if not text:
        return None

    return {
        "text": text,
        "source": source,
    }


async def write_jsonl(path: Path, records: Iterable[Dict]):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Merge pipeline
# -------------------------
async def main():
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total = 0

    # 1️⃣ Copy existing arXiv data
    print("➤ Adding base arXiv dataset...")
    async for rec in read_jsonl(BASE_TRAIN):
        await write_jsonl(OUTPUT_FILE, [rec])
        total += 1

    # 2️⃣ Add external datasets
    for filename, cfg in DATASETS.items():
        path = EXTERNAL_DIR / filename
        if not path.exists():
            print(f"⚠️ Skipping missing dataset: {filename}")
            continue

        print(f"➤ Adding {filename} (repeat={cfg['repeat']})")

        for _ in range(cfg["repeat"]):
            buffer = []
            async for rec in read_jsonl(path):
                norm = await normalize_record(
                    rec,
                    source=cfg["source"],
                    text_key=cfg["text_key"],
                    answer_key=cfg.get("answer_key"),
                )
                if norm:
                    buffer.append(norm)
                    total += 1

                # flush periodically
                if len(buffer) >= 1000:
                    await write_jsonl(OUTPUT_FILE, buffer)
                    buffer.clear()

            if buffer:
                await write_jsonl(OUTPUT_FILE, buffer)

    print(f"✅ Done. Total records written: {total:,}")
    print(f"📄 Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())