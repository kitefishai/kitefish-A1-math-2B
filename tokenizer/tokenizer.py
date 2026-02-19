import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os

TOKENIZER_PATH = "./tokenizer"
INPUT_FILE = "data.jsonl"
OUTPUT_DIR = "tokens"
DTYPE = np.uint16  # use uint32 if vocab > 65k

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)

tokens = []
doc_idx = []

offset = 0

with open(INPUT_FILE, "r") as f:
    for line in tqdm(f):
        obj = json.loads(line)
        text = obj["text"].strip()
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(ids)

        offset += len(ids)
        doc_idx.append(offset)

tokens = np.array(tokens, dtype=DTYPE)
doc_idx = np.array(doc_idx, dtype=np.uint64)

tokens.tofile(f"{OUTPUT_DIR}/tokens.bin")
doc_idx.tofile(f"{OUTPUT_DIR}/doc.idx")

print("Total tokens:", len(tokens))
