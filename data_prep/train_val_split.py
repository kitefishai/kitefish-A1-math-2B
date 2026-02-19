import random
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = Path(os.getenv("VOLUME"))
INPUT_DIR   = BASE_DIR / "external"
TRAIN_OUT_DIR     = BASE_DIR / "external" / "train"
VAL_OUT_DIR     = BASE_DIR / "external" / "val"

TRAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

input_file = INPUT_DIR / "ultrachat.jsonl"
train_file = TRAIN_OUT_DIR / "ultrachat_train.jsonl"
val_file = VAL_OUT_DIR / "ultrachat_val.jsonl"

split_ratio = 0.9  # 90% train

print(f"Splitting {input_file} to {train_file} and {val_file}.")
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

split_index = int(len(lines) * split_ratio)

print(f"Splitting {split_index} lines.")
train_lines = lines[:split_index]
val_lines = lines[split_index:]

print(f"Writing {len(train_lines)} lines to {train_file}.")
with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

print(f"Writing {len(val_lines)} lines to {val_file}.")
with open(val_file, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"Train: {len(train_lines)} lines")
print(f"Val: {len(val_lines)} lines")
