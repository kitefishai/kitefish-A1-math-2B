import torch
import json
from torch.utils.data import Dataset

class JsonlDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, seq_length=768, max_samples=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line)

                # Change this depending on your JSON structure
                text = obj["text"]

                self.samples.append(text)

        print(f"Loaded {len(self.samples)} JSONL samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.seq_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0)
        }
