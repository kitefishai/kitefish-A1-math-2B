import os

import torch, pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


load_dotenv()

model_name = os.getenv("MODEL_NAME")


dataset = load_dataset("pubmed_qa", "pqa_labeled")["test"]


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

rows, correct = [], 0
valid = ["yes", "no", "maybe"]

for ex in tqdm(dataset):
    prompt = (
        f"Question: {ex['question']}\n"
        f"Context: {ex['context']}\n"
        f"Answer (Yes / No / Maybe):"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    pred = tokenizer.decode(out[0], skip_special_tokens=True).lower()
    pred = next((v for v in valid if v in pred), None)
    gt = ex["final_decision"].lower()

    ok = pred == gt
    correct += ok

    rows.append({
        "question": ex["question"],
        "gt": gt,
        "pred": pred,
        "correct": ok
    })

print("PubMedQA accuracy:", correct / len(dataset))
pd.DataFrame(rows).to_excel("pubmedqa.xlsx", index=False)
