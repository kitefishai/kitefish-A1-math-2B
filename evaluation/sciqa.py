import os

import torch, pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

load_dotenv()

model_name = os.getenv("MODEL_NAME")



dataset = load_dataset("sciqa")["test"]


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

rows, correct = [], 0

for ex in tqdm(dataset):
    options = "\n".join([f"{k}) {v}" for k, v in ex["choices"].items()])
    prompt = f"Question: {ex['question']}\nOptions:\n{options}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()[:1]
    gt = ex["answer"]

    ok = pred == gt
    correct += ok

    rows.append({
        "question": ex["question"],
        "gt": gt,
        "pred": pred,
        "correct": ok
    })

print("SciQA accuracy:", correct / len(dataset))
pd.DataFrame(rows).to_excel("sciqa.xlsx", index=False)
