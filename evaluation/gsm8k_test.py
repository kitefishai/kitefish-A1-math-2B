import os
import re, torch
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


load_dotenv()

model_name = os.getenv("MODEL_NAME")

dataset = load_dataset("gsm8k", "main")["test"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

def extract_num(text):
    if "####" in text:
        m = re.search(r"####\s*(-?\d+\.?\d*)", text)
        if m: return m.group(1)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

rows, correct = [], 0

for ex in tqdm(dataset):
    prompt = f"Question: {ex['question']}\nAnswer: Let's think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    pred_text = tokenizer.decode(out[0], skip_special_tokens=True)

    gt = extract_num(ex["answer"])
    pred = extract_num(pred_text)
    ok = gt == pred
    correct += ok

    rows.append({
        "question": ex["question"],
        "gt": gt,
        "pred": pred,
        "correct": ok,
        "model_output": pred_text
    })

acc = correct / len(dataset)
print("GSM8K accuracy:", acc)

pd.DataFrame(rows).to_excel("gsm8k.xlsx", index=False)
