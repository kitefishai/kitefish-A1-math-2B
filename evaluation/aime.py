import os
import re, torch, pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

load_dotenv()

dataset = load_dataset("lighteval/aime24")["test"]
model_name = os.getenv("MODEL_NAME")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

def last_int(text):
    nums = re.findall(r"\b\d+\b", text)
    return nums[-1] if nums else None

rows, correct = [], 0

for ex in tqdm(dataset):
    prompt = (
        f"Question: {ex['question']}\n"
        f"Answer: The final answer is an integer between 0 and 999.\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    pred_text = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = last_int(pred_text)
    gt = str(ex["answer"])

    ok = pred == gt
    correct += ok

    rows.append({
        "question": ex["question"],
        "gt": gt,
        "pred": pred,
        "correct": ok
    })

print("AIME accuracy:", correct / len(dataset))
pd.DataFrame(rows).to_excel("aime.xlsx", index=False)
