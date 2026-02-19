import os
import re, torch
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


load_dotenv()

model_name = os.getenv("MODEL_NAME")

dataset = load_dataset("hendrycks/competition_math")["test"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).eval()

def normalize(ans):
    ans = ans.lower()
    ans = re.sub(r"\\boxed{|}", "", ans)
    ans = ans.replace(" ", "")
    return ans

rows, correct = [], 0

for ex in tqdm(dataset):
    prompt = f"Question: {ex['problem']}\nAnswer: Let's think step by step."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    pred = tokenizer.decode(out[0], skip_special_tokens=True)

    gt = normalize(ex["solution"])
    pred_n = normalize(pred)
    ok = gt in pred_n
    correct += ok

    rows.append({
        "problem": ex["problem"],
        "correct": ok,
        "model_output": pred
    })

print("MATH accuracy:", correct / len(dataset))
pd.DataFrame(rows).to_excel("math.xlsx", index=False)
