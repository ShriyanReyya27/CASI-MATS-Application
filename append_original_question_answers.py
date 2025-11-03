import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json

load_dotenv("key.env")
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

INPUT_CSV = "Outputs/decomposed_queries_stored_prompt.csv"  
OUTPUT_CSV = "Outputs/all_decomp_with_orig_answers.csv"
MODEL = "gpt-4"
DELAY_BETWEEN_CALLS = 1.0
MAX_RETRIES = 2

df = pd.read_csv(INPUT_CSV)

def parse_decomposed_cell(cell_value):
    if pd.isna(cell_value):
        return []
    if isinstance(cell_value, (list, dict)):
        return cell_value
    s = str(cell_value).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            fixed = s.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            start = s.find("[")
            end = s.rfind("]") + 1
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end])
                except Exception:
                    return []
            return []

def ask_gpt(question_text, max_retries=MAX_RETRIES):
    question_text = f"{question_text}\n\nPlease answer in at most 2 sentences."
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": question_text}],
                temperature=0
            )
            text = getattr(response, "output_text", None)
            if text:
                return text.strip()
            return str(response).strip()
        except Exception as e:
            print(f"API error on attempt {attempt} for question: {question_text[:50]}..., error: {e}")
            time.sleep(2)
    return "query failed"

all_results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing all questions"):
    orig_q = row['question']
    decomposed_cell = row['decomposed']
    
    orig_ans = ask_gpt(orig_q)
    
    parsed = parse_decomposed_cell(decomposed_cell)
    if not parsed:
        continue
    
    for block in parsed:
        decompositions = block.get("decompositions", []) if isinstance(block, dict) else []
        for decomp in decompositions:
            decomp_qs = decomp.get("decomp_questions", []) if isinstance(decomp, dict) else []
            for sub_q in decomp_qs:
                sub_ans = ask_gpt(sub_q)
                all_results.append({
                    "original_question": orig_q,
                    "sub_question": sub_q,
                    "sub_answer": sub_ans,
                    "original_answer": orig_ans
                })
                time.sleep(DELAY_BETWEEN_CALLS)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved full dataset with original and sub-question answers to {OUTPUT_CSV}")

