import os
from openai import OpenAI
import pandas as pd
import json
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv("key.env")
load_dotenv("prompt.env")

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

INPUT_CSV = "Outputs/decomposed_queries_stored_prompt.csv"  
TEST_OUTPUT_CSV = "Outputs/first_row_decomp_answers.csv"
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
        parsed = json.loads(s)
        return parsed
    except Exception:
        try:
            fixed = s.replace("'", '"')
            parsed = json.loads(fixed)
            return parsed
        except Exception:
            # Try to extract first JSON array substring
            start = s.find("[")
            end = s.rfind("]") + 1
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(s[start:end])
                    return parsed
                except Exception:
                    return []
            return []

def ask_gpt(question_text, max_retries=MAX_RETRIES):
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                input=[{"role": "user", "content": question_text}],
                temperature=0
            )
            # Try the convenient attribute first
            text = getattr(response, "output_text", None)
            if text:
                return text.strip()
            # Fallback: extract from response.output structure
            output_texts = []
            output = getattr(response, "output", None) or (response.get("output") if isinstance(response, dict) else None)
            if output:
                for item in output:
                    content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
                    for c in content:
                        # c may be dicts with 'text' keys
                        if isinstance(c, dict):
                            t = c.get("text") or c.get("value", {}).get("text")
                            if t:
                                output_texts.append(t)
                        elif isinstance(c, str):
                            output_texts.append(c)
            if output_texts:
                return "\n".join(t.strip() for t in output_texts if t and t.strip())
            # Last fallback: return stringified response
            return str(response).strip()
        except Exception as e:
            print(f"API error on attempt {attempt} for question: {question_text[:80]}... error: {e}")
            time.sleep(2)
    return "query failed"
'''
# --- TEST SECTION: only first row ---
if len(df) == 0:
    raise RuntimeError(f"Input CSV '{INPUT_CSV}' is empty or not found.")

first_row = df.loc[0]
orig_question = first_row.get("question", "")
decomposed_cell = first_row.get("decomposed", "")

parsed = parse_decomposed_cell(decomposed_cell)
test_answers = []  # list of dicts: {"question": sub_q, "answer": ans}

print("=== TEST: processing first row decomp_questions ===")
print("Original question (from CSV):", orig_question)
if not parsed:
    print("No decomposed structure found for first row. Parsed value:", decomposed_cell)
else:
    for block in parsed:
        decompositions = block.get("decompositions", []) if isinstance(block, dict) else []
        for decomp in decompositions:
            decomp_qs = decomp.get("decomp_questions", []) if isinstance(decomp, dict) else []
            for sub_q in decomp_qs:
                print("Asking sub-question:", sub_q)
                ans = ask_gpt(sub_q)
                test_answers.append({"question": sub_q, "answer": ans})
                time.sleep(DELAY_BETWEEN_CALLS)

os.makedirs(os.path.dirname(TEST_OUTPUT_CSV), exist_ok=True)
test_df = pd.DataFrame(test_answers)
test_df.to_csv(TEST_OUTPUT_CSV, index=False)
print(f"Saved test answers (first row) to {TEST_OUTPUT_CSV}")
'''
# --- FULL LOOP: all rows ---
FULL_OUTPUT_CSV = "Outputs/all_decomp_answers.csv"
all_results = []  

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing all rows"):
    orig_q = row.get("question", "")
    decomposed_cell = row.get("decomposed", "")
    parsed = parse_decomposed_cell(decomposed_cell)

    if not parsed:
        continue  # skip rows with no decomposed content

    for block in parsed:
        decompositions = block.get("decompositions", []) if isinstance(block, dict) else []
        for decomp in decompositions:
            decomp_qs = decomp.get("decomp_questions", []) if isinstance(decomp, dict) else []
            for sub_q in decomp_qs:
                # Append instruction to limit answer length
                sub_q_with_limit = f"{sub_q}\n\nPlease answer in at most 2 sentences."
                ans = ask_gpt(sub_q_with_limit)
                all_results.append({
                    "original_question": orig_q,
                    "sub_question": sub_q,
                    "answer": ans
                })
                time.sleep(DELAY_BETWEEN_CALLS)

# Save all answers to CSV
os.makedirs(os.path.dirname(FULL_OUTPUT_CSV), exist_ok=True)
all_df = pd.DataFrame(all_results)
all_df.to_csv(FULL_OUTPUT_CSV, index=False)
print(f"Saved all answers to {FULL_OUTPUT_CSV}")

