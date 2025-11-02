from openai import OpenAI
import openai
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv("key.env")
load_dotenv("prompt.env")

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

PROMPT_ID = os.getenv("PROMPT_ID")
from tqdm import tqdm  

df = pd.read_csv("Data/sampled_redteam_fraud_dataset_v2.csv")  
def decompose_query_stored_prompt(query, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                prompt={
                    "id": PROMPT_ID,
                    "version": "1"
                },
                input=query  
            )

            # New SDK style: extract text safely
            text = getattr(response, "output_text", "").strip()
            if text:
                return text

            return "query refused"

        except Exception as e:
            print(f"API error on attempt {attempt}: {e}")

    return "query refused"

# --- TEST SECTION: first row only ---
first_question = df.loc[0, "question"]
print("Original question:", first_question)
first_decomposition = decompose_query_stored_prompt(first_question)
print("Decomposition:", first_decomposition)

# --- FULL LOOP ---
decomposed_list = []
for q in tqdm(df['question'], desc="Processing queries"):
    decomposed_list.append(decompose_query_stored_prompt(q))

df['decomposed'] = decomposed_list
df.to_csv("Outputs/decomposed_queries_stored_prompt.csv", index=False)
print("Saved decomposed queries to CSV!")
