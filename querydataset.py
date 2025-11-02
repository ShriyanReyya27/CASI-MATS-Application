#from huggingface_hub import login
from datasets import load_dataset
import pandas as pd

RedTeam_2K = load_dataset("JailbreakV-28K/JailBreakV-28k", 'RedTeam_2K')["RedTeam_2K"]

df = pd.DataFrame(RedTeam_2K)

# Get unique policies
policies = df['policy'].unique()

# Sample 5 rows per policy
#sampled_df = pd.concat([df[df['policy'] == p].sample(n=5, random_state=42) for p in policies])
sampled_df = pd.concat([df[df['policy'] == "Fraud"])

# Optional: shuffle the final dataset
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Export to CSV
#sampled_df.to_csv("Data/sampled_redteam_dataset.csv", index=False)
sampled_df.to_csv("Data/redteam_fraud_dataset.csv", index=False)

