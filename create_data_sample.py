import pandas as pd

df = pd.read_csv("Data/redteam_fraud_dataset_v2.csv") 
df = df.sample(n = 25, random_state = 42)

df.to_csv("Data/sampled_redteam_fraud_dataset_v2.csv")
