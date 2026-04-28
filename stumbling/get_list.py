import pandas as pd
import json

df = pd.read_csv("rock_vs_control.csv")

rock_ids = (
    df["rock_id"]
    .dropna()
    .astype(int)
    .drop_duplicates()
    .tolist()
)

ctrl_ids = (
    df["ctrl_id"]
    .dropna()
    .astype(int)
    .drop_duplicates()
    .tolist()
)

with open("rock.json", "w") as f:
    json.dump(rock_ids, f)

with open("random.json", "w") as f:
    json.dump(ctrl_ids, f)

print("rock:", len(rock_ids))
print("random:", len(ctrl_ids))