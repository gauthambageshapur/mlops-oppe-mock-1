import json
import os
import sys


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


from infer import predict


TEST_PATH = "data/test.csv"
THRESHOLD = 0.85


df = pd.read_csv(TEST_PATH)
pred = predict(df)
acc = accuracy_score(df["target"], pred)
print(classification_report(df["target"], pred))
print(f"rows={len(df)} accuracy={acc:.4f} threshold={THRESHOLD}")


os.makedirs("results", exist_ok=True)
with open("results/ci_metrics.json", "w") as fh:
    json.dump({"rows": len(df), "accuracy": acc, "threshold": THRESHOLD}, fh, indent=2)


sys.exit(0 if acc >= THRESHOLD else 1)
