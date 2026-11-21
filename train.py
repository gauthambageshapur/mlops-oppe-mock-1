import json
import os
import subprocess


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


V0, V1 = "data/iris_v0.csv", "data/iris_v1.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES_COL = "species"
LABELS = {"setosa": 0, "versicolor": 1, "virginica": 2}
SEED = 42




def dvc_pull(path):
    """Consume data through DVC: pull the file if it is not in the workspace."""
    if not os.path.exists(path):
        subprocess.run(["dvc", "pull", path + ".dvc"], check=True)
    return path




def load(path):
    return pd.read_csv(dvc_pull(path))




def normalize_species(s):
    # "setosa", "Iris-setosa", " Setosa " all become "setosa"
    return s.astype(str).str.strip().str.lower().str.replace("iris-", "", regex=False)




def impute_last10(df):
    """Fill a missing feature value with the mean of the last 10 available values of
    that feature from the same species, taken from earlier rows in file order.
    The data has no timestamp column, so file order is the order.
    Only original (non-missing) values count as available. If no earlier row exists,
    use the first 10 available values after it."""
    df = df.dropna(subset=[SPECIES_COL]).reset_index(drop=True)
    df[SPECIES_COL] = normalize_species(df[SPECIES_COL])
    for f in FEATURES:
        raw = df[f].copy()
        for idx in df.index[raw.isna()]:
            same = df[SPECIES_COL] == df.at[idx, SPECIES_COL]
            prior = raw[same & (df.index < idx)].dropna().tail(10)
            if prior.empty:
                prior = raw[same & (df.index > idx)].dropna().head(10)
            df.at[idx, f] = prior.mean()
    return df

def prepare(df):
    df = impute_last10(df)
    df["target"] = df[SPECIES_COL].map(LABELS)
    assert df["target"].notna().all(), "species values do not match LABELS"
    return df




def run_iteration(name, df, save_split=None):
    df = prepare(df)
    X, y = df[FEATURES], df["target"]
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    model = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(Xtr, ytr)
    pred = model.predict(Xva)
    acc = accuracy_score(yva, pred)
    print(f"\n=== {name}: train={len(Xtr)} val={len(Xva)} accuracy={acc:.4f}")
    print(classification_report(yva, pred))
    if save_split:
        out = Xva.copy()
        out["target"] = yva
        out.to_csv(save_split, index=False)
    return model, acc




if __name__ == "__main__":
    v0, v1 = load(V0), load(V1)
    _, a1 = run_iteration("iteration 1 (iris_v0)", v0)
    merged = pd.concat([v0, v1], ignore_index=True)
    _, a2 = run_iteration("iteration 2 (iris_v0 + iris_v1)", merged,
                          save_split="data/test.csv")
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as fh:
        json.dump({"iteration_1_accuracy": a1, "iteration_2_accuracy": a2}, fh, indent=2)
