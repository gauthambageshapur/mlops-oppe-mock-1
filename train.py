import json
import os
import subprocess
import mlflow
import mlflow.sklearn
import itertools
import joblib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split


V0, V1 = "data/iris_v0.csv", "data/iris_v1.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES_COL = "species"
LABELS = {"setosa": 0, "versicolor": 1, "virginica": 2}
SEED = 42
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-oppe1-mock")


# start with ONE config; expanded in Step 3
GRID = {"n_estimators": [100], "max_depth": [None]}



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
def split(df):
    df = prepare(df)
    X, y = df[FEATURES], df["target"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)




def tune(iteration, df, data_files, save_split=None):
    """One MLflow run per hyperparameter trial."""
    Xtr, Xva, ytr, yva = split(df)
    if save_split:
        out = Xva.copy()
        out["target"] = yva
        out.to_csv(save_split, index=False)


    results = []
    for n, d in itertools.product(GRID["n_estimators"], GRID["max_depth"]):
        with mlflow.start_run(run_name=f"it{iteration}_n{n}_d{d}") as run:
            model = RandomForestClassifier(
                n_estimators=n, max_depth=d, random_state=SEED
            ).fit(Xtr, ytr)
            pred = model.predict(Xva)
            m = {
                "accuracy": accuracy_score(yva, pred),
                "precision_macro": precision_score(yva, pred, average="macro"),
                "recall_macro": recall_score(yva, pred, average="macro"),
                "f1_macro": f1_score(yva, pred, average="macro"),
            }
            mlflow.log_params({
                "n_estimators": n, "max_depth": d, "iteration": iteration,
                "data_files": ",".join(data_files),
                "train_rows": len(Xtr), "val_rows": len(Xva),
            })
            mlflow.set_tag("iteration", str(iteration))
            mlflow.log_metrics(m)
            mlflow.log_text(classification_report(yva, pred), "classification_report.txt")
            for f in data_files:                      # data version lineage
                mlflow.log_artifact(f + ".dvc", artifact_path="data_version")
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
            print(f"it{iteration} n={n} depth={d} acc={m['accuracy']:.4f}")
            results.append({"run_id": run.info.run_id, "iteration": iteration,
                            "n_estimators": n, "max_depth": d, **m, "model": model})
    return results




if __name__ == "__main__":
    v0, v1 = load(V0), load(V1)
    r1 = tune(1, v0, [V0])
    r2 = tune(2, pd.concat([v0, v1], ignore_index=True), [V0, V1],
              save_split="data/test.csv")
    allr = r1 + r2


    best = max(allr, key=lambda r: (r["accuracy"], r["iteration"]))
    rows = [{k: v for k, v in r.items() if k != "model"} for r in allr]
    summary = pd.DataFrame(rows).sort_values(["accuracy", "iteration"], ascending=False)


    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    summary.to_csv("results/runs_summary.csv", index=False)
    joblib.dump(best["model"], "models/model.joblib")


    best_info = {k: v for k, v in best.items() if k != "model"}
    with open("results/best_model.json", "w") as fh:
        json.dump(best_info, fh, indent=2, default=str)
    with open("results/metrics.json", "w") as fh:
        json.dump({
            "iteration_1_best_accuracy": max(r["accuracy"] for r in r1),
            "iteration_2_best_accuracy": max(r["accuracy"] for r in r2),
            "overall_best": best_info,
        }, fh, indent=2, default=str)


    # a final run that records the chosen model
    with mlflow.start_run(run_name="best_model"):
        mlflow.set_tag("role", "best")
        mlflow.log_params({k: best_info[k] for k in ("iteration", "n_estimators", "max_depth")})
        mlflow.log_metrics({k: best_info[k] for k in
                            ("accuracy", "precision_macro", "recall_macro", "f1_macro")})
        mlflow.log_artifact("models/model.joblib")


    print("\n", summary.to_string(index=False))
    print("\nBEST:", best_info)
