import joblib
import pandas as pd


MODEL_PATH = "models/model.joblib"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}




def load_model(path=MODEL_PATH):
    return joblib.load(path)




def predict(df, model=None):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if model is None:
        model = load_model()
    return model.predict(df[FEATURES])




if __name__ == "__main__":
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)
    label = int(predict(sample)[0])
    print("predicted:", label, LABEL_NAMES[label])
