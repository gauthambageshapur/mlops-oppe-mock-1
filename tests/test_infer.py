import os


import pandas as pd
import pytest


from infer import FEATURES, MODEL_PATH, load_model, predict




@pytest.fixture(scope="module")
def model():
    return load_model()




def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)




def test_predict_shape_and_label_range(model):
    df = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2], [6.0, 2.8, 4.5, 1.3], [6.5, 3.0, 5.5, 2.0]],
        columns=FEATURES,
    )
    pred = predict(df, model)
    assert len(pred) == 3
    assert set(pred) <= {0, 1, 2}




def test_clear_samples_get_expected_species(model):
    df = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2], [6.0, 2.8, 4.5, 1.3], [6.5, 3.0, 5.5, 2.0]],
        columns=FEATURES,
    )
    assert list(predict(df, model)) == [0, 1, 2]




def test_missing_column_raises(model):
    with pytest.raises(ValueError):
        predict(pd.DataFrame({"sepal_length": [5.1]}), model)




def test_accuracy_on_dvc_test_data(model):
    path = "data/test.csv"
    if not os.path.exists(path):
        pytest.skip("test data not pulled")
    df = pd.read_csv(path)
    acc = (predict(df, model) == df["target"]).mean()
    assert acc >= 0.85
