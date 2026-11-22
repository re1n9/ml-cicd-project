from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.joblib"
CLASS_NAMES = ("setosa", "versicolor", "virginica")

_model = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_class(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> str:
    model = load_model()
    features = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        dtype=float,
    )
    class_index = int(model.predict(features)[0])
    return CLASS_NAMES[class_index]
