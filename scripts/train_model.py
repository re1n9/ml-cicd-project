from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main() -> None:
    iris = load_iris()
    model = LogisticRegression(max_iter=200, random_state=0)
    model.fit(iris.data, iris.target)

    output = Path(__file__).resolve().parent.parent / "models" / "iris_model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
