from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SETOSA = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_setosa():
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    assert response.json() == {"predicted_class": "setosa"}


def test_missing_field_is_rejected():
    payload = {key: value for key, value in SETOSA.items() if key != "petal_width"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_out_of_range_value_is_rejected():
    payload = {**SETOSA, "sepal_length": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
