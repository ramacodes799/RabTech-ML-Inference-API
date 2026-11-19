
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_valid_prediction():
    response = client.post("/predict", json={
        "features": [
            20000, 2, 2, 1, 25,
            0, 0, 0, 0, 0, 0,
            1000, 1000, 1000, 1000, 1000, 1000,
            500, 500, 500, 500, 500, 500,
            6000, 3000
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 2

def test_invalid_features():
    response = client.post("/predict", json={"features": [1, 2]})
    assert response.status_code == 400

def test_missing_features():
    response = client.post("/predict", json={})
    assert response.status_code == 422


