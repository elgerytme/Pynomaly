# tests/api/test_end_to_end.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import json

@pytest.fixture
def client():
    from pynomaly.presentation.api.app import create_app
    app = create_app()
    return TestClient(app)

@pytest.fixture
def sample_data():
    return {"data": [[0.1, 0.2], [10, 10]]}

@pytest.fixture
def sample_dataset(sample_data, tmp_path):
    filepath = tmp_path / "sample_data.csv"
    with open(filepath, "w") as f:
        f.write("feature1,feature2\n")
        for row in sample_data["data"]:
            f.write(f"{row[0]},{row[1]}\n")
    return str(filepath)

def test_api_flow(client, sample_dataset):
    # Upload dataset
    with open(sample_dataset, "rb") as file:
        response = client.post("/api/v1/datasets/upload", files={"file": file}, data={"name": "test_data"})
    assert response.status_code == 201
    dataset_id = response.json()["dataset_id"]

    # Create detector
    response = client.post("/api/v1/detectors/create", json={"name": "test_detector", "algorithm": "IsolationForest"})
    assert response.status_code == 201
    detector_id = response.json()["detector_id"]

    # Train detector
    train_request = {"detector_id": detector_id, "dataset_id": dataset_id}
    response = client.post("/api/v1/detection/train", json=train_request)
    assert response.status_code == 200

    # Detect anomalies
    detect_request = {"detector_id": detector_id, "dataset_id": dataset_id}
    response = client.post("/api/v1/detection/predict", json=detect_request)
    assert response.status_code == 200
    results = response.json()
    assert results["n_anomalies"] == 1
