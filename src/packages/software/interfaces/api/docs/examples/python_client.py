"""
Pynomaly API Client Example - Python
"""

from typing import Any

import requests


class PynomaliClient:
    """Python client for Pynomaly API."""

    def __init__(self, base_url: str = "https://api.example.com"):
        self.base_url = base_url
        self.session = requests.Session()
        self.access_token = None

    def login(self, username: str, password: str) -> dict[str, Any]:
        """Authenticate and get JWT token."""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password},
        )
        response.raise_for_status()

        token_data = response.json()
        self.access_token = token_data["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self.access_token}"})

        return token_data

    def detect_anomalies(
        self,
        data: list[float],
        algorithm: str = "isolation_forest",
        parameters: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Detect anomalies in data."""
        if parameters is None:
            parameters = {"contamination": 0.1}

        response = self.session.post(
            f"{self.base_url}/api/v1/detection/detect",
            json={"data": data, "algorithm": algorithm, "parameters": parameters},
        )
        response.raise_for_status()
        return response.json()

    def train_model(
        self,
        training_data: str,
        algorithm: str,
        parameters: dict[str, Any] = None,
        model_name: str = None,
    ) -> dict[str, Any]:
        """Train a new anomaly detection model."""
        payload = {"training_data": training_data, "algorithm": algorithm}

        if parameters:
            payload["parameters"] = parameters
        if model_name:
            payload["model_name"] = model_name

        response = self.session.post(
            f"{self.base_url}/api/v1/detection/train", json=payload
        )
        response.raise_for_status()
        return response.json()

    def get_health(self) -> dict[str, Any]:
        """Get system health status."""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    client = PynomaliClient()

    # Login
    client.login("your_username", "your_password")

    # Detect anomalies
    result = client.detect_anomalies([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])
    print(f"Detected anomalies: {result['anomalies']}")

    # Check health
    health = client.get_health()
    print(f"System status: {health['status']}")
