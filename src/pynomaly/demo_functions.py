"""
Demo functions for testing the advanced testing framework.
"""


import numpy as np


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_positive(x: float, y: float) -> float:
    """Multiply two positive numbers."""
    if x <= 0 or y <= 0:
        raise ValueError("Both numbers must be positive")
    return x * y


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to have values between 0 and 1."""
    if len(arr) == 0:
        return arr

    min_val = np.min(arr)
    max_val = np.max(arr)

    if min_val == max_val:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)


def calculate_distance(point1: list[float], point2: list[float]) -> float:
    """Calculate Euclidean distance between two points."""
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimensions")

    return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2, strict=False)))


def detect_anomaly_simple(values: list[float], threshold: float = 2.0) -> list[bool]:
    """Simple anomaly detection based on standard deviation."""
    if not values:
        return []

    mean_val = np.mean(values)
    std_val = np.std(values)

    if std_val == 0:
        return [False] * len(values)

    return [abs(value - mean_val) > threshold * std_val for value in values]


def clean_data(data: list[float | None]) -> list[float]:
    """Clean data by removing None values."""
    return [x for x in data if x is not None]


def score_prediction(prediction: float, actual: float) -> float:
    """Calculate a simple prediction score (0-1 range)."""
    error = abs(prediction - actual)
    max_error = max(abs(prediction), abs(actual), 1.0)
    return max(0.0, 1.0 - (error / max_error))
