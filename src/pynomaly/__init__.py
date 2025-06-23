"""Pynomaly: State-of-the-art Python anomaly detection package."""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "team@pynomaly.io"

from pynomaly.domain.entities import Anomaly, Detector
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate

__all__ = [
    "Anomaly",
    "Detector",
    "AnomalyScore",
    "ContaminationRate",
]