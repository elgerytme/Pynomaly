"""
Advanced Machine Learning Capabilities for Pynomaly Detection
=============================================================

This module provides cutting-edge ML capabilities including:
- AutoML pipeline for automatic algorithm selection
- Deep learning integration with TensorFlow/PyTorch
- Online learning and model adaptation
- Federated learning for distributed training
- Enhanced explainable AI capabilities
"""

from .automl import AutoMLPipeline, AutoMLOptimizer
from .deep_learning import DeepAnomalyDetector, NeuralNetworkEnsemble
from .online_learning import OnlineLearningDetector, StreamingModelAdapter
from .federated_learning import FederatedLearningCoordinator, FederatedDetector
from .explainable_ai import AdvancedExplainer, CausalAnalyzer

__all__ = [
    # AutoML
    'AutoMLPipeline',
    'AutoMLOptimizer',
    
    # Deep Learning
    'DeepAnomalyDetector', 
    'NeuralNetworkEnsemble',
    
    # Online Learning
    'OnlineLearningDetector',
    'StreamingModelAdapter',
    
    # Federated Learning
    'FederatedLearningCoordinator',
    'FederatedDetector',
    
    # Explainable AI
    'AdvancedExplainer',
    'CausalAnalyzer'
]

# Advanced ML Version
__version__ = "1.0.0"

def get_advanced_ml_info():
    """Get advanced ML capabilities information."""
    return {
        "version": __version__,
        "capabilities": {
            "automl": "Automated machine learning pipeline",
            "deep_learning": "Neural network-based anomaly detection",
            "online_learning": "Real-time model adaptation",
            "federated_learning": "Distributed training across multiple sites",
            "explainable_ai": "Advanced model interpretation and causality"
        },
        "supported_frameworks": [
            "scikit-learn", "tensorflow", "pytorch", "xgboost", 
            "lightgbm", "catboost", "optuna", "ray"
        ]
    }