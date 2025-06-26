"""OpenAPI schema examples for API documentation."""

from __future__ import annotations

from typing import Any


class SchemaExamples:
    """Comprehensive schema examples for API documentation."""

    @staticmethod
    def dataset_upload_request() -> dict[str, Any]:
        """Dataset upload request example."""
        return {
            "summary": "Upload a CSV dataset",
            "description": "Upload a CSV file containing numerical data for anomaly detection",
            "value": {
                "name": "Network Traffic Data",
                "description": "Daily network traffic measurements with potential anomalies",
                "source": "production_monitoring",
                "tags": ["network", "traffic", "production"],
                "metadata": {
                    "collection_date": "2024-12-25",
                    "sensor_type": "network_monitor",
                    "location": "datacenter_1"
                }
            }
        }

    @staticmethod
    def dataset_response() -> dict[str, Any]:
        """Dataset response example."""
        return {
            "summary": "Dataset information",
            "description": "Complete dataset metadata and statistics",
            "value": {
                "success": True,
                "data": {
                    "id": "dataset_123",
                    "name": "Network Traffic Data",
                    "description": "Daily network traffic measurements",
                    "rows": 10000,
                    "columns": 15,
                    "features": [
                        "timestamp", "bytes_in", "bytes_out", "packets_in",
                        "packets_out", "connections", "bandwidth_util"
                    ],
                    "statistics": {
                        "numerical_features": 14,
                        "categorical_features": 1,
                        "missing_values": 23,
                        "duplicate_rows": 0
                    },
                    "created_at": "2024-12-25T10:00:00Z",
                    "updated_at": "2024-12-25T10:00:00Z",
                    "file_size": 2457600,
                    "format": "csv"
                },
                "timestamp": "2024-12-25T10:30:00Z"
            }
        }

    @staticmethod
    def detector_create_request() -> dict[str, Any]:
        """Detector creation request example."""
        return {
            "summary": "Create Isolation Forest detector",
            "description": "Configure an Isolation Forest algorithm for anomaly detection",
            "value": {
                "name": "Production Traffic Detector",
                "algorithm": "IsolationForest",
                "parameters": {
                    "n_estimators": 100,
                    "contamination": 0.1,
                    "random_state": 42,
                    "max_features": 1.0
                },
                "description": "Isolation Forest optimized for network traffic anomalies",
                "tags": ["production", "network", "isolation_forest"]
            }
        }

    @staticmethod
    def detector_response() -> dict[str, Any]:
        """Detector response example."""
        return {
            "summary": "Detector configuration",
            "description": "Complete detector configuration and metadata",
            "value": {
                "success": True,
                "data": {
                    "id": "detector_456",
                    "name": "Production Traffic Detector",
                    "algorithm": "IsolationForest",
                    "parameters": {
                        "n_estimators": 100,
                        "contamination": 0.1,
                        "random_state": 42,
                        "max_features": 1.0
                    },
                    "status": "configured",
                    "created_at": "2024-12-25T10:15:00Z",
                    "updated_at": "2024-12-25T10:15:00Z",
                    "training_history": [],
                    "performance_metrics": None
                },
                "timestamp": "2024-12-25T10:30:00Z"
            }
        }

    @staticmethod
    def detection_train_request() -> dict[str, Any]:
        """Detection training request example."""
        return {
            "summary": "Train detector on dataset",
            "description": "Train the configured detector using uploaded dataset",
            "value": {
                "detector_id": "detector_456",
                "dataset_id": "dataset_123",
                "features": [
                    "bytes_in", "bytes_out", "packets_in", "packets_out",
                    "connections", "bandwidth_util"
                ],
                "validation_split": 0.2,
                "preprocessing": {
                    "normalize": True,
                    "handle_missing": "drop",
                    "feature_selection": "auto"
                }
            }
        }

    @staticmethod
    def detection_train_response() -> dict[str, Any]:
        """Detection training response example."""
        return {
            "summary": "Training initiated",
            "description": "Training task started successfully",
            "value": {
                "success": True,
                "data": {
                    "task_id": "train_task_789",
                    "status": "running",
                    "detector_id": "detector_456",
                    "dataset_id": "dataset_123",
                    "progress": 0.0,
                    "estimated_completion": "2024-12-25T10:35:00Z",
                    "started_at": "2024-12-25T10:30:00Z"
                },
                "timestamp": "2024-12-25T10:30:00Z"
            }
        }

    @staticmethod
    def detection_predict_request() -> dict[str, Any]:
        """Detection prediction request example."""
        return {
            "summary": "Detect anomalies in new data",
            "description": "Use trained detector to find anomalies in new data points",
            "value": {
                "detector_id": "detector_456",
                "data": [
                    {
                        "bytes_in": 15000000,
                        "bytes_out": 8000000,
                        "packets_in": 12000,
                        "packets_out": 9500,
                        "connections": 450,
                        "bandwidth_util": 0.85
                    },
                    {
                        "bytes_in": 25000000,
                        "bytes_out": 12000000,
                        "packets_in": 20000,
                        "packets_out": 15000,
                        "connections": 750,
                        "bandwidth_util": 0.95
                    }
                ],
                "threshold": 0.5,
                "explain": True
            }
        }

    @staticmethod
    def detection_predict_response() -> dict[str, Any]:
        """Detection prediction response example."""
        return {
            "summary": "Anomaly detection results",
            "description": "Results of anomaly detection with scores and explanations",
            "value": {
                "success": True,
                "data": {
                    "predictions": [
                        {
                            "index": 0,
                            "anomaly_score": 0.23,
                            "is_anomaly": False,
                            "confidence": 0.89,
                            "explanation": {
                                "feature_importance": {
                                    "bandwidth_util": 0.45,
                                    "bytes_in": 0.32,
                                    "connections": 0.23
                                },
                                "reason": "Normal traffic pattern within expected ranges"
                            }
                        },
                        {
                            "index": 1,
                            "anomaly_score": 0.87,
                            "is_anomaly": True,
                            "confidence": 0.94,
                            "explanation": {
                                "feature_importance": {
                                    "bandwidth_util": 0.67,
                                    "bytes_in": 0.45,
                                    "connections": 0.38
                                },
                                "reason": "Unusual high bandwidth utilization and connection count"
                            }
                        }
                    ],
                    "summary": {
                        "total_predictions": 2,
                        "anomalies_detected": 1,
                        "anomaly_rate": 0.5,
                        "avg_confidence": 0.915
                    }
                },
                "timestamp": "2024-12-25T10:32:00Z"
            }
        }

    @staticmethod
    def experiment_create_request() -> dict[str, Any]:
        """Experiment creation request example."""
        return {
            "summary": "Create new experiment",
            "description": "Set up a new experiment to track detector performance",
            "value": {
                "name": "Network Anomaly Detection Comparison",
                "description": "Compare multiple algorithms on network traffic data",
                "dataset_id": "dataset_123",
                "detectors": ["detector_456", "detector_789", "detector_101"],
                "metrics": ["precision", "recall", "f1_score", "auc"],
                "tags": ["comparison", "network", "production"]
            }
        }

    @staticmethod
    def user_login_request() -> dict[str, Any]:
        """User login request example."""
        return {
            "summary": "User authentication",
            "description": "Authenticate user and receive JWT token",
            "value": {
                "username": "admin",
                "password": "secure_password",
                "remember_me": True
            }
        }

    @staticmethod
    def user_login_response() -> dict[str, Any]:
        """User login response example."""
        return {
            "summary": "Authentication successful",
            "description": "JWT token and user information",
            "value": {
                "success": True,
                "data": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 3600,
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "user": {
                        "id": "user_123",
                        "username": "admin",
                        "email": "admin@company.com",
                        "roles": ["admin", "detector_manager"],
                        "permissions": ["read", "write", "admin"]
                    }
                },
                "timestamp": "2024-12-25T10:30:00Z"
            }
        }

    @staticmethod
    def health_check_response() -> dict[str, Any]:
        """Health check response example."""
        return {
            "summary": "System health status",
            "description": "Overall system health and service statuses",
            "value": {
                "status": "healthy",
                "version": "1.0.0",
                "environment": "production",
                "uptime": 86400.0,
                "services": {
                    "database": "healthy",
                    "cache": "healthy",
                    "storage": "healthy",
                    "auth_service": "healthy",
                    "ml_backends": "healthy"
                },
                "performance": {
                    "avg_response_time": 0.145,
                    "requests_per_second": 127.5,
                    "active_connections": 23
                },
                "timestamp": "2024-12-25T10:30:00Z"
            }
        }

    @staticmethod
    def error_response_example() -> dict[str, Any]:
        """Generic error response example."""
        return {
            "summary": "Error response",
            "description": "Standard error response format",
            "value": {
                "success": False,
                "error": "Resource not found",
                "error_code": "RESOURCE_NOT_FOUND",
                "details": {
                    "resource_type": "detector",
                    "resource_id": "detector_999",
                    "available_resources": ["detector_456", "detector_789"]
                },
                "timestamp": "2024-12-25T10:30:00Z",
                "request_id": "req_12345"
            }
        }

    @staticmethod
    def pagination_response_example() -> dict[str, Any]:
        """Paginated response example."""
        return {
            "summary": "Paginated results",
            "description": "Standard pagination response format",
            "value": {
                "success": True,
                "data": [
                    {"id": "dataset_1", "name": "Dataset 1", "rows": 1000},
                    {"id": "dataset_2", "name": "Dataset 2", "rows": 2500},
                    {"id": "dataset_3", "name": "Dataset 3", "rows": 750}
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 20,
                    "total_items": 157,
                    "total_pages": 8,
                    "has_next": True,
                    "has_previous": False
                },
                "timestamp": "2024-12-25T10:30:00Z"
            }
        }
