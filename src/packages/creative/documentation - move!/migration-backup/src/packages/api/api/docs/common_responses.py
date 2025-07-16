"""Common API response models and configurations."""

from typing import Any

# Common response status codes and descriptions
COMMON_RESPONSES: dict[int, dict[str, Any]] = {
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "example": "Invalid request parameters",
                        }
                    },
                }
            }
        },
    },
    401: {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "example": "Authentication required",
                        }
                    },
                }
            }
        },
    },
    403: {
        "description": "Forbidden",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "string",
                            "example": "Insufficient permissions",
                        }
                    },
                }
            }
        },
    },
    404: {
        "description": "Not Found",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Resource not found"}
                    },
                }
            }
        },
    },
    422: {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "loc": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "msg": {"type": "string"},
                                    "type": {"type": "string"},
                                },
                            },
                        }
                    },
                }
            }
        },
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "detail": {"type": "string", "example": "Internal server error"}
                    },
                }
            }
        },
    },
}

# Endpoint metadata for documentation
ENDPOINT_METADATA = {
    "health": {
        "summary": "Health Check",
        "description": "Check the health status of the API and its dependencies",
        "tags": ["Health"],
    },
    "explainability": {
        "summary": "Explainability Operations",
        "description": "Operations for generating and managing model explanations",
        "tags": ["Explainability"],
    },
}


def configure_api_docs(app, **kwargs):
    """Configure API documentation settings."""
    # This is a placeholder function for API documentation configuration
    # The actual configuration is handled by FastAPI's automatic OpenAPI generation
    pass
