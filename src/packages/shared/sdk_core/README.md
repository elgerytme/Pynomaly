# Core SDK Package

This package provides the foundational components for all monorepo client SDKs, including authentication, HTTP client functionality, error handling, and common utilities.

## Features

- 🔐 **JWT Authentication**: Automatic token management with refresh
- 🌐 **HTTP Client**: Async/sync HTTP client with retry logic and rate limiting
- 🛡️ **Error Handling**: Comprehensive exception hierarchy
- 📊 **Response Models**: Pydantic models for all API responses
- ⚡ **Performance**: Connection pooling and request optimization
- 🔧 **Configuration**: Flexible configuration management
- 📝 **Logging**: Structured logging with request tracking

## Architecture

The SDK follows a modular architecture with clear separation of concerns:

```
sdk_core/
├── auth/              # Authentication components
├── client/            # HTTP client implementation  
├── models/            # Pydantic response models
├── exceptions/        # Exception hierarchy
├── config/            # Configuration management
├── utils/             # Common utilities
└── types/             # Type definitions
```

## Usage

```python
from sdk_core import BaseClient, JWTAuth

client = BaseClient(
    base_url="https://api.platform.com",
    auth=JWTAuth(api_key="your-api-key")
)

# The core client handles authentication, retries, and error handling
response = await client.get("/api/v1/health")
```

## Installation

```bash
pip install -e .
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy src/
```