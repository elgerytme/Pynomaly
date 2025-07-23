"""External service adapters and integrations.

This module provides adapters for external services, APIs, and third-party
integrations. It implements the Adapter pattern to decouple the application
from external dependencies.

Example usage:
    from infrastructure.adapters import HTTPClient, DatabaseAdapter
    
    http_client = HTTPClient()
    response = await http_client.get("https://api.example.com/users")
    
    db_adapter = DatabaseAdapter()
    result = await db_adapter.execute_query("SELECT * FROM users")
"""

from .http_client import HTTPClient
from .database_adapter import DatabaseAdapter
from .file_storage import FileStorageAdapter
from .email_adapter import EmailAdapter

__all__ = [
    "HTTPClient",
    "DatabaseAdapter",
    "FileStorageAdapter",
    "EmailAdapter"
]