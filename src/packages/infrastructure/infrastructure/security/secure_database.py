"""
Secure Database Operations

This module provides secure database operations to prevent SQL injection
and other database-related security vulnerabilities.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class SecureDatabaseManager:
    """
    Secure database manager with SQL injection prevention.

    This class provides secure database operations using parameterized queries
    and input validation to prevent SQL injection attacks.
    """

    def __init__(self, engine: Engine):
        """Initialize with database engine."""
        self.engine = engine
        self._query_cache = {}
        self._max_query_time = 30.0  # 30 seconds max query time

    @contextmanager
    def get_secure_session(self):
        """Get a secure database session with proper error handling."""
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            raise
        finally:
            session.close()

    def execute_secure_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        session: Session | None = None,
    ) -> Any:
        """
        Execute a parameterized query safely.

        Args:
            query: SQL query with named parameters
            parameters: Dictionary of parameters
            session: Optional existing session

        Returns:
            Query result

        Raises:
            ValueError: If query contains dangerous patterns
            SQLAlchemyError: If query execution fails
        """
        # Validate query for dangerous patterns
        self._validate_query_security(query)

        # Sanitize parameters
        safe_params = self._sanitize_parameters(parameters or {})

        start_time = time.time()

        try:
            if session:
                result = session.execute(text(query), safe_params)
            else:
                with self.get_secure_session() as sess:
                    result = sess.execute(text(query), safe_params)

            # Check query execution time
            execution_time = time.time() - start_time
            if execution_time > self._max_query_time:
                logger.warning(f"Slow query detected: {execution_time:.2f}s")

            return result

        except SQLAlchemyError as e:
            logger.error(f"Secure query execution failed: {str(e)}")
            raise

    def execute_secure_migration(
        self, migration_queries: list[tuple[str, dict[str, Any]]]
    ) -> None:
        """
        Execute migration queries securely.

        Args:
            migration_queries: List of (query, parameters) tuples
        """
        with self.get_secure_session() as session:
            for query, params in migration_queries:
                try:
                    self.execute_secure_query(query, params, session)
                    logger.info("Migration query executed successfully")
                except Exception as e:
                    logger.error(f"Migration query failed: {str(e)}")
                    raise

    def _validate_query_security(self, query: str) -> None:
        """
        Validate query for security issues.

        Args:
            query: SQL query to validate

        Raises:
            ValueError: If query contains dangerous patterns
        """
        # Normalize query for checking
        normalized_query = query.upper().strip()

        # Check for dynamic SQL construction (should use parameters)
        if any(pattern in query for pattern in ["%s", "%d", "{", "}"]):
            raise ValueError("Query contains dynamic SQL construction patterns")

        # Check for comment injection
        if "--" in query or "/*" in query or "*/" in query:
            raise ValueError("Query contains SQL comments")

        # Check for semicolon injection (multiple statements)
        if ";" in query.rstrip(";"):  # Allow trailing semicolon
            raise ValueError("Query contains multiple statements")

        # Log query for audit (without parameters)
        logger.debug(f"Executing secure query: {query[:100]}...")

    def _sanitize_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize query parameters.

        Args:
            parameters: Raw parameters

        Returns:
            Sanitized parameters
        """
        sanitized = {}

        for key, value in parameters.items():
            # Validate parameter names (alphanumeric + underscore only)
            if not key.replace("_", "").isalnum():
                raise ValueError(f"Invalid parameter name: {key}")

            # Sanitize parameter values
            sanitized[key] = self._sanitize_parameter_value(value)

        return sanitized

    def _sanitize_parameter_value(self, value: Any) -> Any:
        """Sanitize individual parameter value."""
        if value is None:
            return None
        elif isinstance(value, str):
            # Remove null bytes
            sanitized = value.replace("\x00", "")
            # Limit length
            if len(sanitized) > 10000:
                raise ValueError("Parameter value too long")
            return sanitized
        elif isinstance(value, (int, float)):
            # Check for reasonable bounds
            if isinstance(value, float):
                if value != value:  # NaN check
                    raise ValueError("NaN parameter value not allowed")
                if abs(value) == float("inf"):
                    raise ValueError("Infinite parameter value not allowed")
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_parameter_value(item) for item in value]
        else:
            # Convert other types to string and sanitize
            return self._sanitize_parameter_value(str(value))


class SecureMigrationManager:
    """
    Secure database migration manager.

    This class provides secure migration operations using parameterized queries
    to fix SQL injection vulnerabilities in the original migration code.
    """

    def __init__(self, db_manager: SecureDatabaseManager):
        """Initialize with secure database manager."""
        self.db_manager = db_manager

    def create_roles_securely(self, roles: list[str]) -> None:
        """
        Create roles securely using parameterized queries.

        Args:
            roles: List of role names to create
        """
        migration_queries = []

        for role in roles:
            # Validate role name
            if not role.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Invalid role name: {role}")

            # Use parameterized query
            query = "INSERT INTO roles (name) VALUES (:role_name) ON CONFLICT (name) DO NOTHING"
            parameters = {"role_name": role}
            migration_queries.append((query, parameters))

        self.db_manager.execute_secure_migration(migration_queries)

    def create_permissions_securely(self, permissions: list[dict[str, str]]) -> None:
        """
        Create permissions securely.

        Args:
            permissions: List of permission dictionaries
        """
        migration_queries = []

        for perm in permissions:
            # Validate permission structure
            required_fields = ["name", "description", "resource", "action"]
            if not all(field in perm for field in required_fields):
                raise ValueError(f"Invalid permission structure: {perm}")

            # Use parameterized query
            query = """
                INSERT INTO permissions (name, description, resource, action)
                VALUES (:name, :description, :resource, :action)
                ON CONFLICT (name) DO NOTHING
            """
            parameters = {
                "name": perm["name"],
                "description": perm["description"],
                "resource": perm["resource"],
                "action": perm["action"],
            }
            migration_queries.append((query, parameters))

        self.db_manager.execute_secure_migration(migration_queries)

    def assign_role_permissions_securely(
        self, role_permissions: list[tuple[str, str]]
    ) -> None:
        """
        Assign permissions to roles securely.

        Args:
            role_permissions: List of (role_name, permission_name) tuples
        """
        migration_queries = []

        for role_name, permission_name in role_permissions:
            # Use parameterized query with subqueries for IDs
            query = """
                INSERT INTO role_permissions (role_id, permission_id)
                SELECT r.id, p.id
                FROM roles r, permissions p
                WHERE r.name = :role_name AND p.name = :permission_name
                ON CONFLICT DO NOTHING
            """
            parameters = {"role_name": role_name, "permission_name": permission_name}
            migration_queries.append((query, parameters))

        self.db_manager.execute_secure_migration(migration_queries)

    def create_default_admin_securely(
        self, username: str, email: str, password_hash: str
    ) -> None:
        """
        Create default admin user securely.

        Args:
            username: Admin username
            email: Admin email
            password_hash: Pre-hashed password
        """
        # Validate inputs
        if not username.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid username format")

        if "@" not in email or len(email) > 255:
            raise ValueError("Invalid email format")

        migration_queries = [
            # Create user
            (
                """
                INSERT INTO users (username, email, password_hash, is_active, is_verified)
                VALUES (:username, :email, :password_hash, true, true)
                ON CONFLICT (username) DO NOTHING
                """,
                {"username": username, "email": email, "password_hash": password_hash},
            ),
            # Assign admin role
            (
                """
                INSERT INTO user_roles (user_id, role_id)
                SELECT u.id, r.id
                FROM users u, roles r
                WHERE u.username = :username AND r.name = 'admin'
                ON CONFLICT DO NOTHING
                """,
                {"username": username},
            ),
        ]

        self.db_manager.execute_secure_migration(migration_queries)


class DatabaseQueryAuditor:
    """
    Database query auditing and monitoring.

    This class provides auditing capabilities for database operations
    to detect potential security issues.
    """

    def __init__(self):
        """Initialize auditor."""
        self.suspicious_patterns = [
            "UNION SELECT",
            "DROP TABLE",
            "DELETE FROM",
            "UPDATE SET",
            "ALTER TABLE",
            "CREATE USER",
            "GRANT ALL",
            "REVOKE",
            "--",
            "/*",
            "xp_cmdshell",
            "sp_executesql",
        ]
        self.query_log = []
        self.max_log_size = 1000

    def audit_query(
        self, query: str, parameters: dict[str, Any], user_id: str | None = None
    ) -> bool:
        """
        Audit a database query for security issues.

        Args:
            query: SQL query
            parameters: Query parameters
            user_id: User executing the query

        Returns:
            True if query is safe, False if suspicious
        """
        timestamp = time.time()

        # Check for suspicious patterns
        is_suspicious = any(
            pattern.upper() in query.upper() for pattern in self.suspicious_patterns
        )

        # Log the query
        log_entry = {
            "timestamp": timestamp,
            "query": query[:200],  # Truncate for logging
            "parameters": list(parameters.keys()) if parameters else [],
            "user_id": user_id,
            "is_suspicious": is_suspicious,
        }

        self.query_log.append(log_entry)

        # Maintain log size
        if len(self.query_log) > self.max_log_size:
            self.query_log = self.query_log[-self.max_log_size :]

        # Alert on suspicious queries
        if is_suspicious:
            logger.warning(f"Suspicious database query detected: {query[:100]}...")

        return not is_suspicious

    def get_recent_suspicious_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent suspicious queries."""
        suspicious = [entry for entry in self.query_log if entry["is_suspicious"]]
        return suspicious[-limit:]

    def get_query_statistics(self) -> dict[str, Any]:
        """Get query execution statistics."""
        total_queries = len(self.query_log)
        suspicious_queries = len(
            [entry for entry in self.query_log if entry["is_suspicious"]]
        )

        return {
            "total_queries": total_queries,
            "suspicious_queries": suspicious_queries,
            "suspicious_percentage": (suspicious_queries / total_queries * 100)
            if total_queries > 0
            else 0,
        }


# Global instances
_secure_db_manager = None
_migration_manager = None
_query_auditor = DatabaseQueryAuditor()


def get_secure_db_manager(engine: Engine) -> SecureDatabaseManager:
    """Get secure database manager instance."""
    global _secure_db_manager
    if _secure_db_manager is None:
        _secure_db_manager = SecureDatabaseManager(engine)
    return _secure_db_manager


def get_migration_manager(db_manager: SecureDatabaseManager) -> SecureMigrationManager:
    """Get secure migration manager instance."""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = SecureMigrationManager(db_manager)
    return _migration_manager


def get_query_auditor() -> DatabaseQueryAuditor:
    """Get query auditor instance."""
    return _query_auditor
