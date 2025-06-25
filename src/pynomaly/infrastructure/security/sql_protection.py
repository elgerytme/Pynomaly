"""SQL injection protection and query sanitization.

This module provides comprehensive protection against SQL injection attacks:
- Query parameter validation
- SQL statement analysis
- Safe query building
- Dynamic query sanitization
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import MetaData

logger = logging.getLogger(__name__)


class SQLInjectionError(Exception):
    """Raised when SQL injection is detected."""

    def __init__(
        self,
        message: str,
        query: str | None = None,
        parameters: dict | None = None,
    ):
        self.query = query
        self.parameters = parameters
        super().__init__(message)


class QueryType(str, Enum):
    """Types of SQL queries."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"
    UNKNOWN = "UNKNOWN"


class QueryAnalysisResult(BaseModel):
    """Result of SQL query analysis."""

    query_type: QueryType
    is_safe: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    detected_threats: list[str]
    safe_parameters: dict[str, Any]
    recommendations: list[str]


class SQLInjectionProtector:
    """Service for detecting and preventing SQL injection attacks."""

    def __init__(self):
        """Initialize SQL injection protector."""
        # Common SQL injection patterns
        self.injection_patterns = [
            # Classic injection patterns
            (r"'\s*(?:OR|AND)\s*'.*?'", "Boolean-based injection"),
            (r"'\s*(?:OR|AND)\s*\d+\s*=\s*\d+", "Tautology injection"),
            (r"'\s*(?:OR|AND)\s*\d+\s*<>\s*\d+", "Inequality injection"),
            (r"'\s*(?:OR|AND)\s*'.*?'\s*=\s*'.*?'", "String comparison injection"),
            # UNION-based injection
            (r"UNION\s+(?:ALL\s+)?SELECT", "UNION-based injection"),
            # Time-based injection
            (r"(?:SLEEP|WAITFOR|DELAY)\s*\(", "Time-based injection"),
            (r"BENCHMARK\s*\(", "MySQL benchmark injection"),
            # Error-based injection
            (r"(?:CAST|CONVERT)\s*\(.*AS\s+(?:INT|CHAR)", "Type conversion injection"),
            (r"(?:EXTRACTVALUE|UPDATEXML)\s*\(", "XML injection"),
            # Blind injection
            (
                r"(?:SUBSTRING|SUBSTR|MID)\s*\(.*,\s*\d+\s*,\s*\d+\s*\)",
                "Substring injection",
            ),
            (r"(?:ASCII|ORD|CHAR)\s*\(", "Character-based injection"),
            # Comment-based evasion
            (r"--\s*$", "SQL comment"),
            (r"/\*.*?\*/", "Block comment"),
            (r"#.*$", "Hash comment"),
            # Function-based injection
            (
                r"(?:DATABASE|VERSION|USER|CURRENT_USER)\s*\(\s*\)",
                "Information gathering",
            ),
            (r"(?:LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)", "File operation"),
            # Stacked queries
            (r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)", "Stacked query"),
            # NoSQL injection (for document databases)
            (r"\$(?:where|regex|ne|gt|lt|gte|lte|in|nin)", "NoSQL injection"),
            (r"(?:this\.|db\.)", "JavaScript injection in NoSQL"),
        ]

        # Compile patterns for performance
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), description)
            for pattern, description in self.injection_patterns
        ]

        # Dangerous SQL keywords
        self.dangerous_keywords = {
            "DROP",
            "DELETE",
            "TRUNCATE",
            "ALTER",
            "CREATE",
            "EXEC",
            "EXECUTE",
            "xp_",
            "sp_",
            "OPENROWSET",
            "OPENDATASOURCE",
            "BULK",
            "LOAD_FILE",
            "INTO OUTFILE",
            "INTO DUMPFILE",
            "SCRIPT",
            "SHUTDOWN",
        }

        # Safe SQL functions (whitelist)
        self.safe_functions = {
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "UPPER",
            "LOWER",
            "TRIM",
            "LENGTH",
            "ROUND",
            "ABS",
            "CEIL",
            "FLOOR",
            "NOW",
            "CURDATE",
            "CURTIME",
            "DATE",
            "TIME",
            "YEAR",
            "MONTH",
            "DAY",
        }

    def analyze_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> QueryAnalysisResult:
        """Analyze SQL query for injection risks.

        Args:
            query: SQL query to analyze
            parameters: Query parameters

        Returns:
            Analysis result with risk assessment
        """
        detected_threats = []
        risk_level = "LOW"
        recommendations = []

        # Determine query type
        query_type = self._determine_query_type(query)

        # Check for injection patterns
        for pattern, description in self.compiled_patterns:
            if pattern.search(query):
                detected_threats.append(description)
                risk_level = "HIGH"

        # Check for dangerous keywords
        query_upper = query.upper()
        for keyword in self.dangerous_keywords:
            if keyword in query_upper:
                detected_threats.append(f"Dangerous keyword: {keyword}")
                risk_level = "CRITICAL"

        # Analyze parameters
        safe_parameters = {}
        if parameters:
            safe_parameters, param_threats = self._analyze_parameters(parameters)
            detected_threats.extend(param_threats)
            if param_threats:
                risk_level = max(
                    risk_level,
                    "MEDIUM",
                    key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x),
                )

        # Generate recommendations
        if detected_threats:
            recommendations.extend(
                [
                    "Use parameterized queries with bound parameters",
                    "Validate and sanitize all user inputs",
                    "Implement input length limits",
                    "Use stored procedures where possible",
                    "Apply principle of least privilege to database connections",
                ]
            )

        # Additional checks based on query type
        if query_type in [QueryType.DELETE, QueryType.DROP, QueryType.ALTER]:
            if "WHERE" not in query_upper and query_type == QueryType.DELETE:
                detected_threats.append("DELETE without WHERE clause")
                risk_level = "HIGH"
                recommendations.append("Always use WHERE clause with DELETE statements")

        is_safe = not detected_threats or risk_level == "LOW"

        return QueryAnalysisResult(
            query_type=query_type,
            is_safe=is_safe,
            risk_level=risk_level,
            detected_threats=detected_threats,
            safe_parameters=safe_parameters,
            recommendations=recommendations,
        )

    def _determine_query_type(self, query: str) -> QueryType:
        """Determine the type of SQL query."""
        query_clean = query.strip().upper()

        if query_clean.startswith("SELECT"):
            return QueryType.SELECT
        elif query_clean.startswith("INSERT"):
            return QueryType.INSERT
        elif query_clean.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_clean.startswith("DELETE"):
            return QueryType.DELETE
        elif query_clean.startswith("CREATE"):
            return QueryType.CREATE
        elif query_clean.startswith("DROP"):
            return QueryType.DROP
        elif query_clean.startswith("ALTER"):
            return QueryType.ALTER
        else:
            return QueryType.UNKNOWN

    def _analyze_parameters(
        self, parameters: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Analyze query parameters for injection risks."""
        safe_parameters = {}
        threats = []

        for key, value in parameters.items():
            if isinstance(value, str):
                # Check parameter value for injection
                for pattern, description in self.compiled_patterns:
                    if pattern.search(value):
                        threats.append(
                            f"Parameter '{key}' contains potential injection: {description}"
                        )
                        continue

                # Check for suspicious characters
                if any(char in value for char in ["'", '"', ";", "--", "/*", "*/"]):
                    threats.append(f"Parameter '{key}' contains suspicious characters")

                safe_parameters[key] = value
            else:
                safe_parameters[key] = value

        return safe_parameters, threats

    def validate_query_safety(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> bool:
        """Quick validation of query safety.

        Args:
            query: SQL query
            parameters: Query parameters

        Returns:
            True if query appears safe

        Raises:
            SQLInjectionError: If injection is detected
        """
        analysis = self.analyze_query(query, parameters)

        if analysis.risk_level in ["HIGH", "CRITICAL"]:
            raise SQLInjectionError(
                f"SQL injection detected: {', '.join(analysis.detected_threats)}",
                query=query,
                parameters=parameters,
            )

        return analysis.is_safe


class QuerySanitizer:
    """Service for sanitizing SQL queries and parameters."""

    def __init__(self):
        """Initialize query sanitizer."""
        self.protector = SQLInjectionProtector()

    def sanitize_query(self, query: str) -> str:
        """Sanitize SQL query by removing dangerous elements.

        Args:
            query: Query to sanitize

        Returns:
            Sanitized query
        """
        # Remove SQL comments
        query = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
        query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
        query = re.sub(r"#.*$", "", query, flags=re.MULTILINE)

        # Remove suspicious function calls
        dangerous_functions = [
            "xp_cmdshell",
            "sp_execute",
            "OPENROWSET",
            "OPENDATASOURCE",
            "LOAD_FILE",
            "INTO OUTFILE",
            "INTO DUMPFILE",
        ]

        for func in dangerous_functions:
            query = re.sub(rf"\b{func}\b", "", query, flags=re.IGNORECASE)

        # Remove stacked queries (multiple statements)
        statements = query.split(";")
        if len(statements) > 1:
            # Keep only the first statement
            query = statements[0]

        return query.strip()

    def sanitize_parameter(self, value: Any, param_name: str = "") -> Any:
        """Sanitize a query parameter.

        Args:
            value: Parameter value
            param_name: Parameter name for error reporting

        Returns:
            Sanitized parameter value

        Raises:
            SQLInjectionError: If parameter is unsafe
        """
        if not isinstance(value, str):
            return value

        # Check for injection patterns
        analysis = self.protector._analyze_parameters({param_name: value})
        if analysis[1]:  # If threats detected
            raise SQLInjectionError(
                f"Unsafe parameter '{param_name}': {', '.join(analysis[1])}",
                parameters={param_name: value},
            )

        # Basic sanitization
        # Escape single quotes
        value = value.replace("'", "''")

        # Remove null bytes
        value = value.replace("\x00", "")

        return value


class SafeQueryBuilder:
    """Builder for constructing safe SQL queries."""

    def __init__(self, metadata: MetaData | None = None):
        """Initialize safe query builder.

        Args:
            metadata: SQLAlchemy metadata for table definitions
        """
        self.metadata = metadata
        self.sanitizer = QuerySanitizer()
        self.protector = SQLInjectionProtector()

    def build_select(
        self,
        table_name: str,
        columns: list[str] | None = None,
        where_conditions: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a safe SELECT query.

        Args:
            table_name: Table name
            columns: Columns to select (None = all)
            where_conditions: WHERE conditions as dict
            order_by: ORDER BY columns
            limit: LIMIT value

        Returns:
            Tuple of (query, parameters)
        """
        # Validate table name
        safe_table = self._validate_identifier(table_name)

        # Build column list
        if columns:
            safe_columns = [self._validate_identifier(col) for col in columns]
            column_list = ", ".join(safe_columns)
        else:
            column_list = "*"

        # Start building query
        query_parts = [f"SELECT {column_list} FROM {safe_table}"]
        parameters = {}

        # Add WHERE clause
        if where_conditions:
            where_clause, where_params = self._build_where_clause(where_conditions)
            query_parts.append(f"WHERE {where_clause}")
            parameters.update(where_params)

        # Add ORDER BY
        if order_by:
            safe_order_cols = [self._validate_identifier(col) for col in order_by]
            query_parts.append(f"ORDER BY {', '.join(safe_order_cols)}")

        # Add LIMIT
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise SQLInjectionError("Invalid LIMIT value")
            query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)

        # Final safety check
        self.protector.validate_query_safety(query, parameters)

        return query, parameters

    def build_insert(
        self, table_name: str, data: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Build a safe INSERT query.

        Args:
            table_name: Table name
            data: Data to insert

        Returns:
            Tuple of (query, parameters)
        """
        safe_table = self._validate_identifier(table_name)

        if not data:
            raise SQLInjectionError("No data provided for INSERT")

        # Validate and sanitize column names and values
        safe_columns = []
        safe_values = []
        parameters = {}

        for i, (column, value) in enumerate(data.items()):
            safe_column = self._validate_identifier(column)
            safe_columns.append(safe_column)

            param_name = f"param_{i}"
            safe_values.append(f":{param_name}")
            parameters[param_name] = self.sanitizer.sanitize_parameter(value, column)

        query = f"INSERT INTO {safe_table} ({', '.join(safe_columns)}) VALUES ({', '.join(safe_values)})"

        # Final safety check
        self.protector.validate_query_safety(query, parameters)

        return query, parameters

    def _build_where_clause(
        self, conditions: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Build WHERE clause from conditions."""
        clauses = []
        parameters = {}

        for i, (column, value) in enumerate(conditions.items()):
            safe_column = self._validate_identifier(column)
            param_name = f"where_param_{i}"

            if isinstance(value, (list, tuple)):
                # Handle IN clause
                in_params = []
                for j, v in enumerate(value):
                    in_param = f"{param_name}_{j}"
                    in_params.append(f":{in_param}")
                    parameters[in_param] = self.sanitizer.sanitize_parameter(v, column)

                clauses.append(f"{safe_column} IN ({', '.join(in_params)})")
            else:
                # Handle equality
                clauses.append(f"{safe_column} = :{param_name}")
                parameters[param_name] = self.sanitizer.sanitize_parameter(
                    value, column
                )

        return " AND ".join(clauses), parameters

    def _validate_identifier(self, identifier: str) -> str:
        """Validate SQL identifier (table/column name).

        Args:
            identifier: SQL identifier

        Returns:
            Validated identifier

        Raises:
            SQLInjectionError: If identifier is invalid
        """
        if not identifier:
            raise SQLInjectionError("Empty identifier")

        # Check for valid identifier pattern
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            raise SQLInjectionError(f"Invalid identifier: {identifier}")

        # Check length
        if len(identifier) > 64:  # Common DB limit
            raise SQLInjectionError(f"Identifier too long: {identifier}")

        # Check against SQL keywords (basic list)
        sql_keywords = {
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TABLE",
            "INDEX",
            "VIEW",
            "PROCEDURE",
            "FUNCTION",
            "TRIGGER",
            "DATABASE",
            "SCHEMA",
            "USER",
            "ROLE",
            "GRANT",
            "REVOKE",
        }

        if identifier.upper() in sql_keywords:
            raise SQLInjectionError(f"Identifier cannot be SQL keyword: {identifier}")

        return identifier


# Global instances
_protector: SQLInjectionProtector | None = None
_sanitizer: QuerySanitizer | None = None


def get_sql_protector() -> SQLInjectionProtector:
    """Get global SQL injection protector."""
    global _protector
    if _protector is None:
        _protector = SQLInjectionProtector()
    return _protector


def get_query_sanitizer() -> QuerySanitizer:
    """Get global query sanitizer."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = QuerySanitizer()
    return _sanitizer
