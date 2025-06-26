"""Database loader for SQL databases with comprehensive connection support."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.shared.protocols import DatabaseLoaderProtocol, BatchDataLoaderProtocol


class DatabaseLoader(DatabaseLoaderProtocol, BatchDataLoaderProtocol):
    """Database loader supporting multiple SQL databases."""
    
    # Supported database types and their default ports
    SUPPORTED_DATABASES = {
        "postgresql": {"default_port": 5432, "driver": "psycopg2"},
        "mysql": {"default_port": 3306, "driver": "pymysql"},
        "sqlite": {"default_port": None, "driver": "pysqlite"},
        "mssql": {"default_port": 1433, "driver": "pyodbc"},
        "oracle": {"default_port": 1521, "driver": "cx_oracle"},
        "snowflake": {"default_port": 443, "driver": "snowflake"},
    }
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        query_timeout: int = 300,
        batch_size: int = 10000,
        use_connection_pooling: bool = True,
    ):
        """Initialize database loader.
        
        Args:
            connection_string: SQLAlchemy connection string
            engine_kwargs: Additional engine configuration
            query_timeout: Query timeout in seconds
            batch_size: Default batch size for large queries
            use_connection_pooling: Whether to use connection pooling
        """
        self.connection_string = connection_string
        self.engine_kwargs = engine_kwargs or {}
        self.query_timeout = query_timeout
        self.batch_size = batch_size
        self.use_connection_pooling = use_connection_pooling
        self.logger = logging.getLogger(__name__)
        
        # Connection management
        self._engine: Optional[sa.Engine] = None
        self._connection_cache: Dict[str, sa.Engine] = {}
    
    @property
    def supported_formats(self) -> List[str]:
        """Get supported database types."""
        return list(self.SUPPORTED_DATABASES.keys())
    
    def load(
        self, 
        source: Union[str, Path], 
        name: Optional[str] = None, 
        **kwargs: Any
    ) -> Dataset:
        """Load data using connection string as source.
        
        Args:
            source: Database connection string or table name
            name: Optional dataset name
            **kwargs: Additional loading options (query, table_name, etc.)
            
        Returns:
            Loaded dataset
        """
        # Determine if source is a connection string or table name
        if self._is_connection_string(str(source)):
            connection = str(source)
            table_name = kwargs.get("table_name")
            query = kwargs.get("query")
            
            if not table_name and not query:
                raise DataValidationError(
                    "Either table_name or query must be provided when using connection string"
                )
        else:
            connection = self.connection_string
            if not connection:
                raise DataValidationError(
                    "No connection string provided and source is not a connection string"
                )
            
            # Assume source is a table name
            table_name = str(source)
            query = kwargs.get("query")
        
        # Load data based on what's provided
        if query:
            return self.load_query(query, connection, name, **kwargs)
        else:
            return self.load_table(table_name, connection, name=name, **kwargs)
    
    def validate(self, source: Union[str, Path]) -> bool:
        """Validate database connection or table.
        
        Args:
            source: Connection string or table name
            
        Returns:
            True if valid
        """
        try:
            if self._is_connection_string(str(source)):
                # Test connection
                engine = self._get_engine(str(source))
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
            else:
                # Test if table exists
                if not self.connection_string:
                    return False
                
                engine = self._get_engine(self.connection_string)
                inspector = inspect(engine)
                return str(source) in inspector.get_table_names()
                
        except Exception:
            return False
    
    def load_query(
        self,
        query: str,
        connection: Union[str, sa.Engine],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load data using SQL query.
        
        Args:
            query: SQL query to execute
            connection: Database connection string or engine
            name: Optional dataset name
            **kwargs: Additional query parameters
            
        Returns:
            Query result as dataset
        """
        self.logger.info(f"Executing query: {query[:100]}...")
        
        try:
            engine = self._get_engine(connection)
            
            # Prepare query parameters
            params = kwargs.get("params", {})
            chunksize = kwargs.get("chunksize")
            
            # Execute query
            if chunksize:
                # Load in chunks
                chunks = []
                for chunk in pd.read_sql(
                    query, 
                    engine, 
                    params=params,
                    chunksize=chunksize
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_sql(query, engine, params=params)
            
            if df.empty:
                self.logger.warning("Query returned no results")
            
            # Create dataset
            dataset_name = name or "query_result"
            
            # Extract metadata
            metadata = {
                "source": "database_query",
                "loader": "DatabaseLoader",
                "query": query,
                "query_params": params,
                "database_type": self._get_database_type(engine.url),
                "row_count": len(df),
                "column_count": len(df.columns),
            }
            
            # Check for target column
            target_column = kwargs.get("target_column")
            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in query results",
                    available_columns=list(df.columns),
                )
            
            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata=metadata,
            )
            
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from query")
            return dataset
            
        except SQLAlchemyError as e:
            raise DataValidationError(
                f"Database query failed: {e}",
                query=query
            ) from e
        except Exception as e:
            raise DataValidationError(
                f"Failed to load data from query: {e}",
                query=query
            ) from e
    
    def load_table(
        self,
        table_name: str,
        connection: Union[str, sa.Engine],
        schema: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dataset:
        """Load entire table as dataset.
        
        Args:
            table_name: Name of the table
            connection: Database connection string or engine
            schema: Optional database schema
            name: Optional dataset name
            **kwargs: Additional parameters (columns, where_clause, etc.)
            
        Returns:
            Table data as dataset
        """
        self.logger.info(f"Loading table: {schema}.{table_name}" if schema else table_name)
        
        try:
            engine = self._get_engine(connection)
            
            # Build query
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            
            # Handle column selection
            columns = kwargs.get("columns")
            if columns:
                columns_str = ", ".join(columns)
            else:
                columns_str = "*"
            
            # Build WHERE clause
            where_clause = kwargs.get("where_clause", "")
            if where_clause and not where_clause.strip().upper().startswith("WHERE"):
                where_clause = f"WHERE {where_clause}"
            
            # Build ORDER BY clause
            order_by = kwargs.get("order_by", "")
            if order_by and not order_by.strip().upper().startswith("ORDER BY"):
                order_by = f"ORDER BY {order_by}"
            
            # Build LIMIT clause
            limit = kwargs.get("limit")
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            # Construct final query
            query = f"SELECT {columns_str} FROM {full_table_name} {where_clause} {order_by} {limit_clause}".strip()
            
            # Load data
            df = pd.read_sql(query, engine)
            
            if df.empty:
                self.logger.warning(f"Table {full_table_name} is empty")
            
            # Get table metadata
            table_info = self.get_table_info(table_name, engine, schema)
            
            # Create dataset
            dataset_name = name or table_name
            
            metadata = {
                "source": "database_table",
                "loader": "DatabaseLoader",
                "table_name": table_name,
                "schema": schema,
                "database_type": self._get_database_type(engine.url),
                "table_info": table_info,
                "query_used": query,
            }
            
            # Check for target column
            target_column = kwargs.get("target_column")
            if target_column and target_column not in df.columns:
                raise DataValidationError(
                    f"Target column '{target_column}' not found in table",
                    table_name=table_name,
                    available_columns=list(df.columns),
                )
            
            dataset = Dataset(
                name=dataset_name,
                data=df,
                target_column=target_column,
                metadata=metadata,
            )
            
            self.logger.info(f"Loaded table {full_table_name}: {len(df)} rows, {len(df.columns)} columns")
            return dataset
            
        except SQLAlchemyError as e:
            raise DataValidationError(
                f"Failed to load table {table_name}: {e}",
                table_name=table_name,
                schema=schema
            ) from e
        except Exception as e:
            raise DataValidationError(
                f"Failed to load table data: {e}",
                table_name=table_name
            ) from e
    
    def get_table_info(
        self, 
        table_name: str, 
        connection: Union[str, sa.Engine], 
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get information about a database table.
        
        Args:
            table_name: Name of the table
            connection: Database connection
            schema: Optional database schema
            
        Returns:
            Table metadata
        """
        try:
            engine = self._get_engine(connection)
            inspector = inspect(engine)
            
            # Get column information
            columns = inspector.get_columns(table_name, schema=schema)
            
            # Get primary keys
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
            primary_keys = pk_constraint.get("constrained_columns", [])
            
            # Get foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
            
            # Get indexes
            indexes = inspector.get_indexes(table_name, schema=schema)
            
            # Estimate row count (this might be expensive for large tables)
            full_table_name = f"{schema}.{table_name}" if schema else table_name
            try:
                row_count_query = f"SELECT COUNT(*) as row_count FROM {full_table_name}"
                with engine.connect() as conn:
                    result = conn.execute(text(row_count_query))
                    row_count = result.scalar()
            except Exception:
                row_count = "unknown"
            
            return {
                "table_name": table_name,
                "schema": schema,
                "columns": columns,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "row_count": row_count,
                "column_count": len(columns),
            }
            
        except Exception as e:
            return {
                "table_name": table_name,
                "schema": schema,
                "error": str(e),
            }
    
    def load_batch(
        self,
        source: Union[str, Path],
        batch_size: int,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[Dataset]:
        """Load data in batches.
        
        Args:
            source: SQL query or table name
            batch_size: Number of rows per batch
            name: Optional name prefix
            **kwargs: Additional options
            
        Yields:
            Dataset batches
        """
        connection = kwargs.get("connection", self.connection_string)
        if not connection:
            raise DataValidationError("No database connection provided")
        
        engine = self._get_engine(connection)
        
        # Determine if source is a query or table name
        source_str = str(source)
        is_query = any(keyword in source_str.upper() for keyword in ["SELECT", "FROM", "WHERE"])
        
        if is_query:
            query = source_str
        else:
            # Build query for table
            schema = kwargs.get("schema")
            full_table_name = f"{schema}.{source_str}" if schema else source_str
            
            columns = kwargs.get("columns", "*")
            if isinstance(columns, list):
                columns = ", ".join(columns)
            
            where_clause = kwargs.get("where_clause", "")
            if where_clause and not where_clause.strip().upper().startswith("WHERE"):
                where_clause = f"WHERE {where_clause}"
            
            query = f"SELECT {columns} FROM {full_table_name} {where_clause}".strip()
        
        try:
            dataset_name = name or "batch_query"
            
            # Use pandas chunking for batch loading
            chunk_iter = pd.read_sql(
                query,
                engine,
                chunksize=batch_size,
                params=kwargs.get("params", {})
            )
            
            for batch_idx, chunk in enumerate(chunk_iter):
                if chunk.empty:
                    continue
                
                batch_dataset = Dataset(
                    name=f"{dataset_name}_batch_{batch_idx}",
                    data=chunk,
                    target_column=kwargs.get("target_column"),
                    metadata={
                        "source": "database_batch",
                        "loader": "DatabaseLoader",
                        "batch_index": batch_idx,
                        "batch_size": len(chunk),
                        "query": query,
                        "is_batch": True,
                    },
                )
                
                yield batch_dataset
                
        except Exception as e:
            raise DataValidationError(
                f"Failed to load data in batches: {e}",
                query=query
            ) from e
    
    def estimate_size(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Estimate the size of a database query or table.
        
        Args:
            source: SQL query or table name
            
        Returns:
            Size information
        """
        try:
            connection = self.connection_string
            if not connection:
                return {"error": "No database connection provided"}
            
            engine = self._get_engine(connection)
            source_str = str(source)
            
            # Check if it's a query or table name
            is_query = any(keyword in source_str.upper() for keyword in ["SELECT", "FROM", "WHERE"])
            
            if is_query:
                # For queries, we can't easily estimate without running
                return {
                    "type": "query",
                    "estimated_rows": "unknown",
                    "query": source_str,
                    "note": "Run query to get actual size"
                }
            else:
                # For tables, get metadata
                table_info = self.get_table_info(source_str, engine)
                
                # Estimate memory usage (rough)
                row_count = table_info.get("row_count", 0)
                column_count = table_info.get("column_count", 0)
                
                if isinstance(row_count, int) and row_count > 0:
                    # Rough estimate: 50 bytes per cell on average
                    estimated_memory_mb = (row_count * column_count * 50) / (1024 * 1024)
                else:
                    estimated_memory_mb = "unknown"
                
                return {
                    "type": "table",
                    "table_name": source_str,
                    "estimated_rows": row_count,
                    "columns": column_count,
                    "estimated_memory_mb": estimated_memory_mb,
                    "table_info": table_info,
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def list_tables(
        self, 
        connection: Optional[Union[str, sa.Engine]] = None,
        schema: Optional[str] = None
    ) -> List[str]:
        """List all tables in the database.
        
        Args:
            connection: Database connection (uses default if None)
            schema: Optional schema to filter by
            
        Returns:
            List of table names
        """
        engine = self._get_engine(connection or self.connection_string)
        inspector = inspect(engine)
        
        if schema:
            return inspector.get_table_names(schema=schema)
        else:
            return inspector.get_table_names()
    
    def list_schemas(self, connection: Optional[Union[str, sa.Engine]] = None) -> List[str]:
        """List all schemas in the database.
        
        Args:
            connection: Database connection (uses default if None)
            
        Returns:
            List of schema names
        """
        engine = self._get_engine(connection or self.connection_string)
        inspector = inspect(engine)
        
        try:
            return inspector.get_schema_names()
        except Exception:
            # Some databases don't support schema listing
            return []
    
    def close_connections(self) -> None:
        """Close all cached connections."""
        for engine in self._connection_cache.values():
            engine.dispose()
        self._connection_cache.clear()
        
        if self._engine:
            self._engine.dispose()
            self._engine = None
    
    @contextmanager
    def connection(self, connection_string: Optional[str] = None):
        """Context manager for database connections.
        
        Args:
            connection_string: Optional specific connection string
            
        Yields:
            SQLAlchemy connection
        """
        engine = self._get_engine(connection_string or self.connection_string)
        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def _get_engine(self, connection: Union[str, sa.Engine]) -> sa.Engine:
        """Get or create SQLAlchemy engine.
        
        Args:
            connection: Connection string or existing engine
            
        Returns:
            SQLAlchemy engine
        """
        if isinstance(connection, sa.Engine):
            return connection
        
        # Check cache
        if connection in self._connection_cache:
            return self._connection_cache[connection]
        
        # Create new engine
        engine_kwargs = self.engine_kwargs.copy()
        
        if self.use_connection_pooling:
            engine_kwargs.setdefault("pool_size", 5)
            engine_kwargs.setdefault("max_overflow", 10)
            engine_kwargs.setdefault("pool_timeout", 30)
        
        try:
            engine = create_engine(connection, **engine_kwargs)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._connection_cache[connection] = engine
            return engine
            
        except Exception as e:
            raise DataValidationError(
                f"Failed to create database connection: {e}",
                connection_string=connection
            ) from e
    
    def _is_connection_string(self, source: str) -> bool:
        """Check if string is a database connection string."""
        # Simple heuristic: contains database schema
        db_schemes = ["postgresql", "mysql", "sqlite", "mssql", "oracle", "snowflake"]
        return any(source.startswith(f"{scheme}://") for scheme in db_schemes)
    
    def _get_database_type(self, url: sa.engine.URL) -> str:
        """Extract database type from SQLAlchemy URL."""
        return url.drivername.split("+")[0]
    
    @classmethod
    def create_connection_string(
        cls,
        database_type: str,
        username: str,
        password: str,
        host: str,
        database: str,
        port: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Create a connection string for the specified database type.
        
        Args:
            database_type: Type of database (postgresql, mysql, etc.)
            username: Database username
            password: Database password
            host: Database host
            database: Database name
            port: Database port (uses default if None)
            **kwargs: Additional connection parameters
            
        Returns:
            SQLAlchemy connection string
        """
        if database_type not in cls.SUPPORTED_DATABASES:
            raise ValueError(f"Unsupported database type: {database_type}")
        
        db_info = cls.SUPPORTED_DATABASES[database_type]
        
        # Use default port if not specified
        if port is None:
            port = db_info["default_port"]
        
        # Handle SQLite specially (file-based)
        if database_type == "sqlite":
            return f"sqlite:///{database}"
        
        # Build connection string
        driver = db_info["driver"]
        connection_string = f"{database_type}+{driver}://{username}:{password}@{host}"
        
        if port:
            connection_string += f":{port}"
        
        connection_string += f"/{database}"
        
        # Add additional parameters
        if kwargs:
            params = "&".join(f"{k}={v}" for k, v in kwargs.items())
            connection_string += f"?{params}"
        
        return connection_string