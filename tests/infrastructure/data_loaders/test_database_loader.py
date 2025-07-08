"""Test cases for database loader."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import sqlalchemy as sa
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError
from pynomaly.infrastructure.data_loaders.database_loader import DatabaseLoader


class TestDatabaseLoader:
    """Test cases for DatabaseLoader."""

    def test_init_default(self):
        """Test default initialization."""
        loader = DatabaseLoader()

        assert loader.connection_string is None
        assert loader.engine_kwargs == {}
        assert loader.query_timeout == 300
        assert loader.batch_size == 10000
        assert loader.use_connection_pooling is True
        assert loader._engine is None

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        engine_kwargs = {"pool_size": 10}
        loader = DatabaseLoader(
            connection_string="sqlite:///test.db",
            engine_kwargs=engine_kwargs,
            query_timeout=600,
            batch_size=5000,
            use_connection_pooling=False,
        )

        assert loader.connection_string == "sqlite:///test.db"
        assert loader.engine_kwargs == engine_kwargs
        assert loader.query_timeout == 600
        assert loader.batch_size == 5000
        assert loader.use_connection_pooling is False

    def test_supported_formats(self):
        """Test supported database formats."""
        loader = DatabaseLoader()

        expected_formats = [
            "postgresql",
            "mysql",
            "sqlite",
            "mssql",
            "oracle",
            "snowflake",
        ]
        assert loader.supported_formats == expected_formats

    def test_supported_databases_constant(self):
        """Test SUPPORTED_DATABASES constant."""
        expected_databases = {
            "postgresql": {"default_port": 5432, "driver": "psycopg2"},
            "mysql": {"default_port": 3306, "driver": "pymysql"},
            "sqlite": {"default_port": None, "driver": "pysqlite"},
            "mssql": {"default_port": 1433, "driver": "pyodbc"},
            "oracle": {"default_port": 1521, "driver": "cx_oracle"},
            "snowflake": {"default_port": 443, "driver": "snowflake"},
        }

        assert DatabaseLoader.SUPPORTED_DATABASES == expected_databases

    def test_is_connection_string_valid(self):
        """Test valid connection string detection."""
        loader = DatabaseLoader()

        assert loader._is_connection_string("postgresql://user:pass@host/db") is True
        assert loader._is_connection_string("mysql://user:pass@host/db") is True
        assert loader._is_connection_string("sqlite:///path/to/db.sqlite") is True
        assert loader._is_connection_string("mssql://user:pass@host/db") is True
        assert loader._is_connection_string("oracle://user:pass@host/db") is True
        assert loader._is_connection_string("snowflake://user:pass@host/db") is True

    def test_is_connection_string_invalid(self):
        """Test invalid connection string detection."""
        loader = DatabaseLoader()

        assert loader._is_connection_string("table_name") is False
        assert loader._is_connection_string("not_a_connection_string") is False
        assert loader._is_connection_string("http://example.com") is False

    @patch("sqlalchemy.create_engine")
    def test_get_engine_new_connection(self, mock_create_engine):
        """Test creating new engine for connection."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_connection.execute.return_value = Mock()
        mock_create_engine.return_value = mock_engine

        loader = DatabaseLoader(use_connection_pooling=True)

        result = loader._get_engine("sqlite:///test.db")

        assert result == mock_engine
        mock_create_engine.assert_called_once()

        # Check pooling configuration
        call_kwargs = mock_create_engine.call_args[1]
        assert call_kwargs["pool_size"] == 5
        assert call_kwargs["max_overflow"] == 10
        assert call_kwargs["pool_timeout"] == 30

    @patch("sqlalchemy.create_engine")
    def test_get_engine_cached_connection(self, mock_create_engine):
        """Test using cached engine."""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        loader = DatabaseLoader()

        # First call creates engine
        with patch.object(loader, "_connection_cache", {}) as mock_cache:
            engine1 = loader._get_engine("sqlite:///test.db")
            mock_cache["sqlite:///test.db"] = mock_engine

            # Second call should use cached engine
            engine2 = loader._get_engine("sqlite:///test.db")

        assert engine1 == engine2
        assert mock_create_engine.call_count == 1  # Only called once

    @patch("sqlalchemy.create_engine")
    def test_get_engine_connection_failure(self, mock_create_engine):
        """Test engine creation failure."""
        mock_create_engine.side_effect = Exception("Connection failed")

        loader = DatabaseLoader()

        with pytest.raises(
            DataValidationError, match="Failed to create database connection"
        ):
            loader._get_engine("invalid://connection")

    def test_get_engine_existing_engine(self):
        """Test passing existing engine."""
        mock_engine = Mock(spec=sa.Engine)
        loader = DatabaseLoader()

        result = loader._get_engine(mock_engine)

        assert result is mock_engine

    def test_get_database_type(self):
        """Test database type extraction from URL."""
        loader = DatabaseLoader()

        # Mock URL objects
        postgresql_url = Mock()
        postgresql_url.drivername = "postgresql+psycopg2"
        assert loader._get_database_type(postgresql_url) == "postgresql"

        mysql_url = Mock()
        mysql_url.drivername = "mysql+pymysql"
        assert loader._get_database_type(mysql_url) == "mysql"

        sqlite_url = Mock()
        sqlite_url.drivername = "sqlite"
        assert loader._get_database_type(sqlite_url) == "sqlite"

    @patch("pandas.read_sql")
    def test_load_query_success(self, mock_read_sql):
        """Test successful query loading."""
        mock_df = pd.DataFrame(
            {"id": [1, 2, 3], "value": [10.0, 20.0, 30.0], "label": [0, 1, 0]}
        )
        mock_read_sql.return_value = mock_df

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "postgresql"

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.load_query(
                "SELECT * FROM test_table",
                connection="postgresql://user:pass@host/db",
                name="test_query",
                target_column="label",
            )

        assert isinstance(result, Dataset)
        assert result.name == "test_query"
        assert result.target_column == "label"
        assert len(result.data) == 3
        assert "source" in result.metadata
        assert result.metadata["source"] == "database_query"
        assert result.metadata["database_type"] == "postgresql"

    @patch("pandas.read_sql")
    def test_load_query_empty_result(self, mock_read_sql):
        """Test query with empty result."""
        mock_read_sql.return_value = pd.DataFrame()

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "sqlite"

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.load_query(
                "SELECT * FROM empty_table", connection="sqlite:///test.db"
            )

        assert isinstance(result, Dataset)
        assert len(result.data) == 0

    @patch("pandas.read_sql")
    def test_load_query_with_chunking(self, mock_read_sql):
        """Test query loading with chunking."""
        # Mock chunked results
        chunk1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        chunk2 = pd.DataFrame({"id": [3, 4], "value": [30, 40]})
        mock_read_sql.return_value = [chunk1, chunk2]

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "postgresql"

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.load_query(
                "SELECT * FROM large_table",
                connection="postgresql://user:pass@host/db",
                chunksize=2,
            )

        assert isinstance(result, Dataset)
        assert len(result.data) == 4  # Combined chunks

    def test_load_query_target_column_not_found(self):
        """Test error when target column not found in query results."""
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "sqlite"

        with (
            patch("pandas.read_sql", return_value=mock_df),
            patch.object(loader, "_get_engine", return_value=mock_engine),
        ):
            with pytest.raises(
                DataValidationError, match="Target column 'missing' not found"
            ):
                loader.load_query(
                    "SELECT col1, col2 FROM test_table",
                    connection="sqlite:///test.db",
                    target_column="missing",
                )

    @patch("pandas.read_sql")
    def test_load_query_sql_error(self, mock_read_sql):
        """Test SQL execution error."""
        mock_read_sql.side_effect = sa.exc.SQLAlchemyError("Invalid SQL")

        loader = DatabaseLoader()

        mock_engine = Mock()
        with patch.object(loader, "_get_engine", return_value=mock_engine):
            with pytest.raises(DataValidationError, match="Database query failed"):
                loader.load_query("INVALID SQL", connection="sqlite:///test.db")

    @patch("pandas.read_sql")
    def test_load_table_success(self, mock_read_sql):
        """Test successful table loading."""
        mock_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        )
        mock_read_sql.return_value = mock_df

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "mysql"

        with (
            patch.object(loader, "_get_engine", return_value=mock_engine),
            patch.object(loader, "get_table_info", return_value={"columns": 3}),
        ):
            result = loader.load_table(
                "users", connection="mysql://user:pass@host/db", name="users_table"
            )

        assert isinstance(result, Dataset)
        assert result.name == "users_table"
        assert len(result.data) == 3
        assert "source" in result.metadata
        assert result.metadata["source"] == "database_table"
        assert result.metadata["table_name"] == "users"

    @patch("pandas.read_sql")
    def test_load_table_with_schema(self, mock_read_sql):
        """Test table loading with schema."""
        mock_df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        mock_read_sql.return_value = mock_df

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "postgresql"

        with (
            patch.object(loader, "_get_engine", return_value=mock_engine),
            patch.object(loader, "get_table_info", return_value={}),
        ):
            result = loader.load_table(
                "test_table",
                connection="postgresql://user:pass@host/db",
                schema="public",
            )

        # Verify SQL query included schema
        mock_read_sql.assert_called_once()
        sql_query = mock_read_sql.call_args[0][0]
        assert "public.test_table" in sql_query

    @patch("pandas.read_sql")
    def test_load_table_with_filters(self, mock_read_sql):
        """Test table loading with WHERE clause and other filters."""
        mock_df = pd.DataFrame({"id": [1, 2], "active": [True, True]})
        mock_read_sql.return_value = mock_df

        loader = DatabaseLoader()

        mock_engine = Mock()
        mock_engine.url.drivername = "sqlite"

        with (
            patch.object(loader, "_get_engine", return_value=mock_engine),
            patch.object(loader, "get_table_info", return_value={}),
        ):
            result = loader.load_table(
                "users",
                connection="sqlite:///test.db",
                columns=["id", "active"],
                where_clause="active = 1",
                order_by="id DESC",
                limit=100,
            )

        # Verify SQL query structure
        mock_read_sql.assert_called_once()
        sql_query = mock_read_sql.call_args[0][0]
        assert "SELECT id, active FROM" in sql_query
        assert "WHERE active = 1" in sql_query
        assert "ORDER BY id DESC" in sql_query
        assert "LIMIT 100" in sql_query

    @patch("sqlalchemy.inspect")
    def test_get_table_info_success(self, mock_inspect):
        """Test successful table information retrieval."""
        mock_inspector = Mock()
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR(50)"},
        ]
        mock_inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
        mock_inspector.get_foreign_keys.return_value = []
        mock_inspector.get_indexes.return_value = []
        mock_inspect.return_value = mock_inspector

        loader = DatabaseLoader()

        mock_engine = Mock()

        # Mock row count query
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1000
        mock_connection.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.get_table_info("test_table", mock_engine)

        assert result["table_name"] == "test_table"
        assert result["row_count"] == 1000
        assert result["column_count"] == 2
        assert result["primary_keys"] == ["id"]

    @patch("sqlalchemy.inspect")
    def test_get_table_info_error(self, mock_inspect):
        """Test table information retrieval with error."""
        mock_inspect.side_effect = Exception("Inspector error")

        loader = DatabaseLoader()

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.get_table_info("test_table", mock_engine)

        assert result["table_name"] == "test_table"
        assert "error" in result

    @patch("pandas.read_sql")
    def test_load_batch_query(self, mock_read_sql):
        """Test batch loading with SQL query."""
        # Mock chunked iterator
        chunk1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        chunk2 = pd.DataFrame({"id": [3, 4], "value": [30, 40]})
        mock_read_sql.return_value = [chunk1, chunk2]

        loader = DatabaseLoader()

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            batches = list(
                loader.load_batch(
                    "SELECT * FROM test_table",
                    batch_size=2,
                    connection="sqlite:///test.db",
                )
            )

        assert len(batches) == 2
        assert isinstance(batches[0], Dataset)
        assert batches[0].name == "batch_query_batch_0"
        assert len(batches[0].data) == 2
        assert batches[1].name == "batch_query_batch_1"

    @patch("pandas.read_sql")
    def test_load_batch_table(self, mock_read_sql):
        """Test batch loading with table name."""
        chunk1 = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        chunk2 = pd.DataFrame({"id": [3, 4], "name": ["C", "D"]})
        mock_read_sql.return_value = [chunk1, chunk2]

        loader = DatabaseLoader(connection_string="postgresql://user:pass@host/db")

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            batches = list(
                loader.load_batch(
                    "users_table",
                    batch_size=2,
                    schema="public",
                    where_clause="active = true",
                )
            )

        assert len(batches) == 2

        # Verify SQL was constructed for table
        mock_read_sql.assert_called()
        sql_query = mock_read_sql.call_args[0][0]
        assert "SELECT * FROM public.users_table" in sql_query
        assert "WHERE active = true" in sql_query

    def test_load_batch_no_connection(self):
        """Test batch loading error when no connection provided."""
        loader = DatabaseLoader()

        with pytest.raises(
            DataValidationError, match="No database connection provided"
        ):
            list(loader.load_batch("SELECT * FROM test", batch_size=100))

    def test_estimate_size_query(self):
        """Test size estimation for SQL query."""
        loader = DatabaseLoader(connection_string="sqlite:///test.db")

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.estimate_size(
                "SELECT * FROM large_table WHERE condition = 1"
            )

        assert result["type"] == "query"
        assert result["estimated_rows"] == "unknown"
        assert "note" in result

    def test_estimate_size_table(self):
        """Test size estimation for table."""
        loader = DatabaseLoader(connection_string="postgresql://user:pass@host/db")

        mock_engine = Mock()
        table_info = {"row_count": 100000, "column_count": 10}

        with (
            patch.object(loader, "_get_engine", return_value=mock_engine),
            patch.object(loader, "get_table_info", return_value=table_info),
        ):
            result = loader.estimate_size("users")

        assert result["type"] == "table"
        assert result["table_name"] == "users"
        assert result["estimated_rows"] == 100000
        assert result["columns"] == 10
        assert "estimated_memory_mb" in result

    def test_estimate_size_no_connection(self):
        """Test size estimation error when no connection."""
        loader = DatabaseLoader()

        result = loader.estimate_size("test_table")

        assert "error" in result
        assert "No database connection provided" in result["error"]

    @patch("sqlalchemy.inspect")
    def test_list_tables(self, mock_inspect):
        """Test listing database tables."""
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["table1", "table2", "table3"]
        mock_inspect.return_value = mock_inspector

        loader = DatabaseLoader(connection_string="sqlite:///test.db")

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.list_tables()

        assert result == ["table1", "table2", "table3"]

    @patch("sqlalchemy.inspect")
    def test_list_tables_with_schema(self, mock_inspect):
        """Test listing tables in specific schema."""
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["schema_table1", "schema_table2"]
        mock_inspect.return_value = mock_inspector

        loader = DatabaseLoader()

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.list_tables(connection=mock_engine, schema="public")

        assert result == ["schema_table1", "schema_table2"]
        mock_inspector.get_table_names.assert_called_with(schema="public")

    @patch("sqlalchemy.inspect")
    def test_list_schemas(self, mock_inspect):
        """Test listing database schemas."""
        mock_inspector = Mock()
        mock_inspector.get_schema_names.return_value = [
            "public",
            "private",
            "analytics",
        ]
        mock_inspect.return_value = mock_inspector

        loader = DatabaseLoader(connection_string="postgresql://user:pass@host/db")

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.list_schemas()

        assert result == ["public", "private", "analytics"]

    @patch("sqlalchemy.inspect")
    def test_list_schemas_not_supported(self, mock_inspect):
        """Test listing schemas when not supported."""
        mock_inspector = Mock()
        mock_inspector.get_schema_names.side_effect = Exception("Not supported")
        mock_inspect.return_value = mock_inspector

        loader = DatabaseLoader()

        mock_engine = Mock()

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            result = loader.list_schemas(connection=mock_engine)

        assert result == []

    def test_close_connections(self):
        """Test closing all cached connections."""
        loader = DatabaseLoader()

        # Mock cached engines
        mock_engine1 = Mock()
        mock_engine2 = Mock()
        loader._connection_cache = {"conn1": mock_engine1, "conn2": mock_engine2}
        loader._engine = Mock()

        loader.close_connections()

        mock_engine1.dispose.assert_called_once()
        mock_engine2.dispose.assert_called_once()
        loader._engine.dispose.assert_called_once()
        assert len(loader._connection_cache) == 0
        assert loader._engine is None

    def test_connection_context_manager(self):
        """Test database connection context manager."""
        loader = DatabaseLoader(connection_string="sqlite:///test.db")

        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value = mock_connection

        with patch.object(loader, "_get_engine", return_value=mock_engine):
            with loader.connection() as conn:
                assert conn == mock_connection

        mock_connection.close.assert_called_once()

    def test_create_connection_string_postgresql(self):
        """Test PostgreSQL connection string creation."""
        result = DatabaseLoader.create_connection_string(
            database_type="postgresql",
            username="user",
            password="pass",
            host="localhost",
            database="testdb",
            port=5432,
        )

        expected = "postgresql+psycopg2://user:pass@localhost:5432/testdb"
        assert result == expected

    def test_create_connection_string_mysql_default_port(self):
        """Test MySQL connection string with default port."""
        result = DatabaseLoader.create_connection_string(
            database_type="mysql",
            username="user",
            password="pass",
            host="localhost",
            database="testdb",
        )

        expected = "mysql+pymysql://user:pass@localhost:3306/testdb"
        assert result == expected

    def test_create_connection_string_sqlite(self):
        """Test SQLite connection string creation."""
        result = DatabaseLoader.create_connection_string(
            database_type="sqlite",
            username="",
            password="",
            host="",
            database="/path/to/database.db",
        )

        expected = "sqlite:////path/to/database.db"
        assert result == expected

    def test_create_connection_string_with_params(self):
        """Test connection string creation with additional parameters."""
        result = DatabaseLoader.create_connection_string(
            database_type="postgresql",
            username="user",
            password="pass",
            host="localhost",
            database="testdb",
            sslmode="require",
            charset="utf8",
        )

        expected = "postgresql+psycopg2://user:pass@localhost:5432/testdb?sslmode=require&charset=utf8"
        assert result == expected

    def test_create_connection_string_unsupported_type(self):
        """Test error for unsupported database type."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseLoader.create_connection_string(
                database_type="unsupported",
                username="user",
                password="pass",
                host="localhost",
                database="testdb",
            )

    @patch("sqlalchemy.create_engine")
    def test_validate_connection_string_valid(self, mock_create_engine):
        """Test validation with valid connection string."""
        mock_engine = Mock()
        mock_connection = Mock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
        mock_connection.execute.return_value = Mock()
        mock_create_engine.return_value = mock_engine

        loader = DatabaseLoader()

        assert loader.validate("postgresql://user:pass@host/db") is True

    def test_validate_table_name_valid(self):
        """Test validation with valid table name."""
        loader = DatabaseLoader(connection_string="sqlite:///test.db")

        mock_engine = Mock()
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["test_table", "other_table"]

        with (
            patch.object(loader, "_get_engine", return_value=mock_engine),
            patch("sqlalchemy.inspect", return_value=mock_inspector),
        ):
            assert loader.validate("test_table") is True

    def test_validate_table_name_invalid(self):
        """Test validation with invalid table name."""
        loader = DatabaseLoader(connection_string="sqlite:///test.db")

        mock_engine = Mock()
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ["other_table"]

        with (
            patch.object(loader, "_get_engine", return_value=mock_engine),
            patch("sqlalchemy.inspect", return_value=mock_inspector),
        ):
            assert loader.validate("nonexistent_table") is False

    def test_validate_no_connection_string(self):
        """Test validation failure when no connection string."""
        loader = DatabaseLoader()

        assert loader.validate("table_name") is False

    def test_validate_connection_error(self):
        """Test validation failure on connection error."""
        loader = DatabaseLoader()

        with patch.object(
            loader, "_get_engine", side_effect=Exception("Connection failed")
        ):
            assert loader.validate("postgresql://invalid") is False

    def test_load_connection_string_with_query(self):
        """Test load method with connection string and query."""
        mock_df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})

        loader = DatabaseLoader()

        with patch.object(loader, "load_query", return_value=Mock()) as mock_load_query:
            loader.load(
                "postgresql://user:pass@host/db", query="SELECT * FROM test_table"
            )

        mock_load_query.assert_called_once()

    def test_load_connection_string_with_table(self):
        """Test load method with connection string and table name."""
        loader = DatabaseLoader()

        with patch.object(loader, "load_table", return_value=Mock()) as mock_load_table:
            loader.load("postgresql://user:pass@host/db", table_name="test_table")

        mock_load_table.assert_called_once()

    def test_load_connection_string_no_query_or_table(self):
        """Test error when no query or table provided with connection string."""
        loader = DatabaseLoader()

        with pytest.raises(
            DataValidationError, match="Either table_name or query must be provided"
        ):
            loader.load("postgresql://user:pass@host/db")

    def test_load_table_name_with_default_connection(self):
        """Test load method with table name using default connection."""
        loader = DatabaseLoader(connection_string="sqlite:///test.db")

        with patch.object(loader, "load_table", return_value=Mock()) as mock_load_table:
            loader.load("test_table")

        mock_load_table.assert_called_once_with(
            "test_table", "sqlite:///test.db", name=None
        )

    def test_load_table_name_no_default_connection(self):
        """Test error when loading table name without default connection."""
        loader = DatabaseLoader()

        with pytest.raises(DataValidationError, match="No connection string provided"):
            loader.load("test_table")
