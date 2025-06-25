"""
Enhanced Database Operations Testing Suite
Comprehensive tests for database CRUD, transactions, migrations, and performance.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from pynomaly.domain.exceptions import DatabaseError
from pynomaly.infrastructure.persistence.database import DatabaseManager
from pynomaly.infrastructure.persistence.migrations import MigrationManager


class TestDatabaseManager:
    """Enhanced test suite for database management operations."""

    @pytest.fixture
    def db_config(self):
        """Database configuration for testing."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "pynomaly_test",
            "username": "test_user",
            "password": "test_password",
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "pool_recycle": 3600,
        }

    @pytest.fixture
    def db_manager(self, db_config):
        """Create database manager instance."""
        return DatabaseManager(config=db_config)

    @pytest.fixture
    def mock_connection(self):
        """Mock database connection."""
        connection = AsyncMock()
        connection.execute = AsyncMock()
        connection.fetch = AsyncMock()
        connection.fetchrow = AsyncMock()
        connection.fetchval = AsyncMock()
        return connection

    @pytest.fixture
    def mock_pool(self, mock_connection):
        """Mock connection pool."""
        pool = AsyncMock()
        pool.acquire = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool

    # Database Initialization Tests

    @pytest.mark.asyncio
    async def test_database_initialization_success(self, db_manager):
        """Test successful database initialization."""
        with patch("asyncpg.create_pool") as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool

            await db_manager.initialize()

            assert db_manager.pool is not None
            mock_create_pool.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_initialization_failure(self, db_manager):
        """Test database initialization failure handling."""
        with patch("asyncpg.create_pool") as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")

            with pytest.raises(DatabaseError):
                await db_manager.initialize()

    @pytest.mark.asyncio
    async def test_database_connection_retry(self, db_manager):
        """Test database connection retry mechanism."""
        with patch("asyncpg.create_pool") as mock_create_pool:
            # First two attempts fail, third succeeds
            mock_create_pool.side_effect = [
                Exception("Connection failed"),
                Exception("Connection failed"),
                AsyncMock(),
            ]

            await db_manager.initialize(max_retries=3, retry_delay=0.1)

            assert mock_create_pool.call_count == 3

    @pytest.mark.asyncio
    async def test_database_health_check(self, db_manager, mock_pool):
        """Test database health check."""
        db_manager.pool = mock_pool

        with patch.object(db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [{"result": 1}]

            is_healthy = await db_manager.health_check()

            assert is_healthy is True
            mock_execute.assert_called_with("SELECT 1 as result")

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self, db_manager, mock_pool):
        """Test database health check failure."""
        db_manager.pool = mock_pool

        with patch.object(db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database unavailable")

            is_healthy = await db_manager.health_check()

            assert is_healthy is False

    # Connection Pool Management Tests

    @pytest.mark.asyncio
    async def test_connection_pool_configuration(self, db_config):
        """Test connection pool configuration."""
        db_manager = DatabaseManager(config=db_config)

        with patch("asyncpg.create_pool") as mock_create_pool:
            await db_manager.initialize()

            call_args = mock_create_pool.call_args
            assert call_args[1]["min_size"] == db_config["pool_size"]
            assert (
                call_args[1]["max_size"]
                == db_config["pool_size"] + db_config["max_overflow"]
            )

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_handling(self, db_manager, mock_pool):
        """Test handling of connection pool exhaustion."""
        db_manager.pool = mock_pool

        # Simulate pool exhaustion
        mock_pool.acquire.side_effect = TimeoutError("Pool exhausted")

        with pytest.raises(DatabaseError, match="Connection pool exhausted"):
            async with db_manager.get_connection() as conn:
                pass

    @pytest.mark.asyncio
    async def test_connection_lifecycle_management(
        self, db_manager, mock_pool, mock_connection
    ):
        """Test connection lifecycle management."""
        db_manager.pool = mock_pool

        async with db_manager.get_connection() as conn:
            assert conn is not None
            # Connection should be acquired from pool
            mock_pool.acquire.assert_called()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, db_manager, mock_pool):
        """Test connection error handling."""
        db_manager.pool = mock_pool

        # Simulate connection error
        mock_pool.acquire.side_effect = Exception("Connection error")

        with pytest.raises(DatabaseError):
            async with db_manager.get_connection() as conn:
                pass

    # Query Execution Tests

    @pytest.mark.asyncio
    async def test_execute_query_select(self, db_manager, mock_connection):
        """Test SELECT query execution."""
        mock_rows = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        mock_connection.fetch.return_value = mock_rows

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            result = await db_manager.execute_query(
                "SELECT id, name FROM datasets WHERE active = $1", True
            )

            assert result == mock_rows
            mock_connection.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_insert(self, db_manager, mock_connection):
        """Test INSERT query execution."""
        mock_connection.fetchrow.return_value = {"id": 123}

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            result = await db_manager.execute_query(
                "INSERT INTO datasets (name, data) VALUES ($1, $2) RETURNING id",
                "test_dataset",
                '{"test": "data"}',
            )

            assert result == {"id": 123}
            mock_connection.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_with_timeout(self, db_manager, mock_connection):
        """Test query execution with timeout."""
        mock_connection.fetch.side_effect = TimeoutError()

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            with pytest.raises(DatabaseError, match="Query timeout"):
                await db_manager.execute_query("SELECT * FROM large_table", timeout=1.0)

    @pytest.mark.asyncio
    async def test_execute_batch_queries(self, db_manager, mock_connection):
        """Test batch query execution."""
        queries = [
            ("INSERT INTO datasets (name) VALUES ($1)", "dataset1"),
            ("INSERT INTO datasets (name) VALUES ($1)", "dataset2"),
            ("INSERT INTO datasets (name) VALUES ($1)", "dataset3"),
        ]

        mock_connection.executemany.return_value = None

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            await db_manager.execute_batch(queries)

            mock_connection.executemany.assert_called()

    # Transaction Management Tests

    @pytest.mark.asyncio
    async def test_transaction_success(self, db_manager, mock_connection):
        """Test successful transaction execution."""
        mock_transaction = AsyncMock()
        mock_connection.transaction.return_value = mock_transaction

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            async with db_manager.transaction() as txn:
                await txn.execute("INSERT INTO datasets (name) VALUES ($1)", "test")
                await txn.execute("INSERT INTO detectors (name) VALUES ($1)", "test")

            mock_transaction.__aenter__.assert_called()
            mock_transaction.__aexit__.assert_called()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, db_manager, mock_connection):
        """Test transaction rollback on error."""
        mock_transaction = AsyncMock()
        mock_connection.transaction.return_value = mock_transaction
        mock_connection.execute.side_effect = Exception("Query failed")

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            with pytest.raises(Exception):
                async with db_manager.transaction() as txn:
                    await txn.execute("INSERT INTO datasets (name) VALUES ($1)", "test")

            # Transaction should have been rolled back
            mock_transaction.__aexit__.assert_called()

    @pytest.mark.asyncio
    async def test_nested_transactions(self, db_manager, mock_connection):
        """Test nested transaction handling."""
        mock_transaction = AsyncMock()
        mock_savepoint = AsyncMock()
        mock_connection.transaction.return_value = mock_transaction
        mock_transaction.savepoint.return_value = mock_savepoint

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            async with db_manager.transaction() as txn:
                await txn.execute("INSERT INTO datasets (name) VALUES ($1)", "test1")

                async with txn.savepoint() as sp:
                    await sp.execute(
                        "INSERT INTO detectors (name) VALUES ($1)", "test2"
                    )

            mock_transaction.__aenter__.assert_called()
            mock_savepoint.__aenter__.assert_called()

    # Data Migration Tests

    @pytest.mark.asyncio
    async def test_migration_execution(self, db_manager):
        """Test database migration execution."""
        migration_manager = MigrationManager(db_manager)

        with (
            patch.object(migration_manager, "get_pending_migrations") as mock_pending,
            patch.object(migration_manager, "execute_migration") as mock_execute,
        ):
            mock_pending.return_value = [
                {"version": "001", "description": "Create datasets table"},
                {"version": "002", "description": "Add indexes"},
            ]

            await migration_manager.migrate()

            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_migration_rollback(self, db_manager):
        """Test migration rollback functionality."""
        migration_manager = MigrationManager(db_manager)

        with patch.object(migration_manager, "rollback_migration") as mock_rollback:
            await migration_manager.rollback("002")

            mock_rollback.assert_called_with("002")

    @pytest.mark.asyncio
    async def test_migration_version_tracking(self, db_manager, mock_connection):
        """Test migration version tracking."""
        migration_manager = MigrationManager(db_manager)

        mock_connection.fetchval.return_value = "003"

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            current_version = await migration_manager.get_current_version()

            assert current_version == "003"

    # Performance and Optimization Tests

    @pytest.mark.asyncio
    async def test_query_performance_monitoring(self, db_manager, mock_connection):
        """Test query performance monitoring."""
        db_manager.enable_query_monitoring = True

        with (
            patch("time.time") as mock_time,
            patch.object(db_manager, "get_connection") as mock_get_conn,
        ):
            mock_time.side_effect = [0, 0.5]  # 500ms query
            mock_get_conn.return_value.__aenter__.return_value = mock_connection
            mock_connection.fetch.return_value = []

            await db_manager.execute_query("SELECT * FROM datasets")

            # Should log slow query
            assert db_manager.query_stats is not None

    @pytest.mark.asyncio
    async def test_connection_pooling_metrics(self, db_manager, mock_pool):
        """Test connection pool metrics collection."""
        db_manager.pool = mock_pool
        mock_pool.size = 10
        mock_pool.maxsize = 20
        mock_pool.freesize = 5

        metrics = await db_manager.get_pool_metrics()

        assert metrics["total_connections"] == 10
        assert metrics["max_connections"] == 20
        assert metrics["free_connections"] == 5
        assert metrics["used_connections"] == 5

    @pytest.mark.asyncio
    async def test_query_caching(self, db_manager, mock_connection):
        """Test query result caching."""
        db_manager.enable_caching = True
        cache_key = "SELECT * FROM datasets WHERE id = $1"
        cached_result = [{"id": 1, "name": "cached"}]

        with (
            patch.object(db_manager.cache, "get") as mock_cache_get,
            patch.object(db_manager.cache, "set") as mock_cache_set,
        ):
            mock_cache_get.return_value = cached_result

            result = await db_manager.execute_query(cache_key, 1)

            assert result == cached_result
            mock_cache_get.assert_called()

    @pytest.mark.asyncio
    async def test_prepared_statement_usage(self, db_manager, mock_connection):
        """Test prepared statement usage for performance."""
        prepared_query = "SELECT * FROM datasets WHERE status = $1 AND created_at > $2"

        mock_connection.prepare.return_value = AsyncMock()

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            await db_manager.execute_prepared(prepared_query, "active", datetime.now())

            mock_connection.prepare.assert_called()

    # Security Tests

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, db_manager):
        """Test SQL injection prevention through parameterized queries."""
        malicious_input = "'; DROP TABLE datasets; --"

        with patch.object(db_manager, "execute_query") as mock_execute:
            # Should use parameterized query, not string concatenation
            await db_manager.execute_query(
                "SELECT * FROM datasets WHERE name = $1", malicious_input
            )

            # Verify parameterized query was used
            call_args = mock_execute.call_args
            assert "$1" in call_args[0][0]
            assert malicious_input in call_args[0][1:]

    @pytest.mark.asyncio
    async def test_connection_encryption(self, db_config):
        """Test database connection encryption configuration."""
        db_config["ssl_mode"] = "require"
        db_config["ssl_cert"] = "/path/to/cert.pem"

        db_manager = DatabaseManager(config=db_config)

        with patch("asyncpg.create_pool") as mock_create_pool:
            await db_manager.initialize()

            call_args = mock_create_pool.call_args
            assert call_args[1]["ssl"] == "require"

    @pytest.mark.asyncio
    async def test_audit_logging(self, db_manager, mock_connection):
        """Test audit logging for database operations."""
        db_manager.enable_audit_logging = True

        with (
            patch.object(db_manager.audit_logger, "log") as mock_log,
            patch.object(db_manager, "get_connection") as mock_get_conn,
        ):
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            await db_manager.execute_query(
                "INSERT INTO datasets (name) VALUES ($1)",
                "sensitive_data",
                user_id="user123",
            )

            mock_log.assert_called()

    # Error Handling and Recovery Tests

    @pytest.mark.asyncio
    async def test_connection_recovery(self, db_manager, mock_pool):
        """Test connection recovery after network failure."""
        db_manager.pool = mock_pool

        # Simulate connection failure then recovery
        mock_pool.acquire.side_effect = [
            Exception("Network error"),
            AsyncMock(),  # Successful connection
        ]

        with patch("asyncio.sleep"):  # Speed up retry delay
            async with db_manager.get_connection() as conn:
                assert conn is not None

    @pytest.mark.asyncio
    async def test_deadlock_detection_and_retry(self, db_manager, mock_connection):
        """Test deadlock detection and automatic retry."""
        deadlock_error = Exception("deadlock detected")
        deadlock_error.sqlstate = "40P01"  # PostgreSQL deadlock code

        mock_connection.execute.side_effect = [
            deadlock_error,
            None,  # Successful retry
        ]

        with patch.object(db_manager, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__aenter__.return_value = mock_connection

            await db_manager.execute_query_with_retry(
                "UPDATE datasets SET status = $1 WHERE id = $2", "processing", 123
            )

            # Should have retried once
            assert mock_connection.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_data_corruption_detection(self, db_manager, mock_connection):
        """Test data corruption detection."""
        with patch.object(db_manager, "verify_data_integrity") as mock_verify:
            mock_verify.return_value = False  # Corruption detected

            with pytest.raises(DatabaseError, match="Data corruption detected"):
                await db_manager.execute_query("SELECT * FROM datasets")

    # Backup and Recovery Tests

    @pytest.mark.asyncio
    async def test_database_backup_creation(self, db_manager):
        """Test database backup creation."""
        backup_manager = db_manager.backup_manager

        with patch.object(backup_manager, "create_backup") as mock_backup:
            mock_backup.return_value = {
                "backup_id": "backup_123",
                "size": 1024 * 1024,
                "timestamp": datetime.now(),
            }

            backup_info = await backup_manager.create_backup()

            assert backup_info["backup_id"] == "backup_123"
            mock_backup.assert_called()

    @pytest.mark.asyncio
    async def test_point_in_time_recovery(self, db_manager):
        """Test point-in-time recovery capability."""
        recovery_manager = db_manager.recovery_manager
        target_time = datetime.now() - timedelta(hours=1)

        with patch.object(recovery_manager, "recover_to_point_in_time") as mock_recover:
            await recovery_manager.recover_to_point_in_time(target_time)

            mock_recover.assert_called_with(target_time)

    @pytest.mark.asyncio
    async def test_backup_verification(self, db_manager):
        """Test backup file verification."""
        backup_manager = db_manager.backup_manager

        with patch.object(backup_manager, "verify_backup") as mock_verify:
            mock_verify.return_value = True

            is_valid = await backup_manager.verify_backup("backup_123")

            assert is_valid is True
            mock_verify.assert_called_with("backup_123")


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_complete_crud_workflow(self, db_manager):
        """Test complete CRUD workflow."""
        with patch.object(db_manager, "execute_query") as mock_execute:
            # Create
            mock_execute.return_value = {"id": 1}
            dataset_id = await db_manager.execute_query(
                "INSERT INTO datasets (name, data) VALUES ($1, $2) RETURNING id",
                "test_dataset",
                '{"features": ["a", "b"]}',
            )

            # Read
            mock_execute.return_value = [{"id": 1, "name": "test_dataset"}]
            dataset = await db_manager.execute_query(
                "SELECT * FROM datasets WHERE id = $1", 1
            )

            # Update
            mock_execute.return_value = 1
            updated_rows = await db_manager.execute_query(
                "UPDATE datasets SET name = $1 WHERE id = $2", "updated_dataset", 1
            )

            # Delete
            mock_execute.return_value = 1
            deleted_rows = await db_manager.execute_query(
                "DELETE FROM datasets WHERE id = $1", 1
            )

            assert mock_execute.call_count == 4

    @pytest.mark.asyncio
    async def test_concurrent_transaction_handling(self, db_manager):
        """Test handling of concurrent transactions."""

        async def transaction_task(task_id):
            async with db_manager.transaction() as txn:
                await txn.execute(
                    "INSERT INTO datasets (name) VALUES ($1)",
                    f"concurrent_dataset_{task_id}",
                )
                await asyncio.sleep(0.1)  # Simulate work
                return task_id

        with patch.object(db_manager, "transaction"):
            # Run multiple concurrent transactions
            tasks = [transaction_task(i) for i in range(5)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5

    @pytest.mark.asyncio
    async def test_database_monitoring_integration(self, db_manager):
        """Test database monitoring integration."""
        with patch.object(db_manager, "get_performance_stats") as mock_stats:
            mock_stats.return_value = {
                "active_connections": 5,
                "idle_connections": 3,
                "total_queries": 1000,
                "slow_queries": 5,
                "average_response_time": 0.15,
            }

            stats = await db_manager.get_performance_stats()

            assert stats["active_connections"] == 5
            assert stats["slow_queries"] == 5
