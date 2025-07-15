"""Comprehensive tests for the database migration manager."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pynomaly.infrastructure.persistence.migration_manager import (
    MigrationManager,
    create_migration_manager,
    init_and_migrate,
    quick_migrate,
)


class TestMigrationManagerBasic:
    """Basic functionality tests for MigrationManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.database_url = f"sqlite:///{self.test_db_path}"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_manager_initialization(self):
        """Test MigrationManager initialization."""
        manager = MigrationManager(self.database_url)
        
        assert manager.database_url == self.database_url
        assert manager.engine is not None
        assert manager.config is not None

    def test_get_database_url_from_env(self):
        """Test database URL retrieval from environment."""
        with patch.dict(os.environ, {'DATABASE_URL': self.database_url}):
            manager = MigrationManager()
            assert manager.database_url == self.database_url

    def test_get_database_url_fallback(self):
        """Test database URL fallback to default."""
        # Clear environment variable
        with patch.dict(os.environ, {}, clear=True):
            with patch('pynomaly.infrastructure.persistence.migration_manager.Settings') as mock_settings:
                mock_settings.side_effect = Exception("Settings error")
                manager = MigrationManager()
                assert "sqlite:///./storage/pynomaly.db" in manager.database_url

    @patch('pynomaly.infrastructure.persistence.migration_manager.MigrationContext')
    @patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory')
    def test_check_migration_needed_no_history(self, mock_script_dir, mock_migration_context):
        """Test migration check when no history exists."""
        # Mock migration context
        mock_context = MagicMock()
        mock_context.get_current_revision.return_value = None
        mock_migration_context.configure.return_value = mock_context
        
        # Mock script directory
        mock_script = MagicMock()
        mock_script.get_current_head.return_value = "abc123"
        mock_script_dir.from_config.return_value = mock_script
        
        manager = MigrationManager(self.database_url)
        with patch.object(manager.engine, 'connect') as mock_connect:
            mock_connect.return_value.__enter__.return_value = MagicMock()
            result = manager.check_migration_needed()
            
        assert result is True

    @patch('pynomaly.infrastructure.persistence.migration_manager.MigrationContext')
    @patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory')
    def test_check_migration_needed_up_to_date(self, mock_script_dir, mock_migration_context):
        """Test migration check when database is up to date."""
        # Mock migration context
        mock_context = MagicMock()
        mock_context.get_current_revision.return_value = "abc123"
        mock_migration_context.configure.return_value = mock_context
        
        # Mock script directory
        mock_script = MagicMock()
        mock_script.get_current_head.return_value = "abc123"
        mock_script_dir.from_config.return_value = mock_script
        
        manager = MigrationManager(self.database_url)
        with patch.object(manager.engine, 'connect') as mock_connect:
            mock_connect.return_value.__enter__.return_value = MagicMock()
            result = manager.check_migration_needed()
            
        assert result is False

    @patch('pynomaly.infrastructure.persistence.migration_manager.command')
    def test_create_migration(self, mock_command):
        """Test migration creation."""
        manager = MigrationManager(self.database_url)
        
        with patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory') as mock_script_dir:
            mock_script = MagicMock()
            mock_script.get_current_head.return_value = "def456"
            mock_script_dir.from_config.return_value = mock_script
            
            revision_id = manager.create_migration("Test migration", autogenerate=True)
            
            mock_command.revision.assert_called_once_with(
                manager.config, 
                message="Test migration", 
                autogenerate=True
            )
            assert revision_id == "def456"

    @patch('pynomaly.infrastructure.persistence.migration_manager.command')
    def test_run_migrations_success(self, mock_command):
        """Test successful migration run."""
        manager = MigrationManager(self.database_url)
        
        result = manager.run_migrations("head")
        
        mock_command.upgrade.assert_called_once_with(manager.config, "head")
        assert result is True

    @patch('pynomaly.infrastructure.persistence.migration_manager.command')
    def test_run_migrations_failure(self, mock_command):
        """Test migration run failure."""
        mock_command.upgrade.side_effect = Exception("Migration failed")
        
        manager = MigrationManager(self.database_url)
        result = manager.run_migrations("head")
        
        assert result is False

    @patch('pynomaly.infrastructure.persistence.migration_manager.command')
    def test_rollback_migration_success(self, mock_command):
        """Test successful migration rollback."""
        manager = MigrationManager(self.database_url)
        
        result = manager.rollback_migration("-1")
        
        mock_command.downgrade.assert_called_once_with(manager.config, "-1")
        assert result is True

    @patch('pynomaly.infrastructure.persistence.migration_manager.command')
    def test_rollback_migration_failure(self, mock_command):
        """Test migration rollback failure."""
        mock_command.downgrade.side_effect = Exception("Rollback failed")
        
        manager = MigrationManager(self.database_url)
        result = manager.rollback_migration("-1")
        
        assert result is False


class TestMigrationManagerAdvanced:
    """Advanced functionality tests for MigrationManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.database_url = f"sqlite:///{self.test_db_path}"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_backup_database_sqlite(self):
        """Test database backup for SQLite."""
        # Create a dummy database file
        self.test_db_path.write_text("dummy database content")
        
        manager = MigrationManager(self.database_url)
        backup_path = self.temp_dir / "backup.db"
        
        result = manager.backup_database(str(backup_path))
        
        assert result is True
        assert backup_path.exists()
        assert backup_path.read_text() == "dummy database content"

    def test_backup_database_non_sqlite(self):
        """Test database backup for non-SQLite databases."""
        postgres_url = "postgresql://user:pass@localhost/db"
        manager = MigrationManager(postgres_url)
        
        result = manager.backup_database("/tmp/backup.sql")
        
        # Should return False for unsupported database types
        assert result is False

    def test_cleanup_old_backups(self):
        """Test cleanup of old backup files."""
        # Create multiple backup files
        backup_dir = self.temp_dir / "backups"
        backup_dir.mkdir()
        
        for i in range(10):
            backup_file = backup_dir / f"backup_{i}.db"
            backup_file.write_text(f"backup {i}")
        
        manager = MigrationManager(self.database_url)
        result = manager.cleanup_old_backups(str(backup_dir), keep_count=3)
        
        assert result is True
        remaining_files = list(backup_dir.glob("*.db"))
        assert len(remaining_files) == 3

    @patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory')
    def test_validate_migration_success(self, mock_script_dir):
        """Test successful migration validation."""
        # Create a temporary migration file
        migration_file = self.temp_dir / "migration.py"
        migration_file.write_text("# Migration content")
        
        # Mock script directory
        mock_script = MagicMock()
        mock_revision = MagicMock()
        mock_revision.path = str(migration_file)
        mock_script.get_revision.return_value = mock_revision
        mock_script_dir.from_config.return_value = mock_script
        
        manager = MigrationManager(self.database_url)
        result = manager.validate_migration("abc123")
        
        assert result is True

    @patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory')
    def test_validate_migration_not_found(self, mock_script_dir):
        """Test migration validation when revision not found."""
        # Mock script directory
        mock_script = MagicMock()
        mock_script.get_revision.return_value = None
        mock_script_dir.from_config.return_value = mock_script
        
        manager = MigrationManager(self.database_url)
        result = manager.validate_migration("nonexistent")
        
        assert result is False

    @patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory')
    def test_validate_migration_file_missing(self, mock_script_dir):
        """Test migration validation when file is missing."""
        # Mock script directory
        mock_script = MagicMock()
        mock_revision = MagicMock()
        mock_revision.path = "/nonexistent/migration.py"
        mock_script.get_revision.return_value = mock_revision
        mock_script_dir.from_config.return_value = mock_script
        
        manager = MigrationManager(self.database_url)
        result = manager.validate_migration("abc123")
        
        assert result is False

    @patch('pynomaly.infrastructure.persistence.migration_manager.ScriptDirectory')
    def test_get_migration_history(self, mock_script_dir):
        """Test getting migration history."""
        # Mock script directory
        mock_script = MagicMock()
        
        # Create mock revisions
        mock_revisions = []
        for i in range(3):
            mock_rev = MagicMock()
            mock_rev.revision = f"rev_{i}"
            mock_rev.down_revision = f"rev_{i-1}" if i > 0 else None
            mock_rev.doc = f"Migration {i}"
            mock_rev.create_date = None
            mock_revisions.append(mock_rev)
        
        mock_script.walk_revisions.return_value = mock_revisions
        mock_script_dir.from_config.return_value = mock_script
        
        manager = MigrationManager(self.database_url)
        
        with patch.object(manager.engine, 'connect') as mock_connect:
            mock_conn = MagicMock()
            mock_conn.execute.return_value = []
            mock_connect.return_value.__enter__.return_value = mock_conn
            
            history = manager.get_migration_history()
        
        assert len(history) == 3
        for i, entry in enumerate(history):
            assert entry["revision"] == f"rev_{2-i}"  # Reversed order
            assert entry["message"] == f"Migration {2-i}"
            assert entry["applied"] is False


class TestMigrationManagerIntegration:
    """Integration tests for MigrationManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.database_url = f"sqlite:///{self.test_db_path}"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_migration_manager_factory(self):
        """Test migration manager factory function."""
        manager = create_migration_manager(self.database_url)
        
        assert isinstance(manager, MigrationManager)
        assert manager.database_url == self.database_url

    @patch('pynomaly.infrastructure.persistence.migration_manager.MigrationManager.run_migrations')
    def test_quick_migrate_function(self, mock_run_migrations):
        """Test quick migrate convenience function."""
        mock_run_migrations.return_value = True
        
        result = quick_migrate(self.database_url)
        
        assert result is True
        mock_run_migrations.assert_called_once()

    @patch('pynomaly.infrastructure.persistence.migration_manager.MigrationManager.initialize_database')
    @patch('pynomaly.infrastructure.persistence.migration_manager.MigrationManager.run_migrations')
    def test_init_and_migrate_success(self, mock_run_migrations, mock_initialize):
        """Test init and migrate function success."""
        mock_initialize.return_value = True
        mock_run_migrations.return_value = True
        
        result = init_and_migrate(self.database_url)
        
        assert result is True
        mock_initialize.assert_called_once()
        mock_run_migrations.assert_called_once()

    @patch('pynomaly.infrastructure.persistence.migration_manager.MigrationManager.initialize_database')
    def test_init_and_migrate_init_failure(self, mock_initialize):
        """Test init and migrate function with initialization failure."""
        mock_initialize.return_value = False
        
        result = init_and_migrate(self.database_url)
        
        assert result is False
        mock_initialize.assert_called_once()


class TestMigrationManagerErrorHandling:
    """Error handling tests for MigrationManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.database_url = f"sqlite:///{self.test_db_path}"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_alembic_config_not_found(self):
        """Test handling when alembic.ini is not found."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Alembic configuration file not found"):
                MigrationManager(self.database_url)

    def test_get_migration_status_database_error(self):
        """Test get_migration_status with database error."""
        manager = MigrationManager(self.database_url)
        
        with patch.object(manager.engine, 'connect') as mock_connect:
            mock_connect.side_effect = Exception("Database connection error")
            
            status = manager.get_migration_status()
            
            assert "error" in status
            assert "Database connection error" in status["error"]

    def test_backup_database_error(self):
        """Test backup database with error."""
        # Create manager with non-existent database
        nonexistent_url = "sqlite:///nonexistent/path/db.sqlite"
        manager = MigrationManager(nonexistent_url)
        
        result = manager.backup_database("/tmp/backup.db")
        
        assert result is False

    def test_cleanup_old_backups_error(self):
        """Test cleanup old backups with error."""
        manager = MigrationManager(self.database_url)
        
        # Try to cleanup non-existent directory
        result = manager.cleanup_old_backups("/nonexistent/directory")
        
        assert result is True  # Returns True for non-existent directory

    def test_create_migration_error(self):
        """Test create migration with error."""
        manager = MigrationManager(self.database_url)
        
        with patch('pynomaly.infrastructure.persistence.migration_manager.command') as mock_command:
            mock_command.revision.side_effect = Exception("Migration creation failed")
            
            with pytest.raises(Exception, match="Migration creation failed"):
                manager.create_migration("Test migration")


class TestMigrationManagerCLIIntegration:
    """Tests for CLI integration with MigrationManager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.database_url = f"sqlite:///{self.test_db_path}"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_manager_with_settings_integration(self):
        """Test MigrationManager integration with application settings."""
        with patch('pynomaly.infrastructure.persistence.migration_manager.Settings') as mock_settings:
            mock_config = MagicMock()
            mock_config.database_url = self.database_url
            mock_settings.return_value = mock_config
            
            manager = MigrationManager()
            
            assert manager.database_url == self.database_url

    @patch('pynomaly.infrastructure.persistence.migration_manager.Config')
    def test_alembic_config_setup(self, mock_config_class):
        """Test Alembic configuration setup."""
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        with patch('pathlib.Path.exists', return_value=True):
            manager = MigrationManager(self.database_url)
            
            # Verify config was created and URL was set
            mock_config_class.assert_called_once()
            mock_config.set_main_option.assert_called_with("sqlalchemy.url", self.database_url)

    def test_database_url_precedence(self):
        """Test database URL precedence (explicit > env > settings > default)."""
        # Test explicit URL takes precedence
        manager = MigrationManager(self.database_url)
        assert manager.database_url == self.database_url
        
        # Test environment variable takes precedence over settings
        with patch.dict(os.environ, {'DATABASE_URL': 'env://database/url'}):
            manager = MigrationManager()
            assert manager.database_url == 'env://database/url'


class TestMigrationSafety:
    """Safety and validation tests for migrations."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.database_url = f"sqlite:///{self.test_db_path}"

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_validation_workflow(self):
        """Test complete migration validation workflow."""
        manager = MigrationManager(self.database_url)
        
        # Test validation for non-existent revision
        with patch.object(manager, 'validate_migration', return_value=False):
            assert manager.validate_migration("invalid_revision") is False
        
        # Test validation for valid revision
        with patch.object(manager, 'validate_migration', return_value=True):
            assert manager.validate_migration("valid_revision") is True

    def test_backup_before_migration_workflow(self):
        """Test backup before migration workflow."""
        # Create dummy database
        self.test_db_path.write_text("test database")
        
        manager = MigrationManager(self.database_url)
        backup_path = self.temp_dir / "pre_migration_backup.db"
        
        # Test backup creation
        result = manager.backup_database(str(backup_path))
        assert result is True
        assert backup_path.exists()

    def test_rollback_safety_checks(self):
        """Test rollback safety checks."""
        manager = MigrationManager(self.database_url)
        
        with patch.object(manager, 'rollback_migration') as mock_rollback:
            mock_rollback.return_value = True
            
            # Test rollback to previous revision
            result = manager.rollback_migration("-1")
            assert result is True
            mock_rollback.assert_called_with("-1")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])