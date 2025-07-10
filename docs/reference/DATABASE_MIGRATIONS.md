# Database Migration System Documentation

## Overview

Pynomaly includes a comprehensive database migration system built on Alembic that provides safe, version-controlled database schema evolution. The system supports PostgreSQL and SQLite databases with automatic migration generation, rollback capabilities, and safety validation.

## Architecture

### Core Components

1. **MigrationManager**: Core migration management with Alembic integration
2. **MigrationValidator**: Safety validation and risk assessment
3. **CLI Commands**: User-friendly migration management interface
4. **Automatic Migration**: Startup migration detection and execution

### Database Support

- **PostgreSQL**: Production recommended, full feature support
- **SQLite**: Development and testing, with backup capabilities

## Migration System Features

### ✅ Core Features

- **Version Control**: Track database schema changes with revision history
- **Auto-generation**: Automatically detect model changes and generate migrations
- **Rollback Support**: Safe rollback to previous schema versions
- **Safety Validation**: Comprehensive migration safety checks
- **Backup Integration**: Automatic database backups before migrations
- **CLI Interface**: Rich command-line interface for migration management
- **Multi-environment**: Support for development, staging, and production
- **Error Handling**: Robust error handling with detailed logging

### 🛡️ Safety Features

- **Migration Validation**: AST analysis of migration files
- **Dangerous Pattern Detection**: Identify potentially destructive operations
- **Database Readiness Checks**: Verify database state before migrations
- **Rollback Testing**: Validate upgrade/downgrade roundtrips
- **Lock Detection**: Check for database locks before execution
- **Data Integrity Validation**: Post-migration integrity checks

## CLI Commands

### Basic Commands

```bash
# Show migration status
pynomaly migrate status

# Run all pending migrations
pynomaly migrate run

# Create a new migration
pynomaly migrate create "add user preferences table"

# Rollback last migration
pynomaly migrate rollback

# Show migration history
pynomaly migrate history
```

### Advanced Commands

```bash
# Initialize database with migrations
pynomaly migrate init

# Reset database (DESTRUCTIVE)
pynomaly migrate reset

# Validate specific migration
pynomaly migrate validate abc123

# Quick setup (init + run)
pynomaly migrate quick

# Check if migration needed
pynomaly migrate check

# Create database backup
pynomaly migrate backup /path/to/backup.db
```

### Command Options

```bash
# Specify database URL
pynomaly migrate status --database-url postgresql://user:pass@host/db

# Target specific revision
pynomaly migrate run --target abc123

# Auto-generate migration
pynomaly migrate create "description" --autogenerate

# Force operations (skip confirmations)
pynomaly migrate rollback --force

# Verbose output
pynomaly migrate run --verbose
```

## Migration File Structure

### Standard Migration Template

```python
"""Add user preferences table

Revision ID: abc123def456
Revises: 789ghi012jkl
Create Date: 2024-01-15 10:30:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic
revision = 'abc123def456'
down_revision = '789ghi012jkl'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('user_id', sa.UUID(), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('preferences', sa.JSON(), default={}),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    # Create indexes
    op.create_index('idx_user_preferences_user_id', 'user_preferences', ['user_id'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('idx_user_preferences_user_id')
    op.drop_table('user_preferences')
```

### Best Practices for Migrations

#### ✅ Do's

1. **Descriptive Names**: Use clear, descriptive migration names
2. **Reversible Operations**: Always provide downgrade functions
3. **Index Management**: Create/drop indexes appropriately
4. **Foreign Keys**: Properly handle foreign key constraints
5. **Data Types**: Use appropriate data types for your database
6. **Documentation**: Include docstrings and comments

#### ❌ Don'ts

1. **Manual Editing**: Never manually edit applied migrations
2. **Data Loss**: Avoid operations that permanently delete data
3. **Large Transactions**: Split large migrations into smaller chunks
4. **Production Data**: Don't include production data in migrations
5. **Hard-coded Values**: Avoid hard-coded IDs or specific data

## Safety Validation

### Automatic Safety Checks

The migration system performs comprehensive safety validation:

```python
# Example validation output
{
    "valid": True,
    "safety_score": 85,
    "warnings": [
        "⚠️ DROP COLUMN operation found",
        "🔶 Adding NOT NULL constraint found"
    ],
    "errors": [],
    "recommendations": [
        "📋 Create database backup before running",
        "⏰ Schedule during maintenance window"
    ]
}
```

### Safety Score Interpretation

- **90-100**: Very Safe - Low risk operations
- **70-89**: Safe - Standard operations with minor risks
- **50-69**: Moderate Risk - Review carefully, test first
- **30-49**: High Risk - Staging environment testing required
- **0-29**: Very High Risk - Expert review and planning needed

### Dangerous Operations Detection

The system automatically detects potentially dangerous operations:

- **DROP TABLE**: Complete table removal
- **DROP COLUMN**: Column deletion (potential data loss)
- **ALTER TABLE**: Schema modifications that may lock tables
- **TRUNCATE**: Data deletion operations
- **Mass Updates**: Bulk data modifications

## Environment Configuration

### Database URL Configuration

```bash
# Environment variable (highest priority)
export DATABASE_URL="postgresql://user:pass@localhost/pynomaly"

# Settings file configuration
PYNOMALY_DATABASE_URL="postgresql://user:pass@localhost/pynomaly"

# Default development
# sqlite:///./storage/pynomaly.db
```

### Migration Directory Structure

```
alembic/
├── env.py                 # Alembic environment configuration
├── script.py.mako        # Migration template
└── versions/             # Migration files
    ├── 001_initial_schema.py
    ├── 002_security_hardening.py
    └── 003_user_preferences.py
```

## Production Deployment

### Pre-deployment Checklist

1. **Backup Database**: Always backup before migration
2. **Test Staging**: Run migrations on staging environment
3. **Validate Safety**: Check migration safety score
4. **Schedule Window**: Plan maintenance window for risky operations
5. **Monitor Resources**: Ensure sufficient disk space and memory
6. **Rollback Plan**: Prepare rollback procedures

### Deployment Commands

```bash
# 1. Backup database
pynomaly migrate backup /backups/pre_migration_$(date +%Y%m%d_%H%M%S).db

# 2. Validate readiness
pynomaly migrate check

# 3. Run migrations
pynomaly migrate run

# 4. Verify success
pynomaly migrate status
```

### Zero-Downtime Migrations

For zero-downtime deployments:

1. **Backward Compatible**: Ensure new migrations are backward compatible
2. **Feature Flags**: Use feature flags for new functionality
3. **Gradual Rollout**: Deploy schema changes before application changes
4. **Column Additions**: Add nullable columns first, populate later
5. **Index Creation**: Create indexes with `CONCURRENTLY` where supported

## Troubleshooting

### Common Issues

#### Migration Failed
```bash
# Check migration status
pynomaly migrate status

# Validate specific migration
pynomaly migrate validate <revision_id>

# Manual rollback if needed
pynomaly migrate rollback
```

#### Database Lock Issues
```bash
# Check for active connections
# PostgreSQL:
SELECT * FROM pg_stat_activity WHERE state = 'active';

# Terminate blocking queries (if safe)
SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
WHERE state = 'active' AND query != '<IDLE>';
```

#### Alembic Version Conflicts
```bash
# Reset to specific revision
alembic stamp <revision_id>

# Force revision update
pynomaly migrate run --target <revision_id>
```

### Recovery Procedures

#### Restore from Backup
```bash
# Stop application
systemctl stop pynomaly

# Restore database
# PostgreSQL:
psql -U user -d database < backup.sql

# SQLite:
cp backup.db current.db

# Restart application
systemctl start pynomaly
```

#### Manual Migration Fix
```python
# If migration partially applied, create fix migration
def upgrade():
    # Check if change already applied
    conn = op.get_bind()
    inspector = inspect(conn)
    
    if 'new_column' not in inspector.get_columns('table_name'):
        op.add_column('table_name', sa.Column('new_column', sa.String(255)))
```

## API Integration

### Programmatic Migration Management

```python
from pynomaly.infrastructure.persistence.migration_manager import (
    create_migration_manager,
    init_and_migrate,
    quick_migrate
)
from pynomaly.infrastructure.persistence.migration_validator import (
    MigrationValidator,
    validate_migration_safety
)

# Create migration manager
manager = create_migration_manager("postgresql://user:pass@host/db")

# Check migration status
status = manager.get_migration_status()
print(f"Current revision: {status['current_revision']}")

# Run migrations programmatically
success = manager.run_migrations()

# Validate migration safety
validator = MigrationValidator("postgresql://user:pass@host/db")
safety_result = validator.validate_migration_file("migration.py")
print(f"Safety score: {safety_result['safety_score']}")
```

### Application Startup Integration

```python
# In application startup
from pynomaly.infrastructure.persistence.migration_manager import init_and_migrate

def initialize_database():
    """Initialize database with migrations on startup."""
    try:
        success = init_and_migrate()
        if success:
            logger.info("Database migrations completed successfully")
        else:
            logger.error("Database migration failed")
            raise RuntimeError("Failed to initialize database")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise
```

## Advanced Features

### Custom Migration Templates

Create custom migration templates in `script.py.mako`:

```python
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    """Apply migration changes."""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """Revert migration changes."""
    ${downgrades if downgrades else "pass"}
```

### Migration Hooks

```python
# In env.py, add migration hooks
def run_migrations_online():
    """Run migrations in 'online' mode."""
    
    def process_revision_directives(context, revision, directives):
        """Process revision directives with custom logic."""
        # Add custom validation or modification logic
        pass
    
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            process_revision_directives=process_revision_directives
        )
        
        with context.begin_transaction():
            context.run_migrations()
```

### Branching and Merging

```bash
# Create branch
alembic revision --branch-label feature_branch -m "Start feature branch"

# Merge branches
alembic merge -m "Merge feature branch" head1 head2
```

## Monitoring and Observability

### Migration Metrics

Monitor key migration metrics:

- **Migration Duration**: Track time for each migration
- **Rollback Frequency**: Monitor rollback rates
- **Safety Scores**: Track migration safety trends
- **Database Size**: Monitor database growth
- **Lock Duration**: Track table lock times

### Logging Configuration

```python
# Configure migration logging
import logging

# Migration manager logs
logging.getLogger('pynomaly.infrastructure.persistence.migration_manager').setLevel(logging.INFO)

# Alembic logs
logging.getLogger('alembic').setLevel(logging.INFO)

# SQLAlchemy logs (for debugging)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
```

### Integration with APM

```python
# Example with OpenTelemetry
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def run_migrations_with_tracing():
    with tracer.start_as_current_span("database_migration") as span:
        span.set_attribute("migration.target", "head")
        success = manager.run_migrations()
        span.set_attribute("migration.success", success)
        return success
```

## Testing Migrations

### Testing Framework

```python
import pytest
from pynomaly.infrastructure.persistence.migration_validator import MigrationTestRunner

@pytest.fixture
def test_migration_runner():
    """Create migration test runner."""
    return MigrationTestRunner("sqlite:///test.db")

def test_migration_roundtrip(test_migration_runner, migration_manager):
    """Test migration upgrade and downgrade."""
    result = test_migration_runner.test_migration_roundtrip(
        migration_manager, 
        "abc123"
    )
    assert result["success"]
    assert result["upgrade_success"]
    assert result["downgrade_success"]

def test_data_integrity(test_migration_runner):
    """Test data integrity after migration."""
    integrity = test_migration_runner.validate_data_integrity()
    assert integrity["valid"]
    assert len(integrity["issues"]) == 0
```

### CI/CD Integration

```yaml
# .github/workflows/migrations.yml
name: Test Migrations
on: [push, pull_request]

jobs:
  test-migrations:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: pip install -r requirements.txt
        
      - name: Test migrations
        run: |
          # Run migration tests
          pytest tests/infrastructure/persistence/test_migration*
          
          # Validate all migrations
          for migration in alembic/versions/*.py; do
            python -c "
            from pynomaly.infrastructure.persistence.migration_validator import validate_migration_safety
            result = validate_migration_safety('$migration', 'sqlite:///test.db')
            assert result['overall_safety'], f'Migration {migration} failed safety check'
            "
          done
```

## Security Considerations

### Access Control

- **Database Permissions**: Use dedicated migration user with limited permissions
- **Environment Isolation**: Separate migration credentials per environment
- **Audit Logging**: Log all migration activities for compliance
- **Secret Management**: Store database credentials securely

### Migration Security

```python
# Secure migration example
def upgrade():
    """Secure migration with proper checks."""
    # Check current environment
    if os.getenv('ENVIRONMENT') == 'production':
        # Additional safety checks for production
        pass
    
    # Use parameterized queries
    op.execute(
        text("UPDATE users SET status = :status WHERE inactive_days > :days"),
        {"status": "inactive", "days": 90}
    )
```

## Performance Optimization

### Large Table Migrations

```python
def upgrade():
    """Handle large table migration efficiently."""
    # For large tables, process in batches
    batch_size = 10000
    offset = 0
    
    while True:
        result = op.execute(
            text(f"""
                UPDATE large_table 
                SET new_column = calculate_value(old_column)
                WHERE id IN (
                    SELECT id FROM large_table 
                    WHERE new_column IS NULL 
                    LIMIT {batch_size} OFFSET {offset}
                )
            """)
        )
        
        if result.rowcount == 0:
            break
            
        offset += batch_size
        
        # Optional: Add delay to reduce load
        import time
        time.sleep(0.1)
```

### Index Creation

```python
def upgrade():
    """Create indexes efficiently."""
    # Create index concurrently (PostgreSQL)
    op.create_index(
        'idx_users_email', 
        'users', 
        ['email'], 
        postgresql_concurrently=True
    )
    
    # For very large tables, consider:
    # 1. Creating partial indexes
    # 2. Using background job queues
    # 3. Splitting into multiple migrations
```

## Migration Patterns

### Common Patterns

1. **Add Column**: Safe, backward compatible
2. **Drop Column**: Requires careful planning
3. **Rename Column**: Use temporary column approach
4. **Split Table**: Create new table, migrate data, drop old
5. **Merge Tables**: Combine data, update references
6. **Change Data Type**: May require data conversion

### Pattern Examples

```python
# Safe column addition
def upgrade():
    op.add_column('users', sa.Column('middle_name', sa.String(100), nullable=True))

# Safe column removal (two-step process)
# Migration 1: Make column nullable and stop using
def upgrade_step1():
    op.alter_column('users', 'old_column', nullable=True)

# Migration 2 (later): Remove column
def upgrade_step2():
    op.drop_column('users', 'old_column')

# Safe column rename (three-step process)
# Step 1: Add new column
def upgrade_rename_1():
    op.add_column('users', sa.Column('new_name', sa.String(255)))

# Step 2: Copy data and update application
def upgrade_rename_2():
    op.execute("UPDATE users SET new_name = old_name")

# Step 3: Remove old column
def upgrade_rename_3():
    op.drop_column('users', 'old_name')
```

## Conclusion

The Pynomaly migration system provides a robust, safe, and comprehensive solution for database schema management. By following the guidelines and best practices outlined in this documentation, you can ensure reliable database evolution throughout your application's lifecycle.

For additional help or specific use cases not covered in this documentation, please refer to the [Alembic documentation](https://alembic.sqlalchemy.org/) or contact the development team.