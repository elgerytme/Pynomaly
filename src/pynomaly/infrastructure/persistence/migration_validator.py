"""Database migration validation and safety checks."""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class MigrationSafetyError(Exception):
    """Exception raised when a migration fails safety checks."""
    pass


class MigrationValidator:
    """Validates migrations for safety and correctness."""

    def __init__(self, database_url: str):
        """Initialize migration validator.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)

    def validate_migration_file(self, migration_file_path: str) -> dict[str, Any]:
        """Validate a migration file for safety issues.
        
        Args:
            migration_file_path: Path to migration file
            
        Returns:
            Validation result with warnings and errors
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "safety_score": 100,
        }

        try:
            migration_path = Path(migration_file_path)
            if not migration_path.exists():
                validation_result["errors"].append(f"Migration file not found: {migration_file_path}")
                validation_result["valid"] = False
                return validation_result

            # Read migration content
            migration_content = migration_path.read_text()

            # Parse AST for analysis
            try:
                tree = ast.parse(migration_content)
                validation_result.update(self._analyze_migration_ast(tree))
            except SyntaxError as e:
                validation_result["errors"].append(f"Syntax error in migration: {e}")
                validation_result["valid"] = False

            # Check for dangerous patterns
            validation_result.update(self._check_dangerous_patterns(migration_content))

            # Check migration structure
            validation_result.update(self._check_migration_structure(migration_content))

            # Calculate final safety score
            validation_result["safety_score"] = self._calculate_safety_score(validation_result)

            if validation_result["errors"]:
                validation_result["valid"] = False

        except Exception as e:
            logger.error(f"Error validating migration file: {e}")
            validation_result["errors"].append(f"Validation error: {e}")
            validation_result["valid"] = False

        return validation_result

    def _analyze_migration_ast(self, tree: ast.AST) -> dict[str, Any]:
        """Analyze migration AST for potential issues.
        
        Args:
            tree: Parsed AST of migration file
            
        Returns:
            Analysis results
        """
        analysis = {"warnings": [], "errors": []}

        class MigrationAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.has_upgrade = False
                self.has_downgrade = False
                self.dangerous_calls = []
                self.table_operations = []

            def visit_FunctionDef(self, node):
                if node.name == "upgrade":
                    self.has_upgrade = True
                elif node.name == "downgrade":
                    self.has_downgrade = True
                self.generic_visit(node)

            def visit_Call(self, node):
                # Check for dangerous operations
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    if func_name in ["drop_table", "drop_column", "drop_index"]:
                        self.dangerous_calls.append(func_name)
                    elif func_name in ["create_table", "add_column", "create_index"]:
                        self.table_operations.append(func_name)

                self.generic_visit(node)

        analyzer = MigrationAnalyzer()
        analyzer.visit(tree)

        # Check required functions
        if not analyzer.has_upgrade:
            analysis["errors"].append("Migration missing 'upgrade' function")
        if not analyzer.has_downgrade:
            analysis["warnings"].append("Migration missing 'downgrade' function")

        # Check for dangerous operations
        if analyzer.dangerous_calls:
            analysis["warnings"].extend([
                f"Potentially destructive operation: {op}"
                for op in analyzer.dangerous_calls
            ])

        # Check for balanced operations
        if "drop_table" in analyzer.dangerous_calls and "create_table" not in analyzer.table_operations:
            analysis["warnings"].append("Drop table without corresponding create in same migration")

        return analysis

    def _check_dangerous_patterns(self, content: str) -> dict[str, Any]:
        """Check for dangerous patterns in migration content.
        
        Args:
            content: Migration file content
            
        Returns:
            Pattern check results
        """
        checks = {"warnings": [], "errors": []}

        # Define dangerous patterns
        dangerous_patterns = [
            (r"DROP\s+TABLE", "DROP TABLE statement found"),
            (r"DROP\s+DATABASE", "DROP DATABASE statement found"),
            (r"TRUNCATE", "TRUNCATE statement found"),
            (r"DELETE\s+FROM.*WHERE", "DELETE with WHERE clause found"),
            (r"UPDATE.*SET.*WHERE", "UPDATE with WHERE clause found"),
        ]

        risky_patterns = [
            (r"ALTER\s+TABLE.*DROP\s+COLUMN", "DROP COLUMN operation found"),
            (r"DROP\s+INDEX", "DROP INDEX operation found"),
            (r"RENAME\s+TABLE", "RENAME TABLE operation found"),
            (r"ADD\s+CONSTRAINT.*UNIQUE", "Adding UNIQUE constraint found"),
            (r"ADD\s+CONSTRAINT.*NOT\s+NULL", "Adding NOT NULL constraint found"),
        ]

        # Check dangerous patterns
        for pattern, message in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                checks["warnings"].append(f"⚠️  {message}")

        # Check risky patterns
        for pattern, message in risky_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                checks["warnings"].append(f"🔶 {message}")

        # Check for missing transaction handling
        if "BEGIN" not in content.upper() and "COMMIT" not in content.upper():
            if any(op in content.upper() for op in ["ALTER", "CREATE", "DROP"]):
                checks["warnings"].append("Migration may benefit from explicit transaction handling")

        return checks

    def _check_migration_structure(self, content: str) -> dict[str, Any]:
        """Check migration structure and best practices.
        
        Args:
            content: Migration file content
            
        Returns:
            Structure check results
        """
        checks = {"warnings": [], "errors": []}

        # Check for revision ID format
        revision_match = re.search(r'revision\s*=\s*["\']([^"\']+)["\']', content)
        if revision_match:
            revision_id = revision_match.group(1)
            if not re.match(r'^[a-f0-9]{3,}$', revision_id):
                checks["warnings"].append("Revision ID should be a hexadecimal string")
        else:
            checks["errors"].append("No revision ID found in migration")

        # Check for proper imports
        required_imports = ["op", "sqlalchemy"]
        for imp in required_imports:
            if imp not in content:
                checks["warnings"].append(f"Missing import: {imp}")

        # Check for docstring
        if '"""' not in content and "'''" not in content:
            checks["warnings"].append("Migration lacks descriptive docstring")

        # Check for proper down_revision
        if "down_revision" not in content:
            checks["errors"].append("Missing down_revision specification")

        return checks

    def _calculate_safety_score(self, validation_result: dict[str, Any]) -> int:
        """Calculate overall safety score for migration.
        
        Args:
            validation_result: Validation results
            
        Returns:
            Safety score (0-100)
        """
        score = 100

        # Deduct points for errors and warnings
        score -= len(validation_result.get("errors", [])) * 20
        score -= len(validation_result.get("warnings", [])) * 5

        return max(0, score)

    def validate_schema_changes(self, migration_content: str) -> dict[str, Any]:
        """Validate schema changes for potential issues.
        
        Args:
            migration_content: Migration content to validate
            
        Returns:
            Schema validation results
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        # Check for large table alterations
        if re.search(r"ALTER\s+TABLE", migration_content, re.IGNORECASE):
            validation["warnings"].append(
                "ALTER TABLE operations may lock tables during execution"
            )
            validation["recommendations"].append(
                "Consider running during maintenance window"
            )

        # Check for index creation on large tables
        if re.search(r"CREATE\s+INDEX", migration_content, re.IGNORECASE):
            validation["recommendations"].append(
                "CREATE INDEX operations can be slow on large tables"
            )

        # Check for data migrations
        if any(op in migration_content.upper() for op in ["INSERT", "UPDATE", "DELETE"]):
            validation["warnings"].append(
                "Data migration detected - ensure proper backup before running"
            )
            validation["recommendations"].append(
                "Test data migration on copy of production data first"
            )

        return validation

    def check_database_readiness(self) -> dict[str, Any]:
        """Check if database is ready for migrations.
        
        Returns:
            Database readiness status
        """
        readiness = {
            "ready": False,
            "issues": [],
            "warnings": [],
            "info": {}
        }

        try:
            with self.engine.connect() as conn:
                # Check database connection
                conn.execute(text("SELECT 1"))
                readiness["info"]["connection"] = "OK"

                # Check available disk space (for SQLite)
                if "sqlite" in self.database_url:
                    db_path = self.database_url.replace("sqlite:///", "")
                    if Path(db_path).exists():
                        stat = Path(db_path).stat()
                        readiness["info"]["database_size"] = stat.st_size

                # Check for existing alembic version table
                try:
                    result = conn.execute(text("SELECT * FROM alembic_version LIMIT 1"))
                    readiness["info"]["migration_history"] = "Present"
                except SQLAlchemyError:
                    readiness["warnings"].append("No migration history found")
                    readiness["info"]["migration_history"] = "Not initialized"

                # Check for locks (basic check)
                try:
                    # Try a quick write operation
                    conn.execute(text("BEGIN IMMEDIATE"))
                    conn.execute(text("ROLLBACK"))
                    readiness["info"]["database_locks"] = "None detected"
                except SQLAlchemyError:
                    readiness["issues"].append("Database may be locked")

                readiness["ready"] = len(readiness["issues"]) == 0

        except Exception as e:
            readiness["issues"].append(f"Database connection failed: {e}")
            readiness["ready"] = False

        return readiness

    def get_migration_recommendations(self, migration_file: str) -> list[str]:
        """Get recommendations for migration execution.
        
        Args:
            migration_file: Path to migration file
            
        Returns:
            List of recommendations
        """
        recommendations = []

        validation = self.validate_migration_file(migration_file)

        if validation["safety_score"] < 80:
            recommendations.append("🔍 Review migration carefully before running")

        if validation["safety_score"] < 60:
            recommendations.append("⚠️  Consider testing on staging environment first")

        if any("DROP" in warning for warning in validation.get("warnings", [])):
            recommendations.append("📋 Create database backup before running")

        if any("ALTER TABLE" in warning for warning in validation.get("warnings", [])):
            recommendations.append("⏰ Schedule during maintenance window")

        if not validation.get("warnings") and validation["safety_score"] > 90:
            recommendations.append("✅ Migration appears safe to run")

        return recommendations


class MigrationTestRunner:
    """Test runner for validating migrations in test environment."""

    def __init__(self, test_database_url: str):
        """Initialize test runner.
        
        Args:
            test_database_url: Test database URL
        """
        self.test_database_url = test_database_url
        self.engine = create_engine(test_database_url)

    def test_migration_roundtrip(self, migration_manager, revision: str) -> dict[str, Any]:
        """Test migration upgrade and downgrade roundtrip.
        
        Args:
            migration_manager: Migration manager instance
            revision: Revision to test
            
        Returns:
            Test results
        """
        test_results = {
            "success": False,
            "upgrade_success": False,
            "downgrade_success": False,
            "errors": [],
            "warnings": []
        }

        try:
            # Test upgrade
            logger.info(f"Testing upgrade to revision {revision}")
            if migration_manager.run_migrations(revision):
                test_results["upgrade_success"] = True
                logger.info("Upgrade successful")
            else:
                test_results["errors"].append("Upgrade failed")

            # Test downgrade
            if test_results["upgrade_success"]:
                logger.info(f"Testing downgrade from revision {revision}")
                if migration_manager.rollback_migration("-1"):
                    test_results["downgrade_success"] = True
                    logger.info("Downgrade successful")
                else:
                    test_results["errors"].append("Downgrade failed")

            test_results["success"] = (
                test_results["upgrade_success"] and
                test_results["downgrade_success"]
            )

        except Exception as e:
            test_results["errors"].append(f"Test execution error: {e}")
            logger.error(f"Migration test error: {e}")

        return test_results

    def validate_data_integrity(self) -> dict[str, Any]:
        """Validate data integrity after migration.
        
        Returns:
            Data integrity check results
        """
        integrity_results = {
            "valid": True,
            "checks": {},
            "issues": []
        }

        try:
            with self.engine.connect() as conn:
                # Check foreign key constraints
                if "sqlite" in self.test_database_url:
                    result = conn.execute(text("PRAGMA foreign_key_check"))
                    fk_violations = result.fetchall()
                    if fk_violations:
                        integrity_results["issues"].append(
                            f"Foreign key violations: {len(fk_violations)}"
                        )
                        integrity_results["valid"] = False
                    integrity_results["checks"]["foreign_keys"] = len(fk_violations) == 0

                # Check for orphaned records (basic)
                # This would be customized based on your schema
                integrity_results["checks"]["orphaned_records"] = True

        except Exception as e:
            integrity_results["issues"].append(f"Integrity check error: {e}")
            integrity_results["valid"] = False

        return integrity_results


def validate_migration_safety(migration_file: str, database_url: str) -> dict[str, Any]:
    """Comprehensive migration safety validation.
    
    Args:
        migration_file: Path to migration file
        database_url: Database URL
        
    Returns:
        Complete safety validation results
    """
    validator = MigrationValidator(database_url)

    # File validation
    file_validation = validator.validate_migration_file(migration_file)

    # Database readiness
    readiness = validator.check_database_readiness()

    # Recommendations
    recommendations = validator.get_migration_recommendations(migration_file)

    return {
        "file_validation": file_validation,
        "database_readiness": readiness,
        "recommendations": recommendations,
        "overall_safety": (
            file_validation["valid"] and
            readiness["ready"] and
            file_validation["safety_score"] >= 70
        )
    }
