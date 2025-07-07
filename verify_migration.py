#!/usr/bin/env python3
"""Script to verify the migration results."""

import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from pynomaly.infrastructure.persistence.database import init_database, SQLITE_FILE_URL
from pynomaly.infrastructure.persistence.database_repositories import RoleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Verify the migration results."""
    try:
        logger.info("Verifying migration results...")
        
        # Initialize database connection
        db_manager = init_database(SQLITE_FILE_URL, echo=False)
        
        # Check table existence
        engine = db_manager.engine
        from sqlalchemy import inspect
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        expected_tables = ['users', 'roles', 'tenants', 'user_roles', 'datasets', 'detectors', 'detection_results']
        
        logger.info(f"Existing tables: {table_names}")
        
        for table in expected_tables:
            if table in table_names:
                logger.info(f"✓ Table '{table}' exists")
            else:
                logger.error(f"✗ Table '{table}' missing")
        
        # Check roles
        session_gen = db_manager.get_session()
        session = next(session_gen)
        try:
            roles = session.query(RoleModel).all()
            logger.info(f"\nRoles in database ({len(roles)} total):")
            
            for role in roles:
                role_type = "SYSTEM" if role.is_system_role else "CUSTOM"
                perm_count = len(role.permissions) if role.permissions else 0
                logger.info(f"  - {role.name} ({role_type}): {perm_count} permissions")
                logger.info(f"    Description: {role.description}")
                
        finally:
            session.close()
            
        logger.info("\nMigration verification completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Error verifying migration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
