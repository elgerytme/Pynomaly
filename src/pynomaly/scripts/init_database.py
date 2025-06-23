#!/usr/bin/env python3
"""Database initialization script for Pynomaly."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pynomaly.infrastructure.config.settings import Settings
from pynomaly.infrastructure.persistence.migrations import create_database_migrator
from pynomaly.infrastructure.persistence.database import (
    SQLITE_FILE_URL,
    POSTGRESQL_LOCAL_URL,
    SQLITE_MEMORY_URL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for database initialization."""
    parser = argparse.ArgumentParser(description="Initialize Pynomaly database")
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL (overrides config)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database (drop and recreate tables)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check database status, don't modify"
    )
    parser.add_argument(
        "--preset",
        choices=["sqlite-file", "sqlite-memory", "postgresql-local"],
        help="Use predefined database configuration"
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo SQL statements"
    )
    
    args = parser.parse_args()
    
    # Determine database URL
    database_url = None
    
    if args.preset:
        preset_urls = {
            "sqlite-file": SQLITE_FILE_URL,
            "sqlite-memory": SQLITE_MEMORY_URL,
            "postgresql-local": POSTGRESQL_LOCAL_URL
        }
        database_url = preset_urls[args.preset]
        logger.info(f"Using preset '{args.preset}': {database_url}")
    
    elif args.database_url:
        database_url = args.database_url
        logger.info(f"Using provided URL: {database_url}")
    
    else:
        # Try to load from settings
        try:
            settings = Settings()
            if settings.database_url:
                database_url = settings.database_url
                logger.info(f"Using URL from settings: {database_url}")
            else:
                logger.error("No database URL configured")
                logger.info("Use --database-url, --preset, or configure PYNOMALY_DATABASE_URL")
                return 1
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return 1
    
    # Create migrator
    try:
        migrator = create_database_migrator(database_url, echo=args.echo)
    except Exception as e:
        logger.error(f"Failed to create database migrator: {e}")
        return 1
    
    # Check database status
    logger.info("Checking database status...")
    info = migrator.get_database_info()
    
    print("\n=== Database Status ===")
    print(f"Database URL: {info['database_url']}")
    print(f"Connection working: {info['connection_working']}")
    print(f"Required tables exist: {info['tables_exist']}")
    print(f"Total tables: {info['table_count']}")
    
    if info.get('tables'):
        print(f"Tables: {', '.join(info['tables'])}")
    
    if info.get('engine_info'):
        engine_info = info['engine_info']
        print(f"Database driver: {engine_info.get('driver', 'unknown')}")
        print(f"Database dialect: {engine_info.get('dialect', 'unknown')}")
    
    # If only checking, exit here
    if args.check_only:
        return 0 if info['connection_working'] else 1
    
    # Perform database operations
    if args.reset:
        logger.info("Resetting database...")
        if migrator.reset_database():
            logger.info("Database reset completed successfully")
        else:
            logger.error("Database reset failed")
            return 1
    
    else:
        logger.info("Initializing database...")
        if migrator.initialize_database():
            logger.info("Database initialization completed successfully")
        else:
            logger.error("Database initialization failed")
            return 1
    
    # Final status check
    final_info = migrator.get_database_info()
    if final_info['connection_working'] and final_info['tables_exist']:
        logger.info("Database is ready for use")
        return 0
    else:
        logger.error("Database is not ready")
        return 1


if __name__ == "__main__":
    sys.exit(main())