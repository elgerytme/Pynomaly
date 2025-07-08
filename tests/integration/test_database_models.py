#!/usr/bin/env python3
"""Test that database models can be created without import errors."""

import sys
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the src directory to the path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Import database models directly to check for metadata issues
    from pynomaly.infrastructure.persistence.database_repositories import (
        Base, UserModel, TenantModel, RoleModel, UserRoleModel, MetricModel
    )
    print("✓ Successfully imported all database models")

    # Test that models can be created with SQLAlchemy
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    print("✓ Successfully created all tables in memory database")

    # Test some basic operations
    Session = sessionmaker(bind=engine)
    session = Session()

    # Try to query the tables (should be empty)
    users_count = session.query(UserModel).count()
    tenants_count = session.query(TenantModel).count()
    roles_count = session.query(RoleModel).count()
    metrics_count = session.query(MetricModel).count()

    print(f"✓ Database query tests passed: {users_count} users, {tenants_count} tenants, {roles_count} roles, {metrics_count} metrics")

    session.close()
    print("\n🎉 All database model tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
