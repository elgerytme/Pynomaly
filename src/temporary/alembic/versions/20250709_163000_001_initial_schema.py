"""Initial database schema

Revision ID: 001
Revises:
Create Date: 2025-07-09 16:30:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True, index=True),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()
        ),
        sa.Column("last_login_at", sa.DateTime),
        sa.Column("email_verified_at", sa.DateTime),
        sa.Column("settings", sa.JSON, default={}),
    )

    # Create tenants table
    op.create_table(
        "tenants",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("domain", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("plan", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()
        ),
        sa.Column("expires_at", sa.DateTime),
        sa.Column("contact_email", sa.String(255), default=""),
        sa.Column("billing_email", sa.String(255), default=""),
        sa.Column("settings", sa.JSON, default={}),
        # Limits
        sa.Column("max_users", sa.Integer, default=10),
        sa.Column("max_datasets", sa.Integer, default=100),
        sa.Column("max_models", sa.Integer, default=50),
        sa.Column("max_detections_per_month", sa.Integer, default=10000),
        sa.Column("max_storage_gb", sa.Integer, default=10),
        sa.Column("max_api_calls_per_minute", sa.Integer, default=100),
        sa.Column("max_concurrent_detections", sa.Integer, default=5),
        # Usage
        sa.Column("users_count", sa.Integer, default=0),
        sa.Column("datasets_count", sa.Integer, default=0),
        sa.Column("models_count", sa.Integer, default=0),
        sa.Column("detections_this_month", sa.Integer, default=0),
        sa.Column("storage_used_gb", sa.Float, default=0.0),
        sa.Column("api_calls_this_minute", sa.Integer, default=0),
        sa.Column("concurrent_detections", sa.Integer, default=0),
        sa.Column("usage_last_updated", sa.DateTime, default=sa.func.now()),
    )

    # Create user_tenant_roles association table
    op.create_table(
        "user_tenant_roles",
        sa.Column(
            "user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), primary_key=True
        ),
        sa.Column(
            "tenant_id",
            UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            primary_key=True,
        ),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("permissions", sa.JSON),
        sa.Column("granted_at", sa.DateTime, default=sa.func.now()),
        sa.Column("granted_by", UUID(as_uuid=True)),
        sa.Column("expires_at", sa.DateTime),
    )

    # Create user_sessions table
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime, nullable=False),
        sa.Column("ip_address", sa.String(45), default=""),
        sa.Column("user_agent", sa.Text, default=""),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("last_activity", sa.DateTime, default=sa.func.now()),
    )

    # Create datasets table
    op.create_table(
        "datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("file_path", sa.String(500)),
        sa.Column("file_type", sa.String(50)),
        sa.Column("size_bytes", sa.BigInteger),
        sa.Column("row_count", sa.Integer),
        sa.Column("column_count", sa.Integer),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()
        ),
        sa.Column("metadata", sa.JSON, default={}),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
    )

    # Create detectors table
    op.create_table(
        "detectors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("algorithm_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("parameters", sa.JSON, default={}),
        sa.Column("is_fitted", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime, default=sa.func.now(), onupdate=sa.func.now()
        ),
        sa.Column("metadata", sa.JSON, default={}),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
    )

    # Create detection_results table
    op.create_table(
        "detection_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "detector_id",
            UUID(as_uuid=True),
            sa.ForeignKey("detectors.id"),
            nullable=False,
        ),
        sa.Column(
            "dataset_id",
            UUID(as_uuid=True),
            sa.ForeignKey("datasets.id"),
            nullable=False,
        ),
        sa.Column("anomaly_scores", sa.JSON),
        sa.Column("anomaly_labels", sa.JSON),
        sa.Column("execution_time_ms", sa.Float),
        sa.Column("performance_metrics", sa.JSON),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column("metadata", sa.JSON, default={}),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
    )

    # Create indexes
    op.create_index("idx_users_email", "users", ["email"])
    op.create_index("idx_users_username", "users", ["username"])
    op.create_index("idx_tenants_domain", "tenants", ["domain"])
    op.create_index("idx_user_sessions_user_id", "user_sessions", ["user_id"])
    op.create_index("idx_user_sessions_expires_at", "user_sessions", ["expires_at"])
    op.create_index("idx_datasets_tenant_id", "datasets", ["tenant_id"])
    op.create_index("idx_detectors_tenant_id", "detectors", ["tenant_id"])
    op.create_index(
        "idx_detection_results_detector_id", "detection_results", ["detector_id"]
    )
    op.create_index(
        "idx_detection_results_dataset_id", "detection_results", ["dataset_id"]
    )
    op.create_index(
        "idx_detection_results_tenant_id", "detection_results", ["tenant_id"]
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("idx_detection_results_tenant_id")
    op.drop_index("idx_detection_results_dataset_id")
    op.drop_index("idx_detection_results_detector_id")
    op.drop_index("idx_detectors_tenant_id")
    op.drop_index("idx_datasets_tenant_id")
    op.drop_index("idx_user_sessions_expires_at")
    op.drop_index("idx_user_sessions_user_id")
    op.drop_index("idx_tenants_domain")
    op.drop_index("idx_users_username")
    op.drop_index("idx_users_email")

    # Drop tables
    op.drop_table("detection_results")
    op.drop_table("detectors")
    op.drop_table("datasets")
    op.drop_table("user_sessions")
    op.drop_table("user_tenant_roles")
    op.drop_table("tenants")
    op.drop_table("users")
