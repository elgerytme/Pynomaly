"""Security hardening and additional constraints

Revision ID: 002
Revises: 001
Create Date: 2025-07-09 16:40:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add additional security constraints

    # Add password policy tracking
    op.add_column(
        "users", sa.Column("password_changed_at", sa.DateTime, default=sa.func.now())
    )
    op.add_column("users", sa.Column("password_reset_token", sa.String(255)))
    op.add_column("users", sa.Column("password_reset_expires", sa.DateTime))
    op.add_column("users", sa.Column("failed_login_attempts", sa.Integer, default=0))
    op.add_column("users", sa.Column("locked_until", sa.DateTime))
    op.add_column("users", sa.Column("two_factor_enabled", sa.Boolean, default=False))
    op.add_column("users", sa.Column("two_factor_secret", sa.String(255)))

    # Add audit logging table
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50)),
        sa.Column("resource_id", sa.String(255)),
        sa.Column("details", sa.JSON),
        sa.Column("ip_address", sa.String(45)),
        sa.Column("user_agent", sa.Text),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime, default=sa.func.now()),
        sa.Column("session_id", sa.String(255)),
    )

    # Add API keys table
    op.create_table(
        "api_keys",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("key_hash", sa.String(255), nullable=False, unique=True),
        sa.Column("key_prefix", sa.String(10), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("permissions", sa.JSON, default={}),
        sa.Column("last_used_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("rate_limit_per_minute", sa.Integer, default=100),
    )

    # Add security events table
    op.create_table(
        "security_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("tenant_id", UUID(as_uuid=True), sa.ForeignKey("tenants.id")),
        sa.Column("ip_address", sa.String(45)),
        sa.Column("user_agent", sa.Text),
        sa.Column("details", sa.JSON),
        sa.Column("timestamp", sa.DateTime, default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime),
        sa.Column("resolved_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
    )

    # Add tenant security settings
    op.add_column("tenants", sa.Column("security_settings", sa.JSON, default={}))
    op.add_column("tenants", sa.Column("password_policy", sa.JSON, default={}))
    op.add_column(
        "tenants", sa.Column("session_timeout_minutes", sa.Integer, default=60)
    )
    op.add_column(
        "tenants", sa.Column("max_concurrent_sessions", sa.Integer, default=5)
    )
    op.add_column("tenants", sa.Column("require_2fa", sa.Boolean, default=False))
    op.add_column("tenants", sa.Column("allowed_ip_ranges", sa.JSON, default=[]))

    # Add model versioning table
    op.create_table(
        "model_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "detector_id",
            UUID(as_uuid=True),
            sa.ForeignKey("detectors.id"),
            nullable=False,
        ),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("model_path", sa.String(500)),
        sa.Column("model_hash", sa.String(255)),
        sa.Column("performance_metrics", sa.JSON),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("is_active", sa.Boolean, default=False),
        sa.Column("metadata", sa.JSON, default={}),
    )

    # Add data lineage table
    op.create_table(
        "data_lineage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), nullable=False),
        sa.Column("target_type", sa.String(50), nullable=False),
        sa.Column("relationship_type", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime, default=sa.func.now()),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
        sa.Column("metadata", sa.JSON, default={}),
    )

    # Create additional indexes for security and performance
    op.create_index("idx_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("idx_audit_logs_tenant_id", "audit_logs", ["tenant_id"])
    op.create_index("idx_audit_logs_timestamp", "audit_logs", ["timestamp"])
    op.create_index("idx_audit_logs_action", "audit_logs", ["action"])

    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("idx_api_keys_tenant_id", "api_keys", ["tenant_id"])
    op.create_index("idx_api_keys_key_hash", "api_keys", ["key_hash"])
    op.create_index("idx_api_keys_expires_at", "api_keys", ["expires_at"])

    op.create_index("idx_security_events_event_type", "security_events", ["event_type"])
    op.create_index("idx_security_events_severity", "security_events", ["severity"])
    op.create_index("idx_security_events_timestamp", "security_events", ["timestamp"])
    op.create_index("idx_security_events_user_id", "security_events", ["user_id"])

    op.create_index("idx_model_versions_detector_id", "model_versions", ["detector_id"])
    op.create_index("idx_model_versions_version", "model_versions", ["version"])
    op.create_index("idx_model_versions_is_active", "model_versions", ["is_active"])

    op.create_index(
        "idx_data_lineage_source", "data_lineage", ["source_id", "source_type"]
    )
    op.create_index(
        "idx_data_lineage_target", "data_lineage", ["target_id", "target_type"]
    )

    # Add constraints
    op.create_check_constraint(
        "ck_users_failed_login_attempts", "users", "failed_login_attempts >= 0"
    )

    op.create_check_constraint(
        "ck_security_events_severity",
        "security_events",
        "severity IN ('low', 'medium', 'high', 'critical')",
    )

    op.create_check_constraint(
        "ck_audit_logs_outcome",
        "audit_logs",
        "outcome IN ('success', 'failure', 'warning')",
    )


def downgrade() -> None:
    # Drop constraints
    op.drop_constraint("ck_audit_logs_outcome", "audit_logs")
    op.drop_constraint("ck_security_events_severity", "security_events")
    op.drop_constraint("ck_users_failed_login_attempts", "users")

    # Drop indexes
    op.drop_index("idx_data_lineage_target")
    op.drop_index("idx_data_lineage_source")
    op.drop_index("idx_model_versions_is_active")
    op.drop_index("idx_model_versions_version")
    op.drop_index("idx_model_versions_detector_id")
    op.drop_index("idx_security_events_user_id")
    op.drop_index("idx_security_events_timestamp")
    op.drop_index("idx_security_events_severity")
    op.drop_index("idx_security_events_event_type")
    op.drop_index("idx_api_keys_expires_at")
    op.drop_index("idx_api_keys_key_hash")
    op.drop_index("idx_api_keys_tenant_id")
    op.drop_index("idx_api_keys_user_id")
    op.drop_index("idx_audit_logs_action")
    op.drop_index("idx_audit_logs_timestamp")
    op.drop_index("idx_audit_logs_tenant_id")
    op.drop_index("idx_audit_logs_user_id")

    # Drop tables
    op.drop_table("data_lineage")
    op.drop_table("model_versions")
    op.drop_table("security_events")
    op.drop_table("api_keys")
    op.drop_table("audit_logs")

    # Drop tenant columns
    op.drop_column("tenants", "allowed_ip_ranges")
    op.drop_column("tenants", "require_2fa")
    op.drop_column("tenants", "max_concurrent_sessions")
    op.drop_column("tenants", "session_timeout_minutes")
    op.drop_column("tenants", "password_policy")
    op.drop_column("tenants", "security_settings")

    # Drop user columns
    op.drop_column("users", "two_factor_secret")
    op.drop_column("users", "two_factor_enabled")
    op.drop_column("users", "locked_until")
    op.drop_column("users", "failed_login_attempts")
    op.drop_column("users", "password_reset_expires")
    op.drop_column("users", "password_reset_token")
    op.drop_column("users", "password_changed_at")
