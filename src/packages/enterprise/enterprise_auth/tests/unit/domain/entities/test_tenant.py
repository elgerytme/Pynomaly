"""
Unit tests for Tenant domain entity.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_auth.domain.entities.tenant import (
    Tenant, TenantInvitation, TenantStatus, TenantPlan, TenantFeature
)


class TestTenant:
    """Test cases for Tenant entity."""
    
    def test_tenant_creation_with_defaults(self):
        """Test tenant creation with default values."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com"
        )
        
        assert isinstance(tenant.id, UUID)
        assert tenant.name == "Test Company"
        assert tenant.slug == "test-company"
        assert tenant.admin_email == "admin@test.com"
        assert tenant.plan == TenantPlan.FREE
        assert tenant.status == TenantStatus.TRIAL
        assert tenant.enabled_features == []
        assert tenant.max_users == 5
        assert tenant.max_data_retention_days == 30
        assert tenant.max_api_calls_per_month == 1000
        assert tenant.max_storage_gb == 1
        assert tenant.settings == {}
        assert tenant.password_policy == {}
        assert tenant.session_timeout_minutes == 480
        assert tenant.mfa_required is False
        assert tenant.ip_whitelist == []
        assert tenant.sso_enabled is False
        assert tenant.current_user_count == 0
        assert tenant.current_storage_gb == 0.0
        assert tenant.api_calls_this_month == 0
        
    def test_tenant_creation_with_all_fields(self):
        """Test tenant creation with all fields specified."""
        created_by = uuid4()
        features = [TenantFeature.SSO, TenantFeature.SAML]
        settings = {"theme": "dark", "timezone": "UTC"}
        password_policy = {"min_length": 8, "require_uppercase": True}
        ip_whitelist = ["192.168.1.0/24", "10.0.0.0/8"]
        saml_config = {"entity_id": "test", "sso_url": "https://sso.test.com"}
        
        trial_ends = datetime.utcnow() + timedelta(days=30)
        
        tenant = Tenant(
            name="Enterprise Corp",
            slug="ENTERPRISE_CORP",
            description="Large enterprise tenant",
            admin_email="ADMIN@ENTERPRISE.COM",
            support_email="support@enterprise.com",
            website="https://enterprise.com",
            phone="+1234567890",
            plan=TenantPlan.ENTERPRISE,
            status=TenantStatus.ACTIVE,
            trial_ends_at=trial_ends,
            enabled_features=features,
            max_users=1000,
            max_data_retention_days=1095,
            max_api_calls_per_month=1000000,
            max_storage_gb=1000,
            settings=settings,
            custom_domain="app.enterprise.com",
            logo_url="https://enterprise.com/logo.png",
            theme_config={"primary_color": "#003366"},
            password_policy=password_policy,
            session_timeout_minutes=240,
            mfa_required=True,
            ip_whitelist=ip_whitelist,
            sso_enabled=True,
            saml_config=saml_config,
            current_user_count=500,
            current_storage_gb=750.5,
            api_calls_this_month=50000,
            created_by=created_by
        )
        
        assert tenant.name == "Enterprise Corp"
        assert tenant.slug == "enterprise_corp"  # Should be lowercase
        assert tenant.description == "Large enterprise tenant"
        assert tenant.admin_email == "admin@enterprise.com"  # Should be lowercase
        assert tenant.support_email == "support@enterprise.com"
        assert tenant.website == "https://enterprise.com"
        assert tenant.phone == "+1234567890"
        assert tenant.plan == TenantPlan.ENTERPRISE
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.trial_ends_at == trial_ends
        assert tenant.enabled_features == features
        assert tenant.max_users == 1000
        assert tenant.max_data_retention_days == 1095
        assert tenant.max_api_calls_per_month == 1000000
        assert tenant.max_storage_gb == 1000
        assert tenant.settings == settings
        assert tenant.custom_domain == "app.enterprise.com"
        assert tenant.logo_url == "https://enterprise.com/logo.png"
        assert tenant.theme_config == {"primary_color": "#003366"}
        assert tenant.password_policy == password_policy
        assert tenant.session_timeout_minutes == 240
        assert tenant.mfa_required is True
        assert tenant.ip_whitelist == ip_whitelist
        assert tenant.sso_enabled is True
        assert tenant.saml_config == saml_config
        assert tenant.current_user_count == 500
        assert tenant.current_storage_gb == 750.5
        assert tenant.api_calls_this_month == 50000
        assert tenant.created_by == created_by
        
    def test_slug_validation_lowercase(self):
        """Test slug is converted to lowercase."""
        tenant = Tenant(
            name="Test Company",
            slug="TEST-COMPANY_123",
            admin_email="admin@test.com"
        )
        assert tenant.slug == "test-company_123"
        
    def test_slug_validation_invalid_characters(self):
        """Test slug validation with invalid characters."""
        with pytest.raises(ValueError, match="Slug must contain only alphanumeric characters"):
            Tenant(
                name="Test Company",
                slug="test@company",
                admin_email="admin@test.com"
            )
            
    def test_admin_email_validation_lowercase(self):
        """Test admin email is converted to lowercase."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="ADMIN@TEST.COM"
        )
        assert tenant.admin_email == "admin@test.com"
        
    def test_enabled_features_string_conversion(self):
        """Test enabled features string is converted to list."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            enabled_features="sso"
        )
        assert tenant.enabled_features == ["sso"]
        
    def test_enabled_features_none_conversion(self):
        """Test enabled features None is converted to empty list."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            enabled_features=None
        )
        assert tenant.enabled_features == []
        
    def test_has_feature(self):
        """Test has_feature method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            enabled_features=[TenantFeature.SSO, TenantFeature.SAML]
        )
        
        assert tenant.has_feature(TenantFeature.SSO) is True
        assert tenant.has_feature(TenantFeature.SAML) is True
        assert tenant.has_feature(TenantFeature.LDAP) is False
        
    def test_enable_feature(self):
        """Test enable_feature method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com"
        )
        
        original_updated_at = tenant.updated_at
        tenant.enable_feature(TenantFeature.SSO)
        
        assert TenantFeature.SSO in tenant.enabled_features
        assert tenant.updated_at > original_updated_at
        
    def test_enable_feature_already_enabled(self):
        """Test enable_feature method with already enabled feature."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            enabled_features=[TenantFeature.SSO]
        )
        
        original_updated_at = tenant.updated_at
        tenant.enable_feature(TenantFeature.SSO)
        
        # Should not add duplicate
        assert tenant.enabled_features.count(TenantFeature.SSO) == 1
        assert tenant.updated_at > original_updated_at
        
    def test_disable_feature(self):
        """Test disable_feature method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            enabled_features=[TenantFeature.SSO, TenantFeature.SAML]
        )
        
        original_updated_at = tenant.updated_at
        tenant.disable_feature(TenantFeature.SSO)
        
        assert TenantFeature.SSO not in tenant.enabled_features
        assert TenantFeature.SAML in tenant.enabled_features
        assert tenant.updated_at > original_updated_at
        
    def test_disable_feature_not_enabled(self):
        """Test disable_feature method with non-enabled feature."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com"
        )
        
        # Should not raise error
        tenant.disable_feature(TenantFeature.SSO)
        assert TenantFeature.SSO not in tenant.enabled_features
        
    def test_is_active_true(self):
        """Test is_active returns True for active tenant."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE
        )
        assert tenant.is_active() is True
        
    def test_is_active_trial(self):
        """Test is_active returns True for trial tenant."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL,
            trial_ends_at=datetime.utcnow() + timedelta(days=30)
        )
        assert tenant.is_active() is True
        
    def test_is_active_false_suspended(self):
        """Test is_active returns False for suspended tenant."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.SUSPENDED
        )
        assert tenant.is_active() is False
        
    def test_is_active_false_deleted(self):
        """Test is_active returns False for deleted tenant."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE,
            deleted_at=datetime.utcnow()
        )
        assert tenant.is_active() is False
        
    def test_is_active_false_expired(self):
        """Test is_active returns False for expired tenant."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL,
            trial_ends_at=datetime.utcnow() - timedelta(days=1)
        )
        assert tenant.is_active() is False
        
    def test_is_expired_false_no_expiration(self):
        """Test is_expired returns False when no expiration date."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE
        )
        assert tenant.is_expired() is False
        
    def test_is_expired_false_future_trial(self):
        """Test is_expired returns False for future trial expiration."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL,
            trial_ends_at=datetime.utcnow() + timedelta(days=30)
        )
        assert tenant.is_expired() is False
        
    def test_is_expired_true_past_trial(self):
        """Test is_expired returns True for past trial expiration."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL,
            trial_ends_at=datetime.utcnow() - timedelta(days=1)
        )
        assert tenant.is_expired() is True
        
    def test_is_expired_false_future_subscription(self):
        """Test is_expired returns False for future subscription expiration."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE,
            subscription_ends_at=datetime.utcnow() + timedelta(days=30)
        )
        assert tenant.is_expired() is False
        
    def test_is_expired_true_past_subscription(self):
        """Test is_expired returns True for past subscription expiration."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE,
            subscription_ends_at=datetime.utcnow() - timedelta(days=1)
        )
        assert tenant.is_expired() is True
        
    def test_is_trial(self):
        """Test is_trial method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL
        )
        assert tenant.is_trial() is True
        
        tenant.status = TenantStatus.ACTIVE
        assert tenant.is_trial() is False
        
    def test_can_add_user_true(self):
        """Test can_add_user returns True when under limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_users=10,
            current_user_count=5
        )
        assert tenant.can_add_user() is True
        
    def test_can_add_user_false(self):
        """Test can_add_user returns False when at limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_users=10,
            current_user_count=10
        )
        assert tenant.can_add_user() is False
        
    def test_can_use_storage_true(self):
        """Test can_use_storage returns True when under limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_storage_gb=100,
            current_storage_gb=50.0
        )
        assert tenant.can_use_storage(25.0) is True
        
    def test_can_use_storage_false(self):
        """Test can_use_storage returns False when over limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_storage_gb=100,
            current_storage_gb=80.0
        )
        assert tenant.can_use_storage(25.0) is False
        
    def test_can_make_api_call_true(self):
        """Test can_make_api_call returns True when under limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_api_calls_per_month=1000,
            api_calls_this_month=500
        )
        assert tenant.can_make_api_call() is True
        
    def test_can_make_api_call_false(self):
        """Test can_make_api_call returns False when at limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_api_calls_per_month=1000,
            api_calls_this_month=1000
        )
        assert tenant.can_make_api_call() is False
        
    def test_increment_user_count(self):
        """Test increment_user_count method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            current_user_count=5
        )
        
        original_updated_at = tenant.updated_at
        tenant.increment_user_count()
        
        assert tenant.current_user_count == 6
        assert tenant.updated_at > original_updated_at
        
    def test_decrement_user_count(self):
        """Test decrement_user_count method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            current_user_count=5
        )
        
        original_updated_at = tenant.updated_at
        tenant.decrement_user_count()
        
        assert tenant.current_user_count == 4
        assert tenant.updated_at > original_updated_at
        
    def test_decrement_user_count_zero(self):
        """Test decrement_user_count method with zero count."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            current_user_count=0
        )
        
        tenant.decrement_user_count()
        
        assert tenant.current_user_count == 0  # Should not go negative
        
    def test_increment_api_calls(self):
        """Test increment_api_calls method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            api_calls_this_month=100
        )
        
        original_updated_at = tenant.updated_at
        tenant.increment_api_calls(5)
        
        assert tenant.api_calls_this_month == 105
        assert tenant.last_activity_at is not None
        assert tenant.updated_at > original_updated_at
        
    def test_increment_api_calls_default(self):
        """Test increment_api_calls method with default increment."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            api_calls_this_month=100
        )
        
        tenant.increment_api_calls()
        
        assert tenant.api_calls_this_month == 101
        
    def test_update_storage_usage(self):
        """Test update_storage_usage method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            current_storage_gb=50.0
        )
        
        original_updated_at = tenant.updated_at
        tenant.update_storage_usage(75.5)
        
        assert tenant.current_storage_gb == 75.5
        assert tenant.updated_at > original_updated_at
        
    def test_upgrade_plan(self):
        """Test upgrade_plan method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            plan=TenantPlan.FREE,
            status=TenantStatus.TRIAL
        )
        
        updated_by = uuid4()
        original_updated_at = tenant.updated_at
        
        tenant.upgrade_plan(TenantPlan.PROFESSIONAL, updated_by)
        
        assert tenant.plan == TenantPlan.PROFESSIONAL
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.subscription_starts_at is not None
        assert tenant.updated_by == updated_by
        assert tenant.updated_at > original_updated_at
        
        # Check limits are updated
        assert tenant.max_users == 100
        assert tenant.max_api_calls_per_month == 100000
        
        # Check features are updated
        assert TenantFeature.API_ACCESS in tenant.enabled_features
        assert TenantFeature.SSO in tenant.enabled_features
        
    def test_upgrade_plan_custom(self):
        """Test upgrade_plan method with custom plan."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            plan=TenantPlan.PROFESSIONAL,
            max_users=500  # Custom limit
        )
        
        tenant.upgrade_plan(TenantPlan.CUSTOM)
        
        assert tenant.plan == TenantPlan.CUSTOM
        assert tenant.max_users == 500  # Should maintain custom limits
        
    def test_suspend(self):
        """Test suspend method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE
        )
        
        suspended_by = uuid4()
        reason = "Payment failed"
        original_updated_at = tenant.updated_at
        
        tenant.suspend(suspended_by, reason)
        
        assert tenant.status == TenantStatus.SUSPENDED
        assert tenant.updated_by == suspended_by
        assert tenant.updated_at > original_updated_at
        assert tenant.settings["suspension_reason"] == reason
        assert "suspended_at" in tenant.settings
        
    def test_reactivate(self):
        """Test reactivate method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.SUSPENDED,
            settings={"suspension_reason": "Payment failed", "suspended_at": "2023-01-01"}
        )
        
        reactivated_by = uuid4()
        original_updated_at = tenant.updated_at
        
        tenant.reactivate(reactivated_by)
        
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.updated_by == reactivated_by
        assert tenant.updated_at > original_updated_at
        assert "suspension_reason" not in tenant.settings
        assert "suspended_at" not in tenant.settings
        
    def test_soft_delete(self):
        """Test soft_delete method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE
        )
        
        deleted_by = uuid4()
        tenant.soft_delete(deleted_by)
        
        assert tenant.deleted_at is not None
        assert tenant.deleted_by == deleted_by
        assert tenant.status == TenantStatus.DELETED
        
    def test_restore(self):
        """Test restore method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.DELETED,
            deleted_at=datetime.utcnow(),
            deleted_by=uuid4()
        )
        
        tenant.restore()
        
        assert tenant.deleted_at is None
        assert tenant.deleted_by is None
        assert tenant.status == TenantStatus.INACTIVE
        
    def test_reset_monthly_usage(self):
        """Test reset_monthly_usage method."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            api_calls_this_month=5000
        )
        
        original_updated_at = tenant.updated_at
        tenant.reset_monthly_usage()
        
        assert tenant.api_calls_this_month == 0
        assert tenant.updated_at > original_updated_at
        
    def test_is_deleted_property(self):
        """Test is_deleted property."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com"
        )
        
        assert tenant.is_deleted is False
        
        tenant.deleted_at = datetime.utcnow()
        assert tenant.is_deleted is True
        
    def test_days_until_trial_expiry(self):
        """Test days_until_trial_expiry property."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL,
            trial_ends_at=datetime.utcnow() + timedelta(days=15)
        )
        
        days = tenant.days_until_trial_expiry
        assert days is not None
        assert 14 <= days <= 15  # Account for timing differences
        
    def test_days_until_trial_expiry_not_trial(self):
        """Test days_until_trial_expiry property for non-trial tenant."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE
        )
        
        assert tenant.days_until_trial_expiry is None
        
    def test_days_until_trial_expiry_expired(self):
        """Test days_until_trial_expiry property for expired trial."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.TRIAL,
            trial_ends_at=datetime.utcnow() - timedelta(days=5)
        )
        
        assert tenant.days_until_trial_expiry == 0
        
    def test_storage_usage_percentage(self):
        """Test storage_usage_percentage property."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_storage_gb=100,
            current_storage_gb=75.0
        )
        
        assert tenant.storage_usage_percentage == 75.0
        
    def test_storage_usage_percentage_zero_limit(self):
        """Test storage_usage_percentage property with zero limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_storage_gb=0,
            current_storage_gb=50.0
        )
        
        assert tenant.storage_usage_percentage == 0.0
        
    def test_api_usage_percentage(self):
        """Test api_usage_percentage property."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_api_calls_per_month=10000,
            api_calls_this_month=2500
        )
        
        assert tenant.api_usage_percentage == 25.0
        
    def test_api_usage_percentage_zero_limit(self):
        """Test api_usage_percentage property with zero limit."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            max_api_calls_per_month=0,
            api_calls_this_month=1000
        )
        
        assert tenant.api_usage_percentage == 0.0
        
    def test_to_dict_without_sensitive(self):
        """Test to_dict method without sensitive fields."""
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            saml_config={"entity_id": "secret"},
            oauth_config={"client_secret": "secret"},
            ldap_config={"password": "secret"},
            password_policy={"rules": "complex"},
            ip_whitelist=["192.168.1.0/24"]
        )
        
        data = tenant.to_dict(include_sensitive=False)
        
        assert data["saml_config"] == "***REDACTED***"
        assert data["oauth_config"] == "***REDACTED***"
        assert data["ldap_config"] == "***REDACTED***"
        assert data["password_policy"] == "***REDACTED***"
        assert data["ip_whitelist"] == "***REDACTED***"
        assert data["name"] == "Test Company"
        
    def test_to_dict_with_sensitive(self):
        """Test to_dict method with sensitive fields."""
        saml_config = {"entity_id": "secret"}
        tenant = Tenant(
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            saml_config=saml_config
        )
        
        data = tenant.to_dict(include_sensitive=True)
        
        assert data["saml_config"] == saml_config
        

class TestTenantInvitation:
    """Test cases for TenantInvitation entity."""
    
    def test_tenant_invitation_creation(self):
        """Test tenant invitation creation."""
        tenant_id = uuid4()
        invited_by = uuid4()
        expires_at = datetime.utcnow() + timedelta(days=7)
        
        invitation = TenantInvitation(
            tenant_id=tenant_id,
            email="newuser@example.com",
            role="analyst",
            invited_by=invited_by,
            invitation_token="token123",
            expires_at=expires_at
        )
        
        assert isinstance(invitation.id, UUID)
        assert invitation.tenant_id == tenant_id
        assert invitation.email == "newuser@example.com"
        assert invitation.role == "analyst"
        assert invitation.invited_by == invited_by
        assert invitation.invitation_token == "token123"
        assert invitation.expires_at == expires_at
        assert invitation.status == "pending"
        assert invitation.message is None
        
    def test_tenant_invitation_with_message(self):
        """Test tenant invitation creation with message."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() + timedelta(days=7),
            message="Welcome to our team!"
        )
        
        assert invitation.message == "Welcome to our team!"
        
    def test_is_valid_true(self):
        """Test is_valid returns True for valid invitation."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        assert invitation.is_valid() is True
        
    def test_is_valid_false_accepted(self):
        """Test is_valid returns False for accepted invitation."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            status="accepted"
        )
        
        assert invitation.is_valid() is False
        
    def test_is_valid_false_expired(self):
        """Test is_valid returns False for expired invitation."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        assert invitation.is_valid() is False
        
    def test_accept(self):
        """Test accept method."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        invitation.accept()
        
        assert invitation.status == "accepted"
        assert invitation.accepted_at is not None
        
    def test_revoke(self):
        """Test revoke method."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        invitation.revoke()
        
        assert invitation.status == "revoked"
        assert invitation.revoked_at is not None
        
    def test_is_expired_false(self):
        """Test is_expired returns False for valid invitation."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        assert invitation.is_expired() is False
        
    def test_is_expired_true(self):
        """Test is_expired returns True for expired invitation."""
        invitation = TenantInvitation(
            tenant_id=uuid4(),
            email="newuser@example.com",
            invited_by=uuid4(),
            invitation_token="token123",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        assert invitation.is_expired() is True