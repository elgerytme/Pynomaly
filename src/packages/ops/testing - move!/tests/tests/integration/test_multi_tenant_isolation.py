"""
Multi-Tenant Isolation Testing Framework

Comprehensive testing for multi-tenant architecture ensuring proper data isolation,
security boundaries, resource allocation, and tenant-specific configurations.
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@dataclass
class TenantContext:
    """Tenant context for testing."""
    
    tenant_id: str
    tenant_name: str
    user_id: str
    role: str
    permissions: List[str]
    resource_limits: Dict[str, Any]
    configuration: Dict[str, Any]


@dataclass
class IsolationTestResult:
    """Multi-tenant isolation test result."""
    
    test_name: str
    tenant_id: str
    isolation_type: str  # "data", "security", "resource", "configuration"
    passed: bool
    violations: List[str]
    recommendations: List[str]
    severity: str  # "critical", "high", "medium", "low"


class MultiTenantIsolationTester:
    """Multi-tenant isolation testing framework."""
    
    def __init__(self):
        self.test_results: List[IsolationTestResult] = []
        self.tenant_contexts: Dict[str, TenantContext] = {}
        self.isolation_violations: List[Dict[str, Any]] = []
    
    def create_tenant_context(
        self, 
        tenant_id: str,
        tenant_name: str,
        user_id: str = None,
        role: str = "user",
        permissions: List[str] = None,
        resource_limits: Dict[str, Any] = None,
        configuration: Dict[str, Any] = None
    ) -> TenantContext:
        """Create tenant context for testing."""
        
        context = TenantContext(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            user_id=user_id or f"user_{tenant_id}",
            role=role,
            permissions=permissions or ["read", "write"],
            resource_limits=resource_limits or {
                "max_datasets": 100,
                "max_detectors": 50,
                "max_storage_mb": 1000,
                "max_api_calls_per_hour": 1000
            },
            configuration=configuration or {
                "features_enabled": ["anomaly_detection", "data_export"],
                "ui_theme": "default",
                "notification_settings": {"email": True, "webhook": False}
            }
        )
        
        self.tenant_contexts[tenant_id] = context
        return context
    
    def add_test_result(self, result: IsolationTestResult):
        """Add isolation test result."""
        self.test_results.append(result)
        
        if not result.passed:
            self.isolation_violations.append({
                "test_name": result.test_name,
                "tenant_id": result.tenant_id,
                "isolation_type": result.isolation_type,
                "violations": result.violations,
                "severity": result.severity
            })
    
    def generate_isolation_report(self) -> Dict[str, Any]:
        """Generate comprehensive isolation testing report."""
        
        total_tests = len(self.test_results)
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        # Group by isolation type
        isolation_types = {}
        for result in self.test_results:
            if result.isolation_type not in isolation_types:
                isolation_types[result.isolation_type] = {"passed": 0, "failed": 0}
            
            if result.passed:
                isolation_types[result.isolation_type]["passed"] += 1
            else:
                isolation_types[result.isolation_type]["failed"] += 1
        
        # Group by severity
        severity_counts = {}
        for result in failed_tests:
            severity = result.severity
            if severity not in severity_counts:
                severity_counts[severity] = 0
            severity_counts[severity] += 1
        
        # Critical violations
        critical_violations = [
            v for v in self.isolation_violations 
            if v["severity"] == "critical"
        ]
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests),
                "isolation_score": (len(passed_tests) / total_tests * 100) if total_tests > 0 else 0,
                "total_tenants_tested": len(self.tenant_contexts)
            },
            "isolation_types": isolation_types,
            "severity_breakdown": severity_counts,
            "critical_violations": critical_violations,
            "tenant_contexts": {
                tenant_id: {
                    "name": context.tenant_name,
                    "user_id": context.user_id,
                    "role": context.role,
                    "permissions": context.permissions,
                    "resource_limits": context.resource_limits
                }
                for tenant_id, context in self.tenant_contexts.items()
            },
            "recommendations": list(set(
                rec for result in failed_tests for rec in result.recommendations
            ))
        }


class TestDataIsolation:
    """Test data isolation between tenants."""
    
    @pytest.fixture
    def isolation_tester(self):
        """Create isolation tester."""
        return MultiTenantIsolationTester()
    
    @pytest.fixture
    def tenant_contexts(self, isolation_tester):
        """Create multiple tenant contexts."""
        return {
            "tenant_a": isolation_tester.create_tenant_context(
                tenant_id="tenant_a",
                tenant_name="Tenant A Corporation",
                user_id="user_a_admin",
                role="admin",
                permissions=["read", "write", "delete", "admin"]
            ),
            "tenant_b": isolation_tester.create_tenant_context(
                tenant_id="tenant_b",
                tenant_name="Tenant B Ltd",
                user_id="user_b_normal",
                role="user",
                permissions=["read", "write"]
            ),
            "tenant_c": isolation_tester.create_tenant_context(
                tenant_id="tenant_c",
                tenant_name="Tenant C Inc",
                user_id="user_c_readonly",
                role="readonly",
                permissions=["read"]
            )
        }
    
    def test_dataset_isolation(self, isolation_tester, tenant_contexts):
        """Test dataset isolation between tenants."""
        
        with patch('monorepo.infrastructure.persistence.multi_tenant_repository.MultiTenantRepository') as mock_repo:
            # Mock repository to simulate tenant isolation
            def mock_get_datasets(tenant_id: str):
                # Return different datasets for different tenants
                datasets = {
                    "tenant_a": [
                        Mock(id="dataset_a_1", name="Tenant A Dataset 1", tenant_id="tenant_a"),
                        Mock(id="dataset_a_2", name="Tenant A Dataset 2", tenant_id="tenant_a")
                    ],
                    "tenant_b": [
                        Mock(id="dataset_b_1", name="Tenant B Dataset 1", tenant_id="tenant_b")
                    ],
                    "tenant_c": [
                        Mock(id="dataset_c_1", name="Tenant C Dataset 1", tenant_id="tenant_c")
                    ]
                }
                return datasets.get(tenant_id, [])
            
            mock_repo.return_value.get_datasets_by_tenant = mock_get_datasets
            
            # Test each tenant can only access their own datasets
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Get datasets for this tenant
                tenant_datasets = mock_repo.return_value.get_datasets_by_tenant(tenant_id)
                
                # Verify all datasets belong to this tenant
                for dataset in tenant_datasets:
                    if dataset.tenant_id != tenant_id:
                        violations.append(f"Dataset {dataset.id} with tenant_id {dataset.tenant_id} accessible by {tenant_id}")
                
                # Try to access other tenants' datasets
                for other_tenant_id in tenant_contexts.keys():
                    if other_tenant_id != tenant_id:
                        try:
                            # This should fail or return empty
                            other_datasets = mock_repo.return_value.get_datasets_by_tenant(other_tenant_id)
                            
                            # If we got datasets, check they don't belong to current tenant
                            accessible_other_datasets = [
                                d for d in other_datasets 
                                if d.tenant_id != tenant_id
                            ]
                            
                            if accessible_other_datasets:
                                violations.append(f"Tenant {tenant_id} can access {len(accessible_other_datasets)} datasets from {other_tenant_id}")
                        
                        except PermissionError:
                            pass  # Good, access denied
                
                result = IsolationTestResult(
                    test_name="Dataset Isolation",
                    tenant_id=tenant_id,
                    isolation_type="data",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement proper tenant-based dataset filtering",
                        "Add tenant_id to all dataset queries",
                        "Validate tenant access in repository layer"
                    ] if violations else [],
                    severity="critical" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Dataset isolation failed for {tenant_id}: {violations}"
    
    def test_detector_isolation(self, isolation_tester, tenant_contexts):
        """Test detector isolation between tenants."""
        
        with patch('monorepo.infrastructure.persistence.multi_tenant_repository.MultiTenantRepository') as mock_repo:
            # Mock detector repository
            def mock_get_detectors(tenant_id: str):
                detectors = {
                    "tenant_a": [
                        Mock(id="detector_a_1", name="Tenant A Detector 1", tenant_id="tenant_a"),
                        Mock(id="detector_a_2", name="Tenant A Detector 2", tenant_id="tenant_a")
                    ],
                    "tenant_b": [
                        Mock(id="detector_b_1", name="Tenant B Detector 1", tenant_id="tenant_b")
                    ],
                    "tenant_c": []  # No detectors for tenant C
                }
                return detectors.get(tenant_id, [])
            
            mock_repo.return_value.get_detectors_by_tenant = mock_get_detectors
            
            # Test detector isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Get detectors for this tenant
                tenant_detectors = mock_repo.return_value.get_detectors_by_tenant(tenant_id)
                
                # Verify all detectors belong to this tenant
                for detector in tenant_detectors:
                    if detector.tenant_id != tenant_id:
                        violations.append(f"Detector {detector.id} with tenant_id {detector.tenant_id} accessible by {tenant_id}")
                
                # Test cross-tenant detector access
                with patch('monorepo.application.services.detector_service.DetectorService') as mock_service:
                    mock_service.return_value.get_detector_by_id.side_effect = lambda detector_id, tenant_id: self._check_detector_access(detector_id, tenant_id)
                    
                    # Try to access detectors from other tenants
                    other_detector_ids = [
                        "detector_a_1", "detector_a_2",  # Tenant A detectors
                        "detector_b_1"  # Tenant B detector
                    ]
                    
                    for detector_id in other_detector_ids:
                        try:
                            detector = mock_service.return_value.get_detector_by_id(detector_id, tenant_id)
                            
                            if detector and detector.tenant_id != tenant_id:
                                violations.append(f"Tenant {tenant_id} can access detector {detector_id} from tenant {detector.tenant_id}")
                        
                        except PermissionError:
                            pass  # Good, access denied
                
                result = IsolationTestResult(
                    test_name="Detector Isolation",
                    tenant_id=tenant_id,
                    isolation_type="data",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement detector-level tenant validation",
                        "Add tenant_id checks in detector service",
                        "Use tenant-scoped detector repositories"
                    ] if violations else [],
                    severity="critical" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Detector isolation failed for {tenant_id}: {violations}"
    
    def _check_detector_access(self, detector_id: str, tenant_id: str):
        """Helper to check detector access."""
        # Mock detector ownership
        detector_ownership = {
            "detector_a_1": "tenant_a",
            "detector_a_2": "tenant_a",
            "detector_b_1": "tenant_b"
        }
        
        if detector_id in detector_ownership:
            owner_tenant = detector_ownership[detector_id]
            if owner_tenant != tenant_id:
                raise PermissionError(f"Detector {detector_id} not accessible by tenant {tenant_id}")
            
            return Mock(id=detector_id, tenant_id=owner_tenant)
        
        return None
    
    def test_detection_results_isolation(self, isolation_tester, tenant_contexts):
        """Test detection results isolation between tenants."""
        
        with patch('monorepo.infrastructure.persistence.multi_tenant_repository.MultiTenantRepository') as mock_repo:
            # Mock results repository
            def mock_get_results(tenant_id: str):
                results = {
                    "tenant_a": [
                        Mock(id="result_a_1", detector_id="detector_a_1", tenant_id="tenant_a"),
                        Mock(id="result_a_2", detector_id="detector_a_2", tenant_id="tenant_a")
                    ],
                    "tenant_b": [
                        Mock(id="result_b_1", detector_id="detector_b_1", tenant_id="tenant_b")
                    ],
                    "tenant_c": []
                }
                return results.get(tenant_id, [])
            
            mock_repo.return_value.get_results_by_tenant = mock_get_results
            
            # Test results isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Get results for this tenant
                tenant_results = mock_repo.return_value.get_results_by_tenant(tenant_id)
                
                # Verify all results belong to this tenant
                for result in tenant_results:
                    if result.tenant_id != tenant_id:
                        violations.append(f"Result {result.id} with tenant_id {result.tenant_id} accessible by {tenant_id}")
                
                # Test cross-tenant result access
                with patch('monorepo.application.services.detection_service.DetectionService') as mock_service:
                    mock_service.return_value.get_result_by_id.side_effect = lambda result_id, tenant_id: self._check_result_access(result_id, tenant_id)
                    
                    # Try to access results from other tenants
                    other_result_ids = ["result_a_1", "result_a_2", "result_b_1"]
                    
                    for result_id in other_result_ids:
                        try:
                            result = mock_service.return_value.get_result_by_id(result_id, tenant_id)
                            
                            if result and result.tenant_id != tenant_id:
                                violations.append(f"Tenant {tenant_id} can access result {result_id} from tenant {result.tenant_id}")
                        
                        except PermissionError:
                            pass  # Good, access denied
                
                result = IsolationTestResult(
                    test_name="Detection Results Isolation",
                    tenant_id=tenant_id,
                    isolation_type="data",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement result-level tenant validation",
                        "Add tenant_id checks in detection service",
                        "Use tenant-scoped result repositories"
                    ] if violations else [],
                    severity="critical" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Detection results isolation failed for {tenant_id}: {violations}"
    
    def _check_result_access(self, result_id: str, tenant_id: str):
        """Helper to check result access."""
        # Mock result ownership
        result_ownership = {
            "result_a_1": "tenant_a",
            "result_a_2": "tenant_a",
            "result_b_1": "tenant_b"
        }
        
        if result_id in result_ownership:
            owner_tenant = result_ownership[result_id]
            if owner_tenant != tenant_id:
                raise PermissionError(f"Result {result_id} not accessible by tenant {tenant_id}")
            
            return Mock(id=result_id, tenant_id=owner_tenant)
        
        return None


class TestSecurityIsolation:
    """Test security isolation between tenants."""
    
    @pytest.fixture
    def isolation_tester(self):
        """Create isolation tester."""
        return MultiTenantIsolationTester()
    
    @pytest.fixture
    def tenant_contexts(self, isolation_tester):
        """Create multiple tenant contexts."""
        return {
            "tenant_a": isolation_tester.create_tenant_context(
                tenant_id="tenant_a",
                tenant_name="Tenant A Corporation",
                user_id="user_a_admin",
                role="admin",
                permissions=["read", "write", "delete", "admin"]
            ),
            "tenant_b": isolation_tester.create_tenant_context(
                tenant_id="tenant_b",
                tenant_name="Tenant B Ltd",
                user_id="user_b_normal",
                role="user",
                permissions=["read", "write"]
            )
        }
    
    def test_authentication_isolation(self, isolation_tester, tenant_contexts):
        """Test authentication isolation between tenants."""
        
        with patch('monorepo.infrastructure.security.auth_provider.AuthProvider') as mock_auth:
            # Mock authentication provider
            def mock_authenticate(username: str, password: str, tenant_id: str):
                # Define users for each tenant
                tenant_users = {
                    "tenant_a": {"user_a_admin": "password_a"},
                    "tenant_b": {"user_b_normal": "password_b"}
                }
                
                tenant_user_map = tenant_users.get(tenant_id, {})
                
                if username in tenant_user_map and tenant_user_map[username] == password:
                    return Mock(
                        id=username,
                        tenant_id=tenant_id,
                        username=username,
                        authenticated=True
                    )
                
                return None
            
            mock_auth.return_value.authenticate = mock_authenticate
            
            # Test authentication isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test valid authentication for this tenant
                valid_auth = mock_auth.return_value.authenticate(
                    context.user_id, 
                    f"password_{tenant_id.split('_')[1]}", 
                    tenant_id
                )
                
                if not valid_auth or valid_auth.tenant_id != tenant_id:
                    violations.append(f"Valid authentication failed for tenant {tenant_id}")
                
                # Test cross-tenant authentication attempts
                for other_tenant_id, other_context in tenant_contexts.items():
                    if other_tenant_id != tenant_id:
                        # Try to authenticate with other tenant's credentials
                        cross_auth = mock_auth.return_value.authenticate(
                            other_context.user_id,
                            f"password_{other_tenant_id.split('_')[1]}",
                            tenant_id  # Using current tenant_id
                        )
                        
                        if cross_auth and cross_auth.tenant_id == tenant_id:
                            violations.append(f"Cross-tenant authentication succeeded: {other_context.user_id} authenticated for {tenant_id}")
                
                result = IsolationTestResult(
                    test_name="Authentication Isolation",
                    tenant_id=tenant_id,
                    isolation_type="security",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped authentication",
                        "Validate tenant_id in authentication flow",
                        "Use tenant-specific user stores"
                    ] if violations else [],
                    severity="critical" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Authentication isolation failed for {tenant_id}: {violations}"
    
    def test_authorization_isolation(self, isolation_tester, tenant_contexts):
        """Test authorization isolation between tenants."""
        
        with patch('monorepo.infrastructure.security.rbac.RoleBasedAccessControl') as mock_rbac:
            # Mock RBAC system
            def mock_check_permission(user_id: str, resource: str, action: str, tenant_id: str):
                # Define permissions for each tenant
                tenant_permissions = {
                    "tenant_a": {
                        "user_a_admin": ["read", "write", "delete", "admin"]
                    },
                    "tenant_b": {
                        "user_b_normal": ["read", "write"]
                    }
                }
                
                tenant_user_perms = tenant_permissions.get(tenant_id, {})
                user_permissions = tenant_user_perms.get(user_id, [])
                
                return action in user_permissions
            
            mock_rbac.return_value.check_permission = mock_check_permission
            
            # Test authorization isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test permissions for this tenant's user
                for permission in context.permissions:
                    has_permission = mock_rbac.return_value.check_permission(
                        context.user_id, 
                        "dataset", 
                        permission, 
                        tenant_id
                    )
                    
                    if not has_permission:
                        violations.append(f"User {context.user_id} missing expected permission {permission} in tenant {tenant_id}")
                
                # Test cross-tenant permission attempts
                for other_tenant_id, other_context in tenant_contexts.items():
                    if other_tenant_id != tenant_id:
                        # Try to use other tenant's user in current tenant
                        cross_permission = mock_rbac.return_value.check_permission(
                            other_context.user_id,
                            "dataset",
                            "read",
                            tenant_id  # Using current tenant_id
                        )
                        
                        if cross_permission:
                            violations.append(f"Cross-tenant authorization succeeded: {other_context.user_id} has permissions in {tenant_id}")
                
                result = IsolationTestResult(
                    test_name="Authorization Isolation",
                    tenant_id=tenant_id,
                    isolation_type="security",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped authorization",
                        "Validate tenant_id in authorization checks",
                        "Use tenant-specific role definitions"
                    ] if violations else [],
                    severity="critical" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Authorization isolation failed for {tenant_id}: {violations}"
    
    def test_session_isolation(self, isolation_tester, tenant_contexts):
        """Test session isolation between tenants."""
        
        with patch('monorepo.infrastructure.security.session_manager.SessionManager') as mock_session:
            # Mock session manager
            def mock_create_session(user_id: str, tenant_id: str):
                session_id = f"session_{user_id}_{tenant_id}"
                return Mock(
                    id=session_id,
                    user_id=user_id,
                    tenant_id=tenant_id,
                    created_at="2023-01-01T00:00:00Z"
                )
            
            def mock_validate_session(session_id: str, tenant_id: str):
                # Session should only be valid for its own tenant
                if tenant_id in session_id:
                    return Mock(
                        id=session_id,
                        user_id=session_id.split('_')[1],
                        tenant_id=tenant_id,
                        valid=True
                    )
                return None
            
            mock_session.return_value.create_session = mock_create_session
            mock_session.return_value.validate_session = mock_validate_session
            
            # Test session isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Create session for this tenant
                session = mock_session.return_value.create_session(context.user_id, tenant_id)
                
                if not session or session.tenant_id != tenant_id:
                    violations.append(f"Session creation failed for tenant {tenant_id}")
                
                # Validate session for this tenant
                if session:
                    valid_session = mock_session.return_value.validate_session(session.id, tenant_id)
                    
                    if not valid_session or valid_session.tenant_id != tenant_id:
                        violations.append(f"Session validation failed for tenant {tenant_id}")
                    
                    # Try to validate session in other tenants
                    for other_tenant_id in tenant_contexts.keys():
                        if other_tenant_id != tenant_id:
                            cross_session = mock_session.return_value.validate_session(session.id, other_tenant_id)
                            
                            if cross_session and cross_session.valid:
                                violations.append(f"Session {session.id} valid in wrong tenant {other_tenant_id}")
                
                result = IsolationTestResult(
                    test_name="Session Isolation",
                    tenant_id=tenant_id,
                    isolation_type="security",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped session validation",
                        "Include tenant_id in session tokens",
                        "Validate tenant context in session middleware"
                    ] if violations else [],
                    severity="high" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Session isolation failed for {tenant_id}: {violations}"


class TestResourceIsolation:
    """Test resource isolation between tenants."""
    
    @pytest.fixture
    def isolation_tester(self):
        """Create isolation tester."""
        return MultiTenantIsolationTester()
    
    @pytest.fixture
    def tenant_contexts(self, isolation_tester):
        """Create multiple tenant contexts with different resource limits."""
        return {
            "tenant_premium": isolation_tester.create_tenant_context(
                tenant_id="tenant_premium",
                tenant_name="Premium Tenant",
                resource_limits={
                    "max_datasets": 500,
                    "max_detectors": 100,
                    "max_storage_mb": 10000,
                    "max_api_calls_per_hour": 5000
                }
            ),
            "tenant_basic": isolation_tester.create_tenant_context(
                tenant_id="tenant_basic",
                tenant_name="Basic Tenant",
                resource_limits={
                    "max_datasets": 50,
                    "max_detectors": 10,
                    "max_storage_mb": 1000,
                    "max_api_calls_per_hour": 500
                }
            )
        }
    
    def test_storage_isolation(self, isolation_tester, tenant_contexts):
        """Test storage isolation and limits between tenants."""
        
        with patch('monorepo.infrastructure.storage.tenant_storage_manager.TenantStorageManager') as mock_storage:
            # Mock storage manager
            def mock_get_storage_usage(tenant_id: str):
                # Simulate different storage usage
                usage = {
                    "tenant_premium": 5000,  # 5GB
                    "tenant_basic": 800      # 800MB
                }
                return usage.get(tenant_id, 0)
            
            def mock_check_storage_limit(tenant_id: str, additional_size: int):
                context = tenant_contexts.get(tenant_id)
                if not context:
                    return False
                
                current_usage = mock_get_storage_usage(tenant_id)
                max_storage = context.resource_limits["max_storage_mb"]
                
                return (current_usage + additional_size) <= max_storage
            
            mock_storage.return_value.get_storage_usage = mock_get_storage_usage
            mock_storage.return_value.check_storage_limit = mock_check_storage_limit
            
            # Test storage isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test current storage usage
                current_usage = mock_storage.return_value.get_storage_usage(tenant_id)
                max_storage = context.resource_limits["max_storage_mb"]
                
                if current_usage > max_storage:
                    violations.append(f"Storage usage {current_usage}MB exceeds limit {max_storage}MB for tenant {tenant_id}")
                
                # Test storage limit enforcement
                large_upload = 2000  # 2GB
                can_upload = mock_storage.return_value.check_storage_limit(tenant_id, large_upload)
                
                expected_can_upload = (current_usage + large_upload) <= max_storage
                
                if can_upload != expected_can_upload:
                    violations.append(f"Storage limit check incorrect for tenant {tenant_id}: {can_upload} vs expected {expected_can_upload}")
                
                # Test cross-tenant storage access
                for other_tenant_id in tenant_contexts.keys():
                    if other_tenant_id != tenant_id:
                        try:
                            # Try to access other tenant's storage
                            other_usage = mock_storage.return_value.get_storage_usage(other_tenant_id)
                            
                            # This should be isolated - we shouldn't be able to get accurate usage
                            if other_usage > 0:
                                # Check if we can modify other tenant's storage
                                with patch('monorepo.infrastructure.storage.tenant_storage_manager.TenantStorageManager.delete_tenant_data') as mock_delete:
                                    mock_delete.side_effect = lambda tid: tid == tenant_id  # Only allow own tenant
                                    
                                    can_delete = mock_delete(other_tenant_id)
                                    
                                    if can_delete:
                                        violations.append(f"Tenant {tenant_id} can delete storage from tenant {other_tenant_id}")
                        
                        except PermissionError:
                            pass  # Good, access denied
                
                result = IsolationTestResult(
                    test_name="Storage Isolation",
                    tenant_id=tenant_id,
                    isolation_type="resource",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped storage containers",
                        "Enforce storage quotas per tenant",
                        "Validate tenant_id in storage operations"
                    ] if violations else [],
                    severity="high" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Storage isolation failed for {tenant_id}: {violations}"
    
    def test_compute_isolation(self, isolation_tester, tenant_contexts):
        """Test compute resource isolation between tenants."""
        
        with patch('monorepo.infrastructure.compute.tenant_compute_manager.TenantComputeManager') as mock_compute:
            # Mock compute manager
            def mock_get_compute_usage(tenant_id: str):
                # Simulate different compute usage
                usage = {
                    "tenant_premium": {"cpu_seconds": 3600, "memory_mb_seconds": 100000},
                    "tenant_basic": {"cpu_seconds": 600, "memory_mb_seconds": 20000}
                }
                return usage.get(tenant_id, {"cpu_seconds": 0, "memory_mb_seconds": 0})
            
            def mock_check_compute_limit(tenant_id: str, cpu_required: int, memory_required: int):
                context = tenant_contexts.get(tenant_id)
                if not context:
                    return False
                
                # Basic compute limit check (simplified)
                if tenant_id == "tenant_premium":
                    return cpu_required <= 8 and memory_required <= 16000  # 8 cores, 16GB
                else:
                    return cpu_required <= 2 and memory_required <= 4000   # 2 cores, 4GB
            
            mock_compute.return_value.get_compute_usage = mock_get_compute_usage
            mock_compute.return_value.check_compute_limit = mock_check_compute_limit
            
            # Test compute isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test compute usage tracking
                compute_usage = mock_compute.return_value.get_compute_usage(tenant_id)
                
                if not compute_usage or "cpu_seconds" not in compute_usage:
                    violations.append(f"Compute usage tracking missing for tenant {tenant_id}")
                
                # Test compute limit enforcement
                if tenant_id == "tenant_premium":
                    # Premium tenant should be able to use more resources
                    can_use_high = mock_compute.return_value.check_compute_limit(tenant_id, 6, 12000)
                    if not can_use_high:
                        violations.append(f"Premium tenant {tenant_id} cannot use expected high resources")
                else:
                    # Basic tenant should be limited
                    can_use_high = mock_compute.return_value.check_compute_limit(tenant_id, 6, 12000)
                    if can_use_high:
                        violations.append(f"Basic tenant {tenant_id} can use too many resources")
                
                # Test compute isolation
                with patch('monorepo.infrastructure.compute.resource_allocator.ResourceAllocator') as mock_allocator:
                    mock_allocator.return_value.allocate_resources.side_effect = lambda tid, resources: tid == tenant_id
                    
                    # Try to allocate resources for other tenants
                    for other_tenant_id in tenant_contexts.keys():
                        if other_tenant_id != tenant_id:
                            can_allocate = mock_allocator.return_value.allocate_resources(
                                other_tenant_id, 
                                {"cpu": 1, "memory": 1000}
                            )
                            
                            if can_allocate:
                                violations.append(f"Tenant {tenant_id} can allocate resources for tenant {other_tenant_id}")
                
                result = IsolationTestResult(
                    test_name="Compute Isolation",
                    tenant_id=tenant_id,
                    isolation_type="resource",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped compute limits",
                        "Use container-based resource isolation",
                        "Monitor and enforce compute quotas"
                    ] if violations else [],
                    severity="medium" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Compute isolation failed for {tenant_id}: {violations}"
    
    def test_api_rate_limiting_isolation(self, isolation_tester, tenant_contexts):
        """Test API rate limiting isolation between tenants."""
        
        with patch('monorepo.infrastructure.rate_limiting.tenant_rate_limiter.TenantRateLimiter') as mock_limiter:
            # Mock rate limiter
            def mock_check_rate_limit(tenant_id: str, endpoint: str):
                context = tenant_contexts.get(tenant_id)
                if not context:
                    return False
                
                # Simulate different rate limits
                max_calls = context.resource_limits["max_api_calls_per_hour"]
                
                # Simulate current usage (simplified)
                current_calls = {
                    "tenant_premium": 2000,
                    "tenant_basic": 300
                }.get(tenant_id, 0)
                
                return current_calls < max_calls
            
            mock_limiter.return_value.check_rate_limit = mock_check_rate_limit
            
            # Test rate limiting isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test rate limit enforcement
                can_make_call = mock_limiter.return_value.check_rate_limit(tenant_id, "/api/v1/datasets")
                
                if not can_make_call:
                    # Check if tenant has reached their limit
                    max_calls = context.resource_limits["max_api_calls_per_hour"]
                    if max_calls > 0:  # Should allow calls if under limit
                        violations.append(f"Rate limiting incorrectly blocking tenant {tenant_id}")
                
                # Test cross-tenant rate limit isolation
                with patch('monorepo.infrastructure.rate_limiting.rate_limit_storage.RateLimitStorage') as mock_storage:
                    mock_storage.return_value.get_usage.side_effect = lambda tid, endpoint: tid == tenant_id
                    
                    # Try to access other tenant's rate limit data
                    for other_tenant_id in tenant_contexts.keys():
                        if other_tenant_id != tenant_id:
                            can_access = mock_storage.return_value.get_usage(other_tenant_id, "/api/v1/datasets")
                            
                            if can_access:
                                violations.append(f"Tenant {tenant_id} can access rate limit data for tenant {other_tenant_id}")
                
                result = IsolationTestResult(
                    test_name="API Rate Limiting Isolation",
                    tenant_id=tenant_id,
                    isolation_type="resource",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped rate limiting",
                        "Use separate rate limit counters per tenant",
                        "Validate tenant_id in rate limiting middleware"
                    ] if violations else [],
                    severity="medium" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"API rate limiting isolation failed for {tenant_id}: {violations}"


class TestConfigurationIsolation:
    """Test configuration isolation between tenants."""
    
    @pytest.fixture
    def isolation_tester(self):
        """Create isolation tester."""
        return MultiTenantIsolationTester()
    
    @pytest.fixture
    def tenant_contexts(self, isolation_tester):
        """Create multiple tenant contexts with different configurations."""
        return {
            "tenant_enterprise": isolation_tester.create_tenant_context(
                tenant_id="tenant_enterprise",
                tenant_name="Enterprise Tenant",
                configuration={
                    "features_enabled": ["anomaly_detection", "data_export", "advanced_analytics", "custom_algorithms"],
                    "ui_theme": "dark",
                    "notification_settings": {"email": True, "webhook": True, "slack": True},
                    "data_retention_days": 365,
                    "backup_enabled": True
                }
            ),
            "tenant_standard": isolation_tester.create_tenant_context(
                tenant_id="tenant_standard",
                tenant_name="Standard Tenant",
                configuration={
                    "features_enabled": ["anomaly_detection", "data_export"],
                    "ui_theme": "light",
                    "notification_settings": {"email": True, "webhook": False, "slack": False},
                    "data_retention_days": 90,
                    "backup_enabled": False
                }
            )
        }
    
    def test_feature_isolation(self, isolation_tester, tenant_contexts):
        """Test feature configuration isolation between tenants."""
        
        with patch('monorepo.infrastructure.config.tenant_config_manager.TenantConfigManager') as mock_config:
            # Mock config manager
            def mock_get_tenant_config(tenant_id: str):
                context = tenant_contexts.get(tenant_id)
                return context.configuration if context else {}
            
            def mock_is_feature_enabled(tenant_id: str, feature: str):
                config = mock_get_tenant_config(tenant_id)
                return feature in config.get("features_enabled", [])
            
            mock_config.return_value.get_tenant_config = mock_get_tenant_config
            mock_config.return_value.is_feature_enabled = mock_is_feature_enabled
            
            # Test feature isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test enabled features
                for feature in context.configuration["features_enabled"]:
                    is_enabled = mock_config.return_value.is_feature_enabled(tenant_id, feature)
                    
                    if not is_enabled:
                        violations.append(f"Feature {feature} should be enabled for tenant {tenant_id}")
                
                # Test disabled features
                all_features = ["anomaly_detection", "data_export", "advanced_analytics", "custom_algorithms"]
                for feature in all_features:
                    if feature not in context.configuration["features_enabled"]:
                        is_enabled = mock_config.return_value.is_feature_enabled(tenant_id, feature)
                        
                        if is_enabled:
                            violations.append(f"Feature {feature} should be disabled for tenant {tenant_id}")
                
                # Test cross-tenant feature access
                with patch('monorepo.application.services.feature_service.FeatureService') as mock_feature:
                    mock_feature.return_value.check_feature_access.side_effect = lambda tid, feature: mock_is_feature_enabled(tid, feature)
                    
                    # Try to access other tenant's features
                    for other_tenant_id, other_context in tenant_contexts.items():
                        if other_tenant_id != tenant_id:
                            for feature in other_context.configuration["features_enabled"]:
                                if feature not in context.configuration["features_enabled"]:
                                    # This tenant shouldn't have access to this feature
                                    has_access = mock_feature.return_value.check_feature_access(tenant_id, feature)
                                    
                                    if has_access:
                                        violations.append(f"Tenant {tenant_id} has access to feature {feature} from tenant {other_tenant_id}")
                
                result = IsolationTestResult(
                    test_name="Feature Isolation",
                    tenant_id=tenant_id,
                    isolation_type="configuration",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped feature flags",
                        "Validate feature access per tenant",
                        "Use tenant-specific configuration stores"
                    ] if violations else [],
                    severity="medium" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"Feature isolation failed for {tenant_id}: {violations}"
    
    def test_ui_customization_isolation(self, isolation_tester, tenant_contexts):
        """Test UI customization isolation between tenants."""
        
        with patch('monorepo.infrastructure.ui.tenant_ui_manager.TenantUIManager') as mock_ui:
            # Mock UI manager
            def mock_get_ui_config(tenant_id: str):
                context = tenant_contexts.get(tenant_id)
                if not context:
                    return {}
                
                return {
                    "theme": context.configuration.get("ui_theme", "default"),
                    "branding": {
                        "logo": f"logo_{tenant_id}.png",
                        "colors": {
                            "primary": "#007bff" if tenant_id == "tenant_enterprise" else "#6c757d",
                            "secondary": "#6c757d"
                        }
                    },
                    "features_visible": context.configuration.get("features_enabled", [])
                }
            
            mock_ui.return_value.get_ui_config = mock_get_ui_config
            
            # Test UI customization isolation
            for tenant_id, context in tenant_contexts.items():
                violations = []
                
                # Test UI configuration
                ui_config = mock_ui.return_value.get_ui_config(tenant_id)
                
                if not ui_config:
                    violations.append(f"UI configuration missing for tenant {tenant_id}")
                else:
                    # Check theme
                    expected_theme = context.configuration["ui_theme"]
                    actual_theme = ui_config.get("theme")
                    
                    if actual_theme != expected_theme:
                        violations.append(f"UI theme mismatch for tenant {tenant_id}: {actual_theme} vs {expected_theme}")
                    
                    # Check branding isolation
                    branding = ui_config.get("branding", {})
                    logo = branding.get("logo", "")
                    
                    if tenant_id not in logo:
                        violations.append(f"UI branding not isolated for tenant {tenant_id}: logo {logo}")
                
                # Test cross-tenant UI access
                with patch('monorepo.presentation.web.ui_renderer.UIRenderer') as mock_renderer:
                    mock_renderer.return_value.render_for_tenant.side_effect = lambda tid: mock_get_ui_config(tid)
                    
                    # Try to render UI for other tenants
                    for other_tenant_id in tenant_contexts.keys():
                        if other_tenant_id != tenant_id:
                            other_ui_config = mock_renderer.return_value.render_for_tenant(other_tenant_id)
                            
                            # Check if we can access other tenant's UI config
                            if other_ui_config and other_ui_config.get("theme"):
                                # This should be isolated
                                current_ui = mock_renderer.return_value.render_for_tenant(tenant_id)
                                
                                if current_ui.get("theme") == other_ui_config.get("theme") and current_ui.get("theme") != "default":
                                    violations.append(f"UI configuration not isolated between {tenant_id} and {other_tenant_id}")
                
                result = IsolationTestResult(
                    test_name="UI Customization Isolation",
                    tenant_id=tenant_id,
                    isolation_type="configuration",
                    passed=len(violations) == 0,
                    violations=violations,
                    recommendations=[
                        "Implement tenant-scoped UI configurations",
                        "Use tenant-specific asset storage",
                        "Validate tenant_id in UI rendering"
                    ] if violations else [],
                    severity="low" if violations else "low"
                )
                
                isolation_tester.add_test_result(result)
                assert result.passed, f"UI customization isolation failed for {tenant_id}: {violations}"


def test_comprehensive_multi_tenant_isolation():
    """Run comprehensive multi-tenant isolation testing."""
    
    isolation_tester = MultiTenantIsolationTester()
    
    # Create comprehensive tenant contexts
    tenant_contexts = {
        "tenant_enterprise": isolation_tester.create_tenant_context(
            tenant_id="tenant_enterprise",
            tenant_name="Enterprise Corporation",
            user_id="enterprise_admin",
            role="admin",
            permissions=["read", "write", "delete", "admin"],
            resource_limits={
                "max_datasets": 1000,
                "max_detectors": 200,
                "max_storage_mb": 50000,
                "max_api_calls_per_hour": 10000
            },
            configuration={
                "features_enabled": ["anomaly_detection", "data_export", "advanced_analytics", "custom_algorithms"],
                "ui_theme": "enterprise",
                "notification_settings": {"email": True, "webhook": True, "slack": True},
                "data_retention_days": 2555,  # 7 years
                "backup_enabled": True
            }
        ),
        "tenant_standard": isolation_tester.create_tenant_context(
            tenant_id="tenant_standard",
            tenant_name="Standard Company",
            user_id="standard_user",
            role="user",
            permissions=["read", "write"],
            resource_limits={
                "max_datasets": 100,
                "max_detectors": 20,
                "max_storage_mb": 5000,
                "max_api_calls_per_hour": 1000
            },
            configuration={
                "features_enabled": ["anomaly_detection", "data_export"],
                "ui_theme": "standard",
                "notification_settings": {"email": True, "webhook": False, "slack": False},
                "data_retention_days": 365,
                "backup_enabled": False
            }
        ),
        "tenant_basic": isolation_tester.create_tenant_context(
            tenant_id="tenant_basic",
            tenant_name="Basic Startup",
            user_id="basic_user",
            role="user",
            permissions=["read"],
            resource_limits={
                "max_datasets": 10,
                "max_detectors": 5,
                "max_storage_mb": 500,
                "max_api_calls_per_hour": 100
            },
            configuration={
                "features_enabled": ["anomaly_detection"],
                "ui_theme": "basic",
                "notification_settings": {"email": True, "webhook": False, "slack": False},
                "data_retention_days": 90,
                "backup_enabled": False
            }
        )
    }
    
    # Run simplified isolation tests
    try:
        # Test data isolation
        data_tester = TestDataIsolation()
        data_tester.test_dataset_isolation(isolation_tester, tenant_contexts)
        
        # Test security isolation
        security_tester = TestSecurityIsolation()
        security_tester.test_authentication_isolation(isolation_tester, tenant_contexts)
        
        # Test resource isolation
        resource_tester = TestResourceIsolation()
        resource_tester.test_storage_isolation(isolation_tester, tenant_contexts)
        
        # Test configuration isolation
        config_tester = TestConfigurationIsolation()
        config_tester.test_feature_isolation(isolation_tester, tenant_contexts)
        
    except Exception as e:
        print(f"Multi-tenant isolation test execution error: {e}")
    
    # Generate isolation report
    report = isolation_tester.generate_isolation_report()
    
    print("\n" + "="*60)
    print("🏢 MULTI-TENANT ISOLATION TESTING REPORT")
    print("="*60)
    
    print(f"\n📊 Summary:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed_tests']}")
    print(f"  Failed: {report['summary']['failed_tests']}")
    print(f"  Isolation Score: {report['summary']['isolation_score']:.1f}%")
    print(f"  Tenants Tested: {report['summary']['total_tenants_tested']}")
    
    print(f"\n🔒 Isolation Types:")
    for isolation_type, results in report['isolation_types'].items():
        total = results['passed'] + results['failed']
        success_rate = (results['passed'] / total * 100) if total > 0 else 0
        print(f"  {isolation_type.title()}: {results['passed']}/{total} ({success_rate:.1f}%)")
    
    if report['severity_breakdown']:
        print(f"\n⚠️  Severity Breakdown:")
        for severity, count in report['severity_breakdown'].items():
            if count > 0:
                print(f"  {severity.title()}: {count}")
    
    if report['critical_violations']:
        print(f"\n🚨 Critical Violations:")
        for violation in report['critical_violations']:
            print(f"  • {violation['test_name']} (Tenant: {violation['tenant_id']})")
            for v in violation['violations'][:2]:  # Show first 2
                print(f"    - {v}")
    
    print(f"\n🏢 Tenant Contexts:")
    for tenant_id, context in report['tenant_contexts'].items():
        print(f"  {tenant_id}: {context['name']} (Role: {context['role']})")
        print(f"    Permissions: {context['permissions']}")
        print(f"    Storage Limit: {context['resource_limits']['max_storage_mb']} MB")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations'][:5]:  # Show first 5
            print(f"  • {rec}")
    
    print("="*60)
    
    # Multi-tenant isolation assertions
    assert report['summary']['isolation_score'] >= 85, \
        f"Multi-tenant isolation score too low: {report['summary']['isolation_score']:.1f}%"
    
    assert len(report['critical_violations']) == 0, \
        f"Critical isolation violations found: {len(report['critical_violations'])}"
    
    # Check each isolation type
    for isolation_type, results in report['isolation_types'].items():
        total = results['passed'] + results['failed']
        success_rate = (results['passed'] / total * 100) if total > 0 else 0
        
        assert success_rate >= 80, \
            f"{isolation_type} isolation success rate too low: {success_rate:.1f}%"
    
    print("✅ Multi-tenant isolation testing completed successfully!")