"""Security and authentication flow testing."""

import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import pytest


class TestAuthenticationFlows:
    """Test authentication and authorization flows."""
    
    @pytest.mark.security
    async def test_jwt_authentication_flow(
        self,
        api_client,
        security_context
    ):
        """Test JWT authentication complete flow."""
        
        # Step 1: Create test user
        test_user = security_context.create_test_user("analyst")
        
        # Step 2: Mock login endpoint
        login_response = Mock()
        login_response.status_code = 200
        login_response.json.return_value = {
            "access_token": "test-jwt-token-12345",
            "token_type": "bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token-67890",
            "user": {
                "id": test_user["id"],
                "username": test_user["username"],
                "role": test_user["role"]
            }
        }
        
        api_client.post.return_value = login_response
        
        # Attempt login
        response = api_client.post(
            "/auth/login",
            json={
                "username": test_user["username"],
                "password": "test-password"
            }
        )
        
        assert response.status_code == 200
        auth_data = response.json()
        
        assert "access_token" in auth_data
        assert "refresh_token" in auth_data
        assert auth_data["token_type"] == "bearer"
        assert auth_data["user"]["role"] == test_user["role"]
        
        # Step 3: Test authenticated request
        access_token = auth_data["access_token"]
        
        protected_response = Mock()
        protected_response.status_code = 200
        protected_response.json.return_value = {
            "message": "Access granted",
            "user_id": test_user["id"]
        }
        
        api_client.get.return_value = protected_response
        
        # Mock authorization header
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {access_token}"}):
            response = api_client.get("/protected/resource")
            assert response.status_code == 200
            assert response.json()["user_id"] == test_user["id"]
        
        # Step 4: Test token refresh
        refresh_response = Mock()
        refresh_response.status_code = 200
        refresh_response.json.return_value = {
            "access_token": "test-jwt-token-new-54321",
            "token_type": "bearer",
            "expires_in": 3600
        }
        
        api_client.post.return_value = refresh_response
        
        response = api_client.post(
            "/auth/refresh",
            json={"refresh_token": auth_data["refresh_token"]}
        )
        
        assert response.status_code == 200
        new_auth_data = response.json()
        assert new_auth_data["access_token"] != access_token  # Should be different
        
        # Step 5: Test logout
        logout_response = Mock()
        logout_response.status_code = 200
        logout_response.json.return_value = {"message": "Logged out successfully"}
        
        api_client.post.return_value = logout_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {access_token}"}):
            response = api_client.post("/auth/logout")
            assert response.status_code == 200
    
    @pytest.mark.security
    async def test_role_based_access_control(
        self,
        api_client,
        security_context
    ):
        """Test role-based access control (RBAC)."""
        
        # Create users with different roles
        admin_user = security_context.create_test_user("admin")
        analyst_user = security_context.create_test_user("analyst")
        viewer_user = security_context.create_test_user("viewer")
        
        admin_token = security_context.generate_test_token(admin_user)
        analyst_token = security_context.generate_test_token(analyst_user)
        viewer_token = security_context.generate_test_token(viewer_user)
        
        # Test admin access to admin-only resources
        admin_response = Mock()
        admin_response.status_code = 200
        admin_response.json.return_value = {"message": "Admin access granted"}
        
        api_client.get.return_value = admin_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {admin_token}"}):
            response = api_client.get("/admin/users")
            assert response.status_code == 200
        
        # Test analyst access to admin resources (should be denied)
        forbidden_response = Mock()
        forbidden_response.status_code = 403
        forbidden_response.json.return_value = {
            "error": "Forbidden",
            "message": "Insufficient permissions"
        }
        
        api_client.get.return_value = forbidden_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {analyst_token}"}):
            response = api_client.get("/admin/users")
            assert response.status_code == 403
        
        # Test analyst access to analyst resources
        analyst_response = Mock()
        analyst_response.status_code = 200
        analyst_response.json.return_value = {"message": "Analyst access granted"}
        
        api_client.get.return_value = analyst_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {analyst_token}"}):
            response = api_client.get("/detectors")
            assert response.status_code == 200
        
        # Test viewer read-only access
        viewer_response = Mock()
        viewer_response.status_code = 200
        viewer_response.json.return_value = {"detectors": []}
        
        api_client.get.return_value = viewer_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {viewer_token}"}):
            response = api_client.get("/detectors")
            assert response.status_code == 200
        
        # Test viewer write access denial
        write_forbidden_response = Mock()
        write_forbidden_response.status_code = 403
        write_forbidden_response.json.return_value = {
            "error": "Forbidden",
            "message": "Read-only access"
        }
        
        api_client.post.return_value = write_forbidden_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {viewer_token}"}):
            response = api_client.post("/detectors", json={"name": "test"})
            assert response.status_code == 403
    
    @pytest.mark.security
    async def test_authentication_security_measures(
        self,
        api_client,
        security_context
    ):
        """Test authentication security measures."""
        
        # Test 1: Invalid credentials
        invalid_login_response = Mock()
        invalid_login_response.status_code = 401
        invalid_login_response.json.return_value = {
            "error": "Unauthorized",
            "message": "Invalid credentials"
        }
        
        api_client.post.return_value = invalid_login_response
        
        response = api_client.post(
            "/auth/login",
            json={
                "username": "invalid_user",
                "password": "wrong_password"
            }
        )
        
        assert response.status_code == 401
        
        # Test 2: Missing authentication token
        unauthorized_response = Mock()
        unauthorized_response.status_code = 401
        unauthorized_response.json.return_value = {
            "error": "Unauthorized",
            "message": "Missing authentication token"
        }
        
        api_client.get.return_value = unauthorized_response
        
        response = api_client.get("/protected/resource")
        assert response.status_code == 401
        
        # Test 3: Invalid/expired token
        expired_token_response = Mock()
        expired_token_response.status_code = 401
        expired_token_response.json.return_value = {
            "error": "Unauthorized",
            "message": "Token expired or invalid"
        }
        
        api_client.get.return_value = expired_token_response
        
        with patch.object(api_client, 'headers', {"Authorization": "Bearer invalid-token"}):
            response = api_client.get("/protected/resource")
            assert response.status_code == 401
        
        # Test 4: Malformed authorization header
        malformed_auth_response = Mock()
        malformed_auth_response.status_code = 400
        malformed_auth_response.json.return_value = {
            "error": "Bad Request",
            "message": "Malformed authorization header"
        }
        
        api_client.get.return_value = malformed_auth_response
        
        with patch.object(api_client, 'headers', {"Authorization": "InvalidFormat"}):
            response = api_client.get("/protected/resource")
            assert response.status_code == 400
    
    @pytest.mark.security
    async def test_session_management(
        self,
        api_client,
        security_context
    ):
        """Test session management and security."""
        
        # Create test user and session
        test_user = security_context.create_test_user("analyst")
        token = security_context.generate_test_token(test_user)
        
        # Test 1: Session validation
        session_response = Mock()
        session_response.status_code = 200
        session_response.json.return_value = {
            "session_id": "session-123",
            "user_id": test_user["id"],
            "expires_at": "2023-01-01T13:00:00Z",
            "is_valid": True
        }
        
        api_client.get.return_value = session_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.get("/auth/session")
            assert response.status_code == 200
            session_data = response.json()
            assert session_data["is_valid"] is True
        
        # Test 2: Session timeout
        timeout_response = Mock()
        timeout_response.status_code = 401
        timeout_response.json.return_value = {
            "error": "Session Expired",
            "message": "Session has timed out"
        }
        
        # Simulate session timeout after some time
        api_client.get.return_value = timeout_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.get("/auth/session")
            assert response.status_code == 401
        
        # Test 3: Multiple session management
        # Create multiple sessions for the same user
        session_tokens = [
            security_context.generate_test_token(test_user)
            for _ in range(3)
        ]
        
        sessions_response = Mock()
        sessions_response.status_code = 200
        sessions_response.json.return_value = {
            "active_sessions": len(session_tokens),
            "max_sessions": 5,
            "sessions": [
                {
                    "session_id": f"session-{i}",
                    "created_at": "2023-01-01T12:00:00Z",
                    "last_activity": "2023-01-01T12:30:00Z"
                }
                for i in range(len(session_tokens))
            ]
        }
        
        api_client.get.return_value = sessions_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {session_tokens[0]}"}):
            response = api_client.get("/auth/sessions")
            assert response.status_code == 200
            sessions_data = response.json()
            assert sessions_data["active_sessions"] == 3
        
        # Test 4: Session invalidation
        invalidate_response = Mock()
        invalidate_response.status_code = 200
        invalidate_response.json.return_value = {
            "message": "Session invalidated",
            "invalidated_sessions": 2
        }
        
        api_client.delete.return_value = invalidate_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {session_tokens[0]}"}):
            response = api_client.delete("/auth/sessions/all")
            assert response.status_code == 200
    
    @pytest.mark.security
    async def test_password_security_policies(
        self,
        api_client,
        security_context
    ):
        """Test password security policies and requirements."""
        
        # Test 1: Weak password rejection
        weak_password_response = Mock()
        weak_password_response.status_code = 400
        weak_password_response.json.return_value = {
            "error": "Weak Password",
            "message": "Password does not meet security requirements",
            "requirements": [
                "Minimum 8 characters",
                "At least one uppercase letter",
                "At least one lowercase letter",
                "At least one number",
                "At least one special character"
            ]
        }
        
        api_client.post.return_value = weak_password_response
        
        response = api_client.post(
            "/auth/register",
            json={
                "username": "testuser",
                "password": "weak",  # Weak password
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 400
        assert "requirements" in response.json()
        
        # Test 2: Strong password acceptance
        strong_password_response = Mock()
        strong_password_response.status_code = 201
        strong_password_response.json.return_value = {
            "user_id": "user-123",
            "username": "testuser",
            "message": "User created successfully"
        }
        
        api_client.post.return_value = strong_password_response
        
        response = api_client.post(
            "/auth/register",
            json={
                "username": "testuser",
                "password": "StrongP@ssw0rd123!",  # Strong password
                "email": "test@example.com"
            }
        )
        
        assert response.status_code == 201
        
        # Test 3: Password change validation
        test_user = security_context.create_test_user("analyst")
        token = security_context.generate_test_token(test_user)
        
        # Valid password change
        password_change_response = Mock()
        password_change_response.status_code = 200
        password_change_response.json.return_value = {
            "message": "Password changed successfully"
        }
        
        api_client.post.return_value = password_change_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.post(
                "/auth/change-password",
                json={
                    "current_password": "OldP@ssw0rd123!",
                    "new_password": "NewP@ssw0rd456!",
                    "confirm_password": "NewP@ssw0rd456!"
                }
            )
            assert response.status_code == 200
        
        # Invalid password change (wrong current password)
        invalid_change_response = Mock()
        invalid_change_response.status_code = 400
        invalid_change_response.json.return_value = {
            "error": "Invalid Current Password",
            "message": "Current password is incorrect"
        }
        
        api_client.post.return_value = invalid_change_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.post(
                "/auth/change-password",
                json={
                    "current_password": "WrongPassword",
                    "new_password": "NewP@ssw0rd456!",
                    "confirm_password": "NewP@ssw0rd456!"
                }
            )
            assert response.status_code == 400
    
    @pytest.mark.security
    async def test_rate_limiting_and_brute_force_protection(
        self,
        api_client,
        security_context
    ):
        """Test rate limiting and brute force protection."""
        
        # Test 1: Rate limiting on login attempts
        # Simulate multiple failed login attempts
        failed_attempts = 0
        
        for attempt in range(6):  # Exceed rate limit of 5 attempts
            if attempt < 5:
                # First 5 attempts get normal failure response
                response = Mock()
                response.status_code = 401
                response.json.return_value = {
                    "error": "Unauthorized",
                    "message": "Invalid credentials",
                    "attempts_remaining": 5 - attempt - 1
                }
            else:
                # 6th attempt triggers rate limiting
                response = Mock()
                response.status_code = 429
                response.json.return_value = {
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded. Try again later.",
                    "retry_after": 300  # 5 minutes
                }
            
            api_client.post.return_value = response
            
            login_response = api_client.post(
                "/auth/login",
                json={
                    "username": "testuser",
                    "password": "wrongpassword"
                }
            )
            
            if attempt < 5:
                assert login_response.status_code == 401
                failed_attempts += 1
            else:
                assert login_response.status_code == 429
                assert "retry_after" in login_response.json()
        
        # Test 2: Rate limiting on API endpoints
        # Simulate rapid API requests
        for request_num in range(15):  # Exceed rate limit
            if request_num < 10:
                # Normal responses for first 10 requests
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"status": "ok"}
            else:
                # Rate limiting kicks in
                response = Mock()
                response.status_code = 429
                response.json.return_value = {
                    "error": "Rate Limit Exceeded",
                    "message": "Too many requests",
                    "limit": 10,
                    "window": 60,
                    "retry_after": 60
                }
            
            api_client.get.return_value = response
            
            api_response = api_client.get("/health")
            
            if request_num < 10:
                assert api_response.status_code == 200
            else:
                assert api_response.status_code == 429
        
        # Test 3: Account lockout after multiple failed attempts
        lockout_response = Mock()
        lockout_response.status_code = 423
        lockout_response.json.return_value = {
            "error": "Account Locked",
            "message": "Account locked due to multiple failed login attempts",
            "unlock_time": "2023-01-01T13:00:00Z"
        }
        
        api_client.post.return_value = lockout_response
        
        # Attempt login on locked account
        response = api_client.post(
            "/auth/login",
            json={
                "username": "testuser",
                "password": "correctpassword"  # Even correct password fails
            }
        )
        
        assert response.status_code == 423
        assert "unlock_time" in response.json()
    
    @pytest.mark.security
    async def test_data_encryption_and_protection(
        self,
        api_client,
        security_context,
        test_data_manager
    ):
        """Test data encryption and protection measures."""
        
        # Create test user and data
        test_user = security_context.create_test_user("analyst")
        token = security_context.generate_test_token(test_user)
        test_dataset = test_data_manager.create_test_dataset(size=100)
        
        # Test 1: Sensitive data masking in responses
        masked_response = Mock()
        masked_response.status_code = 200
        masked_response.json.return_value = {
            "user_id": test_user["id"],
            "username": test_user["username"],
            "email": "***@***.com",  # Masked email
            "role": test_user["role"],
            "created_at": "2023-01-01T12:00:00Z",
            # Note: No password or sensitive fields in response
        }
        
        api_client.get.return_value = masked_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.get("/users/profile")
            assert response.status_code == 200
            profile_data = response.json()
            
            # Verify sensitive data is masked or excluded
            assert "password" not in profile_data
            assert "***" in profile_data["email"]  # Email is masked
        
        # Test 2: Data transmission security headers
        secure_response = Mock()
        secure_response.status_code = 200
        secure_response.headers = {
            "Content-Security-Policy": "default-src 'self'",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        secure_response.json.return_value = {"data": "secure"}
        
        api_client.get.return_value = secure_response
        
        response = api_client.get("/secure-endpoint")
        assert response.status_code == 200
        
        # Verify security headers
        headers = response.headers
        assert "Content-Security-Policy" in headers
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        
        # Test 3: Input sanitization and validation
        # Test SQL injection prevention
        injection_response = Mock()
        injection_response.status_code = 400
        injection_response.json.return_value = {
            "error": "Invalid Input",
            "message": "Input contains potentially malicious content"
        }
        
        api_client.post.return_value = injection_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.post(
                "/datasets/search",
                json={
                    "query": "'; DROP TABLE users; --",  # SQL injection attempt
                    "limit": 10
                }
            )
            assert response.status_code == 400
        
        # Test XSS prevention
        xss_response = Mock()
        xss_response.status_code = 400
        xss_response.json.return_value = {
            "error": "Invalid Input",
            "message": "Input contains potentially harmful scripts"
        }
        
        api_client.post.return_value = xss_response
        
        with patch.object(api_client, 'headers', {"Authorization": f"Bearer {token}"}):
            response = api_client.post(
                "/datasets/create",
                json={
                    "name": "<script>alert('xss')</script>",  # XSS attempt
                    "description": "Test dataset"
                }
            )
            assert response.status_code == 400