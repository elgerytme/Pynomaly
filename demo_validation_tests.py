#!/usr/bin/env python3
"""
Demo script to showcase validation, error, and edge-case tests for API endpoints.

This demonstrates the three main testing categories:
1. Malformed JSON, missing required fields, wrong enum values → 422 responses
2. Boundary values (max string length, numeric limits) 
3. Internal errors via monkeypatching → 5xx handling and error body
"""

import json
import tempfile
import os
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from pynomaly.presentation.api.app import create_app


def demo_malformed_json_tests():
    """Demonstrate malformed JSON validation tests."""
    print("=" * 60)
    print("1. MALFORMED JSON & VALIDATION TESTS (422 Responses)")
    print("=" * 60)
    
    app = create_app()
    client = TestClient(app)
    
    # Test 1: Malformed JSON
    print("\n🔴 Test 1: Malformed JSON in auth register")
    response = client.post(
        "/api/v1/auth/register",
        data='{"username": "test", "email": "test@',  # Malformed JSON
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {response.status_code} (Expected: 422)")
    print(f"   Response: {response.json() if response.status_code != 422 else 'Malformed JSON detected'}")
    
    # Test 2: Missing required fields
    print("\n🔴 Test 2: Missing required fields")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "password123"
            # Missing required 'username' field
        }
    )
    print(f"   Status: {response.status_code} (Expected: 422)")
    if response.status_code == 422:
        errors = response.json()["detail"]
        username_error = any("username" in str(error) for error in errors)
        print(f"   Username validation error found: {username_error}")
    
    # Test 3: Invalid field types
    print("\n🔴 Test 3: Invalid email format")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "invalid-email",  # Invalid email format
            "password": "password123"
        }
    )
    print(f"   Status: {response.status_code} (Expected: 422)")
    
    # Test 4: Invalid UUID format
    print("\n🔴 Test 4: Invalid UUID in path parameter")
    response = client.get("/api/v1/datasets/invalid-uuid-format")
    print(f"   Status: {response.status_code} (Expected: 422)")


def demo_boundary_value_tests():
    """Demonstrate boundary value tests."""
    print("\n\n" + "=" * 60)
    print("2. BOUNDARY VALUE TESTS")
    print("=" * 60)
    
    app = create_app()
    client = TestClient(app)
    
    # Test 1: Maximum string length
    print("\n🟡 Test 1: Very long username (1000 chars)")
    long_username = "a" * 1000
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": long_username,
            "email": "test@example.com", 
            "password": "password123"
        }
    )
    print(f"   Status: {response.status_code} (Expected: 422)")
    
    # Test 2: Empty required fields
    print("\n🟡 Test 2: Empty username")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "",  # Empty string
            "email": "test@example.com",
            "password": "password123"
        }
    )
    print(f"   Status: {response.status_code} (Expected: 422)")
    
    # Test 3: Password too short
    print("\n🟡 Test 3: Password too short")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "123"  # Too short
        }
    )
    print(f"   Status: {response.status_code} (Expected: 422)")
    
    # Test 4: Negative numeric values
    print("\n🟡 Test 4: Negative limit parameter")
    response = client.get("/api/v1/datasets/", params={"limit": -1})
    print(f"   Status: {response.status_code} (Expected: 422)")
    
    # Test 5: Very large numeric values
    print("\n🟡 Test 5: Very large limit parameter")
    response = client.get("/api/v1/datasets/", params={"limit": 999999})
    print(f"   Status: {response.status_code} (Expected: 422)")


def demo_internal_error_tests():
    """Demonstrate internal error handling via monkeypatching."""
    print("\n\n" + "=" * 60)
    print("3. INTERNAL ERROR HANDLING TESTS (5xx Responses)")
    print("=" * 60)
    
    app = create_app()
    client = TestClient(app)
    
    # Test 1: Database connection error
    print("\n🔥 Test 1: Database connection error (mocked)")
    with patch('pynomaly.infrastructure.config.Container.dataset_repository') as mock_repo:
        mock_repo.return_value.find_all.side_effect = ConnectionError("Database connection failed")
        
        response = client.get("/api/v1/datasets/")
        print(f"   Status: {response.status_code} (Expected: 500)")
        if response.status_code == 500:
            error_data = response.json()
            print(f"   Error structure contains 'detail': {'detail' in error_data}")
    
    # Test 2: Service layer exception
    print("\n🔥 Test 2: Service layer exception (mocked)")
    with patch('pynomaly.infrastructure.config.Container.dataset_repository') as mock_repo:
        mock_repo.return_value.find_by_id.side_effect = RuntimeError("Service unavailable")
        
        test_id = str(uuid4())
        response = client.get(f"/api/v1/datasets/{test_id}")
        print(f"   Status: {response.status_code} (Expected: 500)")
    
    # Test 3: Auth service unavailable
    print("\n🔥 Test 3: Authentication service unavailable (mocked)")
    with patch('pynomaly.infrastructure.auth.get_auth') as mock_auth:
        mock_auth.return_value = None  # Simulate auth service unavailable
        
        response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "admin123"
        })
        print(f"   Status: {response.status_code} (Expected: 503)")
        if response.status_code == 503:
            error_data = response.json()
            contains_service_msg = "service not available" in error_data["detail"].lower()
            print(f"   Contains 'service not available': {contains_service_msg}")
    
    # Test 4: File processing error
    print("\n🔥 Test 4: File processing error (mocked)")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n1,2\n3,4\n")
        temp_file = f.name
    
    try:
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = Exception("File processing failed")
            
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test.csv", file, "text/csv")},
                    data={"name": "test_dataset"}
                )
            
            print(f"   Status: {response.status_code} (Expected: 400)")
            if response.status_code == 400:
                error_data = response.json()
                contains_failed = "failed" in error_data["detail"].lower()
                print(f"   Contains 'failed': {contains_failed}")
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    # Test 5: Memory error
    print("\n🔥 Test 5: Memory error (mocked)")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n1,2\n3,4\n")
        temp_file = f.name
    
    try:
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = MemoryError("Not enough memory")
            
            with open(temp_file, 'rb') as file:
                response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test.csv", file, "text/csv")},
                    data={"name": "test_dataset"}
                )
            
            print(f"   Status: {response.status_code} (Expected: 500)")
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass


def demo_error_response_structure():
    """Demonstrate error response structure consistency."""
    print("\n\n" + "=" * 60)
    print("4. ERROR RESPONSE STRUCTURE VALIDATION")
    print("=" * 60)
    
    app = create_app()
    client = TestClient(app)
    
    # Test 422 error structure
    print("\n📋 Test 1: 422 Error Structure")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "",  # Empty required field
            "email": "test@example.com",
            "password": "password123"
        }
    )
    
    if response.status_code == 422:
        error_data = response.json()
        print(f"   Has 'detail' field: {'detail' in error_data}")
        print(f"   Detail is list: {isinstance(error_data['detail'], list)}")
        
        # Check validation error structure
        if error_data["detail"]:
            first_error = error_data["detail"][0]
            has_required_fields = all(field in first_error for field in ["loc", "msg", "type"])
            print(f"   Validation errors have required fields: {has_required_fields}")
    
    # Test 401 error structure
    print("\n📋 Test 2: 401 Error Structure")
    response = client.get("/api/v1/auth/me")
    
    if response.status_code == 401:
        error_data = response.json()
        print(f"   Has 'detail' field: {'detail' in error_data}")
        print(f"   Detail is string: {isinstance(error_data['detail'], str)}")
    
    # Test 404 error structure
    print("\n📋 Test 3: 404 Error Structure")
    response = client.get("/api/v1/nonexistent")
    
    if response.status_code == 404:
        error_data = response.json()
        print(f"   Has 'detail' field: {'detail' in error_data}")
        print(f"   Detail is string: {isinstance(error_data['detail'], str)}")


def demo_edge_case_scenarios():
    """Demonstrate edge case scenarios."""
    print("\n\n" + "=" * 60)
    print("5. EDGE CASE SCENARIOS")
    print("=" * 60)
    
    app = create_app()
    client = TestClient(app)
    
    # Test 1: Unicode characters
    print("\n🌐 Test 1: Unicode characters in request")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "用户名",  # Chinese characters
            "email": "test@example.com",
            "password": "password123"
        }
    )
    print(f"   Status: {response.status_code} (Should handle unicode)")
    
    # Test 2: SQL injection attempt
    print("\n🌐 Test 2: SQL injection attempt in parameters")
    response = client.get(
        "/api/v1/datasets/",
        params={"limit": "1; DROP TABLE datasets;"}
    )
    print(f"   Status: {response.status_code} (Expected: 422 - invalid parameter type)")
    
    # Test 3: XSS attempt
    print("\n🌐 Test 3: XSS attempt in registration")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "<script>alert('xss')</script>",
            "email": "test@example.com",
            "password": "password123"
        }
    )
    print(f"   Status: {response.status_code} (Should handle XSS safely)")
    
    # Test 4: Empty file upload
    print("\n🌐 Test 4: Empty file upload")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write nothing to file
        temp_file = f.name
    
    try:
        with open(temp_file, 'rb') as file:
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("empty.csv", file, "text/csv")},
                data={"name": "empty_dataset"}
            )
        
        print(f"   Status: {response.status_code} (Expected: 400 - empty file)")
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    # Test 5: Multiple validation errors
    print("\n🌐 Test 5: Multiple validation errors in single request")
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": "",           # Empty username
            "email": "invalid-email", # Invalid email
            "password": "123"         # Too short password
        }
    )
    
    if response.status_code == 422:
        error_data = response.json()
        error_count = len(error_data["detail"])
        print(f"   Multiple errors detected: {error_count > 1} ({error_count} errors)")


def main():
    """Run all validation test demonstrations."""
    print("🚀 VALIDATION, ERROR, AND EDGE-CASE TESTS DEMONSTRATION")
    print("This showcases comprehensive API endpoint testing covering:")
    print("  1. Malformed JSON & missing fields → 422 responses")
    print("  2. Boundary values → appropriate error handling") 
    print("  3. Internal errors via mocking → 5xx responses")
    print("  4. Error response structure consistency")
    print("  5. Edge case scenarios")
    
    try:
        demo_malformed_json_tests()
        demo_boundary_value_tests()
        demo_internal_error_tests()
        demo_error_response_structure()
        demo_edge_case_scenarios()
        
        print("\n\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nAll test categories have been demonstrated:")
        print("✓ Malformed JSON validation (422 responses)")
        print("✓ Boundary value testing")
        print("✓ Internal error handling via monkeypatching (5xx responses)")
        print("✓ Error response structure consistency")
        print("✓ Edge case scenario handling")
        
        print("\nTo run the actual test suite:")
        print("pytest tests/presentation/api/test_validation_error_edge_cases.py -v")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This may be due to missing dependencies or configuration issues.")


if __name__ == "__main__":
    main()
