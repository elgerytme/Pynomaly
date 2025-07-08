"""
Tests for the forward-reference-free dependency injection system.
"""

import pytest
from typing import Any, Dict
from fastapi import Depends

from pynomaly.infrastructure.dependencies import (
    DependencyWrapper,
    register_dependency,
    register_dependency_provider,
    clear_dependencies,
    initialize_dependencies,
    get_dependency,
    is_dependency_available,
    test_dependency_context,
    create_mock_dependencies,
    setup_test_dependencies,
    validate_standard_dependencies,
)


class TestDependencyWrapper:
    """Test the DependencyWrapper class."""
    
    def test_dependency_wrapper_creation(self):
        """Test that DependencyWrapper can be created."""
        wrapper = DependencyWrapper("test_service")
        assert wrapper.dependency_key == "test_service"
        assert wrapper.optional == False
        
        optional_wrapper = DependencyWrapper("optional_service", optional=True)
        assert optional_wrapper.dependency_key == "optional_service"
        assert optional_wrapper.optional == True
    
    def test_dependency_wrapper_call_returns_depends(self):
        """Test that calling the wrapper returns a Depends object."""
        with test_dependency_context():
            # Register a test service
            test_service = {"name": "test"}
            register_dependency("test_service", test_service)
            
            # Create wrapper and call it
            wrapper = DependencyWrapper("test_service")
            depends_obj = wrapper()
            
            # Should return a Depends object
            assert isinstance(depends_obj, Depends)
    
    def test_optional_dependency_wrapper(self):
        """Test optional dependency wrapper behavior."""
        with test_dependency_context():
            # Don't register the service
            
            # Create optional wrapper
            wrapper = DependencyWrapper("missing_service", optional=True)
            depends_obj = wrapper()
            
            # Should still return a Depends object
            assert isinstance(depends_obj, Depends)


class TestDependencyRegistry:
    """Test the dependency registry functionality."""
    
    def test_register_and_get_dependency(self):
        """Test registering and retrieving dependencies."""
        with test_dependency_context():
            # Register a dependency
            test_service = {"name": "test_service"}
            register_dependency("test_service", test_service)
            
            # Retrieve the dependency
            retrieved = get_dependency("test_service")
            assert retrieved == test_service
    
    def test_register_provider_function(self):
        """Test registering a provider function."""
        with test_dependency_context():
            # Define a provider function
            def create_service():
                return {"name": "provider_service", "created": True}
            
            # Register the provider
            register_dependency_provider("provider_service", create_service)
            
            # Retrieve the dependency (should call the provider)
            retrieved = get_dependency("provider_service")
            assert retrieved["name"] == "provider_service"
            assert retrieved["created"] == True
    
    def test_dependency_availability_check(self):
        """Test checking if dependencies are available."""
        with test_dependency_context():
            # Initially, no dependencies should be available
            assert not is_dependency_available("test_service")
            
            # Register a dependency
            register_dependency("test_service", {"name": "test"})
            
            # Now it should be available
            assert is_dependency_available("test_service")
    
    def test_clear_dependencies(self):
        """Test clearing all dependencies."""
        with test_dependency_context():
            # Register some dependencies
            register_dependency("service1", {"name": "service1"})
            register_dependency("service2", {"name": "service2"})
            
            # Both should be available
            assert is_dependency_available("service1")
            assert is_dependency_available("service2")
            
            # Clear dependencies
            clear_dependencies()
            
            # Now neither should be available
            assert not is_dependency_available("service1")
            assert not is_dependency_available("service2")


class TestMockDependencies:
    """Test the mock dependency system."""
    
    def test_create_mock_dependencies(self):
        """Test creating mock dependencies."""
        mock_deps = create_mock_dependencies()
        
        # Check that expected services are created
        assert "auth_service" in mock_deps
        assert "detection_service" in mock_deps
        assert "model_service" in mock_deps
        assert "database_service" in mock_deps
        
        # Test that they have expected methods
        assert hasattr(mock_deps["auth_service"], "authenticate")
        assert hasattr(mock_deps["detection_service"], "detect")
        assert hasattr(mock_deps["model_service"], "train")
        assert hasattr(mock_deps["database_service"], "save")
    
    def test_setup_test_dependencies(self):
        """Test setting up mock dependencies."""
        with test_dependency_context():
            # Setup mock dependencies
            setup_test_dependencies()
            
            # Check that services are available
            assert is_dependency_available("auth_service")
            assert is_dependency_available("detection_service")
            assert is_dependency_available("model_service")
            assert is_dependency_available("database_service")
            
            # Test that services work
            auth_service = get_dependency("auth_service")
            assert auth_service.authenticate("valid_token") == True
            assert auth_service.authenticate("invalid_token") == False
            
            detection_service = get_dependency("detection_service")
            result = detection_service.detect("test_data")
            assert result == [1, 2, 3]


class TestDependencyValidation:
    """Test the dependency validation system."""
    
    def test_validate_standard_dependencies(self):
        """Test validating standard dependencies."""
        with test_dependency_context():
            # Initially, no dependencies should be available
            results = validate_standard_dependencies()
            assert results["total_available"] == 0
            assert results["total_expected"] > 0
            assert not results["valid"]
            
            # Setup mock dependencies
            setup_test_dependencies()
            
            # Now some dependencies should be available
            results = validate_standard_dependencies()
            assert results["total_available"] > 0
            assert len(results["available"]) > 0
            assert len(results["missing"]) < results["total_expected"]


class TestCommonServiceWrappers:
    """Test the common service wrapper functions."""
    
    def test_auth_service_wrapper(self):
        """Test the auth service wrapper."""
        from pynomaly.infrastructure.dependencies import auth_service
        
        wrapper = auth_service()
        assert isinstance(wrapper, DependencyWrapper)
        assert wrapper.dependency_key == "auth_service"
        assert wrapper.optional == True  # Auth service is optional
    
    def test_detection_service_wrapper(self):
        """Test the detection service wrapper."""
        from pynomaly.infrastructure.dependencies import detection_service
        
        wrapper = detection_service()
        assert isinstance(wrapper, DependencyWrapper)
        assert wrapper.dependency_key == "detection_service"
        assert wrapper.optional == False  # Detection service is required
    
    def test_model_service_wrapper(self):
        """Test the model service wrapper."""
        from pynomaly.infrastructure.dependencies import model_service
        
        wrapper = model_service()
        assert isinstance(wrapper, DependencyWrapper)
        assert wrapper.dependency_key == "model_service"
        assert wrapper.optional == False  # Model service is required


class TestDependencyContext:
    """Test the dependency context manager."""
    
    def test_dependency_context_isolation(self):
        """Test that dependency context provides isolation."""
        # Register a dependency outside the context
        register_dependency("global_service", {"name": "global"})
        assert is_dependency_available("global_service")
        
        # Use context manager
        with test_dependency_context():
            # Global service should not be available in context
            assert not is_dependency_available("global_service")
            
            # Register a service inside context
            register_dependency("context_service", {"name": "context"})
            assert is_dependency_available("context_service")
        
        # After context, global service should be restored
        assert is_dependency_available("global_service")
        
        # Context service should not be available outside context
        assert not is_dependency_available("context_service")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_get_nonexistent_dependency(self):
        """Test getting a dependency that doesn't exist."""
        with test_dependency_context():
            # Should raise HTTPException for non-existent dependency
            with pytest.raises(Exception):  # HTTPException from FastAPI
                get_dependency("nonexistent_service")
    
    def test_optional_dependency_missing(self):
        """Test optional dependency when service is missing."""
        with test_dependency_context():
            wrapper = DependencyWrapper("missing_service", optional=True)
            depends_obj = wrapper()
            
            # Should return a Depends object that returns None
            assert isinstance(depends_obj, Depends)
    
    def test_provider_function_error(self):
        """Test provider function that raises an error."""
        with test_dependency_context():
            def failing_provider():
                raise ValueError("Provider failed")
            
            register_dependency_provider("failing_service", failing_provider)
            
            # Should raise HTTPException when trying to get the dependency
            with pytest.raises(Exception):
                get_dependency("failing_service")


# Integration test
def test_end_to_end_dependency_usage():
    """Test complete end-to-end dependency usage."""
    with test_dependency_context():
        # Setup mock dependencies
        setup_test_dependencies()
        
        # Create wrappers
        from pynomaly.infrastructure.dependencies import (
            auth_service,
            detection_service,
        )
        
        auth_wrapper = auth_service()
        detection_wrapper = detection_service()
        
        # Get FastAPI Depends objects
        auth_depends = auth_wrapper()
        detection_depends = detection_wrapper()
        
        # Both should be Depends objects
        assert isinstance(auth_depends, Depends)
        assert isinstance(detection_depends, Depends)
        
        # The underlying services should be available
        assert is_dependency_available("auth_service")
        assert is_dependency_available("detection_service")
        
        # Services should work correctly
        auth_svc = get_dependency("auth_service")
        detection_svc = get_dependency("detection_service")
        
        assert auth_svc.authenticate("valid_token") == True
        assert detection_svc.detect("test") == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
