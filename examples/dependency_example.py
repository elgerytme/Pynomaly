"""
Example script demonstrating the forward-reference-free dependency system.

This script shows how to use the new dependency injection system to avoid
circular import issues in FastAPI applications.
"""

import asyncio
from typing import Dict, Any

from fastapi import FastAPI
from pynomaly.infrastructure.dependencies import (
    DependencyWrapper,
    register_dependency,
    register_dependency_provider,
    clear_dependencies,
    initialize_dependencies,
    # Common service wrappers
    auth_service,
    detection_service,
    model_service,
    database_service,
    # Test utilities
    create_mock_dependencies,
    setup_test_dependencies,
    validate_standard_dependencies,
    run_dependency_health_check,
)


def main():
    """Demonstrate the dependency system."""
    print("Forward-Reference-Free Dependency System Demo")
    print("=" * 50)
    
    # 1. Clear any existing dependencies
    clear_dependencies()
    
    # 2. Setup mock dependencies for demonstration
    print("\n1. Setting up mock dependencies...")
    setup_test_dependencies()
    
    # 3. Validate the setup
    print("\n2. Validating dependencies...")
    results = validate_standard_dependencies()
    print(f"   Available: {results['total_available']}/{results['total_expected']}")
    
    # 4. Test individual dependencies
    print("\n3. Testing individual dependencies...")
    
    # Test auth service
    auth_wrapper = auth_service()
    auth_dep = auth_wrapper()  # This returns a FastAPI Depends object
    print(f"   Auth service wrapper created: {type(auth_dep)}")
    
    # Test detection service
    detection_wrapper = detection_service()
    detection_dep = detection_wrapper()
    print(f"   Detection service wrapper created: {type(detection_dep)}")
    
    # 5. Custom dependency example
    print("\n4. Creating custom dependency...")
    
    class CustomService:
        def process(self, data: str) -> str:
            return f"Processed: {data}"
    
    # Register custom dependency
    custom_service = CustomService()
    register_dependency("custom_service", custom_service)
    
    # Create wrapper for custom dependency
    custom_wrapper = DependencyWrapper("custom_service", optional=True)
    custom_dep = custom_wrapper()
    print(f"   Custom service wrapper created: {type(custom_dep)}")
    
    # 6. Provider function example
    print("\n5. Using provider function...")
    
    def create_expensive_service():
        print("   Creating expensive service...")
        return {"expensive": True, "data": "computed"}
    
    register_dependency_provider("expensive_service", create_expensive_service)
    
    expensive_wrapper = DependencyWrapper("expensive_service")
    expensive_dep = expensive_wrapper()
    print(f"   Expensive service wrapper created: {type(expensive_dep)}")
    
    # 7. Run comprehensive health check
    print("\n6. Running health check...")
    run_dependency_health_check()
    
    # 8. Show how to use in FastAPI endpoint
    print("\n7. FastAPI endpoint example...")
    show_fastapi_example()
    
    # 9. Cleanup
    print("\n8. Cleaning up...")
    clear_dependencies()
    print("   Dependencies cleared")


def show_fastapi_example():
    """Show how to use the dependency system in FastAPI endpoints."""
    print("   Creating FastAPI app with dependency injection...")
    
    app = FastAPI(title="Dependency Example")
    
    # This is how you would use dependencies in your router files
    from fastapi import APIRouter
    
    router = APIRouter()
    
    # Declare dependencies without type hints
    get_auth_service = auth_service()
    get_detection_service = detection_service()
    get_custom_service = DependencyWrapper("custom_service", optional=True)
    
    # Example endpoint using the dependencies
    @router.post("/detect")
    async def detect_endpoint(
        data: str,
        # Dependencies without type hints - avoids circular imports
        auth_svc=get_auth_service(),
        detection_svc=get_detection_service(),
        custom_svc=get_custom_service(),
    ) -> Dict[str, Any]:
        """Example endpoint showing dependency usage."""
        result = {
            "success": True,
            "data": data,
            "auth_available": auth_svc is not None,
            "detection_available": detection_svc is not None,
            "custom_available": custom_svc is not None,
        }
        
        # Use services if available
        if auth_svc:
            result["authenticated"] = auth_svc.authenticate("valid_token")
        
        if detection_svc:
            result["anomalies"] = detection_svc.detect(data)
        
        if custom_svc:
            result["custom_result"] = custom_svc.process(data)
        
        return result
    
    # Add router to app
    app.include_router(router, prefix="/api/v1")
    
    print("   FastAPI app created successfully")
    print("   Available endpoints:")
    print("     POST /api/v1/detect")
    
    # In a real application, you would start the server here
    # uvicorn.run(app, host="0.0.0.0", port=8000)


async def async_example():
    """Example showing async usage of the dependency system."""
    print("\nAsync Example:")
    print("-" * 20)
    
    # Setup dependencies
    setup_test_dependencies()
    
    # Simulate async service usage
    from pynomaly.infrastructure.dependencies import get_dependency
    
    try:
        detection_service = get_dependency("detection_service")
        if detection_service:
            # In a real async service, this might be an async call
            result = detection_service.detect("async_data")
            print(f"   Async detection result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Cleanup
    clear_dependencies()


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Run async example
    asyncio.run(async_example())
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)
