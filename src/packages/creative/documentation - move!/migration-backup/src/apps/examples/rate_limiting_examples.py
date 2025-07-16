"""Rate limiting examples and usage patterns for Pynomaly."""

import asyncio
from typing import Any

from monorepo.infrastructure.security import (
    RateLimitAlgorithm,
    RateLimitScope,
    api_rate_limited,
    check_rate_limit_status,
    endpoint_rate_limited,
    rate_limit_context,
    rate_limited,
    user_rate_limited,
)


# Example 1: Basic function rate limiting
@rate_limited(requests_per_second=5.0, burst_capacity=50)
async def process_data(data: dict[str, Any]) -> dict[str, Any]:
    """Example function with basic rate limiting."""
    # Simulate processing
    await asyncio.sleep(0.1)
    return {"processed": True, "data": data}


# Example 2: User-specific rate limiting
@user_rate_limited(requests_per_second=2.0, burst_capacity=20, user_id_key="user_id")
async def process_user_request(
    user_id: str, request_data: dict[str, Any]
) -> dict[str, Any]:
    """Example function with user-specific rate limiting."""
    # Simulate user-specific processing
    await asyncio.sleep(0.2)
    return {"user_id": user_id, "processed": request_data}


# Example 3: API key rate limiting
@api_rate_limited(requests_per_second=100.0, burst_capacity=1000)
async def api_endpoint(api_key: str, query: str) -> dict[str, Any]:
    """Example API endpoint with API key rate limiting."""
    # Simulate API processing
    await asyncio.sleep(0.05)
    return {"api_key": api_key[:8] + "...", "result": f"Processed: {query}"}


# Example 4: Endpoint-specific rate limiting
@endpoint_rate_limited(
    requests_per_second=50.0, burst_capacity=500, endpoint_name="heavy_computation"
)
async def heavy_computation_endpoint(
    computation_params: dict[str, Any],
) -> dict[str, Any]:
    """Example endpoint with specific rate limiting."""
    # Simulate heavy computation
    await asyncio.sleep(1.0)
    return {"computation_result": "completed", "params": computation_params}


# Example 5: Custom rate limiting with exponential backoff
@rate_limited(
    requests_per_second=10.0,
    burst_capacity=100,
    algorithm=RateLimitAlgorithm.EXPONENTIAL_BACKOFF,
    scope=RateLimitScope.GLOBAL,
)
async def unreliable_service_call(service_data: dict[str, Any]) -> dict[str, Any]:
    """Example function using exponential backoff rate limiting."""
    # Simulate unreliable service call
    import random

    if random.random() < 0.3:  # 30% failure rate
        raise Exception("Service temporarily unavailable")

    await asyncio.sleep(0.1)
    return {"service_response": service_data}


# Example 6: Using rate limiting context manager
async def manual_rate_limiting_example(
    user_id: str, operation_data: dict[str, Any]
) -> dict[str, Any]:
    """Example using rate limiting context manager."""
    async with rate_limit_context(
        identifier=user_id,
        requests_per_second=3.0,
        burst_capacity=30,
        scope=RateLimitScope.USER,
        operation="manual_operation",
    ) as status:
        # Protected operation
        await asyncio.sleep(0.5)
        return {
            "user_id": user_id,
            "operation_result": operation_data,
            "rate_limit_remaining": status.remaining,
        }


# Example 7: Checking rate limit status
async def check_rate_limit_example(user_id: str) -> dict[str, Any]:
    """Example of checking rate limit status without consuming tokens."""
    status = await check_rate_limit_status(
        identifier=user_id,
        requests_per_second=5.0,
        burst_capacity=50,
        scope=RateLimitScope.USER,
        operation="status_check",
    )

    return {
        "user_id": user_id,
        "allowed": status.allowed,
        "remaining": status.remaining,
        "reset_time": status.reset_time,
        "retry_after": status.retry_after,
    }


# Example usage and testing
async def main():
    """Main example function demonstrating rate limiting."""
    print("=== Rate Limiting Examples ===\n")

    # Example 1: Basic rate limiting
    print("1. Basic rate limiting:")
    try:
        for i in range(3):
            result = await process_data({"item": i})
            print(f"   Request {i + 1}: {result}")
    except Exception as e:
        print(f"   Rate limit exceeded: {e}")

    print()

    # Example 2: User-specific rate limiting
    print("2. User-specific rate limiting:")
    try:
        for i in range(3):
            result = await process_user_request("user_123", {"request": i})
            print(f"   User request {i + 1}: {result}")
    except Exception as e:
        print(f"   User rate limit exceeded: {e}")

    print()

    # Example 3: API rate limiting
    print("3. API rate limiting:")
    try:
        for i in range(2):
            result = await api_endpoint("api_key_abc123", f"query_{i}")
            print(f"   API request {i + 1}: {result}")
    except Exception as e:
        print(f"   API rate limit exceeded: {e}")

    print()

    # Example 4: Rate limit status check
    print("4. Rate limit status check:")
    status = await check_rate_limit_example("user_456")
    print(f"   Status: {status}")

    print()

    # Example 5: Context manager rate limiting
    print("5. Context manager rate limiting:")
    try:
        result = await manual_rate_limiting_example("user_789", {"data": "test"})
        print(f"   Manual operation: {result}")
    except Exception as e:
        print(f"   Manual rate limit exceeded: {e}")

    print()

    # Example 6: Exponential backoff (simulate failures)
    print("6. Exponential backoff example:")
    for i in range(3):
        try:
            result = await unreliable_service_call({"attempt": i})
            print(f"   Service call {i + 1} succeeded: {result}")
        except Exception as e:
            print(f"   Service call {i + 1} failed: {e}")

    print("\n=== Rate Limiting Examples Complete ===")


# Integration with web frameworks example
def create_fastapi_rate_limiting_example():
    """Example of integrating rate limiting with FastAPI."""
    try:
        from fastapi import FastAPI, HTTPException, Request

        from monorepo.infrastructure.security import create_rate_limit_middleware

        app = FastAPI()

        # Create rate limiting middleware
        def extract_client_ip(request: Request) -> str:
            """Extract client IP from request."""
            return request.client.host if request.client else "unknown"

        rate_limit_middleware = create_rate_limit_middleware(
            default_requests_per_second=10.0,
            default_burst_capacity=100,
            identifier_extractor=extract_client_ip,
            scope=RateLimitScope.IP,
        )

        # Add middleware to app
        app.add_middleware(rate_limit_middleware)

        @app.get("/")
        @rate_limited(requests_per_second=5.0)
        async def root():
            return {"message": "Hello World with rate limiting!"}

        @app.post("/process")
        @user_rate_limited(requests_per_second=2.0, user_id_key="user_id")
        async def process_endpoint(user_id: str, data: dict):
            return await process_user_request(user_id, data)

        return app

    except ImportError:
        print("FastAPI not available - skipping web framework example")
        return None


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())

    # Create FastAPI app example
    fastapi_app = create_fastapi_rate_limiting_example()
    if fastapi_app:
        print("\nFastAPI app with rate limiting created successfully!")
