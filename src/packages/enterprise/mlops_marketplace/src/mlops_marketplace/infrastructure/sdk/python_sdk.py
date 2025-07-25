"""
Python SDK for the MLOps Marketplace.

Provides a comprehensive Python client for interacting with the
marketplace platform, including solution management, deployment,
and monetization features.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import httpx
from pydantic import BaseModel, Field

from mlops_marketplace.infrastructure.sdk.base_sdk import BaseSDK, SDKConfig
from mlops_marketplace.infrastructure.sdk.exceptions import (
    SDKError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    NetworkError,
)
from mlops_marketplace.domain.value_objects import (
    SolutionId,
    ProviderId,
    UserId,
    Price,
    Version,
)


class SolutionSearchRequest(BaseModel):
    """Request model for solution search."""
    query: Optional[str] = None
    categories: Optional[List[str]] = None
    solution_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    providers: Optional[List[str]] = None
    min_rating: Optional[float] = None
    price_range: Optional[Dict[str, float]] = None
    license_types: Optional[List[str]] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    limit: int = 20
    offset: int = 0


class DeploymentRequest(BaseModel):
    """Request model for solution deployment."""
    solution_id: Union[str, SolutionId]
    version_id: Optional[str] = None
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    environment: str = "production"
    scaling_config: Optional[Dict[str, Any]] = None
    resource_limits: Optional[Dict[str, Any]] = None
    environment_variables: Dict[str, str] = Field(default_factory=dict)


class SubscriptionRequest(BaseModel):
    """Request model for creating subscriptions."""
    solution_id: Union[str, SolutionId]
    plan_id: str
    billing_cycle: str = "monthly"
    auto_renew: bool = True
    payment_method_id: Optional[str] = None


class ReviewRequest(BaseModel):
    """Request model for submitting reviews."""
    solution_id: Union[str, SolutionId]
    rating: float = Field(..., ge=1.0, le=5.0)
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10, max_length=2000)
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
    use_case: Optional[str] = None


class PythonSDK(BaseSDK):
    """
    Python SDK for the MLOps Marketplace.
    
    Provides high-level methods for:
    - Solution discovery and search
    - Solution deployment and management
    - User authentication and profile management
    - Subscription and billing operations
    - Review and rating system
    - Analytics and monitoring
    """
    
    def __init__(
        self,
        config: SDKConfig,
        session: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize the Python SDK."""
        super().__init__(config)
        self._session = session or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        self._auth_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close the SDK and cleanup resources."""
        if self._session:
            await self._session.aclose()
    
    # Authentication Methods
    
    async def authenticate(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        oauth_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate with the marketplace.
        
        Args:
            email: User email (for email/password auth)
            password: User password (for email/password auth)
            api_key: API key (for API key auth)
            oauth_token: OAuth token (for OAuth auth)
            
        Returns:
            Authentication response with user info and tokens
        """
        if api_key:
            # API key authentication
            self.config.api_key = api_key
            return await self._verify_api_key()
        
        elif email and password:
            # Email/password authentication
            return await self._authenticate_with_credentials(email, password)
        
        elif oauth_token:
            # OAuth authentication
            return await self._authenticate_with_oauth(oauth_token)
        
        else:
            raise AuthenticationError("No valid authentication method provided")
    
    async def refresh_token(self) -> Dict[str, Any]:
        """Refresh the authentication token."""
        if not self._auth_token:
            raise AuthenticationError("No authentication token to refresh")
        
        response = await self._make_request(
            "POST",
            "/auth/refresh",
            headers={"Authorization": f"Bearer {self._auth_token}"},
        )
        
        self._auth_token = response["access_token"]
        self._token_expires_at = datetime.fromisoformat(response["expires_at"])
        
        return response
    
    # Solution Discovery Methods
    
    async def search_solutions(
        self,
        request: Union[SolutionSearchRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Search for solutions in the marketplace.
        
        Args:
            request: Search request parameters
            
        Returns:
            Search results with solutions and metadata
        """
        if isinstance(request, dict):
            request = SolutionSearchRequest(**request)
        
        return await self._make_request(
            "GET",
            "/solutions/search",
            params=request.dict(exclude_none=True),
        )
    
    async def get_solution(
        self,
        solution_id: Union[str, SolutionId],
        include_versions: bool = True,
        include_reviews: bool = True,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific solution.
        
        Args:
            solution_id: ID of the solution
            include_versions: Whether to include version information
            include_reviews: Whether to include recent reviews
            
        Returns:
            Detailed solution information
        """
        params = {
            "include_versions": include_versions,
            "include_reviews": include_reviews,
        }
        
        return await self._make_request(
            "GET",
            f"/solutions/{solution_id}",
            params=params,
        )
    
    async def get_featured_solutions(
        self,
        category_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get featured solutions.
        
        Args:
            category_id: Optional category filter
            limit: Maximum number of solutions to return
            
        Returns:
            List of featured solutions
        """
        params = {"limit": limit}
        if category_id:
            params["category_id"] = category_id
        
        response = await self._make_request(
            "GET",
            "/solutions/featured",
            params=params,
        )
        
        return response["solutions"]
    
    async def get_trending_solutions(
        self,
        time_period: str = "week",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get trending solutions.
        
        Args:
            time_period: Time period for trending calculation (day/week/month)
            limit: Maximum number of solutions to return
            
        Returns:
            List of trending solutions
        """
        params = {
            "time_period": time_period,
            "limit": limit,
        }
        
        response = await self._make_request(
            "GET",
            "/solutions/trending",
            params=params,
        )
        
        return response["solutions"]
    
    async def get_recommendations(
        self,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        solution_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get personalized solution recommendations.
        
        Args:
            categories: Preferred categories
            tags: Preferred tags
            solution_types: Preferred solution types
            limit: Maximum number of recommendations
            
        Returns:
            Personalized recommendations
        """
        params = {"limit": limit}
        if categories:
            params["categories"] = categories
        if tags:
            params["tags"] = tags
        if solution_types:
            params["solution_types"] = solution_types
        
        return await self._make_request(
            "GET",
            "/recommendations",
            params=params,
        )
    
    # Deployment Methods
    
    async def deploy_solution(
        self,
        request: Union[DeploymentRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Deploy a solution to your environment.
        
        Args:
            request: Deployment request parameters
            
        Returns:
            Deployment information and status
        """
        if isinstance(request, dict):
            request = DeploymentRequest(**request)
        
        return await self._make_request(
            "POST",
            "/deployments",
            json=request.dict(),
        )
    
    async def get_deployment(
        self,
        deployment_id: str,
    ) -> Dict[str, Any]:
        """
        Get deployment status and information.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Deployment status and configuration
        """
        return await self._make_request(
            "GET",
            f"/deployments/{deployment_id}",
        )
    
    async def list_deployments(
        self,
        solution_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List user deployments.
        
        Args:
            solution_id: Filter by solution ID
            status: Filter by deployment status
            limit: Maximum number of deployments to return
            offset: Offset for pagination
            
        Returns:
            List of deployments with metadata
        """
        params = {"limit": limit, "offset": offset}
        if solution_id:
            params["solution_id"] = solution_id
        if status:
            params["status"] = status
        
        return await self._make_request(
            "GET",
            "/deployments",
            params=params,
        )
    
    async def stop_deployment(
        self,
        deployment_id: str,
    ) -> Dict[str, Any]:
        """
        Stop a running deployment.
        
        Args:
            deployment_id: ID of the deployment to stop
            
        Returns:
            Updated deployment status
        """
        return await self._make_request(
            "POST",
            f"/deployments/{deployment_id}/stop",
        )
    
    async def scale_deployment(
        self,
        deployment_id: str,
        replicas: int,
    ) -> Dict[str, Any]:
        """
        Scale a deployment.
        
        Args:
            deployment_id: ID of the deployment to scale
            replicas: Number of replicas
            
        Returns:
            Updated deployment configuration
        """
        return await self._make_request(
            "POST",
            f"/deployments/{deployment_id}/scale",
            json={"replicas": replicas},
        )
    
    # Subscription Methods
    
    async def create_subscription(
        self,
        request: Union[SubscriptionRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create a subscription to a solution.
        
        Args:
            request: Subscription request parameters
            
        Returns:
            Subscription information and payment details
        """
        if isinstance(request, dict):
            request = SubscriptionRequest(**request)
        
        return await self._make_request(
            "POST",
            "/subscriptions",
            json=request.dict(),
        )
    
    async def get_subscription(
        self,
        subscription_id: str,
    ) -> Dict[str, Any]:
        """
        Get subscription details.
        
        Args:
            subscription_id: ID of the subscription
            
        Returns:
            Subscription information and status
        """
        return await self._make_request(
            "GET",
            f"/subscriptions/{subscription_id}",
        )
    
    async def list_subscriptions(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List user subscriptions.
        
        Args:
            status: Filter by subscription status
            limit: Maximum number of subscriptions to return
            offset: Offset for pagination
            
        Returns:
            List of subscriptions
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        return await self._make_request(
            "GET",
            "/subscriptions",
            params=params,
        )
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        immediate: bool = False,
    ) -> Dict[str, Any]:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: ID of the subscription to cancel
            immediate: Whether to cancel immediately or at period end
            
        Returns:
            Updated subscription status
        """
        return await self._make_request(
            "POST",
            f"/subscriptions/{subscription_id}/cancel",
            json={"immediate": immediate},
        )
    
    # Review Methods
    
    async def submit_review(
        self,
        request: Union[ReviewRequest, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Submit a review for a solution.
        
        Args:
            request: Review request parameters
            
        Returns:
            Created review information
        """
        if isinstance(request, dict):
            request = ReviewRequest(**request)
        
        return await self._make_request(
            "POST",
            "/reviews",
            json=request.dict(),
        )
    
    async def get_solution_reviews(
        self,
        solution_id: Union[str, SolutionId],
        rating_filter: Optional[int] = None,
        sort_by: str = "newest",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get reviews for a solution.
        
        Args:
            solution_id: ID of the solution
            rating_filter: Filter by rating (1-5)
            sort_by: Sort order (newest, oldest, rating_high, rating_low)
            limit: Maximum number of reviews to return
            offset: Offset for pagination
            
        Returns:
            List of reviews with metadata
        """
        params = {
            "sort_by": sort_by,
            "limit": limit,
            "offset": offset,
        }
        if rating_filter:
            params["rating"] = rating_filter
        
        return await self._make_request(
            "GET",
            f"/solutions/{solution_id}/reviews",
            params=params,
        )
    
    # Analytics Methods
    
    async def get_usage_analytics(
        self,
        solution_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get usage analytics.
        
        Args:
            solution_id: Filter by solution ID
            deployment_id: Filter by deployment ID
            start_date: Start date for analytics (ISO format)
            end_date: End date for analytics (ISO format)
            
        Returns:
            Usage analytics data
        """
        params = {}
        if solution_id:
            params["solution_id"] = solution_id
        if deployment_id:
            params["deployment_id"] = deployment_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return await self._make_request(
            "GET",
            "/analytics/usage",
            params=params,
        )
    
    async def get_billing_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get billing and cost analytics.
        
        Args:
            start_date: Start date for analytics (ISO format)
            end_date: End date for analytics (ISO format)
            
        Returns:
            Billing analytics data
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return await self._make_request(
            "GET",
            "/analytics/billing",
            params=params,
        )
    
    # Utility Methods
    
    async def get_categories(
        self,
        parent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get solution categories.
        
        Args:
            parent_id: Parent category ID for hierarchical categories
            
        Returns:
            List of categories
        """
        params = {}
        if parent_id:
            params["parent_id"] = parent_id
        
        response = await self._make_request(
            "GET",
            "/categories",
            params=params,
        )
        
        return response["categories"]
    
    async def get_user_profile(self) -> Dict[str, Any]:
        """
        Get current user profile.
        
        Returns:
            User profile information
        """
        return await self._make_request("GET", "/profile")
    
    async def update_user_profile(
        self,
        profile_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update user profile.
        
        Args:
            profile_data: Profile data to update
            
        Returns:
            Updated profile information
        """
        return await self._make_request(
            "PUT",
            "/profile",
            json=profile_data,
        )
    
    # Private Methods
    
    async def _authenticate_with_credentials(
        self,
        email: str,
        password: str,
    ) -> Dict[str, Any]:
        """Authenticate with email and password."""
        response = await self._make_request(
            "POST",
            "/auth/login",
            json={"email": email, "password": password},
        )
        
        self._auth_token = response["access_token"]
        self._token_expires_at = datetime.fromisoformat(response["expires_at"])
        
        return response
    
    async def _authenticate_with_oauth(
        self,
        oauth_token: str,
    ) -> Dict[str, Any]:
        """Authenticate with OAuth token."""
        response = await self._make_request(
            "POST",
            "/auth/oauth",
            json={"token": oauth_token},
        )
        
        self._auth_token = response["access_token"]
        self._token_expires_at = datetime.fromisoformat(response["expires_at"])
        
        return response
    
    async def _verify_api_key(self) -> Dict[str, Any]:
        """Verify API key."""
        return await self._make_request("GET", "/auth/verify")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make HTTP request to the marketplace API."""
        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        
        # Prepare headers
        request_headers = {
            "User-Agent": f"MLOps-Marketplace-Python-SDK/{self.config.version}",
            "Accept": "application/json",
        }
        
        # Add authentication
        if self.config.api_key:
            request_headers["X-API-Key"] = self.config.api_key
        elif self._auth_token:
            request_headers["Authorization"] = f"Bearer {self._auth_token}"
        
        # Add custom headers
        if headers:
            request_headers.update(headers)
        
        try:
            response = await self._session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=request_headers,
            )
            
            # Handle response
            if response.status_code == 401:
                raise AuthenticationError("Authentication failed")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", "API error")
                except:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                
                if response.status_code == 422:
                    raise ValidationError(error_message)
                else:
                    raise SDKError(error_message)
            
            # Return JSON response
            return response.json()
        
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            raise SDKError("Invalid JSON response from API")


# Convenience alias
MarketplaceSDK = PythonSDK