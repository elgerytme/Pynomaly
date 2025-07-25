"""
Solution Catalog Service for the MLOps Marketplace.

Provides comprehensive solution discovery, search, and recommendation
capabilities for the marketplace.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID

from mlops_marketplace.domain.entities import (
    Solution,
    SolutionVersion,
    SolutionCategory,
    MarketplaceUser,
)
from mlops_marketplace.domain.value_objects import (
    SolutionId,
    UserId,
    Rating,
    Price,
)
from mlops_marketplace.domain.repositories import (
    SolutionRepository,
    UserRepository,
    ReviewRepository,
    AnalyticsRepository,
)
from mlops_marketplace.domain.interfaces import (
    SearchEngine,
    RecommendationEngine,
    CacheService,
)
from mlops_marketplace.application.dto import (
    SearchRequestDTO,
    SearchResultDTO,
    SolutionDTO,
    RecommendationRequestDTO,
    RecommendationResultDTO,
)


class SolutionCatalogService:
    """Service for managing solution catalog operations."""
    
    def __init__(
        self,
        solution_repository: SolutionRepository,
        user_repository: UserRepository,
        review_repository: ReviewRepository,
        analytics_repository: AnalyticsRepository,
        search_engine: SearchEngine,
        recommendation_engine: RecommendationEngine,
        cache_service: CacheService,
    ):
        """Initialize the solution catalog service."""
        self.solution_repository = solution_repository
        self.user_repository = user_repository
        self.review_repository = review_repository
        self.analytics_repository = analytics_repository
        self.search_engine = search_engine
        self.recommendation_engine = recommendation_engine
        self.cache_service = cache_service
    
    async def search_solutions(
        self, 
        request: SearchRequestDTO,
        user_id: Optional[UserId] = None
    ) -> SearchResultDTO:
        """Search for solutions with advanced filtering and ranking."""
        # Check cache first
        cache_key = f"search:{hash(str(request))}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return SearchResultDTO.parse_obj(cached_result)
        
        # Perform search
        solutions = await self.search_engine.search(
            query=request.query,
            filters={
                "categories": request.categories,
                "solution_types": request.solution_types,
                "price_range": request.price_range,
                "rating_min": request.min_rating,
                "tags": request.tags,
                "providers": request.providers,
                "license_types": request.license_types,
            },
            sort_by=request.sort_by,
            sort_order=request.sort_order,
            limit=request.limit,
            offset=request.offset,
        )
        
        # Enhance results with additional data
        enhanced_solutions = []
        for solution in solutions:
            enhanced_solution = await self._enhance_solution_data(solution, user_id)
            enhanced_solutions.append(enhanced_solution)
        
        # Get facets for filtering
        facets = await self._get_search_facets(request.query, request.categories)
        
        # Create result
        result = SearchResultDTO(
            solutions=enhanced_solutions,
            total_count=len(solutions),  # This should come from search engine
            facets=facets,
            search_time_ms=0,  # Should be measured
            suggestions=await self._get_search_suggestions(request.query),
        )
        
        # Cache result
        await self.cache_service.set(
            cache_key, 
            result.dict(), 
            ttl_seconds=300  # 5 minutes
        )
        
        # Track search analytics
        if user_id:
            await self.analytics_repository.track_search(
                user_id=user_id,
                query=request.query,
                filters=request.dict(),
                result_count=result.total_count,
            )
        
        return result
    
    async def get_solution_details(
        self, 
        solution_id: SolutionId,
        user_id: Optional[UserId] = None
    ) -> Optional[SolutionDTO]:
        """Get detailed information about a specific solution."""
        solution = await self.solution_repository.get_by_id(solution_id)
        if not solution:
            return None
        
        enhanced_solution = await self._enhance_solution_data(solution, user_id)
        
        # Track view analytics
        if user_id:
            await self.analytics_repository.track_solution_view(
                user_id=user_id,
                solution_id=solution_id,
            )
        
        return enhanced_solution
    
    async def get_recommendations(
        self, 
        request: RecommendationRequestDTO,
        user_id: Optional[UserId] = None
    ) -> RecommendationResultDTO:
        """Get personalized solution recommendations."""
        # Get user preferences if available
        user_preferences = None
        if user_id:
            user = await self.user_repository.get_by_id(user_id)
            user_preferences = user.preferences if user else None
        
        # Get recommendations based on different strategies
        recommendations = await asyncio.gather(
            self._get_collaborative_recommendations(user_id, request.limit // 3),
            self._get_content_based_recommendations(request, user_preferences, request.limit // 3),
            self._get_trending_recommendations(request.categories, request.limit // 3),
        )
        
        # Combine and deduplicate recommendations
        all_recommendations = []
        seen_ids = set()
        
        for rec_list in recommendations:
            for rec in rec_list:
                if rec.solution_id not in seen_ids:
                    all_recommendations.append(rec)
                    seen_ids.add(rec.solution_id)
        
        # Sort by confidence score and limit results
        all_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        final_recommendations = all_recommendations[:request.limit]
        
        return RecommendationResultDTO(
            recommendations=final_recommendations,
            recommendation_types=["collaborative", "content_based", "trending"],
            generated_at=datetime.utcnow(),
        )
    
    async def get_featured_solutions(
        self, 
        category_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[SolutionDTO]:
        """Get featured solutions for the marketplace homepage."""
        cache_key = f"featured_solutions:{category_id}:{limit}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return [SolutionDTO.parse_obj(sol) for sol in cached_result]
        
        solutions = await self.solution_repository.get_featured_solutions(
            category_id=category_id,
            limit=limit
        )
        
        enhanced_solutions = []
        for solution in solutions:
            enhanced_solution = await self._enhance_solution_data(solution)
            enhanced_solutions.append(enhanced_solution)
        
        # Cache for 1 hour
        await self.cache_service.set(
            cache_key,
            [sol.dict() for sol in enhanced_solutions],
            ttl_seconds=3600
        )
        
        return enhanced_solutions
    
    async def get_trending_solutions(
        self, 
        time_period: str = "week",
        limit: int = 10
    ) -> List[SolutionDTO]:
        """Get trending solutions based on recent activity."""
        # Calculate time range
        now = datetime.utcnow()
        if time_period == "day":
            start_time = now - timedelta(days=1)
        elif time_period == "week":
            start_time = now - timedelta(weeks=1)
        elif time_period == "month":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(weeks=1)
        
        # Get trending solution IDs from analytics
        trending_ids = await self.analytics_repository.get_trending_solutions(
            start_time=start_time,
            end_time=now,
            limit=limit
        )
        
        # Get solution details
        solutions = []
        for solution_id in trending_ids:
            solution = await self.solution_repository.get_by_id(solution_id)
            if solution:
                enhanced_solution = await self._enhance_solution_data(solution)
                solutions.append(enhanced_solution)
        
        return solutions
    
    async def get_categories(self, parent_id: Optional[UUID] = None) -> List[SolutionCategory]:
        """Get solution categories with hierarchy support."""
        cache_key = f"categories:{parent_id}"
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return [SolutionCategory.parse_obj(cat) for cat in cached_result]
        
        categories = await self.solution_repository.get_categories(parent_id=parent_id)
        
        # Cache for 1 hour
        await self.cache_service.set(
            cache_key,
            [cat.dict() for cat in categories],
            ttl_seconds=3600
        )
        
        return categories
    
    async def get_similar_solutions(
        self, 
        solution_id: SolutionId,
        limit: int = 5
    ) -> List[SolutionDTO]:
        """Get solutions similar to the specified solution."""
        solution = await self.solution_repository.get_by_id(solution_id)
        if not solution:
            return []
        
        # Use recommendation engine to find similar solutions
        similar_solution_ids = await self.recommendation_engine.get_similar_solutions(
            solution_id=solution_id,
            limit=limit
        )
        
        similar_solutions = []
        for similar_id in similar_solution_ids:
            similar_solution = await self.solution_repository.get_by_id(similar_id)
            if similar_solution:
                enhanced_solution = await self._enhance_solution_data(similar_solution)
                similar_solutions.append(enhanced_solution)
        
        return similar_solutions
    
    async def _enhance_solution_data(
        self, 
        solution: Solution,
        user_id: Optional[UserId] = None
    ) -> SolutionDTO:
        """Enhance solution data with additional information."""
        # Get latest version
        latest_version = solution.get_latest_version()
        
        # Get recent reviews
        recent_reviews = await self.review_repository.get_recent_reviews(
            solution_id=solution.id,
            limit=3
        )
        
        # Get provider information
        provider = await self.solution_repository.get_provider(solution.provider_id)
        
        # Check if user has this solution
        user_has_solution = False
        if user_id:
            user_has_solution = await self.solution_repository.user_has_solution(
                user_id=user_id,
                solution_id=solution.id
            )
        
        return SolutionDTO(
            id=solution.id,
            name=solution.name,
            slug=solution.slug,
            solution_type=solution.solution_type,
            short_description=solution.short_description,
            long_description=solution.long_description,
            logo_url=solution.logo_url,
            banner_url=solution.banner_url,
            screenshots=solution.screenshots,
            metadata=solution.metadata,
            latest_version=latest_version,
            provider=provider,
            category=await self.solution_repository.get_category(solution.category_id),
            average_rating=solution.average_rating,
            total_reviews=solution.total_reviews,
            total_downloads=solution.total_downloads,
            total_deployments=solution.total_deployments,
            recent_reviews=recent_reviews,
            is_featured=solution.is_featured,
            is_verified=solution.is_verified,
            user_has_solution=user_has_solution,
            created_at=solution.created_at,
            updated_at=solution.updated_at,
            published_at=solution.published_at,
        )
    
    async def _get_search_facets(
        self, 
        query: str,
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, any]]]:
        """Get search facets for filtering."""
        return await self.search_engine.get_facets(
            query=query,
            category_filter=categories
        )
    
    async def _get_search_suggestions(self, query: str) -> List[str]:
        """Get search query suggestions."""
        return await self.search_engine.get_suggestions(query)
    
    async def _get_collaborative_recommendations(
        self, 
        user_id: Optional[UserId],
        limit: int
    ) -> List[any]:
        """Get collaborative filtering recommendations."""
        if not user_id:
            return []
        
        return await self.recommendation_engine.get_collaborative_recommendations(
            user_id=user_id,
            limit=limit
        )
    
    async def _get_content_based_recommendations(
        self, 
        request: RecommendationRequestDTO,
        user_preferences: Optional[Dict],
        limit: int
    ) -> List[any]:
        """Get content-based recommendations."""
        return await self.recommendation_engine.get_content_based_recommendations(
            categories=request.categories,
            tags=request.tags,
            solution_types=request.solution_types,
            user_preferences=user_preferences,
            limit=limit
        )
    
    async def _get_trending_recommendations(
        self, 
        categories: Optional[List[str]],
        limit: int
    ) -> List[any]:
        """Get trending solution recommendations."""
        return await self.recommendation_engine.get_trending_recommendations(
            categories=categories,
            limit=limit
        )