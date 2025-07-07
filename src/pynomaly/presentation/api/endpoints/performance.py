"""Performance monitoring and optimization API endpoints."""


from typing import Any

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from pynomaly.infrastructure.config.container import Container
from pynomaly.infrastructure.performance import (
    QueryOptimizer,
)

router = APIRouter(prefix="/performance", tags=["performance"])


class PoolStatsResponse(BaseModel):
    """Pool statistics response model."""

    pool_type: str
    total_connections: int
    active_connections: int
    idle_connections: int
    overflow_connections: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    connections_created: int
    connections_closed: int
    connections_recycled: int
    connection_errors: int
    created_at: float
    last_reset: float


class PoolInfoResponse(BaseModel):
    """Pool information response model."""

    name: str
    type: str
    stats: PoolStatsResponse
    pool_info: dict[str, Any]
    configuration: dict[str, Any] | None = None


class QueryMetricsResponse(BaseModel):
    """Query metrics response model."""

    query_hash: str
    query_type: str
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: float
    cache_hits: int = 0
    cache_misses: int = 0


class PerformanceSummaryResponse(BaseModel):
    """Performance summary response model."""

    total_queries: int
    unique_queries: int
    total_time: float
    avg_time: float
    slow_queries: int
    query_types: dict[str, int]
    slowest_query: float


class IndexRecommendationResponse(BaseModel):
    """Index recommendation response model."""

    table: str
    columns: list[str]
    reason: str
    estimated_benefit: float


class OptimizationReportResponse(BaseModel):
    """Optimization report response model."""

    performance_summary: PerformanceSummaryResponse
    slow_queries: dict[str, Any]
    index_recommendations: list[IndexRecommendationResponse]
    cache_stats: dict[str, Any]
    index_usage: dict[str, Any]
    most_frequent_queries: list[QueryMetricsResponse]


class OptimizationResultResponse(BaseModel):
    """Database optimization result response model."""

    indexes_created: int
    recommendations: list[IndexRecommendationResponse]
    performance_summary: PerformanceSummaryResponse
    cache_stats: dict[str, Any]
    error: str | None = None


@router.get(
    "/pools",
    response_model=list[PoolInfoResponse],
    summary="Get connection pool information",
    description="Retrieve information about all connection pools including statistics and configuration",
)
async def get_pools(
    # pool_manager: ConnectionPoolManager = Depends(Provide[Container.connection_pool_manager])
) -> list[PoolInfoResponse]:
    """Get information about all connection pools."""
    try:
        # TODO: Implement connection pool manager
        # if pool_manager is None:
        #     raise HTTPException(
        #         status_code=503,
        #         detail="Connection pool manager not available"
        #     )
        return []  # Return empty list for now

        # TODO: Restore once connection pool manager is implemented
        # pool_names = pool_manager.list_pools()
        # pools_info = []
        #
        # for name in pool_names:
        #     try:
        #         pool_info = pool_manager.get_pool_info(name)
        #         stats = pool_info["stats"]
        #
        #         pools_info.append(PoolInfoResponse(
        #             name=pool_info["name"],
        #             type=pool_info["type"],
        #             stats=PoolStatsResponse(
        #                 pool_type=stats.pool_type.value,
        #                 total_connections=stats.total_connections,
        #                 active_connections=stats.active_connections,
        #                 idle_connections=stats.idle_connections,
        #                 overflow_connections=stats.overflow_connections,
        #                 total_requests=stats.total_requests,
        #                 successful_requests=stats.successful_requests,
        #                 failed_requests=stats.failed_requests,
        #                 avg_response_time=stats.avg_response_time,
        #                 connections_created=stats.connections_created,
        #                 connections_closed=stats.connections_closed,
        #                 connections_recycled=stats.connections_recycled,
        #                 connection_errors=stats.connection_errors,
        #                 created_at=stats.created_at,
        #                 last_reset=stats.last_reset
        #             ),
        #             pool_info=pool_info["pool_info"],
        #             configuration=pool_info.get("configuration")
        #         ))
        #     except Exception as e:
        #         # Skip pools that can't be accessed
        #         continue
        #
        # return pools_info

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get pool information: {str(e)}"
        )


@router.get(
    "/pools/{pool_name}",
    response_model=PoolInfoResponse,
    summary="Get specific pool information",
    description="Retrieve information about a specific connection pool",
)
async def get_pool(
    pool_name: str,
    # pool_manager: ConnectionPoolManager = Depends(Provide[Container.connection_pool_manager])
) -> PoolInfoResponse:
    """Get information about a specific connection pool."""
    try:
        # TODO: Implement connection pool manager
        raise HTTPException(status_code=404, detail=f"Pool {pool_name} not found")

        # if pool_manager is None:
        #     raise HTTPException(
        #         status_code=503,
        #         detail="Connection pool manager not available"
        #     )

        # pool_info = pool_manager.get_pool_info(pool_name)
        # stats = pool_info["stats"]
        #
        # return PoolInfoResponse(
        #     name=pool_info["name"],
        #     type=pool_info["type"],
        #     stats=PoolStatsResponse(
        #         pool_type=stats.pool_type.value,
        #         total_connections=stats.total_connections,
        #         active_connections=stats.active_connections,
        #         idle_connections=stats.idle_connections,
        #         overflow_connections=stats.overflow_connections,
        #         total_requests=stats.total_requests,
        #         successful_requests=stats.successful_requests,
        #         failed_requests=stats.failed_requests,
        #         avg_response_time=stats.avg_response_time,
        #         connections_created=stats.connections_created,
        #         connections_closed=stats.connections_closed,
        #         connections_recycled=stats.connections_recycled,
        #         connection_errors=stats.connection_errors,
        #         created_at=stats.created_at,
        #         last_reset=stats.last_reset
        #     ),
        #     pool_info=pool_info["pool_info"],
        #     configuration=pool_info.get("configuration")
        # )

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Pool '{pool_name}' not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get pool information: {str(e)}"
        )


@router.post(
    "/pools/{pool_name}/reset",
    summary="Reset pool statistics",
    description="Reset statistics for a specific connection pool",
)
async def reset_pool_stats(
    pool_name: str,
    # pool_manager: ConnectionPoolManager = Depends(Provide[Container.connection_pool_manager])
) -> dict[str, str]:
    """Reset statistics for a specific connection pool."""
    try:
        # TODO: Implement connection pool manager
        return {"message": f"Pool reset not implemented yet for '{pool_name}'"}

        # if pool_manager is None:
        #     raise HTTPException(
        #         status_code=503,
        #         detail="Connection pool manager not available"
        #     )
        #
        # pool_manager.reset_stats(pool_name)
        # return {"message": f"Statistics reset for pool '{pool_name}'"}

    except KeyError:
        raise HTTPException(status_code=404, detail=f"Pool '{pool_name}' not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset pool statistics: {str(e)}"
        )


@router.post(
    "/pools/reset-all",
    summary="Reset all pool statistics",
    description="Reset statistics for all connection pools",
)
async def reset_all_pool_stats(
    # pool_manager: ConnectionPoolManager = Depends(Provide[Container.connection_pool_manager])
) -> dict[str, str]:
    """Reset statistics for all connection pools."""
    try:
        # TODO: Implement connection pool manager
        return {"message": "Pool reset not implemented yet"}

        # if pool_manager is None:
        #     raise HTTPException(
        #         status_code=503,
        #         detail="Connection pool manager not available"
        #     )
        #
        # pool_manager.reset_stats()
        # return {"message": "Statistics reset for all pools"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reset pool statistics: {str(e)}"
        )


@router.get(
    "/queries/summary",
    response_model=PerformanceSummaryResponse,
    summary="Get query performance summary",
    description="Get overall query performance metrics and statistics",
)
@inject
async def get_query_performance_summary(
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> PerformanceSummaryResponse:
    """Get query performance summary."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        summary = optimizer.performance_tracker.get_performance_summary()

        return PerformanceSummaryResponse(
            total_queries=summary["total_queries"],
            unique_queries=summary["unique_queries"],
            total_time=summary["total_time"],
            avg_time=summary["avg_time"],
            slow_queries=summary["slow_queries"],
            query_types=summary["query_types"],
            slowest_query=summary["slowest_query"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get query performance summary: {str(e)}"
        )


@router.get(
    "/queries/slow",
    response_model=list[QueryMetricsResponse],
    summary="Get slow queries",
    description="Get queries that exceed the specified time threshold",
)
@inject
async def get_slow_queries(
    threshold: float = Query(1.0, description="Time threshold in seconds", ge=0.1),
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> list[QueryMetricsResponse]:
    """Get slow queries that exceed the threshold."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        slow_queries = optimizer.performance_tracker.get_slow_queries(threshold)

        return [
            QueryMetricsResponse(
                query_hash=metrics.query_hash,
                query_type=metrics.query_type.value,
                execution_count=metrics.execution_count,
                total_time=metrics.total_time,
                avg_time=metrics.avg_time,
                min_time=metrics.min_time,
                max_time=metrics.max_time,
                last_executed=metrics.last_executed,
                cache_hits=metrics.cache_hits,
                cache_misses=metrics.cache_misses,
            )
            for metrics in slow_queries
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get slow queries: {str(e)}"
        )


@router.get(
    "/queries/frequent",
    response_model=list[QueryMetricsResponse],
    summary="Get most frequent queries",
    description="Get the most frequently executed queries",
)
@inject
async def get_frequent_queries(
    limit: int = Query(10, description="Number of queries to return", ge=1, le=100),
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> list[QueryMetricsResponse]:
    """Get most frequently executed queries."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        frequent_queries = optimizer.performance_tracker.get_most_frequent_queries(
            limit
        )

        return [
            QueryMetricsResponse(
                query_hash=metrics.query_hash,
                query_type=metrics.query_type.value,
                execution_count=metrics.execution_count,
                total_time=metrics.total_time,
                avg_time=metrics.avg_time,
                min_time=metrics.min_time,
                max_time=metrics.max_time,
                last_executed=metrics.last_executed,
                cache_hits=metrics.cache_hits,
                cache_misses=metrics.cache_misses,
            )
            for metrics in frequent_queries
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get frequent queries: {str(e)}"
        )


@router.get(
    "/optimization/report",
    response_model=OptimizationReportResponse,
    summary="Get optimization report",
    description="Get comprehensive performance optimization report",
)
@inject
async def get_optimization_report(
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> OptimizationReportResponse:
    """Get comprehensive optimization report."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        report = await optimizer.get_optimization_report()

        return OptimizationReportResponse(
            performance_summary=PerformanceSummaryResponse(
                **report["performance_summary"]
            ),
            slow_queries=report["slow_queries"],
            index_recommendations=[
                IndexRecommendationResponse(**rec)
                for rec in report["index_recommendations"]
            ],
            cache_stats=report["cache_stats"],
            index_usage=report["index_usage"],
            most_frequent_queries=[
                QueryMetricsResponse(
                    query_hash=q["query_hash"],
                    query_type="select",  # Default since we don't store type in this format
                    execution_count=q["execution_count"],
                    total_time=q["total_time"],
                    avg_time=q["avg_time"],
                    min_time=0.0,  # Not available in this format
                    max_time=0.0,  # Not available in this format
                    last_executed=0.0,  # Not available in this format
                )
                for q in report["most_frequent_queries"]
            ],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get optimization report: {str(e)}"
        )


@router.post(
    "/optimization/optimize",
    response_model=OptimizationResultResponse,
    summary="Optimize database",
    description="Perform database optimization including index creation",
)
@inject
async def optimize_database(
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> OptimizationResultResponse:
    """Perform database optimization."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        result = await optimizer.optimize_database()

        return OptimizationResultResponse(
            indexes_created=result["indexes_created"],
            recommendations=[
                IndexRecommendationResponse(**rec) for rec in result["recommendations"]
            ],
            performance_summary=PerformanceSummaryResponse(
                **result["performance_summary"]
            ),
            cache_stats=result["cache_stats"],
            error=result.get("error"),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to optimize database: {str(e)}"
        )


@router.post(
    "/cache/clear",
    summary="Clear query cache",
    description="Clear all cached query results",
)
@inject
async def clear_query_cache(
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> dict[str, str]:
    """Clear query cache."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        await optimizer.clear_cache()
        return {"message": "Query cache cleared successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear query cache: {str(e)}"
        )


@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    description="Get query cache statistics and performance metrics",
)
@inject
async def get_cache_stats(
    optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer]),
) -> dict[str, Any]:
    """Get query cache statistics."""
    try:
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Query optimizer not available")

        return optimizer.cache.get_stats()

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache statistics: {str(e)}"
        )


@router.get(
    "/system/metrics",
    summary="Get system performance metrics",
    description="Get overall system performance metrics including pools and queries",
)
async def get_system_metrics(
    # pool_manager: ConnectionPoolManager = Depends(Provide[Container.connection_pool_manager]),
    # optimizer: QueryOptimizer = Depends(Provide[Container.query_optimizer])
) -> dict[str, Any]:
    """Get comprehensive system performance metrics."""
    try:
        # TODO: Implement connection pool manager and query optimizer
        return {
            "timestamp": __import__("time").time(),
            "system": "metrics not implemented yet",
        }

        # metrics = {
        #     "timestamp": __import__("time").time(),
        #     "connection_pools": {},
        #     "query_performance": {},
        #     "cache_performance": {}
        # }

        # # Get pool statistics
        # if pool_manager is not None:
        #     try:
        #         all_stats = pool_manager.get_all_stats()
        #         metrics["connection_pools"] = {
        #             name: {
        #                 "type": stats.pool_type.value,
        #                 "active_connections": stats.active_connections,
        #                 "total_requests": stats.total_requests,
        #                 "success_rate": stats.successful_requests / max(1, stats.total_requests),
        #                 "avg_response_time": stats.avg_response_time,
        #                 "connection_errors": stats.connection_errors
        #             }
        #             for name, stats in all_stats.items()
        #         }
        #     except Exception:
        #         metrics["connection_pools"] = {"error": "Unable to retrieve pool statistics"}
        #
        # # Get query performance
        # if optimizer is not None:
        #     try:
        #         query_summary = optimizer.performance_tracker.get_performance_summary()
        #         metrics["query_performance"] = {
        #             "total_queries": query_summary["total_queries"],
        #             "avg_time": query_summary["avg_time"],
        #             "slow_queries": query_summary["slow_queries"],
        #             "unique_queries": query_summary["unique_queries"]
        #         }
        #
        #         # Get cache stats
        #         cache_stats = optimizer.cache.get_stats()
        #         metrics["cache_performance"] = {
        #             "total_entries": cache_stats["total_entries"],
        #             "hit_rate": cache_stats["hit_rate"],
        #             "memory_usage_mb": cache_stats["memory_usage_mb"]
        #         }
        #     except Exception:
        #         metrics["query_performance"] = {"error": "Unable to retrieve query statistics"}
        #         metrics["cache_performance"] = {"error": "Unable to retrieve cache statistics"}
        #
        # return metrics

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get system metrics: {str(e)}"
        )
