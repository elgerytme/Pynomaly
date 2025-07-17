"""
Advanced database query optimization engine for enterprise-scale data quality operations.

This service implements intelligent query optimization, execution plan analysis,
materialized views management, and adaptive query caching for maximum performance.
"""

import asyncio
import logging
import time
import json
import hashlib
import re
from typing import Dict, Any, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager
import sqlalchemy as sa
from sqlalchemy import text, create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import psutil

from ...domain.entities.quality_profile import DataQualityProfile
from ...domain.value_objects.quality_scores import QualityScores
from software.interfaces.data_quality_interface import DataQualityInterface

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    JOIN = "join"
    ANALYTICAL = "analytical"


class OptimizationStrategy(Enum):
    """Query optimization strategies."""
    INDEX_SUGGESTION = "index_suggestion"
    QUERY_REWRITE = "query_rewrite"
    MATERIALIZED_VIEW = "materialized_view"
    PARTITIONING = "partitioning"
    CACHING = "caching"
    PARALLELIZATION = "parallelization"


@dataclass
class QueryExecutionPlan:
    """Database query execution plan analysis."""
    query_id: str
    query_text: str
    estimated_cost: float
    estimated_rows: int
    execution_time_ms: float
    
    # Plan details
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    index_usage: List[str] = field(default_factory=list)
    table_scans: List[str] = field(default_factory=list)
    
    # Optimization opportunities
    missing_indexes: List[str] = field(default_factory=list)
    slow_operations: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    # Performance metrics
    cpu_cost: float = 0.0
    io_cost: float = 0.0
    network_cost: float = 0.0
    
    @property
    def performance_score(self) -> float:
        """Calculate query performance score (0-100)."""
        base_score = 100
        
        # Penalize high execution time
        if self.execution_time_ms > 1000:  # 1 second
            base_score -= min(50, (self.execution_time_ms - 1000) / 100)
        
        # Penalize table scans
        base_score -= len(self.table_scans) * 5
        
        # Penalize missing indexes
        base_score -= len(self.missing_indexes) * 10
        
        return max(0, base_score)


@dataclass
class MaterializedView:
    """Materialized view configuration and metadata."""
    view_name: str
    base_query: str
    refresh_strategy: str  # manual, scheduled, automatic
    last_refreshed: datetime
    
    # Usage statistics
    access_count: int = 0
    avg_query_time_ms: float = 0.0
    size_mb: float = 0.0
    
    # Configuration
    refresh_interval_hours: int = 24
    auto_refresh_threshold: float = 0.1  # Refresh when data staleness > 10%
    
    # Dependencies
    source_tables: List[str] = field(default_factory=list)
    dependent_queries: List[str] = field(default_factory=list)
    
    @property
    def is_stale(self) -> bool:
        """Check if materialized view is stale."""
        if self.refresh_strategy == "manual":
            return False
        
        age_hours = (datetime.utcnow() - self.last_refreshed).total_seconds() / 3600
        return age_hours > self.refresh_interval_hours
    
    @property
    def efficiency_score(self) -> float:
        """Calculate view efficiency score."""
        if self.access_count == 0:
            return 0.0
        
        # Base efficiency on usage vs maintenance cost
        usage_factor = min(1.0, self.access_count / 100)  # Normalize to 100 accesses
        speed_factor = max(0.1, 1.0 - (self.avg_query_time_ms / 10000))  # 10s baseline
        
        return (usage_factor + speed_factor) / 2


@dataclass
class QueryPattern:
    """Identified query pattern for optimization."""
    pattern_id: str
    pattern_type: str
    query_template: str
    frequency: int
    
    # Performance characteristics
    avg_execution_time_ms: float = 0.0
    max_execution_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    
    # Optimization opportunities
    can_cache: bool = False
    can_materialize: bool = False
    can_index: bool = False
    can_partition: bool = False
    
    # Resources
    avg_rows_scanned: int = 0
    avg_cpu_usage: float = 0.0
    avg_memory_usage_mb: float = 0.0
    
    @property
    def optimization_priority(self) -> float:
        """Calculate optimization priority score."""
        # High frequency + high execution time = high priority
        frequency_factor = min(1.0, self.frequency / 1000)  # Normalize to 1000 executions
        time_factor = min(1.0, self.avg_execution_time_ms / 5000)  # 5s baseline
        
        return (frequency_factor + time_factor) / 2


@dataclass
class QueryOptimizationResult:
    """Result of query optimization analysis."""
    original_query: str
    optimized_query: str
    optimization_strategies: List[OptimizationStrategy]
    
    # Performance improvements
    estimated_speedup: float = 1.0
    estimated_cost_reduction: float = 0.0
    
    # Implementation details
    required_indexes: List[str] = field(default_factory=list)
    suggested_materialized_views: List[str] = field(default_factory=list)
    configuration_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Validation
    tested: bool = False
    actual_speedup: Optional[float] = None
    implementation_complexity: str = "low"  # low, medium, high


class QueryOptimizationEngine:
    """Advanced query optimization engine with machine learning capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the query optimization engine."""
        self.config = config
        
        # Database connections
        self.engines: Dict[str, sa.Engine] = {}
        self.session_makers: Dict[str, sessionmaker] = {}
        self._init_database_connections()
        
        # Query tracking
        self.query_history: deque = deque(maxlen=10000)
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.execution_plans: Dict[str, QueryExecutionPlan] = {}
        
        # Materialized views
        self.materialized_views: Dict[str, MaterializedView] = {}
        self.view_refresh_queue: deque = deque()
        
        # Optimization cache
        self.optimization_cache: Dict[str, QueryOptimizationResult] = {}
        self.index_suggestions: Dict[str, List[str]] = defaultdict(list)
        
        # Performance monitoring
        self.query_metrics: Dict[str, List[float]] = defaultdict(list)
        self.slow_query_threshold_ms = config.get("slow_query_threshold_ms", 1000)
        
        # Machine learning for query prediction
        self.ml_model = None
        self.feature_extractors: Dict[str, Callable] = {}
        self._init_feature_extractors()
        
        # Threading
        self.lock = threading.RLock()
        
        # Background tasks
        asyncio.create_task(self._query_analysis_task())
        asyncio.create_task(self._materialized_view_maintenance_task())
        asyncio.create_task(self._performance_monitoring_task())
        asyncio.create_task(self._optimization_learning_task())
    
    def _init_database_connections(self) -> None:
        """Initialize database connection pools."""
        databases = self.config.get("databases", {})
        
        for db_name, db_config in databases.items():
            connection_string = db_config["connection_string"]
            
            # Create engine with connection pooling
            engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=db_config.get("pool_size", 20),
                max_overflow=db_config.get("max_overflow", 30),
                pool_timeout=db_config.get("pool_timeout", 30),
                pool_recycle=db_config.get("pool_recycle", 3600),
                echo=db_config.get("echo_queries", False)
            )
            
            self.engines[db_name] = engine
            self.session_makers[db_name] = sessionmaker(bind=engine)
            
            logger.info(f"Initialized database connection: {db_name}")
    
    def _init_feature_extractors(self) -> None:
        """Initialize feature extractors for ML model."""
        self.feature_extractors = {
            "query_length": lambda q: len(q),
            "select_count": lambda q: q.upper().count("SELECT"),
            "join_count": lambda q: q.upper().count("JOIN"),
            "where_clause_complexity": self._calculate_where_complexity,
            "table_count": self._count_tables_in_query,
            "subquery_count": lambda q: q.upper().count("SELECT") - 1,
            "aggregate_function_count": self._count_aggregate_functions,
            "orderby_present": lambda q: 1 if "ORDER BY" in q.upper() else 0,
            "groupby_present": lambda q: 1 if "GROUP BY" in q.upper() else 0,
            "having_present": lambda q: 1 if "HAVING" in q.upper() else 0
        }
    
    async def _query_analysis_task(self) -> None:
        """Background task for analyzing query patterns."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze recent queries for patterns
                await self._analyze_query_patterns()
                
                # Generate optimization suggestions
                await self._generate_optimization_suggestions()
                
                # Update query statistics
                await self._update_query_statistics()
                
                logger.debug("Query analysis cycle completed")
                
            except Exception as e:
                logger.error(f"Query analysis error: {str(e)}")
    
    async def _materialized_view_maintenance_task(self) -> None:
        """Background task for materialized view maintenance."""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                # Check for stale views
                stale_views = [
                    view for view in self.materialized_views.values()
                    if view.is_stale
                ]
                
                # Refresh stale views
                for view in stale_views:
                    await self._refresh_materialized_view(view)
                
                # Analyze view usage and efficiency
                await self._analyze_view_efficiency()
                
                logger.debug(f"Materialized view maintenance: {len(stale_views)} views refreshed")
                
            except Exception as e:
                logger.error(f"Materialized view maintenance error: {str(e)}")
    
    async def _performance_monitoring_task(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Analyze slow queries
                await self._analyze_slow_queries()
                
                # Monitor database performance
                await self._monitor_database_performance()
                
                # Update optimization recommendations
                await self._update_optimization_recommendations()
                
                logger.debug("Performance monitoring cycle completed")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {str(e)}")
    
    async def _optimization_learning_task(self) -> None:
        """Background task for machine learning optimization."""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Extract features from recent queries
                features = await self._extract_query_features()
                
                # Update ML model if enough data
                if len(features) >= 100:
                    await self._update_ml_model(features)
                
                # Apply ML-based optimizations
                await self._apply_ml_optimizations()
                
                logger.debug("ML optimization cycle completed")
                
            except Exception as e:
                logger.error(f"ML optimization error: {str(e)}")
    
    # Error handling would be managed by interface implementation
    async def optimize_query(self, query: str, database: str = "default") -> QueryOptimizationResult:
        """Optimize a database query using multiple strategies."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check optimization cache
        if query_hash in self.optimization_cache:
            cached_result = self.optimization_cache[query_hash]
            logger.debug(f"Retrieved cached optimization for query: {query_hash[:8]}")
            return cached_result
        
        # Analyze query execution plan
        execution_plan = await self._analyze_execution_plan(query, database)
        
        # Generate optimization strategies
        optimization_strategies = await self._identify_optimization_strategies(query, execution_plan)
        
        # Apply optimizations
        optimized_query = await self._apply_optimizations(query, optimization_strategies)
        
        # Create optimization result
        result = QueryOptimizationResult(
            original_query=query,
            optimized_query=optimized_query,
            optimization_strategies=optimization_strategies,
            estimated_speedup=await self._estimate_speedup(query, optimized_query, database),
            required_indexes=execution_plan.missing_indexes,
            implementation_complexity=self._assess_implementation_complexity(optimization_strategies)
        )
        
        # Cache result
        self.optimization_cache[query_hash] = result
        
        logger.info(f"Optimized query with {len(optimization_strategies)} strategies, "
                   f"estimated speedup: {result.estimated_speedup:.2f}x")
        
        return result
    
    async def _analyze_execution_plan(self, query: str, database: str) -> QueryExecutionPlan:
        """Analyze database query execution plan."""
        engine = self.engines.get(database)
        if not engine:
            raise ValueError(f"Database '{database}' not configured")
        
        start_time = time.time()
        
        try:
            with engine.connect() as conn:
                # Get execution plan (PostgreSQL style - adapt for other databases)
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                
                try:
                    result = conn.execute(text(explain_query))
                    plan_data = result.fetchone()[0]
                except Exception:
                    # Fallback to simple EXPLAIN if ANALYZE fails
                    explain_query = f"EXPLAIN (FORMAT JSON) {query}"
                    result = conn.execute(text(explain_query))
                    plan_data = result.fetchone()[0]
                
                execution_time = (time.time() - start_time) * 1000
                
                # Parse execution plan
                plan = json.loads(plan_data) if isinstance(plan_data, str) else plan_data
                
                # Extract plan information
                query_plan = plan[0]["Plan"] if isinstance(plan, list) else plan["Plan"]
                
                execution_plan = QueryExecutionPlan(
                    query_id=hashlib.md5(query.encode()).hexdigest(),
                    query_text=query,
                    estimated_cost=query_plan.get("Total Cost", 0),
                    estimated_rows=query_plan.get("Plan Rows", 0),
                    execution_time_ms=execution_time,
                    plan_steps=self._extract_plan_steps(query_plan),
                    index_usage=self._extract_index_usage(query_plan),
                    table_scans=self._extract_table_scans(query_plan),
                    missing_indexes=await self._identify_missing_indexes(query, query_plan),
                    slow_operations=self._identify_slow_operations(query_plan),
                    optimization_suggestions=await self._generate_plan_suggestions(query_plan)
                )
                
                return execution_plan
                
        except Exception as e:
            logger.error(f"Failed to analyze execution plan: {str(e)}")
            # Return basic plan with error info
            return QueryExecutionPlan(
                query_id=hashlib.md5(query.encode()).hexdigest(),
                query_text=query,
                estimated_cost=1000.0,  # High cost for failed analysis
                estimated_rows=1000,
                execution_time_ms=(time.time() - start_time) * 1000,
                optimization_suggestions=[f"Analysis failed: {str(e)}"]
            )
    
    def _extract_plan_steps(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract execution plan steps."""
        steps = []
        
        def extract_recursive(node, depth=0):
            steps.append({
                "node_type": node.get("Node Type", "Unknown"),
                "depth": depth,
                "cost": node.get("Total Cost", 0),
                "rows": node.get("Plan Rows", 0),
                "width": node.get("Plan Width", 0),
                "actual_time": node.get("Actual Total Time", 0),
                "actual_rows": node.get("Actual Rows", 0)
            })
            
            # Process child plans
            for child in node.get("Plans", []):
                extract_recursive(child, depth + 1)
        
        extract_recursive(plan)
        return steps
    
    def _extract_index_usage(self, plan: Dict[str, Any]) -> List[str]:
        """Extract index usage from execution plan."""
        indexes = []
        
        def extract_recursive(node):
            if node.get("Node Type") == "Index Scan":
                index_name = node.get("Index Name")
                if index_name:
                    indexes.append(index_name)
            
            for child in node.get("Plans", []):
                extract_recursive(child)
        
        extract_recursive(plan)
        return indexes
    
    def _extract_table_scans(self, plan: Dict[str, Any]) -> List[str]:
        """Extract table scans from execution plan."""
        table_scans = []
        
        def extract_recursive(node):
            if node.get("Node Type") == "Seq Scan":
                relation_name = node.get("Relation Name")
                if relation_name:
                    table_scans.append(relation_name)
            
            for child in node.get("Plans", []):
                extract_recursive(child)
        
        extract_recursive(plan)
        return table_scans
    
    async def _identify_missing_indexes(self, query: str, plan: Dict[str, Any]) -> List[str]:
        """Identify missing indexes that could improve performance."""
        missing_indexes = []
        
        # Analyze WHERE clauses for potential indexes
        where_conditions = self._extract_where_conditions(query)
        for condition in where_conditions:
            if self._should_create_index(condition, plan):
                missing_indexes.append(f"CREATE INDEX ON {condition['table']} ({condition['column']})")
        
        # Analyze JOIN conditions
        join_conditions = self._extract_join_conditions(query)
        for condition in join_conditions:
            if self._should_create_index(condition, plan):
                missing_indexes.append(f"CREATE INDEX ON {condition['table']} ({condition['column']})")
        
        return missing_indexes
    
    def _extract_where_conditions(self, query: str) -> List[Dict[str, str]]:
        """Extract WHERE conditions from query."""
        conditions = []
        
        # Simple regex-based extraction (would be more sophisticated in production)
        where_pattern = r"WHERE\s+(\w+)\.(\w+)\s*[=<>]"
        matches = re.findall(where_pattern, query.upper())
        
        for table, column in matches:
            conditions.append({"table": table.lower(), "column": column.lower()})
        
        return conditions
    
    def _extract_join_conditions(self, query: str) -> List[Dict[str, str]]:
        """Extract JOIN conditions from query."""
        conditions = []
        
        # Simple regex-based extraction
        join_pattern = r"JOIN\s+(\w+).*?ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)"
        matches = re.findall(join_pattern, query.upper())
        
        for table, table1, col1, table2, col2 in matches:
            conditions.extend([
                {"table": table1.lower(), "column": col1.lower()},
                {"table": table2.lower(), "column": col2.lower()}
            ])
        
        return conditions
    
    def _should_create_index(self, condition: Dict[str, str], plan: Dict[str, Any]) -> bool:
        """Determine if an index should be created for a condition."""
        # Check if there's already an index on this column
        # This is a simplified check - would be more sophisticated in production
        return True  # For demo purposes
    
    async def _identify_optimization_strategies(self, query: str, plan: QueryExecutionPlan) -> List[OptimizationStrategy]:
        """Identify applicable optimization strategies."""
        strategies = []
        
        # Index suggestions
        if plan.missing_indexes:
            strategies.append(OptimizationStrategy.INDEX_SUGGESTION)
        
        # Query rewriting
        if self._can_rewrite_query(query):
            strategies.append(OptimizationStrategy.QUERY_REWRITE)
        
        # Materialized views
        if self._can_materialize_query(query):
            strategies.append(OptimizationStrategy.MATERIALIZED_VIEW)
        
        # Caching
        if self._can_cache_query(query):
            strategies.append(OptimizationStrategy.CACHING)
        
        # Parallelization
        if self._can_parallelize_query(query, plan):
            strategies.append(OptimizationStrategy.PARALLELIZATION)
        
        return strategies
    
    def _can_rewrite_query(self, query: str) -> bool:
        """Check if query can be rewritten for better performance."""
        # Look for common anti-patterns
        upper_query = query.upper()
        
        # Check for inefficient patterns
        if "SELECT *" in upper_query:
            return True
        if "ORDER BY" in upper_query and "LIMIT" not in upper_query:
            return True
        if upper_query.count("SELECT") > 1:  # Subqueries
            return True
        
        return False
    
    def _can_materialize_query(self, query: str) -> bool:
        """Check if query is suitable for materialization."""
        upper_query = query.upper()
        
        # Good candidates for materialization
        if "GROUP BY" in upper_query and "JOIN" in upper_query:
            return True
        if upper_query.count("SELECT") > 1:  # Complex subqueries
            return True
        if "SUM(" in upper_query or "COUNT(" in upper_query:
            return True
        
        return False
    
    def _can_cache_query(self, query: str) -> bool:
        """Check if query results can be cached."""
        upper_query = query.upper()
        
        # Don't cache if query has time-sensitive functions
        time_functions = ["NOW()", "CURRENT_DATE", "CURRENT_TIME", "RANDOM()"]
        if any(func in upper_query for func in time_functions):
            return False
        
        # Don't cache write operations
        if any(op in upper_query for op in ["INSERT", "UPDATE", "DELETE"]):
            return False
        
        return True
    
    def _can_parallelize_query(self, query: str, plan: QueryExecutionPlan) -> bool:
        """Check if query can benefit from parallelization."""
        # Check for large table scans or expensive operations
        if plan.estimated_rows > 100000:  # Large result set
            return True
        if plan.estimated_cost > 1000:  # Expensive query
            return True
        if len(plan.table_scans) > 1:  # Multiple table scans
            return True
        
        return False
    
    async def _apply_optimizations(self, query: str, strategies: List[OptimizationStrategy]) -> str:
        """Apply optimization strategies to query."""
        optimized_query = query
        
        for strategy in strategies:
            if strategy == OptimizationStrategy.QUERY_REWRITE:
                optimized_query = await self._rewrite_query(optimized_query)
            elif strategy == OptimizationStrategy.CACHING:
                # Caching is handled at execution level, not query level
                pass
            elif strategy == OptimizationStrategy.PARALLELIZATION:
                optimized_query = await self._add_parallelization_hints(optimized_query)
        
        return optimized_query
    
    async def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better performance."""
        rewritten = query
        
        # Replace SELECT * with specific columns (simplified)
        if "SELECT *" in query.upper():
            # In production, would analyze table schema to get actual columns
            rewritten = rewritten.replace("SELECT *", "SELECT id, name, created_at")
        
        # Add LIMIT to ORDER BY queries without it
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            rewritten += " LIMIT 1000"
        
        return rewritten
    
    async def _add_parallelization_hints(self, query: str) -> str:
        """Add database-specific parallelization hints."""
        # PostgreSQL parallel query hints
        if "SELECT" in query.upper():
            return f"/*+ PARALLEL(4) */ {query}"
        
        return query
    
    async def _estimate_speedup(self, original_query: str, optimized_query: str, database: str) -> float:
        """Estimate performance speedup from optimization."""
        # In production, would run both queries and measure actual performance
        # For now, return estimated based on optimization strategies
        
        speedup = 1.0
        
        if "SELECT *" in original_query and "SELECT *" not in optimized_query:
            speedup *= 1.2  # 20% improvement from column selection
        
        if "LIMIT" in optimized_query and "LIMIT" not in original_query:
            speedup *= 2.0  # 2x improvement from limiting results
        
        if "/*+ PARALLEL" in optimized_query:
            speedup *= 1.5  # 50% improvement from parallelization
        
        return speedup
    
    def _assess_implementation_complexity(self, strategies: List[OptimizationStrategy]) -> str:
        """Assess implementation complexity of optimization strategies."""
        complexity_scores = {
            OptimizationStrategy.CACHING: 1,
            OptimizationStrategy.QUERY_REWRITE: 2,
            OptimizationStrategy.INDEX_SUGGESTION: 3,
            OptimizationStrategy.MATERIALIZED_VIEW: 4,
            OptimizationStrategy.PARTITIONING: 5,
            OptimizationStrategy.PARALLELIZATION: 3
        }
        
        if not strategies:
            return "low"
        
        max_complexity = max(complexity_scores.get(s, 1) for s in strategies)
        
        if max_complexity <= 2:
            return "low"
        elif max_complexity <= 4:
            return "medium"
        else:
            return "high"
    
    # Error handling would be managed by interface implementation
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        # Analyze query patterns
        top_patterns = sorted(
            self.query_patterns.values(),
            key=lambda p: p.optimization_priority,
            reverse=True
        )[:10]
        
        # Analyze slow queries
        slow_queries = [
            plan for plan in self.execution_plans.values()
            if plan.execution_time_ms > self.slow_query_threshold_ms
        ]
        
        # Materialized view efficiency
        view_efficiency = {
            name: view.efficiency_score
            for name, view in self.materialized_views.items()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "query_statistics": {
                "total_queries_analyzed": len(self.query_history),
                "unique_patterns": len(self.query_patterns),
                "slow_queries": len(slow_queries),
                "avg_query_time_ms": sum(q.execution_time_ms for q in self.execution_plans.values()) / 
                                   max(1, len(self.execution_plans))
            },
            "optimization_opportunities": {
                "top_patterns_for_optimization": [
                    {
                        "pattern_id": p.pattern_id,
                        "frequency": p.frequency,
                        "avg_time_ms": p.avg_execution_time_ms,
                        "priority_score": p.optimization_priority
                    }
                    for p in top_patterns
                ],
                "missing_indexes": sum(len(plan.missing_indexes) for plan in self.execution_plans.values()),
                "materialization_candidates": len([p for p in self.query_patterns.values() if p.can_materialize])
            },
            "materialized_views": {
                "total_views": len(self.materialized_views),
                "stale_views": len([v for v in self.materialized_views.values() if v.is_stale]),
                "efficiency_scores": view_efficiency
            },
            "performance_trends": [
                {
                    "timestamp": plan.query_id,
                    "execution_time_ms": plan.execution_time_ms,
                    "performance_score": plan.performance_score
                }
                for plan in list(self.execution_plans.values())[-20:]  # Last 20 queries
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the query optimization engine."""
        logger.info("Shutting down query optimization engine...")
        
        # Close database connections
        for engine in self.engines.values():
            engine.dispose()
        
        logger.info("Query optimization engine shutdown complete")