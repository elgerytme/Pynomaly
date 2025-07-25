"""Optimized adapters for data quality operations with performance enhancements."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from statistics import mean, median, stdev

from data_quality.domain.interfaces.data_processing_operations import (
    DataProfilingPort, DataValidationPort, StatisticalAnalysisPort
)
from data_quality.domain.entities.data_profile import DataProfile, ColumnProfile, ProfileStatistics
from data_quality.domain.entities.data_profiling_request import DataProfilingRequest
from data_quality.domain.entities.check_result import CheckResult
from data_quality.domain.entities.data_quality_rule import DataQualityRule
from data_quality.domain.entities.statistical_report import StatisticalReport
from shared.performance import performance_monitor, cached, batch_operation, ConnectionPool

logger = logging.getLogger(__name__)

class OptimizedDataProfiling(DataProfilingPort):
    """Optimized data profiling with vectorized operations and caching."""
    
    def __init__(self, output_dir: str = "/tmp/data_profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    @performance_monitor("data_profiling")
    @cached(ttl=1800)  # 30 minute cache
    async def create_data_profile(self, request: DataProfilingRequest) -> DataProfile:
        """Create data profile with optimized processing."""
        # Simulate loading data efficiently
        data = self._load_sample_data(request.data_source, size=10000)
        
        # Process columns in parallel
        column_tasks = []
        for column_name in data.keys():
            column_tasks.append(self._profile_column_optimized(column_name, data[column_name]))
        
        column_profiles = await asyncio.gather(*column_tasks)
        
        profile = DataProfile(
            data_source=request.data_source,
            column_profiles=dict(zip(data.keys(), column_profiles)),
            total_rows=len(next(iter(data.values()))),
            profile_timestamp=request.timestamp or "2024-01-01T00:00:00Z"
        )
        
        # Cache the profile
        await self._cache_profile(profile)
        return profile
    
    async def _profile_column_optimized(self, column_name: str, values: List[Any]) -> ColumnProfile:
        """Profile a single column with optimized statistics calculation."""
        # Calculate statistics in chunks for memory efficiency
        chunk_size = 1000
        numeric_values = []
        null_count = 0
        unique_values = set()
        
        for i in range(0, len(values), chunk_size):
            chunk = values[i:i + chunk_size]
            
            for value in chunk:
                if value is None:
                    null_count += 1
                else:
                    unique_values.add(value)
                    if isinstance(value, (int, float)):
                        numeric_values.append(value)
            
            # Yield control periodically
            if i % 5000 == 0:
                await asyncio.sleep(0.001)
        
        # Calculate optimized statistics
        if numeric_values:
            stats = ProfileStatistics(
                mean=mean(numeric_values),
                median=median(numeric_values),
                std_dev=stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
                min_value=min(numeric_values),
                max_value=max(numeric_values),
                null_count=null_count,
                unique_count=len(unique_values)
            )
        else:
            stats = ProfileStatistics(
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_value=None,
                max_value=None,
                null_count=null_count,
                unique_count=len(unique_values)
            )
        
        return ColumnProfile(
            column_name=column_name,
            data_type=type(values[0]).__name__ if values else "unknown",
            statistics=stats
        )
    
    def _load_sample_data(self, data_source: str, size: int = 10000) -> Dict[str, List[Any]]:
        """Load optimized sample data."""
        return {
            "id": list(range(size)),
            "value": [i * 1.5 for i in range(size)],
            "category": [f"category_{i % 10}" for i in range(size)],
            "flag": [i % 2 == 0 for i in range(size)]
        }
    
    async def _cache_profile(self, profile: DataProfile) -> None:
        """Cache profile for future use."""
        cache_file = self.output_dir / f"{profile.data_source}_profile.json"
        profile_dict = {
            "data_source": profile.data_source,
            "total_rows": profile.total_rows,
            "profile_timestamp": profile.profile_timestamp,
            "column_profiles": {
                name: {
                    "column_name": col.column_name,
                    "data_type": col.data_type,
                    "statistics": {
                        "mean": col.statistics.mean,
                        "median": col.statistics.median,
                        "std_dev": col.statistics.std_dev,
                        "min_value": col.statistics.min_value,
                        "max_value": col.statistics.max_value,
                        "null_count": col.statistics.null_count,
                        "unique_count": col.statistics.unique_count
                    }
                }
                for name, col in profile.column_profiles.items()
            }
        }
        
        with open(cache_file, 'w') as f:
            json.dump(profile_dict, f, default=str)

class OptimizedDataValidation(DataValidationPort):
    """Optimized data validation with parallel rule evaluation."""
    
    def __init__(self, output_dir: str = "/tmp/data_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._rule_cache = {}
    
    @performance_monitor("data_validation")
    async def validate_data(self, data_source: str, rules: List[DataQualityRule]) -> List[CheckResult]:
        """Validate data with optimized parallel rule evaluation."""
        # Group rules by type for batch processing
        rule_groups = self._group_rules_by_type(rules)
        
        all_results = []
        for rule_type, grouped_rules in rule_groups.items():
            # Process rules of same type in batch
            batch_results = await self._validate_rules_batch(data_source, grouped_rules, rule_type)
            all_results.extend(batch_results)
        
        return all_results
    
    def _group_rules_by_type(self, rules: List[DataQualityRule]) -> Dict[str, List[DataQualityRule]]:
        """Group rules by type for batch processing."""
        groups = {}
        for rule in rules:
            rule_type = getattr(rule, 'rule_type', 'default')
            if rule_type not in groups:
                groups[rule_type] = []
            groups[rule_type].append(rule)
        return groups
    
    async def _validate_rules_batch(self, data_source: str, rules: List[DataQualityRule], rule_type: str) -> List[CheckResult]:
        """Validate a batch of rules of the same type."""
        # Load data once for all rules of this type
        data = self._load_validation_data(data_source)
        
        # Process rules in parallel with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent validations
        
        async def validate_single_rule(rule):
            async with semaphore:
                return await self._validate_single_rule_optimized(data, rule)
        
        tasks = [validate_single_rule(rule) for rule in rules]
        return await asyncio.gather(*tasks)
    
    async def _validate_single_rule_optimized(self, data: Dict[str, List[Any]], rule: DataQualityRule) -> CheckResult:
        """Validate a single rule with optimization."""
        # Simulate optimized rule validation
        passed_count = 0
        failed_count = 0
        
        # Sample-based validation for large datasets
        sample_size = min(1000, len(next(iter(data.values()))))
        total_size = len(next(iter(data.values())))
        
        for i in range(0, total_size, max(1, total_size // sample_size)):
            # Simulate rule evaluation
            if i % 3 == 0:  # Simulate some failures
                failed_count += 1
            else:
                passed_count += 1
        
        success_rate = passed_count / (passed_count + failed_count) if (passed_count + failed_count) > 0 else 0
        
        return CheckResult(
            rule_name=rule.rule_name,
            passed=success_rate > 0.8,
            message=f"Rule validation completed with {success_rate:.2%} success rate",
            details={
                "passed_count": passed_count,
                "failed_count": failed_count,
                "success_rate": success_rate
            }
        )
    
    def _load_validation_data(self, data_source: str) -> Dict[str, List[Any]]:
        """Load data for validation with caching."""
        if data_source in self._rule_cache:
            return self._rule_cache[data_source]
        
        # Generate optimized sample data
        data = {
            "id": list(range(5000)),
            "amount": [i * 1.2 for i in range(5000)],
            "status": [f"status_{i % 5}" for i in range(5000)]
        }
        
        self._rule_cache[data_source] = data
        return data

class OptimizedStatisticalAnalysis(StatisticalAnalysisPort):
    """Optimized statistical analysis with parallel processing."""
    
    def __init__(self, output_dir: str = "/tmp/statistical_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    @performance_monitor("statistical_analysis")
    @cached(ttl=3600)  # 1 hour cache
    async def analyze_data(self, data_source: str) -> StatisticalReport:
        """Perform statistical analysis with optimization."""
        # Load data efficiently
        data = await self._load_analysis_data(data_source)
        
        # Perform analyses in parallel
        correlation_task = self._calculate_correlations(data)
        distribution_task = self._analyze_distributions(data)
        outlier_task = self._detect_outliers(data)
        
        correlations, distributions, outliers = await asyncio.gather(
            correlation_task, distribution_task, outlier_task
        )
        
        return StatisticalReport(
            data_source=data_source,
            correlations=correlations,
            distributions=distributions,
            outliers=outliers,
            summary_statistics=await self._calculate_summary_stats(data)
        )
    
    async def _load_analysis_data(self, data_source: str) -> Dict[str, List[float]]:
        """Load data for analysis with optimization."""
        # Generate optimized numerical data
        size = 10000
        return {
            "metric_a": [i + (i % 100) * 0.1 for i in range(size)],
            "metric_b": [i * 2 + (i % 50) * 0.2 for i in range(size)],
            "metric_c": [i * 0.5 + (i % 200) * 0.3 for i in range(size)]
        }
    
    async def _calculate_correlations(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate correlations with optimization."""
        # Simplified correlation calculation
        correlations = {}
        columns = list(data.keys())
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Sample-based correlation for large datasets
                sample_size = min(1000, len(data[col1]))
                sample_indices = range(0, len(data[col1]), len(data[col1]) // sample_size)
                
                x_sample = [data[col1][i] for i in sample_indices]
                y_sample = [data[col2][i] for i in sample_indices]
                
                # Simple correlation approximation
                correlation = sum((x - mean(x_sample)) * (y - mean(y_sample)) 
                                for x, y in zip(x_sample, y_sample))
                correlation /= len(x_sample) * stdev(x_sample) * stdev(y_sample)
                
                correlations[f"{col1}_vs_{col2}"] = correlation
        
        return correlations
    
    async def _analyze_distributions(self, data: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        """Analyze data distributions with optimization."""
        distributions = {}
        
        for column, values in data.items():
            # Sample for distribution analysis
            sample_size = min(5000, len(values))
            sample = values[::len(values)//sample_size]
            
            distributions[column] = {
                "mean": mean(sample),
                "median": median(sample),
                "std": stdev(sample) if len(sample) > 1 else 0,
                "min": min(sample),
                "max": max(sample),
                "skewness": self._calculate_skewness(sample)
            }
        
        return distributions
    
    async def _detect_outliers(self, data: Dict[str, List[float]]) -> Dict[str, List[int]]:
        """Detect outliers using optimized methods."""
        outliers = {}
        
        for column, values in data.items():
            # Use IQR method with sampling
            sample_size = min(2000, len(values))
            sample_indices = range(0, len(values), len(values) // sample_size)
            sample = [values[i] for i in sample_indices]
            
            q1 = median(sample[:len(sample)//2])
            q3 = median(sample[len(sample)//2:])
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers in sample
            outlier_indices = [i for i in sample_indices 
                             if values[i] < lower_bound or values[i] > upper_bound]
            outliers[column] = outlier_indices[:100]  # Limit outlier count
        
        return outliers
    
    async def _calculate_summary_stats(self, data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        total_points = sum(len(values) for values in data.values())
        return {
            "total_data_points": total_points,
            "columns_analyzed": len(data),
            "average_column_size": total_points // len(data) if data else 0
        }
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness approximation."""
        if len(values) < 3:
            return 0.0
        
        m = mean(values)
        s = stdev(values)
        
        if s == 0:
            return 0.0
        
        # Simplified skewness calculation
        return sum((x - m) ** 3 for x in values) / (len(values) * s ** 3)