"""Data sampling service for various sampling methodologies."""

import asyncio
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class DataSamplingService:
    """Service for data sampling using various statistical methods."""
    
    def __init__(self):
        self.sampling_methods = {
            'random': self._random_sampling,
            'systematic': self._systematic_sampling,
            'stratified': self._stratified_sampling,
            'cluster': self._cluster_sampling,
            'reservoir': self._reservoir_sampling
        }
    
    async def sample_file(
        self,
        file_path: Path,
        sample_size: int,
        method: str = 'random',
        stratify_column: Optional[str] = None,
        cluster_column: Optional[str] = None,
        seed: Optional[int] = None,
        replacement: bool = False
    ) -> pd.DataFrame:
        """Sample data from a file using specified method."""
        
        if method not in self.sampling_methods:
            raise ValueError(f"Unsupported sampling method: {method}. Available: {list(self.sampling_methods.keys())}")
        
        try:
            logger.info("Starting data sampling",
                       file=str(file_path),
                       method=method,
                       sample_size=sample_size)
            
            # Load data
            data = await self._load_data(file_path)
            
            # Validate sample size
            if sample_size >= len(data) and not replacement:
                logger.warning("Sample size larger than dataset", 
                             sample_size=sample_size, 
                             dataset_size=len(data))
                return data
            
            # Set random seed for reproducibility
            if seed is not None:
                np.random.seed(seed)
            
            # Apply sampling method
            sampler = self.sampling_methods[method]
            
            if method == 'stratified':
                if not stratify_column:
                    raise ValueError("stratify_column is required for stratified sampling")
                sample = await sampler(data, sample_size, stratify_column=stratify_column, replacement=replacement)
            elif method == 'cluster':
                if not cluster_column:
                    raise ValueError("cluster_column is required for cluster sampling")
                sample = await sampler(data, sample_size, cluster_column=cluster_column)
            else:
                sample = await sampler(data, sample_size, replacement=replacement)
            
            logger.info("Data sampling completed",
                       original_size=len(data),
                       sample_size=len(sample),
                       method=method)
            
            return sample
            
        except Exception as e:
            logger.error("Data sampling failed", file=str(file_path), error=str(e))
            raise
    
    async def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load data from file."""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
    
    async def _random_sampling(self, data: pd.DataFrame, sample_size: int, replacement: bool = False) -> pd.DataFrame:
        """Perform simple random sampling."""
        return data.sample(n=sample_size, replace=replacement, random_state=np.random.get_state()[1][0])
    
    async def _systematic_sampling(self, data: pd.DataFrame, sample_size: int, replacement: bool = False) -> pd.DataFrame:
        """Perform systematic sampling."""
        if replacement:
            logger.warning("Systematic sampling with replacement not typical, using random sampling")
            return await self._random_sampling(data, sample_size, replacement=True)
        
        n = len(data)
        if sample_size >= n:
            return data
        
        # Calculate interval
        interval = n // sample_size
        if interval < 1:
            interval = 1
        
        # Random starting point
        start = np.random.randint(0, interval)
        
        # Select indices
        indices = list(range(start, n, interval))[:sample_size]
        
        return data.iloc[indices].reset_index(drop=True)
    
    async def _stratified_sampling(
        self, 
        data: pd.DataFrame, 
        sample_size: int, 
        stratify_column: str,
        replacement: bool = False
    ) -> pd.DataFrame:
        """Perform stratified sampling based on a categorical column."""
        
        if stratify_column not in data.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found in data")
        
        # Get strata proportions
        strata_counts = data[stratify_column].value_counts()
        strata_proportions = strata_counts / len(data)
        
        samples = []
        
        for stratum, proportion in strata_proportions.items():
            stratum_data = data[data[stratify_column] == stratum]
            stratum_sample_size = max(1, int(sample_size * proportion))
            
            # Adjust if stratum is smaller than required sample
            if stratum_sample_size > len(stratum_data) and not replacement:
                stratum_sample_size = len(stratum_data)
            
            if len(stratum_data) > 0:
                stratum_sample = stratum_data.sample(
                    n=stratum_sample_size, 
                    replace=replacement,
                    random_state=np.random.get_state()[1][0]
                )
                samples.append(stratum_sample)
        
        if not samples:
            raise ValueError("No valid strata found for sampling")
        
        # Combine samples
        result = pd.concat(samples, ignore_index=True)
        
        # If we have more samples than requested, randomly select the exact number
        if len(result) > sample_size:
            result = result.sample(n=sample_size, random_state=np.random.get_state()[1][0])
        
        return result
    
    async def _cluster_sampling(
        self, 
        data: pd.DataFrame, 
        sample_size: int, 
        cluster_column: str
    ) -> pd.DataFrame:
        """Perform cluster sampling."""
        
        if cluster_column not in data.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in data")
        
        # Get unique clusters
        clusters = data[cluster_column].unique()
        
        # Determine how many clusters to select
        total_clusters = len(clusters)
        
        # Estimate how many clusters we need to get approximately the sample size
        avg_cluster_size = len(data) / total_clusters
        clusters_needed = max(1, int(sample_size / avg_cluster_size))
        clusters_needed = min(clusters_needed, total_clusters)
        
        # Randomly select clusters
        selected_clusters = np.random.choice(clusters, size=clusters_needed, replace=False)
        
        # Get all data from selected clusters
        cluster_sample = data[data[cluster_column].isin(selected_clusters)]
        
        # If we have too much data, randomly sample from the selected clusters
        if len(cluster_sample) > sample_size:
            cluster_sample = cluster_sample.sample(n=sample_size, random_state=np.random.get_state()[1][0])
        
        return cluster_sample.reset_index(drop=True)
    
    async def _reservoir_sampling(self, data: pd.DataFrame, sample_size: int, replacement: bool = False) -> pd.DataFrame:
        """Perform reservoir sampling (useful for streaming data or large datasets)."""
        
        if replacement:
            logger.warning("Reservoir sampling with replacement not typical, using random sampling")
            return await self._random_sampling(data, sample_size, replacement=True)
        
        n = len(data)
        if sample_size >= n:
            return data
        
        # Initialize reservoir with first k elements
        reservoir_indices = list(range(sample_size))
        
        # Process remaining elements
        for i in range(sample_size, n):
            # Generate random number between 0 and i (inclusive)
            j = np.random.randint(0, i + 1)
            
            # If j is in the first k elements, replace element at index j with element at index i
            if j < sample_size:
                reservoir_indices[j] = i
        
        return data.iloc[reservoir_indices].reset_index(drop=True)
    
    async def sample_multiple_files(
        self,
        file_paths: List[Path],
        sample_size: int,
        method: str = 'random',
        output_dir: Optional[Path] = None,
        **sampling_options
    ) -> List[pd.DataFrame]:
        """Sample multiple files and optionally save results."""
        
        logger.info("Starting batch sampling",
                   files=len(file_paths),
                   sample_size=sample_size,
                   method=method)
        
        results = []
        
        for file_path in file_paths:
            try:
                sample = await self.sample_file(
                    file_path=file_path,
                    sample_size=sample_size,
                    method=method,
                    **sampling_options
                )
                
                results.append(sample)
                
                # Save if output directory specified
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir / f"{file_path.stem}_sample_{method}_{sample_size}.csv"
                    sample.to_csv(output_file, index=False)
                    logger.info("Sample saved", output_file=str(output_file))
                
            except Exception as e:
                logger.error("Failed to sample file", file=str(file_path), error=str(e))
                results.append(pd.DataFrame())  # Empty DataFrame for failed samples
        
        logger.info("Batch sampling completed", 
                   successful_samples=sum(1 for r in results if not r.empty))
        
        return results
    
    async def generate_sampling_report(
        self,
        original_data: pd.DataFrame,
        sampled_data: pd.DataFrame,
        method: str,
        stratify_column: Optional[str] = None
    ) -> dict:
        """Generate a report comparing original and sampled data."""
        
        report = {
            'sampling_metadata': {
                'method': method,
                'original_size': len(original_data),
                'sample_size': len(sampled_data),
                'sampling_ratio': len(sampled_data) / len(original_data) if len(original_data) > 0 else 0,
                'timestamp': datetime.utcnow().isoformat()
            },
            'representativeness_analysis': {}
        }
        
        try:
            # Compare basic statistics for numeric columns
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                numeric_comparison = {}
                
                for col in numeric_cols:
                    if col in sampled_data.columns:
                        orig_stats = original_data[col].describe()
                        sample_stats = sampled_data[col].describe()
                        
                        numeric_comparison[col] = {
                            'original_mean': float(orig_stats['mean']) if not pd.isna(orig_stats['mean']) else None,
                            'sample_mean': float(sample_stats['mean']) if not pd.isna(sample_stats['mean']) else None,
                            'original_std': float(orig_stats['std']) if not pd.isna(orig_stats['std']) else None,
                            'sample_std': float(sample_stats['std']) if not pd.isna(sample_stats['std']) else None,
                            'mean_difference': float(abs(orig_stats['mean'] - sample_stats['mean'])) if not pd.isna(orig_stats['mean']) and not pd.isna(sample_stats['mean']) else None
                        }
                
                report['representativeness_analysis']['numeric_columns'] = numeric_comparison
            
            # Compare categorical distributions
            categorical_cols = original_data.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                categorical_comparison = {}
                
                for col in categorical_cols:
                    if col in sampled_data.columns:
                        orig_dist = original_data[col].value_counts(normalize=True)
                        sample_dist = sampled_data[col].value_counts(normalize=True)
                        
                        # Calculate distribution differences
                        all_categories = set(orig_dist.index) | set(sample_dist.index)
                        distribution_diff = {}
                        
                        for category in all_categories:
                            orig_prop = orig_dist.get(category, 0)
                            sample_prop = sample_dist.get(category, 0)
                            distribution_diff[str(category)] = {
                                'original_proportion': float(orig_prop),
                                'sample_proportion': float(sample_prop),
                                'difference': float(abs(orig_prop - sample_prop))
                            }
                        
                        categorical_comparison[col] = distribution_diff
                
                report['representativeness_analysis']['categorical_columns'] = categorical_comparison
            
            # Special analysis for stratified sampling
            if method == 'stratified' and stratify_column:
                orig_strata = original_data[stratify_column].value_counts(normalize=True)
                sample_strata = sampled_data[stratify_column].value_counts(normalize=True)
                
                strata_comparison = {}
                for stratum in orig_strata.index:
                    orig_prop = orig_strata[stratum]
                    sample_prop = sample_strata.get(stratum, 0)
                    strata_comparison[str(stratum)] = {
                        'original_proportion': float(orig_prop),
                        'sample_proportion': float(sample_prop),
                        'difference': float(abs(orig_prop - sample_prop))
                    }
                
                report['stratification_analysis'] = {
                    'stratify_column': stratify_column,
                    'strata_comparison': strata_comparison,
                    'max_proportion_difference': max(s['difference'] for s in strata_comparison.values()) if strata_comparison else 0
                }
            
            # Overall quality assessment
            quality_score = self._calculate_sampling_quality(report)
            report['quality_assessment'] = {
                'overall_score': quality_score,
                'quality_level': 'excellent' if quality_score > 0.9 else 'good' if quality_score > 0.7 else 'fair' if quality_score > 0.5 else 'poor'
            }
            
        except Exception as e:
            logger.error("Failed to generate sampling report", error=str(e))
            report['error'] = f"Report generation failed: {str(e)}"
        
        return report
    
    def _calculate_sampling_quality(self, report: dict) -> float:
        """Calculate an overall quality score for the sampling."""
        scores = []
        
        # Score based on numeric column representativeness
        numeric_analysis = report.get('representativeness_analysis', {}).get('numeric_columns', {})
        for col_stats in numeric_analysis.values():
            if col_stats.get('mean_difference') is not None:
                # Lower difference is better, score inversely
                mean_diff = col_stats['mean_difference']
                # Normalize assuming differences > 1 are poor
                score = max(0, 1 - min(mean_diff, 1))
                scores.append(score)
        
        # Score based on categorical distributions
        categorical_analysis = report.get('representativeness_analysis', {}).get('categorical_columns', {})
        for col_stats in categorical_analysis.values():
            avg_diff = np.mean([cat['difference'] for cat in col_stats.values()])
            # Lower difference is better
            score = max(0, 1 - min(avg_diff, 1))
            scores.append(score)
        
        # Score based on stratification quality (if applicable)
        if 'stratification_analysis' in report:
            max_strata_diff = report['stratification_analysis'].get('max_proportion_difference', 0)
            strata_score = max(0, 1 - min(max_strata_diff, 1))
            scores.append(strata_score)
        
        # Return average score or 1.0 if no scores calculated
        return np.mean(scores) if scores else 1.0
    
    def get_sampling_methods_info(self) -> dict:
        """Get information about available sampling methods."""
        return {
            'random': {
                'description': 'Simple random sampling - each element has equal probability of selection',
                'parameters': ['sample_size', 'replacement', 'seed'],
                'use_cases': ['General purpose', 'When population is homogeneous']
            },
            'systematic': {
                'description': 'Systematic sampling - select every kth element after random start',
                'parameters': ['sample_size', 'seed'],
                'use_cases': ['When data has natural ordering', 'Quality control sampling']
            },
            'stratified': {
                'description': 'Stratified sampling - sample proportionally from each stratum',
                'parameters': ['sample_size', 'stratify_column', 'replacement', 'seed'],
                'use_cases': ['When population has distinct subgroups', 'Ensuring representation of minorities']
            },
            'cluster': {
                'description': 'Cluster sampling - select entire clusters randomly',
                'parameters': ['sample_size', 'cluster_column', 'seed'],
                'use_cases': ['When natural clusters exist', 'Geographically distributed populations']
            },
            'reservoir': {
                'description': 'Reservoir sampling - useful for streaming data or unknown population size',
                'parameters': ['sample_size', 'seed'],
                'use_cases': ['Streaming data', 'Very large datasets', 'Unknown population size']
            }
        }