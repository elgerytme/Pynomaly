"""Data loader implementations with high-performance processing capabilities."""

from .csv_loader import CSVLoader
from .parquet_loader import ParquetLoader

# High-performance data loaders (optional dependencies)
try:
    from .polars_loader import PolarsLoader, load_with_polars, compare_performance
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

try:
    from .arrow_loader import ArrowLoader, load_with_arrow, stream_with_arrow
    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False

try:
    from .spark_loader import SparkLoader, load_with_spark, load_distributed_with_spark, SparkAnomalyDetector
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False

# Base exports (always available)
__all__ = [
    "CSVLoader",
    "ParquetLoader",
]

# Optional high-performance loaders
if _POLARS_AVAILABLE:
    __all__.extend([
        "PolarsLoader",
        "load_with_polars", 
        "compare_performance"
    ])

if _ARROW_AVAILABLE:
    __all__.extend([
        "ArrowLoader",
        "load_with_arrow",
        "stream_with_arrow"
    ])

if _SPARK_AVAILABLE:
    __all__.extend([
        "SparkLoader",
        "load_with_spark",
        "load_distributed_with_spark",
        "SparkAnomalyDetector"
    ])

# Utility functions
def get_available_loaders() -> dict:
    """Get information about available data loaders.
    
    Returns:
        Dictionary with loader availability and capabilities
    """
    return {
        'pandas_based': {
            'csv': True,
            'parquet': True,
            'description': 'Standard pandas-based loaders (always available)'
        },
        'polars': {
            'available': _POLARS_AVAILABLE,
            'formats': ['.csv', '.parquet', '.json', '.xlsx'] if _POLARS_AVAILABLE else [],
            'description': 'High-performance lazy evaluation with Polars'
        },
        'arrow': {
            'available': _ARROW_AVAILABLE,
            'formats': ['.parquet', '.csv', '.json', '.arrow', '.feather'] if _ARROW_AVAILABLE else [],
            'description': 'Native Arrow columnar processing with compute functions'
        },
        'spark': {
            'available': _SPARK_AVAILABLE,
            'formats': ['.parquet', '.csv', '.json'] if _SPARK_AVAILABLE else [],
            'description': 'Distributed big data processing with Apache Spark'
        }
    }


def get_recommended_loader(file_size_mb: float, file_format: str = '.csv') -> str:
    """Recommend the best loader based on file size and format.
    
    Args:
        file_size_mb: File size in megabytes
        file_format: File format (e.g., '.csv', '.parquet')
        
    Returns:
        Recommended loader name
    """
    if file_size_mb < 10:
        # Small files - use pandas-based loaders
        if file_format in ['.csv']:
            return 'CSVLoader'
        elif file_format in ['.parquet', '.pq']:
            return 'ParquetLoader'
    
    elif file_size_mb < 1000:  # 1GB
        # Medium files - use Polars for best performance
        if _POLARS_AVAILABLE:
            return 'PolarsLoader'
        elif _ARROW_AVAILABLE and file_format in ['.parquet', '.pq']:
            return 'ArrowLoader'
        else:
            return 'CSVLoader' if file_format == '.csv' else 'ParquetLoader'
    
    else:
        # Large files - use Spark for distributed processing
        if _SPARK_AVAILABLE:
            return 'SparkLoader'
        elif _POLARS_AVAILABLE:
            return 'PolarsLoader'  # With streaming mode
        elif _ARROW_AVAILABLE:
            return 'ArrowLoader'  # With streaming
        else:
            return 'CSVLoader'  # Will need chunking


async def load_auto(file_path: str, **kwargs):
    """Automatically select and use the best loader for the given file.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional loading parameters
        
    Returns:
        Dataset loaded with the optimal loader
    """
    from pathlib import Path
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file info
    file_size_mb = path.stat().st_size / (1024 * 1024)
    file_format = path.suffix.lower()
    
    # Get recommended loader
    loader_name = get_recommended_loader(file_size_mb, file_format)
    
    # Load with recommended loader
    if loader_name == 'PolarsLoader' and _POLARS_AVAILABLE:
        loader = PolarsLoader()
        return await loader.load(file_path, **kwargs)
    elif loader_name == 'ArrowLoader' and _ARROW_AVAILABLE:
        loader = ArrowLoader()
        return await loader.load(file_path, **kwargs)
    elif loader_name == 'SparkLoader' and _SPARK_AVAILABLE:
        loader = SparkLoader()
        try:
            return await loader.load(file_path, **kwargs)
        finally:
            loader.stop()
    elif file_format == '.csv':
        loader = CSVLoader()
        return await loader.load(file_path, **kwargs)
    elif file_format in ['.parquet', '.pq']:
        loader = ParquetLoader()
        return await loader.load(file_path, **kwargs)
    else:
        raise ValueError(f"No suitable loader found for {file_format}")


# Add auto loader to exports
__all__.append('load_auto')
__all__.append('get_available_loaders')
__all__.append('get_recommended_loader')