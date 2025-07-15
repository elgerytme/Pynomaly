import pandas as pd
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import logging
from .file_adapter import get_file_adapter, detect_file_format
from .database_adapter import get_database_adapter, DatabaseAdapter

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Data source type enumeration."""
    FILE = "file"
    DATABASE = "database"
    URL = "url"
    STREAM = "stream"


class DataSourceAdapter:
    """Unified adapter for various data sources (files, databases, URLs, etc.)."""
    
    def __init__(self, source_type: DataSourceType, connection_config: Dict[str, Any]):
        self.source_type = source_type
        self.connection_config = connection_config
        self._adapter = None
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load data from the configured source."""
        if self.source_type == DataSourceType.FILE:
            return self._load_from_file(**kwargs)
        elif self.source_type == DataSourceType.DATABASE:
            return self._load_from_database(**kwargs)
        elif self.source_type == DataSourceType.URL:
            return self._load_from_url(**kwargs)
        else:
            raise NotImplementedError(f"Source type {self.source_type} not implemented")
    
    def _load_from_file(self, **kwargs) -> pd.DataFrame:
        """Load data from file source."""
        file_path = self.connection_config.get('path')
        if not file_path:
            raise ValueError("File path is required for file source")
        
        file_options = self.connection_config.get('options', {})
        adapter = get_file_adapter(file_path, file_options)
        return adapter.load(file_path, **kwargs)
    
    def _load_from_database(self, **kwargs) -> pd.DataFrame:
        """Load data from database source."""
        db_type = self.connection_config.get('type')
        if not db_type:
            raise ValueError("Database type is required for database source")
        
        connection_params = self.connection_config.get('connection', {})
        adapter = get_database_adapter(db_type, connection_params)
        
        with adapter:
            if 'table_name' in kwargs:
                return adapter.load_table(**kwargs)
            elif 'query' in kwargs:
                return adapter.load_query(kwargs['query'])
            else:
                raise ValueError("Either 'table_name' or 'query' must be provided for database source")
    
    def _load_from_url(self, **kwargs) -> pd.DataFrame:
        """Load data from URL source."""
        url = self.connection_config.get('url')
        if not url:
            raise ValueError("URL is required for URL source")
        
        # Detect format from URL
        file_format = self.connection_config.get('format')
        if not file_format:
            file_format = detect_file_format(url)
        
        if file_format == 'csv':
            return pd.read_csv(url, **kwargs)
        elif file_format in ['json', 'jsonl']:
            return pd.read_json(url, **kwargs)
        elif file_format == 'parquet':
            return pd.read_parquet(url, **kwargs)
        else:
            # Try to infer format
            try:
                return pd.read_csv(url, **kwargs)
            except Exception:
                try:
                    return pd.read_json(url, **kwargs)
                except Exception:
                    raise ValueError(f"Unable to load data from URL: {url}")
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source."""
        info = {
            'source_type': self.source_type.value,
            'config': self.connection_config
        }
        
        if self.source_type == DataSourceType.FILE:
            file_path = self.connection_config.get('path')
            if file_path:
                adapter = get_file_adapter(file_path)
                info['file_info'] = adapter.get_file_info(file_path)
                info['format'] = detect_file_format(file_path)
        
        elif self.source_type == DataSourceType.DATABASE:
            db_type = self.connection_config.get('type')
            connection_params = self.connection_config.get('connection', {})
            
            try:
                adapter = get_database_adapter(db_type, connection_params)
                with adapter:
                    from .database_adapter import DatabaseProfiler
                    profiler = DatabaseProfiler(adapter)
                    info['database_info'] = profiler.get_table_info()
            except Exception as e:
                info['database_info'] = {'error': str(e)}
        
        return info


class DataSourceFactory:
    """Factory for creating data source adapters."""
    
    @staticmethod
    def create_file_source(file_path: str, options: Optional[Dict[str, Any]] = None) -> DataSourceAdapter:
        """Create a file data source adapter."""
        config = {
            'path': file_path,
            'options': options or {}
        }
        return DataSourceAdapter(DataSourceType.FILE, config)
    
    @staticmethod
    def create_database_source(db_type: str, connection_params: Dict[str, Any]) -> DataSourceAdapter:
        """Create a database data source adapter."""
        config = {
            'type': db_type,
            'connection': connection_params
        }
        return DataSourceAdapter(DataSourceType.DATABASE, config)
    
    @staticmethod
    def create_url_source(url: str, file_format: Optional[str] = None, 
                         options: Optional[Dict[str, Any]] = None) -> DataSourceAdapter:
        """Create a URL data source adapter."""
        config = {
            'url': url,
            'format': file_format,
            'options': options or {}
        }
        return DataSourceAdapter(DataSourceType.URL, config)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> DataSourceAdapter:
        """Create data source adapter from configuration dictionary."""
        source_type = config.get('type')
        if not source_type:
            raise ValueError("Source type is required in configuration")
        
        try:
            source_type_enum = DataSourceType(source_type)
        except ValueError:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return DataSourceAdapter(source_type_enum, config)


class MultiSourceProfiler:
    """Profiler for handling multiple data sources."""
    
    def __init__(self):
        self.sources: List[DataSourceAdapter] = []
    
    def add_source(self, source: DataSourceAdapter) -> None:
        """Add a data source to the profiler."""
        self.sources.append(source)
    
    def add_file_source(self, file_path: str, options: Optional[Dict[str, Any]] = None) -> None:
        """Add a file data source."""
        source = DataSourceFactory.create_file_source(file_path, options)
        self.add_source(source)
    
    def add_database_source(self, db_type: str, connection_params: Dict[str, Any]) -> None:
        """Add a database data source."""
        source = DataSourceFactory.create_database_source(db_type, connection_params)
        self.add_source(source)
    
    def add_url_source(self, url: str, file_format: Optional[str] = None) -> None:
        """Add a URL data source."""
        source = DataSourceFactory.create_url_source(url, file_format)
        self.add_source(source)
    
    def profile_all_sources(self) -> Dict[str, Any]:
        """Profile all configured data sources."""
        results = {}
        
        for i, source in enumerate(self.sources):
            try:
                # Get source info
                source_info = source.get_source_info()
                
                # Load and analyze data
                df = source.load_data()
                
                # Basic profiling information
                profile_data = {
                    'source_info': source_info,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                    'null_counts': df.isnull().sum().to_dict(),
                    'success': True
                }
                
                results[f'source_{i}'] = profile_data
                
            except Exception as e:
                logger.error(f"Failed to profile source {i}: {e}")
                results[f'source_{i}'] = {
                    'source_info': source.get_source_info(),
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def combine_sources(self, join_key: Optional[str] = None) -> pd.DataFrame:
        """Combine data from multiple sources."""
        if not self.sources:
            raise ValueError("No data sources configured")
        
        dataframes = []
        for i, source in enumerate(self.sources):
            try:
                df = source.load_data()
                df['_source_id'] = i  # Add source identifier
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Failed to load source {i}: {e}")
                continue
        
        if not dataframes:
            raise RuntimeError("No data sources could be loaded")
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Combine dataframes
        if join_key:
            # Join on specified key
            combined_df = dataframes[0]
            for df in dataframes[1:]:
                combined_df = pd.merge(combined_df, df, on=join_key, how='outer')
        else:
            # Concatenate vertically
            combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        return combined_df


def create_connection_string(db_type: str, host: str, port: int, database: str, 
                           username: str, password: str) -> str:
    """Create a database connection string."""
    if db_type.lower() in ['postgresql', 'postgres']:
        return f"postgresql://{username}:{password}@{host}:{port}/{database}"
    elif db_type.lower() == 'mysql':
        return f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    elif db_type.lower() == 'sqlite':
        return f"sqlite:///{database}"  # database is the file path for SQLite
    else:
        raise ValueError(f"Unsupported database type for connection string: {db_type}")


def validate_connection_config(source_type: str, config: Dict[str, Any]) -> bool:
    """Validate connection configuration for a data source."""
    if source_type == 'file':
        return 'path' in config
    elif source_type == 'database':
        required_fields = ['type', 'connection']
        return all(field in config for field in required_fields)
    elif source_type == 'url':
        return 'url' in config
    else:
        return False