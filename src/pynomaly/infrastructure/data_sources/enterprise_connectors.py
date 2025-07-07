"""Enterprise data lake connectors for major cloud data platforms with optimized SQL pushdown."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)


class ConnectorType(str, Enum):
    """Types of enterprise data connectors."""
    
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    AZURE_SYNAPSE = "azure_synapse"
    AZURE_DATA_LAKE = "azure_data_lake"
    AWS_S3 = "aws_s3"
    SPARK = "spark"


class CompressionType(str, Enum):
    """Data compression types."""
    
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


class DataFormat(str, Enum):
    """Supported data formats."""
    
    PARQUET = "parquet"
    DELTA = "delta"
    AVRO = "avro"
    ORC = "orc"
    CSV = "csv"
    JSON = "json"
    ARROW = "arrow"


@dataclass
class QueryOptimization:
    """Query optimization settings."""
    
    enable_pushdown: bool = True
    use_column_pruning: bool = True
    enable_predicate_pushdown: bool = True
    use_partition_pruning: bool = True
    enable_vectorized_execution: bool = True
    batch_size: int = 10000
    parallel_degree: int = 4
    cache_results: bool = True
    use_materialized_views: bool = False


@dataclass
class ConnectionConfig:
    """Base configuration for enterprise connectors."""
    
    connector_type: ConnectorType
    connection_string: str
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    warehouse: Optional[str] = None
    role: Optional[str] = None
    region: Optional[str] = None
    ssl_config: Dict[str, Any] = field(default_factory=dict)
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 300
    retry_count: int = 3
    optimization: QueryOptimization = field(default_factory=QueryOptimization)


@dataclass
class DataSourceMetadata:
    """Metadata about data source."""
    
    table_name: str
    schema_name: Optional[str] = None
    database_name: Optional[str] = None
    column_info: Dict[str, str] = field(default_factory=dict)
    partition_columns: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    data_format: Optional[DataFormat] = None
    compression: Optional[CompressionType] = None


class EnterpriseConnector(ABC):
    """Abstract base class for enterprise data connectors."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.connection_pool = []
        self._connection_lock = asyncio.Lock()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute SQL query and return results."""
        pass
    
    @abstractmethod
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get metadata about a table."""
        pass
    
    @abstractmethod
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List available tables."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection health."""
        pass
    
    async def get_anomaly_data(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
        optimize_query: bool = True
    ) -> pd.DataFrame:
        """Get data optimized for anomaly detection."""
        # Build optimized query
        query = self._build_optimized_query(
            table_name, columns, where_clause, limit, optimize_query
        )
        
        return await self.execute_query(query)
    
    def _build_optimized_query(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
        optimize: bool = True
    ) -> str:
        """Build SQL query with optimizations."""
        # Column selection
        if columns:
            column_list = ", ".join(columns)
        else:
            column_list = "*"
        
        # Base query
        query = f"SELECT {column_list} FROM {table_name}"
        
        # Add WHERE clause
        if where_clause:
            query += f" WHERE {where_clause}"
        
        # Add optimizations if enabled
        if optimize and self.config.optimization.enable_pushdown:
            # Add hints for query optimization (varies by platform)
            query = self._add_optimization_hints(query)
        
        # Add LIMIT
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    
    def _add_optimization_hints(self, query: str) -> str:
        """Add platform-specific optimization hints."""
        # Default implementation - override in specific connectors
        return query


class SnowflakeConnector(EnterpriseConnector):
    """Snowflake data warehouse connector with SQL pushdown optimization."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.snowflake_conn = None
    
    async def connect(self) -> bool:
        """Connect to Snowflake."""
        try:
            # Import snowflake-connector-python
            import snowflake.connector
            from snowflake.connector import DictCursor
            
            async with self._connection_lock:
                self.snowflake_conn = snowflake.connector.connect(
                    user=self.config.username,
                    password=self.config.password.get_secret_value() if self.config.password else None,
                    account=self._extract_account_from_connection_string(),
                    warehouse=self.config.warehouse,
                    database=self.config.database,
                    schema=self.config.schema,
                    role=self.config.role,
                    login_timeout=self.config.connection_timeout,
                    network_timeout=self.config.query_timeout
                )
                
                self.is_connected = True
                logger.info("Successfully connected to Snowflake")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Snowflake."""
        if self.snowflake_conn:
            self.snowflake_conn.close()
            self.is_connected = False
            logger.info("Disconnected from Snowflake")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute Snowflake query with optimizations."""
        if not self.is_connected:
            await self.connect()
        
        try:
            cursor = self.snowflake_conn.cursor()
            
            # Enable query result caching
            if self.config.optimization.cache_results:
                cursor.execute("ALTER SESSION SET USE_CACHED_RESULT = TRUE")
            
            # Execute query
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            # Fetch results
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            # Convert to DataFrame
            df = pd.DataFrame(results, columns=columns)
            
            cursor.close()
            return df
            
        except Exception as e:
            logger.error(f"Snowflake query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get Snowflake table metadata."""
        # Get column information
        columns_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = '{table_name.upper()}'
        AND TABLE_SCHEMA = '{self.config.schema.upper()}'
        """
        
        columns_df = await self.execute_query(columns_query)
        column_info = dict(zip(columns_df['COLUMN_NAME'], columns_df['DATA_TYPE']))
        
        # Get table statistics
        stats_query = f"""
        SELECT 
            ROW_COUNT,
            BYTES,
            LAST_ALTERED
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = '{table_name.upper()}'
        AND TABLE_SCHEMA = '{self.config.schema.upper()}'
        """
        
        stats_df = await self.execute_query(stats_query)
        
        return DataSourceMetadata(
            table_name=table_name,
            schema_name=self.config.schema,
            database_name=self.config.database,
            column_info=column_info,
            row_count=stats_df['ROW_COUNT'].iloc[0] if not stats_df.empty else None,
            size_bytes=stats_df['BYTES'].iloc[0] if not stats_df.empty else None,
            last_modified=stats_df['LAST_ALTERED'].iloc[0] if not stats_df.empty else None,
            data_format=DataFormat.PARQUET  # Snowflake uses columnar storage
        )
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List Snowflake tables."""
        schema_name = schema or self.config.schema
        
        query = f"""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{schema_name.upper()}'
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        df = await self.execute_query(query)
        return df['TABLE_NAME'].tolist()
    
    async def test_connection(self) -> bool:
        """Test Snowflake connection."""
        try:
            test_query = "SELECT CURRENT_VERSION()"
            await self.execute_query(test_query)
            return True
        except Exception as e:
            logger.error(f"Snowflake connection test failed: {e}")
            return False
    
    def _extract_account_from_connection_string(self) -> str:
        """Extract Snowflake account from connection string."""
        # Expected format: account.snowflakecomputing.com
        if ".snowflakecomputing.com" in self.config.connection_string:
            return self.config.connection_string.split(".")[0]
        return self.config.connection_string
    
    def _add_optimization_hints(self, query: str) -> str:
        """Add Snowflake-specific optimization hints."""
        optimizations = []
        
        if self.config.optimization.enable_vectorized_execution:
            optimizations.append("/*+ USE_VECTORIZED_EXECUTION */")
        
        if self.config.optimization.parallel_degree > 1:
            optimizations.append(f"/*+ PARALLEL({self.config.optimization.parallel_degree}) */")
        
        if optimizations:
            hints = " ".join(optimizations)
            query = f"{hints}\n{query}"
        
        return query


class DatabricksConnector(EnterpriseConnector):
    """Databricks Delta Lake connector with Spark SQL optimization."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.spark_session = None
    
    async def connect(self) -> bool:
        """Connect to Databricks."""
        try:
            # Import databricks-connect or pyspark
            from pyspark.sql import SparkSession
            
            async with self._connection_lock:
                self.spark_session = SparkSession.builder \
                    .appName("Pynomaly-Databricks-Connector") \
                    .config("spark.databricks.service.address", self.config.connection_string) \
                    .config("spark.databricks.service.token", self.config.password.get_secret_value()) \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                    .getOrCreate()
                
                self.is_connected = True
                logger.info("Successfully connected to Databricks")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Databricks: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Databricks."""
        if self.spark_session:
            self.spark_session.stop()
            self.is_connected = False
            logger.info("Disconnected from Databricks")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute Databricks Spark SQL query."""
        if not self.is_connected:
            await self.connect()
        
        try:
            # Execute Spark SQL query
            if parameters:
                # Format query with parameters (basic implementation)
                formatted_query = query.format(**parameters)
            else:
                formatted_query = query
            
            spark_df = self.spark_session.sql(formatted_query)
            
            # Convert to Pandas DataFrame using Arrow for efficiency
            pandas_df = spark_df.toPandas()
            
            return pandas_df
            
        except Exception as e:
            logger.error(f"Databricks query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get Databricks table metadata."""
        # Get column information
        describe_query = f"DESCRIBE TABLE EXTENDED {table_name}"
        describe_df = await self.execute_query(describe_query)
        
        # Parse column information
        column_info = {}
        for _, row in describe_df.iterrows():
            if row['col_name'] and not row['col_name'].startswith('#'):
                column_info[row['col_name']] = row['data_type']
        
        # Get table details
        detail_query = f"DESCRIBE DETAIL {table_name}"
        detail_df = await self.execute_query(detail_query)
        
        return DataSourceMetadata(
            table_name=table_name,
            column_info=column_info,
            size_bytes=detail_df['sizeInBytes'].iloc[0] if not detail_df.empty else None,
            last_modified=pd.to_datetime(detail_df['lastModified'].iloc[0]) if not detail_df.empty else None,
            data_format=DataFormat.DELTA  # Databricks uses Delta Lake
        )
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List Databricks tables."""
        if schema:
            query = f"SHOW TABLES IN {schema}"
        else:
            query = "SHOW TABLES"
        
        df = await self.execute_query(query)
        return df['tableName'].tolist()
    
    async def test_connection(self) -> bool:
        """Test Databricks connection."""
        try:
            test_query = "SELECT current_timestamp()"
            await self.execute_query(test_query)
            return True
        except Exception as e:
            logger.error(f"Databricks connection test failed: {e}")
            return False
    
    def _add_optimization_hints(self, query: str) -> str:
        """Add Databricks/Spark-specific optimization hints."""
        optimizations = []
        
        if self.config.optimization.enable_vectorized_execution:
            # Spark automatically uses vectorized execution when possible
            pass
        
        if self.config.optimization.use_partition_pruning:
            optimizations.append("/*+ COALESCE(1) */")  # Reduce output partitions
        
        return query


class BigQueryConnector(EnterpriseConnector):
    """Google BigQuery connector with optimized SQL and streaming support."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.client = None
    
    async def connect(self) -> bool:
        """Connect to BigQuery."""
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            async with self._connection_lock:
                # Initialize BigQuery client
                if self.config.ssl_config.get('service_account_path'):
                    credentials = service_account.Credentials.from_service_account_file(
                        self.config.ssl_config['service_account_path']
                    )
                    self.client = bigquery.Client(credentials=credentials)
                else:
                    self.client = bigquery.Client()
                
                self.is_connected = True
                logger.info("Successfully connected to BigQuery")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from BigQuery."""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from BigQuery")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute BigQuery SQL with optimizations."""
        if not self.is_connected:
            await self.connect()
        
        try:
            from google.cloud import bigquery
            
            # Configure query job
            job_config = bigquery.QueryJobConfig()
            
            if self.config.optimization.cache_results:
                job_config.use_query_cache = True
            
            if self.config.optimization.use_materialized_views:
                job_config.use_legacy_sql = False
            
            if parameters:
                # Convert parameters to BigQuery format
                query_parameters = []
                for key, value in parameters.items():
                    param_type = self._get_bigquery_param_type(value)
                    query_parameters.append(
                        bigquery.ScalarQueryParameter(key, param_type, value)
                    )
                job_config.query_parameters = query_parameters
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            
            # Convert results to DataFrame
            df = query_job.to_dataframe()
            
            return df
            
        except Exception as e:
            logger.error(f"BigQuery query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get BigQuery table metadata."""
        # Parse table reference
        if '.' in table_name:
            dataset_id, table_id = table_name.split('.', 1)
        else:
            dataset_id = self.config.schema
            table_id = table_name
        
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        
        # Extract column information
        column_info = {}
        for field in table.schema:
            column_info[field.name] = field.field_type
        
        return DataSourceMetadata(
            table_name=table_id,
            schema_name=dataset_id,
            column_info=column_info,
            row_count=table.num_rows,
            size_bytes=table.num_bytes,
            last_modified=table.modified,
            data_format=DataFormat.PARQUET  # BigQuery uses columnar storage
        )
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List BigQuery tables."""
        dataset_id = schema or self.config.schema
        dataset_ref = self.client.dataset(dataset_id)
        
        tables = list(self.client.list_tables(dataset_ref))
        return [table.table_id for table in tables]
    
    async def test_connection(self) -> bool:
        """Test BigQuery connection."""
        try:
            test_query = "SELECT CURRENT_TIMESTAMP() as current_time"
            await self.execute_query(test_query)
            return True
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            return False
    
    def _get_bigquery_param_type(self, value: Any) -> str:
        """Get BigQuery parameter type from Python value."""
        if isinstance(value, bool):
            return "BOOL"
        elif isinstance(value, int):
            return "INT64"
        elif isinstance(value, float):
            return "FLOAT64"
        elif isinstance(value, str):
            return "STRING"
        elif isinstance(value, datetime):
            return "TIMESTAMP"
        else:
            return "STRING"
    
    def _add_optimization_hints(self, query: str) -> str:
        """Add BigQuery-specific optimization hints."""
        # BigQuery automatically optimizes queries
        # Add partitioning hints if applicable
        if self.config.optimization.use_partition_pruning:
            # This would be context-specific based on table partitioning
            pass
        
        return query


# Factory function to create connectors
def create_enterprise_connector(config: ConnectionConfig) -> EnterpriseConnector:
    """Create enterprise connector based on configuration."""
    
    # Import additional connectors
    from .redshift_azure_connectors import (
        RedshiftConnector,
        AzureSynapseConnector,
        AzureDataLakeConnector,
        AWSS3Connector
    )
    
    connector_map = {
        ConnectorType.SNOWFLAKE: SnowflakeConnector,
        ConnectorType.DATABRICKS: DatabricksConnector,
        ConnectorType.BIGQUERY: BigQueryConnector,
        ConnectorType.REDSHIFT: RedshiftConnector,
        ConnectorType.AZURE_SYNAPSE: AzureSynapseConnector,
        ConnectorType.AZURE_DATA_LAKE: AzureDataLakeConnector,
        ConnectorType.AWS_S3: AWSS3Connector,
    }
    
    connector_class = connector_map.get(config.connector_type)
    if not connector_class:
        raise ValueError(f"Unsupported connector type: {config.connector_type}")
    
    return connector_class(config)


# Connection manager for multiple enterprise sources
class EnterpriseConnectionManager:
    """Manages multiple enterprise data connections."""
    
    def __init__(self):
        self.connectors: Dict[str, EnterpriseConnector] = {}
        self.connection_configs: Dict[str, ConnectionConfig] = {}
    
    async def add_connection(
        self,
        connection_id: str,
        config: ConnectionConfig
    ) -> bool:
        """Add new enterprise connection."""
        try:
            connector = create_enterprise_connector(config)
            success = await connector.connect()
            
            if success:
                self.connectors[connection_id] = connector
                self.connection_configs[connection_id] = config
                logger.info(f"Added enterprise connection: {connection_id}")
                return True
            else:
                logger.error(f"Failed to connect to {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding connection {connection_id}: {e}")
            return False
    
    async def get_connector(self, connection_id: str) -> Optional[EnterpriseConnector]:
        """Get connector by ID."""
        return self.connectors.get(connection_id)
    
    async def list_connections(self) -> List[str]:
        """List all connection IDs."""
        return list(self.connectors.keys())
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test all connections."""
        results = {}
        for connection_id, connector in self.connectors.items():
            results[connection_id] = await connector.test_connection()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all enterprise connections."""
        for connector in self.connectors.values():
            await connector.disconnect()
        
        self.connectors.clear()
        self.connection_configs.clear()
        logger.info("Disconnected all enterprise connections")