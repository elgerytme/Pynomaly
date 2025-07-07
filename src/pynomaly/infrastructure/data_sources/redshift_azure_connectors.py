"""Amazon Redshift and Azure enterprise connectors with advanced optimization."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import SecretStr

from .enterprise_connectors import (
    ConnectionConfig,
    DataFormat,
    DataSourceMetadata,
    EnterpriseConnector
)

logger = logging.getLogger(__name__)


class RedshiftConnector(EnterpriseConnector):
    """Amazon Redshift data warehouse connector with advanced SQL optimization."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.redshift_conn = None
    
    async def connect(self) -> bool:
        """Connect to Amazon Redshift."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            async with self._connection_lock:
                # Parse connection string for Redshift
                connection_params = self._parse_redshift_connection_string()
                
                self.redshift_conn = psycopg2.connect(
                    host=connection_params['host'],
                    port=connection_params.get('port', 5439),
                    database=self.config.database,
                    user=self.config.username,
                    password=self.config.password.get_secret_value() if self.config.password else None,
                    sslmode=self.config.ssl_config.get('sslmode', 'require'),
                    connect_timeout=self.config.connection_timeout,
                    cursor_factory=RealDictCursor
                )
                
                # Set query timeout
                self.redshift_conn.cursor().execute(f"SET statement_timeout = {self.config.query_timeout * 1000}")
                
                self.is_connected = True
                logger.info("Successfully connected to Amazon Redshift")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Redshift: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redshift."""
        if self.redshift_conn:
            self.redshift_conn.close()
            self.is_connected = False
            logger.info("Disconnected from Amazon Redshift")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute Redshift query with optimizations."""
        if not self.is_connected:
            await self.connect()
        
        try:
            cursor = self.redshift_conn.cursor()
            
            # Enable result caching if supported
            if self.config.optimization.cache_results:
                cursor.execute("SET enable_result_cache_for_session = on")
            
            # Execute query with parameters
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            # Fetch results
            results = cursor.fetchall()
            
            if results:
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in results], columns=columns)
            else:
                df = pd.DataFrame()
            
            cursor.close()
            return df
            
        except Exception as e:
            logger.error(f"Redshift query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get Redshift table metadata."""
        # Get column information
        columns_query = f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = '{table_name.lower()}'
        AND table_schema = '{self.config.schema.lower()}'
        ORDER BY ordinal_position
        """
        
        columns_df = await self.execute_query(columns_query)
        column_info = dict(zip(columns_df['column_name'], columns_df['data_type']))
        
        # Get table statistics from system tables
        stats_query = f"""
        SELECT 
            schemaname,
            tablename,
            size,
            tbl_rows,
            sortkey1,
            distkey
        FROM pg_catalog.svv_table_info 
        WHERE schemaname = '{self.config.schema.lower()}'
        AND tablename = '{table_name.lower()}'
        """
        
        stats_df = await self.execute_query(stats_query)
        
        # Get last vacuum/analyze info
        vacuum_query = f"""
        SELECT MAX(starttime) as last_modified
        FROM stl_vacuum 
        WHERE schema = '{self.config.schema.lower()}'
        AND table = '{table_name.lower()}'
        """
        
        vacuum_df = await self.execute_query(vacuum_query)
        
        return DataSourceMetadata(
            table_name=table_name,
            schema_name=self.config.schema,
            database_name=self.config.database,
            column_info=column_info,
            row_count=int(stats_df['tbl_rows'].iloc[0]) if not stats_df.empty else None,
            size_bytes=int(stats_df['size'].iloc[0]) * 1024 * 1024 if not stats_df.empty else None,  # Size in MB to bytes
            last_modified=vacuum_df['last_modified'].iloc[0] if not vacuum_df.empty else None,
            data_format=DataFormat.PARQUET  # Redshift uses columnar storage
        )
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List Redshift tables."""
        schema_name = schema or self.config.schema
        
        query = f"""
        SELECT tablename 
        FROM pg_catalog.pg_tables 
        WHERE schemaname = '{schema_name.lower()}'
        ORDER BY tablename
        """
        
        df = await self.execute_query(query)
        return df['tablename'].tolist()
    
    async def test_connection(self) -> bool:
        """Test Redshift connection."""
        try:
            test_query = "SELECT version()"
            await self.execute_query(test_query)
            return True
        except Exception as e:
            logger.error(f"Redshift connection test failed: {e}")
            return False
    
    def _parse_redshift_connection_string(self) -> Dict[str, Any]:
        """Parse Redshift connection string."""
        # Expected format: host:port or host
        if ':' in self.config.connection_string:
            host, port = self.config.connection_string.split(':')
            return {'host': host, 'port': int(port)}
        else:
            return {'host': self.config.connection_string}
    
    def _add_optimization_hints(self, query: str) -> str:
        """Add Redshift-specific optimization hints."""
        optimizations = []
        
        # Add distribution and sort key optimizations
        if self.config.optimization.enable_pushdown:
            # Redshift automatically uses zone maps and block skipping
            pass
        
        if self.config.optimization.use_column_pruning:
            # Add hint to use only necessary columns
            optimizations.append("-- Use column pruning")
        
        if self.config.optimization.parallel_degree > 1:
            # Redshift automatically parallelizes queries
            pass
        
        return query


class AzureSynapseConnector(EnterpriseConnector):
    """Azure Synapse Analytics connector with dedicated SQL pool optimization."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.synapse_conn = None
    
    async def connect(self) -> bool:
        """Connect to Azure Synapse."""
        try:
            import pyodbc
            
            async with self._connection_lock:
                # Build connection string for Azure Synapse
                connection_string = self._build_synapse_connection_string()
                
                self.synapse_conn = pyodbc.connect(
                    connection_string,
                    timeout=self.config.connection_timeout
                )
                
                # Set query timeout
                self.synapse_conn.timeout = self.config.query_timeout
                
                self.is_connected = True
                logger.info("Successfully connected to Azure Synapse")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Azure Synapse: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Azure Synapse."""
        if self.synapse_conn:
            self.synapse_conn.close()
            self.is_connected = False
            logger.info("Disconnected from Azure Synapse")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute Azure Synapse query with optimizations."""
        if not self.is_connected:
            await self.connect()
        
        try:
            cursor = self.synapse_conn.cursor()
            
            # Set session-level optimizations
            if self.config.optimization.cache_results:
                cursor.execute("SET RESULT_SET_CACHING ON")
            
            # Execute query
            if parameters:
                # Convert parameters to pyodbc format
                param_markers = ['?' for _ in parameters]
                formatted_query = query.format(*param_markers)
                cursor.execute(formatted_query, list(parameters.values()))
            else:
                cursor.execute(query)
            
            # Fetch results
            results = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            
            # Convert to DataFrame
            if results:
                df = pd.DataFrame([list(row) for row in results], columns=columns)
            else:
                df = pd.DataFrame(columns=columns)
            
            cursor.close()
            return df
            
        except Exception as e:
            logger.error(f"Azure Synapse query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get Azure Synapse table metadata."""
        # Get column information
        columns_query = f"""
        SELECT 
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = '{table_name}'
        AND TABLE_SCHEMA = '{self.config.schema}'
        ORDER BY ORDINAL_POSITION
        """
        
        columns_df = await self.execute_query(columns_query)
        column_info = dict(zip(columns_df['COLUMN_NAME'], columns_df['DATA_TYPE']))
        
        # Get table statistics
        stats_query = f"""
        SELECT 
            SUM(row_count) as total_rows,
            SUM(reserved_page_count) * 8192 as size_bytes
        FROM sys.dm_pdw_nodes_db_partition_stats ps
        INNER JOIN sys.tables t ON ps.object_id = t.object_id
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE t.name = '{table_name}'
        AND s.name = '{self.config.schema}'
        """
        
        stats_df = await self.execute_query(stats_query)
        
        # Get last modified time from sys.tables
        modified_query = f"""
        SELECT modify_date
        FROM sys.tables t
        INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
        WHERE t.name = '{table_name}'
        AND s.name = '{self.config.schema}'
        """
        
        modified_df = await self.execute_query(modified_query)
        
        return DataSourceMetadata(
            table_name=table_name,
            schema_name=self.config.schema,
            database_name=self.config.database,
            column_info=column_info,
            row_count=int(stats_df['total_rows'].iloc[0]) if not stats_df.empty else None,
            size_bytes=int(stats_df['size_bytes'].iloc[0]) if not stats_df.empty else None,
            last_modified=modified_df['modify_date'].iloc[0] if not modified_df.empty else None,
            data_format=DataFormat.PARQUET  # Synapse uses columnstore indexes
        )
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List Azure Synapse tables."""
        schema_name = schema or self.config.schema
        
        query = f"""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = '{schema_name}'
        AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        
        df = await self.execute_query(query)
        return df['TABLE_NAME'].tolist()
    
    async def test_connection(self) -> bool:
        """Test Azure Synapse connection."""
        try:
            test_query = "SELECT @@VERSION"
            await self.execute_query(test_query)
            return True
        except Exception as e:
            logger.error(f"Azure Synapse connection test failed: {e}")
            return False
    
    def _build_synapse_connection_string(self) -> str:
        """Build Azure Synapse connection string."""
        # Build ODBC connection string
        connection_parts = [
            "DRIVER={ODBC Driver 17 for SQL Server}",
            f"SERVER={self.config.connection_string}",
            f"DATABASE={self.config.database}",
            f"UID={self.config.username}",
            f"PWD={self.config.password.get_secret_value() if self.config.password else ''}",
            "Encrypt=yes",
            "TrustServerCertificate=no",
            "Connection Timeout=30"
        ]
        
        return ";".join(connection_parts)
    
    def _add_optimization_hints(self, query: str) -> str:
        """Add Azure Synapse-specific optimization hints."""
        optimizations = []
        
        if self.config.optimization.enable_pushdown:
            optimizations.append("OPTION (LABEL = 'Pynomaly Anomaly Detection Query')")
        
        if self.config.optimization.parallel_degree > 1:
            optimizations.append(f"OPTION (MAXDOP {self.config.optimization.parallel_degree})")
        
        if optimizations:
            query += " " + " ".join(optimizations)
        
        return query


class AzureDataLakeConnector(EnterpriseConnector):
    """Azure Data Lake Storage connector with Delta Lake support."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.adls_client = None
        self.spark_session = None
    
    async def connect(self) -> bool:
        """Connect to Azure Data Lake."""
        try:
            from azure.storage.filedatalake import DataLakeServiceClient
            from pyspark.sql import SparkSession
            
            async with self._connection_lock:
                # Initialize ADLS client
                if self.config.ssl_config.get('account_key'):
                    self.adls_client = DataLakeServiceClient(
                        account_url=f"https://{self.config.connection_string}.dfs.core.windows.net",
                        credential=self.config.ssl_config['account_key']
                    )
                elif self.config.ssl_config.get('sas_token'):
                    self.adls_client = DataLakeServiceClient(
                        account_url=f"https://{self.config.connection_string}.dfs.core.windows.net",
                        credential=self.config.ssl_config['sas_token']
                    )
                else:
                    # Use default Azure credential
                    from azure.identity import DefaultAzureCredential
                    credential = DefaultAzureCredential()
                    self.adls_client = DataLakeServiceClient(
                        account_url=f"https://{self.config.connection_string}.dfs.core.windows.net",
                        credential=credential
                    )
                
                # Initialize Spark session for Delta Lake operations
                self.spark_session = SparkSession.builder \
                    .appName("Pynomaly-ADLS-Connector") \
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                    .getOrCreate()
                
                self.is_connected = True
                logger.info("Successfully connected to Azure Data Lake")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Azure Data Lake: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Azure Data Lake."""
        if self.spark_session:
            self.spark_session.stop()
        self.adls_client = None
        self.is_connected = False
        logger.info("Disconnected from Azure Data Lake")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute query on Delta Lake tables in ADLS."""
        if not self.is_connected:
            await self.connect()
        
        try:
            # For ADLS, we primarily work with Delta Lake through Spark
            if parameters:
                formatted_query = query.format(**parameters)
            else:
                formatted_query = query
            
            # Execute Spark SQL on Delta tables
            spark_df = self.spark_session.sql(formatted_query)
            pandas_df = spark_df.toPandas()
            
            return pandas_df
            
        except Exception as e:
            logger.error(f"Azure Data Lake query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get Azure Data Lake table metadata."""
        try:
            # For Delta Lake tables
            detail_query = f"DESCRIBE DETAIL delta.`/mnt/delta/{table_name}`"
            detail_df = await self.execute_query(detail_query)
            
            # Get column information
            describe_query = f"DESCRIBE delta.`/mnt/delta/{table_name}`"
            describe_df = await self.execute_query(describe_query)
            
            column_info = {}
            for _, row in describe_df.iterrows():
                if row['col_name'] and not row['col_name'].startswith('#'):
                    column_info[row['col_name']] = row['data_type']
            
            return DataSourceMetadata(
                table_name=table_name,
                column_info=column_info,
                size_bytes=detail_df['sizeInBytes'].iloc[0] if not detail_df.empty else None,
                last_modified=pd.to_datetime(detail_df['lastModified'].iloc[0]) if not detail_df.empty else None,
                data_format=DataFormat.DELTA
            )
            
        except Exception as e:
            logger.error(f"Failed to get ADLS table metadata: {e}")
            raise
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List Delta Lake tables in ADLS."""
        try:
            # List files in the Delta Lake directory
            file_system_client = self.adls_client.get_file_system_client("delta")
            paths = file_system_client.get_paths()
            
            tables = []
            for path in paths:
                if path.is_directory and not path.name.startswith('.'):
                    tables.append(path.name)
            
            return tables
            
        except Exception as e:
            logger.error(f"Failed to list ADLS tables: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test Azure Data Lake connection."""
        try:
            # Test by listing file systems
            file_systems = self.adls_client.list_file_systems()
            list(file_systems)  # Consume iterator to test connection
            return True
        except Exception as e:
            logger.error(f"Azure Data Lake connection test failed: {e}")
            return False


class AWSS3Connector(EnterpriseConnector):
    """AWS S3 connector with Athena query support."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.s3_client = None
        self.athena_client = None
    
    async def connect(self) -> bool:
        """Connect to AWS S3 and Athena."""
        try:
            import boto3
            
            async with self._connection_lock:
                session = boto3.Session(
                    aws_access_key_id=self.config.username,
                    aws_secret_access_key=self.config.password.get_secret_value() if self.config.password else None,
                    region_name=self.config.region or 'us-east-1'
                )
                
                self.s3_client = session.client('s3')
                self.athena_client = session.client('athena')
                
                self.is_connected = True
                logger.info("Successfully connected to AWS S3/Athena")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to AWS S3/Athena: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from AWS S3/Athena."""
        self.s3_client = None
        self.athena_client = None
        self.is_connected = False
        logger.info("Disconnected from AWS S3/Athena")
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute Athena query on S3 data."""
        if not self.is_connected:
            await self.connect()
        
        try:
            # Format query with parameters
            if parameters:
                formatted_query = query.format(**parameters)
            else:
                formatted_query = query
            
            # Execute Athena query
            response = self.athena_client.start_query_execution(
                QueryString=formatted_query,
                QueryExecutionContext={'Database': self.config.database},
                ResultConfiguration={
                    'OutputLocation': f's3://{self.config.ssl_config.get("results_bucket", "athena-results")}/'
                }
            )
            
            query_execution_id = response['QueryExecutionId']
            
            # Wait for query completion
            while True:
                await asyncio.sleep(1)
                status = self.athena_client.get_query_execution(QueryExecutionId=query_execution_id)
                state = status['QueryExecution']['Status']['State']
                
                if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
            
            if state == 'SUCCEEDED':
                # Get results
                results = self.athena_client.get_query_results(QueryExecutionId=query_execution_id)
                
                # Convert to DataFrame
                rows = results['ResultSet']['Rows']
                if rows:
                    columns = [col['VarCharValue'] for col in rows[0]['Data']]
                    data = []
                    for row in rows[1:]:  # Skip header
                        data.append([col.get('VarCharValue', '') for col in row['Data']])
                    
                    df = pd.DataFrame(data, columns=columns)
                else:
                    df = pd.DataFrame()
                
                return df
            else:
                raise Exception(f"Athena query failed with state: {state}")
            
        except Exception as e:
            logger.error(f"AWS S3/Athena query execution failed: {e}")
            raise
    
    async def get_table_metadata(self, table_name: str) -> DataSourceMetadata:
        """Get S3/Athena table metadata."""
        try:
            # Get table metadata from Glue catalog
            import boto3
            glue_client = boto3.client('glue', region_name=self.config.region)
            
            response = glue_client.get_table(
                DatabaseName=self.config.database,
                Name=table_name
            )
            
            table = response['Table']
            
            # Extract column information
            column_info = {}
            for column in table['StorageDescriptor']['Columns']:
                column_info[column['Name']] = column['Type']
            
            return DataSourceMetadata(
                table_name=table_name,
                database_name=self.config.database,
                column_info=column_info,
                data_format=DataFormat.PARQUET if 'parquet' in table['StorageDescriptor']['InputFormat'].lower() else DataFormat.JSON
            )
            
        except Exception as e:
            logger.error(f"Failed to get S3 table metadata: {e}")
            raise
    
    async def list_tables(self, schema: Optional[str] = None) -> List[str]:
        """List S3/Athena tables."""
        try:
            import boto3
            glue_client = boto3.client('glue', region_name=self.config.region)
            
            response = glue_client.get_tables(DatabaseName=self.config.database)
            return [table['Name'] for table in response['TableList']]
            
        except Exception as e:
            logger.error(f"Failed to list S3 tables: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test AWS S3/Athena connection."""
        try:
            # Test S3 connection
            self.s3_client.list_buckets()
            
            # Test Athena connection
            self.athena_client.list_data_catalogs()
            
            return True
        except Exception as e:
            logger.error(f"AWS S3/Athena connection test failed: {e}")
            return False