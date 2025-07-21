"""
Snowflake Data Platform Connector

Provides integration with Snowflake data warehouse for enterprise
data analytics, anomaly detection, and real-time data processing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from uuid import UUID
import json

from structlog import get_logger
import pandas as pd
import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.pandas_tools import write_pandas
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import asyncio

from ...domain.entities.data_platform import DataPlatformConnection, DataSource, DataQualityResult

logger = get_logger(__name__)


class SnowflakeConnector:
    """
    Snowflake data warehouse connector.
    
    Provides comprehensive integration with Snowflake including
    data ingestion, querying, streaming, and anomaly detection.
    """
    
    def __init__(self, connection: DataPlatformConnection):
        self.connection = connection
        self.snowflake_conn = None
        self.engine = None
        self.logger = logger.bind(connector="snowflake", connection=connection.name)
        
        # Validate connection is for Snowflake
        if connection.platform_type.value != "snowflake":
            raise ValueError(f"Connection {connection.name} is not for Snowflake platform")
        
        self.logger.info("SnowflakeConnector initialized")
    
    async def connect(self) -> bool:
        """Establish connection to Snowflake."""
        self.logger.info("Connecting to Snowflake")
        
        try:
            # Build connection parameters
            conn_params = {
                'account': self.connection.host,
                'user': self.connection.username,
                'database': self.connection.database,
                'schema': self.connection.schema,
                'warehouse': self.connection.warehouse,
                'login_timeout': self.connection.connection_timeout,
                'network_timeout': self.connection.read_timeout
            }
            
            # Add authentication method
            if self.connection.password:
                conn_params['password'] = self.connection.password
            elif self.connection.private_key:
                # Key-pair authentication
                conn_params['private_key'] = self.connection.private_key
                if self.connection.private_key_passphrase:
                    conn_params['private_key_passphrase'] = self.connection.private_key_passphrase
            elif self.connection.oauth_token:
                conn_params['token'] = self.connection.oauth_token
            
            # Add additional properties
            conn_params.update(self.connection.properties)
            
            # Create connection
            self.snowflake_conn = snowflake.connector.connect(**conn_params)
            
            # Create SQLAlchemy engine for pandas integration
            engine_url = URL(
                account=self.connection.host,
                user=self.connection.username,
                password=self.connection.password,
                database=self.connection.database,
                schema=self.connection.schema,
                warehouse=self.connection.warehouse
            )
            self.engine = create_engine(engine_url)
            
            # Test connection
            await self._test_connection()
            
            self.connection.update_status("connected")
            self.logger.info("Successfully connected to Snowflake")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Snowflake: {str(e)}"
            self.logger.error(error_msg)
            self.connection.update_status("error", error_msg)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Snowflake."""
        if self.snowflake_conn:
            try:
                self.snowflake_conn.close()
                self.snowflake_conn = None
                
                if self.engine:
                    self.engine.dispose()
                    self.engine = None
                
                self.connection.update_status("disconnected")
                self.logger.info("Disconnected from Snowflake")
                
            except Exception as e:
                self.logger.error("Error during disconnect", error=str(e))
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_results: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL query on Snowflake."""
        if not self.snowflake_conn:
            raise RuntimeError("Not connected to Snowflake")
        
        self.logger.debug("Executing query", query=query[:100] + "..." if len(query) > 100 else query)
        
        try:
            cursor = self.snowflake_conn.cursor(DictCursor)
            
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            if fetch_results:
                results = cursor.fetchall()
                self.logger.debug("Query executed successfully", rows=len(results))
                return results
            else:
                self.logger.debug("Query executed successfully (no fetch)")
                return None
                
        except Exception as e:
            self.logger.error("Query execution failed", error=str(e), query=query)
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    async def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "append",
        chunk_size: int = 10000
    ) -> Dict[str, Any]:
        """Load pandas DataFrame to Snowflake table."""
        if not self.engine:
            raise RuntimeError("SQLAlchemy engine not available")
        
        self.logger.info("Loading DataFrame to Snowflake", 
                        table=table_name, rows=len(df), schema=schema)
        
        try:
            start_time = datetime.utcnow()
            
            # Use schema from connection if not provided
            target_schema = schema or self.connection.schema
            
            # Load data using write_pandas
            success, nchunks, nrows, _ = write_pandas(
                conn=self.snowflake_conn,
                df=df,
                table_name=table_name.upper(),
                database=self.connection.database.upper(),
                schema=target_schema.upper(),
                chunk_size=chunk_size,
                compression='gzip',
                on_error='abort',
                parallel=4,
                auto_create_table=True,
                overwrite=(if_exists == "replace")
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            if success:
                self.logger.info("DataFrame loaded successfully", 
                               table=table_name, rows=nrows, chunks=nchunks,
                               duration=duration)
                
                return {
                    "success": True,
                    "rows_loaded": nrows,
                    "chunks": nchunks,
                    "duration_seconds": duration,
                    "table": f"{self.connection.database}.{target_schema}.{table_name}".upper()
                }
            else:
                raise RuntimeError("Failed to load DataFrame to Snowflake")
                
        except Exception as e:
            self.logger.error("Failed to load DataFrame", error=str(e))
            raise
    
    async def stream_data(
        self,
        data_source: DataSource,
        batch_size: int = 1000,
        max_batches: Optional[int] = None
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream data from Snowflake table or view."""
        if not self.snowflake_conn:
            raise RuntimeError("Not connected to Snowflake")
        
        self.logger.info("Starting data stream", source=data_source.source_path, 
                        batch_size=batch_size)
        
        try:
            # Build streaming query
            query = self._build_streaming_query(data_source, batch_size)
            
            cursor = self.snowflake_conn.cursor(DictCursor)
            cursor.execute(query)
            
            batch_count = 0
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                
                batch_count += 1
                self.logger.debug("Streaming batch", batch=batch_count, size=len(batch))
                
                yield batch
                
                # Check max batches limit
                if max_batches and batch_count >= max_batches:
                    break
            
            self.logger.info("Data streaming completed", batches=batch_count)
            
        except Exception as e:
            self.logger.error("Data streaming failed", error=str(e))
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
    
    async def create_anomaly_detection_table(
        self,
        table_name: str,
        schema: Optional[str] = None
    ) -> bool:
        """Create table optimized for anomaly detection data."""
        self.logger.info("Creating anomaly detection table", table=table_name)
        
        try:
            target_schema = schema or self.connection.schema
            full_table_name = f"{self.connection.database}.{target_schema}.{table_name}".upper()
            
            create_sql = f"""
            CREATE OR REPLACE TABLE {full_table_name} (
                id VARCHAR(36) NOT NULL,
                tenant_id VARCHAR(36) NOT NULL,
                timestamp TIMESTAMP_NTZ NOT NULL,
                data_source VARCHAR(255) NOT NULL,
                feature_values VARIANT NOT NULL,
                anomaly_score FLOAT,
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_type VARCHAR(100),
                severity VARCHAR(20),
                metadata VARIANT,
                processed_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            CLUSTER BY (tenant_id, timestamp)
            """
            
            await self.execute_query(create_sql, fetch_results=False)
            
            # Create indexes for performance
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_tenant_timestamp 
            ON {full_table_name} (tenant_id, timestamp)
            """
            await self.execute_query(index_sql, fetch_results=False)
            
            self.logger.info("Anomaly detection table created successfully", table=full_table_name)
            return True
            
        except Exception as e:
            self.logger.error("Failed to create anomaly detection table", error=str(e))
            return False
    
    async def store_anomaly_results(
        self,
        results: List[Dict[str, Any]],
        table_name: str = "ANOMALY_RESULTS",
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store anomaly detection results in Snowflake."""
        if not results:
            return {"success": True, "rows_stored": 0}
        
        self.logger.info("Storing anomaly results", count=len(results))
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Ensure required columns
            if 'id' not in df.columns:
                df['id'] = [str(uuid4()) for _ in range(len(df))]
            if 'created_at' not in df.columns:
                df['created_at'] = datetime.utcnow()
            
            # Convert complex fields to JSON
            for col in ['feature_values', 'metadata']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
            
            # Load to Snowflake
            result = await self.load_data_from_dataframe(
                df=df,
                table_name=table_name,
                schema=schema,
                if_exists="append"
            )
            
            self.logger.info("Anomaly results stored successfully", rows=result.get("rows_loaded", 0))
            return result
            
        except Exception as e:
            self.logger.error("Failed to store anomaly results", error=str(e))
            raise
    
    async def query_anomalies(
        self,
        tenant_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        data_source: Optional[str] = None,
        anomaly_types: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        limit: int = 1000,
        table_name: str = "ANOMALY_RESULTS"
    ) -> List[Dict[str, Any]]:
        """Query anomaly detection results from Snowflake."""
        self.logger.info("Querying anomalies", tenant_id=tenant_id, limit=limit)
        
        try:
            # Build query
            schema = self.connection.schema
            full_table_name = f"{self.connection.database}.{schema}.{table_name}".upper()
            
            conditions = ["tenant_id = %s"]
            params = [str(tenant_id)]
            
            if start_time:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= %s")
                params.append(end_time)
            
            if data_source:
                conditions.append("data_source = %s")
                params.append(data_source)
            
            if anomaly_types:
                conditions.append(f"anomaly_type IN ({','.join(['%s'] * len(anomaly_types))})")
                params.extend(anomaly_types)
            
            if min_score:
                conditions.append("anomaly_score >= %s")
                params.append(min_score)
            
            query = f"""
            SELECT 
                id,
                tenant_id,
                timestamp,
                data_source,
                feature_values,
                anomaly_score,
                is_anomaly,
                anomaly_type,
                severity,
                metadata,
                processed_at,
                created_at
            FROM {full_table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            
            results = await self.execute_query(query, dict(zip(range(len(params)), params)))
            
            # Parse JSON fields
            for result in results or []:
                for field in ['feature_values', 'metadata']:
                    if result.get(field):
                        try:
                            result[field] = json.loads(result[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
            
            self.logger.info("Anomaly query completed", results_count=len(results or []))
            return results or []
            
        except Exception as e:
            self.logger.error("Failed to query anomalies", error=str(e))
            raise
    
    async def run_data_quality_checks(
        self,
        data_source: DataSource,
        checks: List[Dict[str, Any]]
    ) -> List[DataQualityResult]:
        """Run data quality checks on Snowflake data."""
        self.logger.info("Running data quality checks", source=data_source.source_path, 
                        checks_count=len(checks))
        
        results = []
        
        try:
            for check in checks:
                start_time = datetime.utcnow()
                
                check_result = await self._execute_quality_check(data_source, check)
                
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds()
                
                # Create quality result
                quality_result = DataQualityResult(
                    data_source_id=data_source.id,
                    tenant_id=data_source.tenant_id,
                    check_name=check.get("name", "unnamed_check"),
                    check_type=check.get("type", "custom"),
                    check_config=check,
                    passed=check_result.get("passed", False),
                    score=check_result.get("score"),
                    total_records=check_result.get("total_records", 0),
                    passed_records=check_result.get("passed_records", 0),
                    failed_records=check_result.get("failed_records", 0),
                    failure_reasons=check_result.get("failure_reasons", []),
                    sample_failures=check_result.get("sample_failures", []),
                    execution_time_seconds=execution_time
                )
                
                results.append(quality_result)
        
            self.logger.info("Data quality checks completed", passed=sum(1 for r in results if r.passed))
            return results
            
        except Exception as e:
            self.logger.error("Data quality checks failed", error=str(e))
            raise
    
    async def get_table_statistics(
        self,
        table_name: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive table statistics from Snowflake."""
        self.logger.info("Getting table statistics", table=table_name)
        
        try:
            target_schema = schema or self.connection.schema
            full_table_name = f"{self.connection.database}.{target_schema}.{table_name}".upper()
            
            # Get basic table info
            info_query = f"""
            SELECT 
                table_catalog,
                table_schema,
                table_name,
                table_type,
                row_count,
                bytes,
                clustering_key
            FROM information_schema.tables
            WHERE table_name = '{table_name.upper()}'
            AND table_schema = '{target_schema.upper()}'
            """
            
            table_info = await self.execute_query(info_query)
            
            if not table_info:
                return {"error": "Table not found"}
            
            info = table_info[0]
            
            # Get column statistics
            columns_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_name = '{table_name.upper()}'
            AND table_schema = '{target_schema.upper()}'
            ORDER BY ordinal_position
            """
            
            columns_info = await self.execute_query(columns_query)
            
            return {
                "table_name": info["TABLE_NAME"],
                "schema": info["TABLE_SCHEMA"],
                "database": info["TABLE_CATALOG"],
                "table_type": info["TABLE_TYPE"],
                "row_count": info["ROW_COUNT"],
                "size_bytes": info["BYTES"],
                "clustering_key": info["CLUSTERING_KEY"],
                "columns": columns_info,
                "column_count": len(columns_info or [])
            }
            
        except Exception as e:
            self.logger.error("Failed to get table statistics", error=str(e))
            raise
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test Snowflake connection."""
        test_query = "SELECT CURRENT_VERSION()"
        result = await self.execute_query(test_query)
        if result:
            version = result[0].get("CURRENT_VERSION()")
            self.logger.info("Connection test successful", snowflake_version=version)
    
    def _build_streaming_query(self, data_source: DataSource, batch_size: int) -> str:
        """Build optimized query for data streaming."""
        base_query = f"SELECT * FROM {data_source.source_path}"
        
        # Add incremental filtering if configured
        if data_source.incremental_column:
            if data_source.last_refresh:
                base_query += f" WHERE {data_source.incremental_column} > '{data_source.last_refresh}'"
        
        # Add ordering for consistent streaming
        if data_source.incremental_column:
            base_query += f" ORDER BY {data_source.incremental_column}"
        
        return base_query
    
    async def _execute_quality_check(
        self,
        data_source: DataSource,
        check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute individual data quality check."""
        check_type = check.get("type", "custom")
        
        if check_type == "null_check":
            return await self._run_null_check(data_source, check)
        elif check_type == "unique_check":
            return await self._run_unique_check(data_source, check)
        elif check_type == "range_check":
            return await self._run_range_check(data_source, check)
        elif check_type == "custom_sql":
            return await self._run_custom_sql_check(data_source, check)
        else:
            return {"passed": False, "error": f"Unknown check type: {check_type}"}
    
    async def _run_null_check(
        self,
        data_source: DataSource,
        check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run null value check."""
        column = check.get("column")
        if not column:
            return {"passed": False, "error": "No column specified for null check"}
        
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT({column}) as non_null_records,
            COUNT(*) - COUNT({column}) as null_records
        FROM {data_source.source_path}
        """
        
        result = await self.execute_query(query)
        if not result:
            return {"passed": False, "error": "Query failed"}
        
        stats = result[0]
        total_records = stats["TOTAL_RECORDS"]
        null_records = stats["NULL_RECORDS"]
        non_null_records = stats["NON_NULL_RECORDS"]
        
        # Check threshold
        null_threshold = check.get("max_null_percentage", 0)
        null_percentage = (null_records / max(total_records, 1)) * 100
        
        passed = null_percentage <= null_threshold
        
        return {
            "passed": passed,
            "total_records": total_records,
            "passed_records": non_null_records,
            "failed_records": null_records,
            "score": 1.0 - (null_percentage / 100),
            "failure_reasons": [f"Null percentage {null_percentage:.2f}% exceeds threshold {null_threshold}%"] if not passed else []
        }
    
    async def _run_unique_check(
        self,
        data_source: DataSource,
        check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run uniqueness check."""
        columns = check.get("columns", [])
        if not columns:
            return {"passed": False, "error": "No columns specified for unique check"}
        
        column_list = ", ".join(columns)
        
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT {column_list}) as unique_records
        FROM {data_source.source_path}
        """
        
        result = await self.execute_query(query)
        if not result:
            return {"passed": False, "error": "Query failed"}
        
        stats = result[0]
        total_records = stats["TOTAL_RECORDS"]
        unique_records = stats["UNIQUE_RECORDS"]
        duplicate_records = total_records - unique_records
        
        passed = duplicate_records == 0
        
        return {
            "passed": passed,
            "total_records": total_records,
            "passed_records": unique_records,
            "failed_records": duplicate_records,
            "score": unique_records / max(total_records, 1),
            "failure_reasons": [f"Found {duplicate_records} duplicate records"] if not passed else []
        }
    
    async def _run_range_check(
        self,
        data_source: DataSource,
        check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run value range check."""
        column = check.get("column")
        min_value = check.get("min_value")
        max_value = check.get("max_value")
        
        if not column:
            return {"passed": False, "error": "No column specified for range check"}
        
        conditions = []
        if min_value is not None:
            conditions.append(f"{column} >= {min_value}")
        if max_value is not None:
            conditions.append(f"{column} <= {max_value}")
        
        if not conditions:
            return {"passed": False, "error": "No range constraints specified"}
        
        range_condition = " AND ".join(conditions)
        
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN {range_condition} THEN 1 END) as in_range_records
        FROM {data_source.source_path}
        WHERE {column} IS NOT NULL
        """
        
        result = await self.execute_query(query)
        if not result:
            return {"passed": False, "error": "Query failed"}
        
        stats = result[0]
        total_records = stats["TOTAL_RECORDS"]
        in_range_records = stats["IN_RANGE_RECORDS"]
        out_of_range_records = total_records - in_range_records
        
        # Check threshold
        threshold = check.get("pass_threshold", 100)
        pass_percentage = (in_range_records / max(total_records, 1)) * 100
        
        passed = pass_percentage >= threshold
        
        return {
            "passed": passed,
            "total_records": total_records,
            "passed_records": in_range_records,
            "failed_records": out_of_range_records,
            "score": pass_percentage / 100,
            "failure_reasons": [f"Pass rate {pass_percentage:.2f}% below threshold {threshold}%"] if not passed else []
        }
    
    async def _run_custom_sql_check(
        self,
        data_source: DataSource,
        check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run custom SQL quality check."""
        sql = check.get("sql")
        if not sql:
            return {"passed": False, "error": "No SQL query specified"}
        
        # Replace placeholder with actual table name
        sql = sql.replace("{{table}}", data_source.source_path)
        
        try:
            result = await self.execute_query(sql)
            
            if result and len(result) > 0:
                first_row = result[0]
                
                # Expected columns: passed (boolean), total_records, passed_records, failed_records
                return {
                    "passed": first_row.get("PASSED", False),
                    "total_records": first_row.get("TOTAL_RECORDS", 0),
                    "passed_records": first_row.get("PASSED_RECORDS", 0),
                    "failed_records": first_row.get("FAILED_RECORDS", 0),
                    "score": first_row.get("SCORE", 0.0),
                    "failure_reasons": [first_row.get("FAILURE_REASON")] if first_row.get("FAILURE_REASON") else []
                }
            else:
                return {"passed": False, "error": "No results from custom SQL check"}
                
        except Exception as e:
            return {"passed": False, "error": f"Custom SQL check failed: {str(e)}"}
    
    def __del__(self):
        """Cleanup on object destruction."""
        if self.snowflake_conn:
            try:
                self.snowflake_conn.close()
            except:
                pass