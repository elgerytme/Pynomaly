"""
Databricks Data Platform Connector

Provides integration with Databricks for enterprise
data analytics, Delta Lake, MLflow, and Spark processing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from uuid import UUID, uuid4
import json
import tempfile
import os

from structlog import get_logger
import pandas as pd
from databricks import sdk
from databricks.sdk.core import DatabricksError
from databricks.sdk.service import workspace, jobs, clusters, sql as databricks_sql
import requests

from ...domain.entities.data_platform import DataPlatformConnection, DataSource, DataQualityResult

logger = get_logger(__name__)


class DatabricksConnector:
    """
    Databricks data platform connector.
    
    Provides comprehensive integration with Databricks including
    Delta Lake, Spark processing, MLflow, and SQL analytics.
    """
    
    def __init__(self, connection: DataPlatformConnection):
        self.connection = connection
        self.databricks_client = None
        self.sql_warehouse_id = None
        self.cluster_id = None
        self.logger = logger.bind(connector="databricks", connection=connection.name)
        
        # Validate connection is for Databricks
        if connection.platform_type.value != "databricks":
            raise ValueError(f"Connection {connection.name} is not for Databricks platform")
        
        self.logger.info("DatabricksConnector initialized")
    
    async def connect(self) -> bool:
        """Establish connection to Databricks."""
        self.logger.info("Connecting to Databricks")
        
        try:
            # Configure Databricks SDK
            config = {}
            
            if self.connection.host:
                config["host"] = f"https://{self.connection.host}"
            
            if self.connection.oauth_token:
                config["token"] = self.connection.oauth_token
            elif self.connection.username and self.connection.password:
                config["username"] = self.connection.username
                config["password"] = self.connection.password
            
            # Add additional properties
            config.update(self.connection.properties)
            
            # Create Databricks client
            self.databricks_client = sdk.WorkspaceClient(**config)
            
            # Extract SQL warehouse and cluster IDs from connection
            self.sql_warehouse_id = self.connection.properties.get("sql_warehouse_id")
            self.cluster_id = self.connection.properties.get("cluster_id")
            
            # Test connection
            await self._test_connection()
            
            self.connection.update_status("connected")
            self.logger.info("Successfully connected to Databricks")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Databricks: {str(e)}"
            self.logger.error(error_msg)
            self.connection.update_status("error", error_msg)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Databricks."""
        if self.databricks_client:
            try:
                # Databricks SDK doesn't require explicit disconnection
                self.databricks_client = None
                
                self.connection.update_status("disconnected")
                self.logger.info("Disconnected from Databricks")
                
            except Exception as e:
                self.logger.error("Error during disconnect", error=str(e))
    
    async def execute_sql_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        warehouse_id: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> pd.DataFrame:
        """Execute SQL query using Databricks SQL warehouse."""
        if not self.databricks_client:
            raise RuntimeError("Not connected to Databricks")
        
        target_warehouse_id = warehouse_id or self.sql_warehouse_id
        if not target_warehouse_id:
            raise RuntimeError("No SQL warehouse configured")
        
        self.logger.debug("Executing SQL query", query=query[:100] + "..." if len(query) > 100 else query)
        
        try:
            # Execute query using SQL warehouse
            result = self.databricks_client.statement_execution.execute_statement(
                warehouse_id=target_warehouse_id,
                statement=query,
                parameters=parameters
            )
            
            # Convert result to DataFrame
            if result.result and result.result.data_array:
                # Get column names
                columns = [col.name for col in result.manifest.schema.columns] if result.manifest and result.manifest.schema else []
                
                # Convert data
                data = []
                for row in result.result.data_array:
                    data.append(row)
                
                df = pd.DataFrame(data, columns=columns)
                
                if max_results and len(df) > max_results:
                    df = df.head(max_results)
                
                self.logger.debug("SQL query executed successfully", rows=len(df))
                return df
            else:
                self.logger.debug("SQL query executed successfully (no results)")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error("SQL query execution failed", error=str(e), query=query)
            raise
    
    async def execute_spark_job(
        self,
        notebook_path: str,
        parameters: Optional[Dict[str, str]] = None,
        cluster_id: Optional[str] = None,
        timeout_seconds: int = 3600
    ) -> Dict[str, Any]:
        """Execute Spark job using Databricks notebook."""
        if not self.databricks_client:
            raise RuntimeError("Not connected to Databricks")
        
        target_cluster_id = cluster_id or self.cluster_id
        if not target_cluster_id:
            raise RuntimeError("No cluster configured")
        
        self.logger.info("Executing Spark job", notebook=notebook_path)
        
        try:
            # Create job configuration
            job_spec = jobs.JobSettings(
                name=f"pynomaly-spark-job-{uuid4().hex[:8]}",
                tasks=[
                    jobs.Task(
                        task_key="main_task",
                        existing_cluster_id=target_cluster_id,
                        notebook_task=jobs.NotebookTask(
                            notebook_path=notebook_path,
                            base_parameters=parameters or {}
                        )
                    )
                ],
                timeout_seconds=timeout_seconds
            )
            
            # Create and run job
            job = self.databricks_client.jobs.create(job_spec)
            run = self.databricks_client.jobs.run_now(job_id=job.job_id)
            
            # Wait for completion
            final_run = self.databricks_client.jobs.wait_get_run_job_terminated_or_skipped(
                run_id=run.run_id
            )
            
            # Clean up job
            self.databricks_client.jobs.delete(job_id=job.job_id)
            
            result = {
                "run_id": final_run.run_id,
                "state": final_run.state.life_cycle_state.value if final_run.state else "UNKNOWN",
                "result_state": final_run.state.result_state.value if final_run.state and final_run.state.result_state else None,
                "start_time": final_run.start_time,
                "end_time": final_run.end_time,
                "execution_duration": final_run.execution_duration,
                "notebook_output": final_run.tasks[0].notebook_output if final_run.tasks else None
            }
            
            self.logger.info("Spark job completed", 
                           run_id=final_run.run_id, 
                           state=result["state"])
            return result
            
        except Exception as e:
            self.logger.error("Spark job execution failed", error=str(e))
            raise
    
    async def load_data_to_delta_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        mode: str = "append",
        partition_columns: Optional[List[str]] = None,
        optimize_write: bool = True
    ) -> Dict[str, Any]:
        """Load pandas DataFrame to Delta Lake table."""
        if not self.databricks_client:
            raise RuntimeError("Not connected to Databricks")
        
        target_schema = schema or self.connection.schema or "default"
        full_table_name = f"{target_schema}.{table_name}"
        
        self.logger.info("Loading DataFrame to Delta table", 
                        table=full_table_name, rows=len(df))
        
        try:
            start_time = datetime.utcnow()
            
            # Create temporary file for data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                tmp_file_path = tmp_file.name
            
            try:
                # Upload file to DBFS
                dbfs_path = f"/tmp/pynomaly_{uuid4().hex[:8]}.csv"
                
                with open(tmp_file_path, 'rb') as file:
                    self.databricks_client.dbfs.upload(
                        path=dbfs_path,
                        contents=file.read(),
                        overwrite=True
                    )
                
                # Build Spark SQL for loading data
                partition_clause = ""
                if partition_columns:
                    partition_clause = f"PARTITIONED BY ({', '.join(partition_columns)})"
                
                # Create or replace table
                create_sql = f"""
                CREATE TABLE IF NOT EXISTS {full_table_name}
                USING DELTA
                {partition_clause}
                AS SELECT * FROM read_files(
                    'dbfs:{dbfs_path}',
                    format => 'csv',
                    header => true,
                    inferSchema => true
                )
                """
                
                if mode == "overwrite":
                    create_sql = create_sql.replace("CREATE TABLE IF NOT EXISTS", "CREATE OR REPLACE TABLE")
                elif mode == "append":
                    # For append mode, first create table if not exists, then insert
                    await self.execute_sql_query(create_sql)
                    create_sql = f"""
                    INSERT INTO {full_table_name}
                    SELECT * FROM read_files(
                        'dbfs:{dbfs_path}',
                        format => 'csv',
                        header => true,
                        inferSchema => true
                    )
                    """
                
                # Execute table creation/insertion
                await self.execute_sql_query(create_sql)
                
                # Optimize table if requested
                if optimize_write:
                    optimize_sql = f"OPTIMIZE {full_table_name}"
                    if partition_columns:
                        optimize_sql += f" ZORDER BY ({', '.join(partition_columns)})"
                    await self.execute_sql_query(optimize_sql)
                
                # Clean up DBFS file
                self.databricks_client.dbfs.delete(path=dbfs_path)
                
            finally:
                # Clean up local temp file
                os.unlink(tmp_file_path)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("DataFrame loaded to Delta table successfully", 
                           table=full_table_name, rows=len(df), duration=duration)
            
            return {
                "success": True,
                "rows_loaded": len(df),
                "duration_seconds": duration,
                "table": full_table_name,
                "optimized": optimize_write
            }
            
        except Exception as e:
            self.logger.error("Failed to load DataFrame to Delta table", error=str(e))
            raise
    
    async def stream_data(
        self,
        data_source: DataSource,
        batch_size: int = 1000,
        max_batches: Optional[int] = None
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream data from Delta table."""
        if not self.databricks_client:
            raise RuntimeError("Not connected to Databricks")
        
        self.logger.info("Starting data stream", source=data_source.source_path, 
                        batch_size=batch_size)
        
        try:
            # Build streaming query
            query = self._build_streaming_query(data_source)
            
            batch_count = 0
            offset = 0
            
            while True:
                # Add pagination to query
                paginated_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                
                # Execute query
                batch_df = await self.execute_sql_query(paginated_query)
                
                if batch_df.empty:
                    break
                
                batch_count += 1
                offset += batch_size
                
                self.logger.debug("Streaming batch", batch=batch_count, size=len(batch_df))
                
                yield batch_df
                
                # Check max batches limit
                if max_batches and batch_count >= max_batches:
                    break
                
                # If we got fewer rows than batch_size, we've reached the end
                if len(batch_df) < batch_size:
                    break
            
            self.logger.info("Data streaming completed", batches=batch_count)
            
        except Exception as e:
            self.logger.error("Data streaming failed", error=str(e))
            raise
    
    async def create_anomaly_detection_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        partition_columns: Optional[List[str]] = None
    ) -> bool:
        """Create Delta table optimized for anomaly detection data."""
        self.logger.info("Creating anomaly detection table", table=table_name)
        
        try:
            target_schema = schema or self.connection.schema or "default"
            full_table_name = f"{target_schema}.{table_name}"
            
            # Default partitioning by tenant and date
            default_partitions = partition_columns or ["tenant_id", "date(timestamp)"]
            partition_clause = f"PARTITIONED BY ({', '.join(default_partitions)})"
            
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {full_table_name} (
                id STRING NOT NULL,
                tenant_id STRING NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data_source STRING NOT NULL,
                feature_values MAP<STRING, DOUBLE> NOT NULL,
                anomaly_score DOUBLE,
                is_anomaly BOOLEAN,
                anomaly_type STRING,
                severity STRING,
                metadata MAP<STRING, STRING>,
                processed_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL
            )
            USING DELTA
            {partition_clause}
            TBLPROPERTIES (
                'delta.autoOptimize.optimizeWrite' = 'true',
                'delta.autoOptimize.autoCompact' = 'true'
            )
            """
            
            await self.execute_sql_query(create_sql)
            
            self.logger.info("Anomaly detection table created successfully", table=full_table_name)
            return True
            
        except Exception as e:
            self.logger.error("Failed to create anomaly detection table", error=str(e))
            return False
    
    async def store_anomaly_results(
        self,
        results: List[Dict[str, Any]],
        table_name: str = "anomaly_results",
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store anomaly detection results in Delta table."""
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
            
            # Load to Delta table
            result = await self.load_data_to_delta_table(
                df=df,
                table_name=table_name,
                schema=schema,
                mode="append"
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
        table_name: str = "anomaly_results",
        schema: Optional[str] = None
    ) -> pd.DataFrame:
        """Query anomaly detection results from Delta table."""
        self.logger.info("Querying anomalies", tenant_id=tenant_id, limit=limit)
        
        try:
            target_schema = schema or self.connection.schema or "default"
            full_table_name = f"{target_schema}.{table_name}"
            
            # Build WHERE conditions
            conditions = [f"tenant_id = '{tenant_id}'"]
            
            if start_time:
                conditions.append(f"timestamp >= '{start_time.isoformat()}'")
            
            if end_time:
                conditions.append(f"timestamp <= '{end_time.isoformat()}'")
            
            if data_source:
                conditions.append(f"data_source = '{data_source}'")
            
            if anomaly_types:
                types_str = "', '".join(anomaly_types)
                conditions.append(f"anomaly_type IN ('{types_str}')")
            
            if min_score:
                conditions.append(f"anomaly_score >= {min_score}")
            
            # Build query
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
            
            df = await self.execute_sql_query(query)
            
            self.logger.info("Anomaly query completed", results_count=len(df))
            return df
            
        except Exception as e:
            self.logger.error("Failed to query anomalies", error=str(e))
            raise
    
    async def run_data_quality_checks(
        self,
        data_source: DataSource,
        checks: List[Dict[str, Any]]
    ) -> List[DataQualityResult]:
        """Run data quality checks on Delta table data."""
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
    
    async def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive table information from Delta Lake."""
        self.logger.info("Getting table information", table=table_name)
        
        try:
            target_schema = schema or self.connection.schema or "default"
            full_table_name = f"{target_schema}.{table_name}"
            
            # Get table details
            describe_query = f"DESCRIBE DETAIL {full_table_name}"
            detail_df = await self.execute_sql_query(describe_query)
            
            if detail_df.empty:
                return {"error": "Table not found"}
            
            detail = detail_df.iloc[0].to_dict()
            
            # Get table schema
            schema_query = f"DESCRIBE {full_table_name}"
            schema_df = await self.execute_sql_query(schema_query)
            
            return {
                "table_name": detail.get("name"),
                "location": detail.get("location"),
                "created_at": detail.get("createdAt"),
                "last_modified": detail.get("lastModified"),
                "partitioning_columns": detail.get("partitionColumns", []),
                "num_files": detail.get("numFiles"),
                "size_bytes": detail.get("sizeInBytes"),
                "min_reader_version": detail.get("minReaderVersion"),
                "min_writer_version": detail.get("minWriterVersion"),
                "schema": [
                    {
                        "column_name": row["col_name"],
                        "data_type": row["data_type"],
                        "comment": row.get("comment", "")
                    }
                    for _, row in schema_df.iterrows()
                    if row["col_name"] and not row["col_name"].startswith("#")
                ]
            }
            
        except Exception as e:
            self.logger.error("Failed to get table information", error=str(e))
            raise
    
    async def optimize_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        zorder_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Optimize Delta table with compaction and Z-ordering."""
        self.logger.info("Optimizing Delta table", table=table_name)
        
        try:
            target_schema = schema or self.connection.schema or "default"
            full_table_name = f"{target_schema}.{table_name}"
            
            # Run OPTIMIZE command
            optimize_sql = f"OPTIMIZE {full_table_name}"
            if zorder_columns:
                optimize_sql += f" ZORDER BY ({', '.join(zorder_columns)})"
            
            result_df = await self.execute_sql_query(optimize_sql)
            
            # Parse optimization results
            if not result_df.empty:
                result = result_df.iloc[0].to_dict()
                return {
                    "success": True,
                    "metrics": result,
                    "table": full_table_name
                }
            else:
                return {
                    "success": True,
                    "table": full_table_name
                }
            
        except Exception as e:
            self.logger.error("Failed to optimize table", error=str(e))
            raise
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test Databricks connection."""
        try:
            # Test connection by listing workspaces or getting current user
            current_user = self.databricks_client.current_user.me()
            self.logger.info("Connection test successful", user=current_user.user_name)
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}")
    
    def _build_streaming_query(self, data_source: DataSource) -> str:
        """Build optimized query for data streaming."""
        base_query = f"SELECT * FROM {data_source.source_path}"
        
        # Add incremental filtering if configured
        if data_source.incremental_column and data_source.last_refresh:
            base_query += f" WHERE {data_source.incremental_column} > '{data_source.last_refresh.isoformat()}'"
        
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
        
        df = await self.execute_sql_query(query)
        if df.empty:
            return {"passed": False, "error": "Query returned no results"}
        
        result = df.iloc[0]
        total_records = int(result["total_records"])
        null_records = int(result["null_records"])
        non_null_records = int(result["non_null_records"])
        
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
        
        df = await self.execute_sql_query(query)
        if df.empty:
            return {"passed": False, "error": "Query returned no results"}
        
        result = df.iloc[0]
        total_records = int(result["total_records"])
        unique_records = int(result["unique_records"])
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
        
        df = await self.execute_sql_query(query)
        if df.empty:
            return {"passed": False, "error": "Query returned no results"}
        
        result = df.iloc[0]
        total_records = int(result["total_records"])
        in_range_records = int(result["in_range_records"])
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
            df = await self.execute_sql_query(sql)
            
            if not df.empty:
                result = df.iloc[0]
                
                # Expected columns: passed (boolean), total_records, passed_records, failed_records
                return {
                    "passed": bool(result.get("passed", False)),
                    "total_records": int(result.get("total_records", 0)),
                    "passed_records": int(result.get("passed_records", 0)),
                    "failed_records": int(result.get("failed_records", 0)),
                    "score": float(result.get("score", 0.0)),
                    "failure_reasons": [result.get("failure_reason")] if result.get("failure_reason") else []
                }
            else:
                return {"passed": False, "error": "No results from custom SQL check"}
                
        except Exception as e:
            return {"passed": False, "error": f"Custom SQL check failed: {str(e)}"}
    
    def __del__(self):
        """Cleanup on object destruction."""
        # Databricks SDK doesn't require explicit cleanup
        pass