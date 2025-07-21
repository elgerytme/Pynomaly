"""
Google BigQuery Data Platform Connector

Provides integration with Google BigQuery for enterprise
data analytics, real-time streaming, and machine learning.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Iterator
from uuid import UUID
import json

from structlog import get_logger
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import SchemaField, LoadJobConfig, QueryJobConfig
from google.cloud.bigquery.job import LoadJob, QueryJob
from google.oauth2 import service_account
import google.auth

from ...domain.entities.data_platform import DataPlatformConnection, DataSource, DataQualityResult

logger = get_logger(__name__)


class BigQueryConnector:
    """
    Google BigQuery data warehouse connector.
    
    Provides comprehensive integration with BigQuery including
    data loading, querying, streaming inserts, and ML integration.
    """
    
    def __init__(self, connection: DataPlatformConnection):
        self.connection = connection
        self.client = None
        self.project_id = None
        self.logger = logger.bind(connector="bigquery", connection=connection.name)
        
        # Validate connection is for BigQuery
        if connection.platform_type.value != "bigquery":
            raise ValueError(f"Connection {connection.name} is not for BigQuery platform")
        
        self.logger.info("BigQueryConnector initialized")
    
    async def connect(self) -> bool:
        """Establish connection to BigQuery."""
        self.logger.info("Connecting to BigQuery")
        
        try:
            # Set up authentication
            if self.connection.service_account_key:
                # Service account key authentication
                credentials_info = json.loads(self.connection.service_account_key)
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                self.project_id = credentials_info.get("project_id") or self.connection.database
            else:
                # Default credentials (ADC)
                credentials, project_id = google.auth.default()
                self.project_id = project_id or self.connection.database
            
            # Create BigQuery client
            self.client = bigquery.Client(
                credentials=credentials,
                project=self.project_id
            )
            
            # Test connection
            await self._test_connection()
            
            self.connection.update_status("connected")
            self.logger.info("Successfully connected to BigQuery", project=self.project_id)
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to BigQuery: {str(e)}"
            self.logger.error(error_msg)
            self.connection.update_status("error", error_msg)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from BigQuery."""
        if self.client:
            try:
                self.client.close()
                self.client = None
                
                self.connection.update_status("disconnected")
                self.logger.info("Disconnected from BigQuery")
                
            except Exception as e:
                self.logger.error("Error during disconnect", error=str(e))
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[List[Any]] = None,
        use_legacy_sql: bool = False,
        dry_run: bool = False,
        max_results: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Execute SQL query on BigQuery."""
        if not self.client:
            raise RuntimeError("Not connected to BigQuery")
        
        self.logger.debug("Executing query", query=query[:100] + "..." if len(query) > 100 else query)
        
        try:
            # Configure job
            job_config = QueryJobConfig()
            job_config.use_legacy_sql = use_legacy_sql
            job_config.dry_run = dry_run
            
            if parameters:
                job_config.query_parameters = parameters
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            
            if dry_run:
                self.logger.info("Dry run completed", 
                               bytes_processed=query_job.total_bytes_processed)
                return None
            
            # Wait for completion
            results = query_job.result(max_results=max_results)
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            self.logger.debug("Query executed successfully", 
                             rows=len(df), 
                             bytes_processed=query_job.total_bytes_processed)
            return df
            
        except Exception as e:
            self.logger.error("Query execution failed", error=str(e), query=query)
            raise
    
    async def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        table_id: str,
        dataset_id: Optional[str] = None,
        if_exists: str = "append",
        schema: Optional[List[SchemaField]] = None,
        clustering_fields: Optional[List[str]] = None,
        partitioning_field: Optional[str] = None,
        partitioning_type: str = "DAY"
    ) -> Dict[str, Any]:
        """Load pandas DataFrame to BigQuery table."""
        if not self.client:
            raise RuntimeError("Not connected to BigQuery")
        
        # Use default dataset from connection if not provided
        target_dataset = dataset_id or self.connection.schema
        table_ref = self.client.dataset(target_dataset).table(table_id)
        
        self.logger.info("Loading DataFrame to BigQuery", 
                        table=f"{target_dataset}.{table_id}", rows=len(df))
        
        try:
            start_time = datetime.utcnow()
            
            # Configure job
            job_config = LoadJobConfig()
            
            if if_exists == "replace":
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            elif if_exists == "append":
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            else:  # if_exists == "fail"
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
            
            # Auto-detect schema or use provided schema
            if schema:
                job_config.schema = schema
            else:
                job_config.autodetect = True
            
            # Configure partitioning
            if partitioning_field:
                if partitioning_type == "DAY":
                    time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field=partitioning_field
                    )
                elif partitioning_type == "MONTH":
                    time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.MONTH,
                        field=partitioning_field
                    )
                elif partitioning_type == "YEAR":
                    time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.YEAR,
                        field=partitioning_field
                    )
                else:
                    time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field=partitioning_field
                    )
                
                job_config.time_partitioning = time_partitioning
            
            # Configure clustering
            if clustering_fields:
                job_config.clustering_fields = clustering_fields
            
            # Load data
            load_job = self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            
            # Wait for completion
            load_job.result()
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Get job statistics
            stats = load_job._properties.get('statistics', {}).get('load', {})
            
            self.logger.info("DataFrame loaded successfully", 
                           table=f"{target_dataset}.{table_id}",
                           rows=stats.get('outputRows', len(df)),
                           bytes=stats.get('outputBytes', 0),
                           duration=duration)
            
            return {
                "success": True,
                "rows_loaded": int(stats.get('outputRows', len(df))),
                "bytes_loaded": int(stats.get('outputBytes', 0)),
                "duration_seconds": duration,
                "job_id": load_job.job_id,
                "table": f"{self.project_id}.{target_dataset}.{table_id}"
            }
            
        except Exception as e:
            self.logger.error("Failed to load DataFrame", error=str(e))
            raise
    
    async def stream_data(
        self,
        data_source: DataSource,
        batch_size: int = 1000,
        max_batches: Optional[int] = None
    ) -> AsyncGenerator[pd.DataFrame, None]:
        """Stream data from BigQuery table using pagination."""
        if not self.client:
            raise RuntimeError("Not connected to BigQuery")
        
        self.logger.info("Starting data stream", source=data_source.source_path, 
                        batch_size=batch_size)
        
        try:
            # Build streaming query
            query = self._build_streaming_query(data_source)
            
            # Configure query job for streaming
            job_config = QueryJobConfig()
            query_job = self.client.query(query, job_config=job_config)
            
            batch_count = 0
            page_token = None
            
            while True:
                # Get page of results
                page = query_job.result(
                    page_size=batch_size,
                    start_index=batch_count * batch_size if not page_token else None,
                    page_token=page_token
                )
                
                # Convert page to DataFrame
                batch_df = page.to_dataframe()
                
                if batch_df.empty:
                    break
                
                batch_count += 1
                self.logger.debug("Streaming batch", batch=batch_count, size=len(batch_df))
                
                yield batch_df
                
                # Check for next page
                page_token = getattr(page, 'next_page_token', None)
                if not page_token:
                    break
                
                # Check max batches limit
                if max_batches and batch_count >= max_batches:
                    break
            
            self.logger.info("Data streaming completed", batches=batch_count)
            
        except Exception as e:
            self.logger.error("Data streaming failed", error=str(e))
            raise
    
    async def create_anomaly_detection_table(
        self,
        table_id: str,
        dataset_id: Optional[str] = None,
        partitioning_field: str = "timestamp",
        clustering_fields: Optional[List[str]] = None
    ) -> bool:
        """Create table optimized for anomaly detection data."""
        self.logger.info("Creating anomaly detection table", table=table_id)
        
        try:
            target_dataset = dataset_id or self.connection.schema
            table_ref = self.client.dataset(target_dataset).table(table_id)
            
            # Define schema
            schema = [
                SchemaField("id", "STRING", mode="REQUIRED"),
                SchemaField("tenant_id", "STRING", mode="REQUIRED"),
                SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
                SchemaField("data_source", "STRING", mode="REQUIRED"),
                SchemaField("feature_values", "JSON", mode="REQUIRED"),
                SchemaField("anomaly_score", "FLOAT64", mode="NULLABLE"),
                SchemaField("is_anomaly", "BOOLEAN", mode="NULLABLE"),
                SchemaField("anomaly_type", "STRING", mode="NULLABLE"),
                SchemaField("severity", "STRING", mode="NULLABLE"),
                SchemaField("metadata", "JSON", mode="NULLABLE"),
                SchemaField("processed_at", "TIMESTAMP", mode="NULLABLE"),
                SchemaField("created_at", "TIMESTAMP", mode="REQUIRED")
            ]
            
            # Configure table
            table = bigquery.Table(table_ref, schema=schema)
            
            # Time partitioning
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partitioning_field,
                expiration_ms=None  # No automatic expiration
            )
            
            # Clustering
            default_clustering = ["tenant_id", "data_source", "is_anomaly"]
            table.clustering_fields = clustering_fields or default_clustering
            
            # Create table
            table = self.client.create_table(table, exists_ok=True)
            
            self.logger.info("Anomaly detection table created successfully", 
                           table=f"{target_dataset}.{table_id}")
            return True
            
        except Exception as e:
            self.logger.error("Failed to create anomaly detection table", error=str(e))
            return False
    
    async def stream_insert_anomalies(
        self,
        results: List[Dict[str, Any]],
        table_id: str = "anomaly_results",
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Stream insert anomaly results to BigQuery."""
        if not results:
            return {"success": True, "rows_inserted": 0}
        
        self.logger.info("Streaming anomaly results", count=len(results))
        
        try:
            target_dataset = dataset_id or self.connection.schema
            table_ref = self.client.dataset(target_dataset).table(table_id)
            
            # Prepare rows for streaming insert
            rows_to_insert = []
            for result in results:
                row = {
                    "id": result.get("id", str(uuid4())),
                    "tenant_id": str(result.get("tenant_id", "")),
                    "timestamp": result.get("timestamp", datetime.utcnow()).isoformat(),
                    "data_source": result.get("data_source", ""),
                    "feature_values": result.get("feature_values", {}),
                    "anomaly_score": result.get("anomaly_score"),
                    "is_anomaly": result.get("is_anomaly", False),
                    "anomaly_type": result.get("anomaly_type"),
                    "severity": result.get("severity"),
                    "metadata": result.get("metadata", {}),
                    "processed_at": datetime.utcnow().isoformat(),
                    "created_at": result.get("created_at", datetime.utcnow()).isoformat()
                }
                rows_to_insert.append(row)
            
            # Perform streaming insert
            errors = self.client.insert_rows_json(
                table_ref, 
                rows_to_insert,
                ignore_unknown_values=True
            )
            
            if errors:
                raise RuntimeError(f"Streaming insert errors: {errors}")
            
            self.logger.info("Anomaly results streamed successfully", rows=len(results))
            return {
                "success": True,
                "rows_inserted": len(results),
                "table": f"{self.project_id}.{target_dataset}.{table_id}"
            }
            
        except Exception as e:
            self.logger.error("Failed to stream anomaly results", error=str(e))
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
        table_id: str = "anomaly_results",
        dataset_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Query anomaly detection results from BigQuery."""
        self.logger.info("Querying anomalies", tenant_id=tenant_id, limit=limit)
        
        try:
            target_dataset = dataset_id or self.connection.schema
            table_name = f"{self.project_id}.{target_dataset}.{table_id}"
            
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
            FROM `{table_name}`
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            
            df = await self.execute_query(query)
            
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
        """Run data quality checks on BigQuery data."""
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
    
    async def create_ml_model(
        self,
        model_name: str,
        model_type: str = "AUTOML_REGRESSOR",
        input_label_cols: List[str] = None,
        data_split_method: str = "AUTO_SPLIT",
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create BigQuery ML model for anomaly detection."""
        self.logger.info("Creating BigQuery ML model", model=model_name, type=model_type)
        
        try:
            target_dataset = dataset_id or self.connection.schema
            model_ref = f"{self.project_id}.{target_dataset}.{model_name}"
            
            # Build CREATE MODEL query
            label_cols = input_label_cols or ["is_anomaly"]
            label_str = ", ".join(label_cols)
            
            query = f"""
            CREATE OR REPLACE MODEL `{model_ref}`
            OPTIONS(
                model_type='{model_type}',
                input_label_cols=['{label_str}'],
                data_split_method='{data_split_method}',
                auto_class_weights=true
            ) AS
            SELECT
                * EXCEPT(id, created_at, processed_at)
            FROM
                `{self.project_id}.{target_dataset}.anomaly_results`
            WHERE
                is_anomaly IS NOT NULL
            """
            
            # Execute model creation
            job = await self.execute_query(query, dry_run=False)
            
            self.logger.info("BigQuery ML model created successfully", model=model_ref)
            
            return {
                "success": True,
                "model_name": model_ref,
                "model_type": model_type,
                "job_id": job.job_id if hasattr(job, 'job_id') else None
            }
            
        except Exception as e:
            self.logger.error("Failed to create ML model", error=str(e))
            raise
    
    async def get_table_info(
        self,
        table_id: str,
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive table information from BigQuery."""
        self.logger.info("Getting table information", table=table_id)
        
        try:
            target_dataset = dataset_id or self.connection.schema
            table_ref = self.client.dataset(target_dataset).table(table_id)
            
            # Get table
            table = self.client.get_table(table_ref)
            
            return {
                "table_id": table.table_id,
                "dataset_id": table.dataset_id,
                "project": table.project,
                "table_type": table.table_type,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None,
                "expires": table.expires.isoformat() if table.expires else None,
                "description": table.description,
                "schema": [
                    {
                        "name": field.name,
                        "field_type": field.field_type,
                        "mode": field.mode,
                        "description": field.description
                    }
                    for field in table.schema
                ],
                "clustering_fields": table.clustering_fields,
                "partitioning": {
                    "type": table.time_partitioning.type_ if table.time_partitioning else None,
                    "field": table.time_partitioning.field if table.time_partitioning else None,
                    "expiration_ms": table.time_partitioning.expiration_ms if table.time_partitioning else None
                } if table.time_partitioning else None,
                "labels": dict(table.labels) if table.labels else {}
            }
            
        except Exception as e:
            self.logger.error("Failed to get table information", error=str(e))
            raise
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test BigQuery connection."""
        try:
            # Try to list datasets as a connection test
            datasets = list(self.client.list_datasets(max_results=1))
            self.logger.info("Connection test successful", project=self.project_id)
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}")
    
    def _build_streaming_query(self, data_source: DataSource) -> str:
        """Build optimized query for data streaming."""
        base_query = f"SELECT * FROM `{data_source.source_path}`"
        
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
        FROM `{data_source.source_path}`
        """
        
        df = await self.execute_query(query)
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
        FROM `{data_source.source_path}`
        """
        
        df = await self.execute_query(query)
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
            COUNTIF({range_condition}) as in_range_records
        FROM `{data_source.source_path}`
        WHERE {column} IS NOT NULL
        """
        
        df = await self.execute_query(query)
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
        sql = sql.replace("{{table}}", f"`{data_source.source_path}`")
        
        try:
            df = await self.execute_query(sql)
            
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
        if self.client:
            try:
                self.client.close()
            except:
                pass