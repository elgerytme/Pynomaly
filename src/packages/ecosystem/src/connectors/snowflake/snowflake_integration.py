"""
Snowflake integration connector.

This module provides comprehensive integration with Snowflake for
cloud data warehousing, analytics, and data sharing.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncIterator

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
import structlog

from ...core.interfaces import (
    IntegrationInterface, IntegrationConfig, ConnectionHealth,
    IntegrationStatus, AuthenticationMethod, DataFlowCapability
)

logger = structlog.get_logger(__name__)


class SnowflakeIntegration(IntegrationInterface):
    """
    Snowflake data warehouse integration.
    
    Provides integration with Snowflake for cloud data warehousing,
    analytics, data sharing, and ML feature engineering.
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Snowflake integration."""
        super().__init__(config)
        
        # Snowflake-specific configuration
        self.account = config.credentials.get("account")
        self.user = config.credentials.get("user")
        self.password = config.credentials.get("password")
        self.warehouse = config.credentials.get("warehouse")
        self.database = config.credentials.get("database")
        self.schema = config.credentials.get("schema", "PUBLIC")
        self.role = config.credentials.get("role")
        
        # Connection management
        self._connection: Optional[snowflake.connector.SnowflakeConnection] = None
        
        self.logger.info(
            "Snowflake integration initialized",
            account=self.account,
            warehouse=self.warehouse,
            database=self.database
        )
    
    # Core connection methods
    
    async def connect(self) -> bool:
        """Establish connection to Snowflake."""
        try:
            await self._set_status(IntegrationStatus.CONNECTING)
            
            # Build connection parameters
            conn_params = {
                "account": self.account,
                "user": self.user,
                "password": self.password,
                "warehouse": self.warehouse,
                "database": self.database,
                "schema": self.schema,
                "timeout": self.config.timeout_seconds
            }
            
            if self.role:
                conn_params["role"] = self.role
            
            # Establish connection
            self._connection = snowflake.connector.connect(**conn_params)
            
            # Test connection
            health = await self.test_connection()
            await self._set_health(health)
            
            if health == ConnectionHealth.HEALTHY:
                await self._set_status(IntegrationStatus.CONNECTED)
                self.logger.info("Connected to Snowflake successfully")
                return True
            else:
                await self._set_status(IntegrationStatus.ERROR)
                return False
                
        except Exception as e:
            self.logger.error("Snowflake connection failed", error=str(e))
            await self._set_status(IntegrationStatus.ERROR)
            await self._set_health(ConnectionHealth.UNHEALTHY)
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to Snowflake."""
        try:
            if self._connection:
                self._connection.close()
                self._connection = None
            
            await self._set_status(IntegrationStatus.DISCONNECTED)
            await self._set_health(ConnectionHealth.UNKNOWN)
            
            self.logger.info("Disconnected from Snowflake")
            return True
            
        except Exception as e:
            self.logger.error("Snowflake disconnection failed", error=str(e))
            return False
    
    async def test_connection(self) -> ConnectionHealth:
        """Test Snowflake connection health."""
        try:
            if not self._connection:
                return ConnectionHealth.UNHEALTHY
            
            # Test with simple query
            cursor = self._connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return ConnectionHealth.HEALTHY
            else:
                return ConnectionHealth.DEGRADED
                
        except Exception as e:
            self.logger.error("Connection health check failed", error=str(e))
            return ConnectionHealth.UNHEALTHY
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get Snowflake integration capabilities."""
        return {
            "platform_type": "data_warehouse",
            "supported_operations": [
                "data_storage",
                "data_querying",
                "data_analytics",
                "data_sharing",
                "ml_features",
                "time_travel",
                "data_governance"
            ],
            "data_capabilities": [
                DataFlowCapability.BIDIRECTIONAL.value,
                DataFlowCapability.BATCH.value,
                DataFlowCapability.STREAMING.value
            ],
            "supported_formats": [
                "json", "csv", "parquet", "avro", "orc", "xml"
            ],
            "authentication_methods": [
                AuthenticationMethod.BASIC_AUTH.value,
                AuthenticationMethod.OAUTH2.value,
                AuthenticationMethod.CERTIFICATE.value
            ]
        }
    
    # Data operations
    
    async def send_data(
        self,
        data: Any,
        destination: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send data to Snowflake table."""
        try:
            if not self._ensure_connected():
                return False
            
            start_time = datetime.now()
            
            # Handle different data types
            if isinstance(data, pd.DataFrame):
                success = await self._write_dataframe(data, destination, options)
            elif isinstance(data, list):
                success = await self._write_records(data, destination, options)
            elif isinstance(data, dict):
                success = await self._write_records([data], destination, options)
            else:
                self.logger.error("Unsupported data type", type=type(data).__name__)
                return False
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._update_metrics(success, response_time)
            
            return success
            
        except Exception as e:
            await self._handle_api_error(e, "send_data")
            return False
    
    async def receive_data(
        self,
        source: str,
        format_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Receive data from Snowflake table."""
        try:
            if not self._ensure_connected():
                return None
            
            start_time = datetime.now()
            
            # Build query
            query = self._build_select_query(source, options)
            
            # Execute query
            cursor = self._connection.cursor()
            cursor.execute(query)
            
            # Return format based on preferences
            return_format = options.get("return_format", "dataframe") if options else "dataframe"
            
            if return_format == "dataframe":
                # Fetch as DataFrame
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                result = pd.DataFrame(rows, columns=columns)
            elif return_format == "records":
                # Fetch as list of dictionaries
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
            else:
                # Fetch as raw tuples
                result = cursor.fetchall()
            
            cursor.close()
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            await self._handle_api_error(e, "receive_data")
            return None
    
    # Event handling (basic implementation)
    
    async def subscribe_to_events(
        self,
        event_types: List[str],
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> str:
        """Subscribe to Snowflake events (polling-based)."""
        subscription_id = str(asyncio.current_task())
        
        # Snowflake doesn't have native event streaming
        # This would need to be implemented using Change Data Capture
        self.logger.info(
            "Event subscription created (CDC-based)",
            subscription_id=subscription_id,
            event_types=event_types
        )
        
        return subscription_id
    
    async def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        self.logger.info(
            "Event subscription cancelled",
            subscription_id=subscription_id
        )
        return True
    
    # Snowflake-specific operations
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        return_results: bool = True
    ) -> Optional[Any]:
        """Execute SQL query in Snowflake."""
        try:
            if not self._ensure_connected():
                return None
            
            cursor = self._connection.cursor()
            
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            if return_results:
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                result = pd.DataFrame(rows, columns=columns) if columns else None
            else:
                result = cursor.rowcount
            
            cursor.close()
            
            self.logger.info(
                "Query executed successfully",
                query_preview=query[:100] + "..." if len(query) > 100 else query
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Query execution failed", error=str(e), query=query[:100])
            return None
    
    async def create_table(
        self,
        table_name: str,
        schema_definition: Dict[str, str],
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create table in Snowflake."""
        try:
            if not self._ensure_connected():
                return False
            
            # Build CREATE TABLE statement
            columns = []
            for column_name, column_type in schema_definition.items():
                columns.append(f"{column_name} {column_type}")
            
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            
            # Add table options
            if options:
                if "cluster_by" in options:
                    create_sql += f" CLUSTER BY ({options['cluster_by']})"
                if "copy_grants" in options:
                    create_sql += " COPY GRANTS"
            
            result = await self.execute_query(create_sql, return_results=False)
            success = result is not None
            
            if success:
                self.logger.info("Table created successfully", table_name=table_name)
            
            return success
            
        except Exception as e:
            self.logger.error("Table creation failed", error=str(e), table_name=table_name)
            return False
    
    async def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table information and metadata."""
        try:
            if not self._ensure_connected():
                return None
            
            # Get table description
            desc_query = f"DESCRIBE TABLE {table_name}"
            table_desc = await self.execute_query(desc_query)
            
            if table_desc is None:
                return None
            
            # Get table statistics
            stats_query = f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT *) as distinct_rows
            FROM {table_name}
            """
            
            table_stats = await self.execute_query(stats_query)
            
            # Combine information
            info = {
                "table_name": table_name,
                "columns": table_desc.to_dict('records') if table_desc is not None else [],
                "row_count": table_stats.iloc[0]["ROW_COUNT"] if table_stats is not None else 0,
                "distinct_rows": table_stats.iloc[0]["DISTINCT_ROWS"] if table_stats is not None else 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            return info
            
        except Exception as e:
            self.logger.error("Failed to get table info", error=str(e), table_name=table_name)
            return None
    
    async def copy_data_from_stage(
        self,
        stage_name: str,
        table_name: str,
        file_pattern: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Copy data from Snowflake stage to table."""
        try:
            if not self._ensure_connected():
                return False
            
            # Build COPY INTO statement
            copy_sql = f"COPY INTO {table_name} FROM @{stage_name}"
            
            if file_pattern:
                copy_sql += f" PATTERN = '{file_pattern}'"
            
            # Add copy options
            if options:
                option_parts = []
                for key, value in options.items():
                    if isinstance(value, str):
                        option_parts.append(f"{key.upper()} = '{value}'")
                    else:
                        option_parts.append(f"{key.upper()} = {value}")
                
                if option_parts:
                    copy_sql += f" ({', '.join(option_parts)})"
            
            result = await self.execute_query(copy_sql, return_results=False)
            success = result is not None
            
            if success:
                self.logger.info(
                    "Data copied from stage successfully",
                    stage=stage_name,
                    table=table_name
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Stage copy failed",
                error=str(e),
                stage=stage_name,
                table=table_name
            )
            return False
    
    async def create_stored_procedure(
        self,
        procedure_name: str,
        procedure_sql: str,
        parameters: Optional[List[str]] = None,
        return_type: str = "STRING"
    ) -> bool:
        """Create stored procedure in Snowflake."""
        try:
            if not self._ensure_connected():
                return False
            
            # Build CREATE PROCEDURE statement
            param_list = f"({', '.join(parameters)})" if parameters else "()"
            
            create_proc_sql = f"""
            CREATE OR REPLACE PROCEDURE {procedure_name}{param_list}
            RETURNS {return_type}
            LANGUAGE SQL
            AS
            $$
            {procedure_sql}
            $$
            """
            
            result = await self.execute_query(create_proc_sql, return_results=False)
            success = result is not None
            
            if success:
                self.logger.info(
                    "Stored procedure created successfully",
                    procedure_name=procedure_name
                )
            
            return success
            
        except Exception as e:
            self.logger.error(
                "Stored procedure creation failed",
                error=str(e),
                procedure_name=procedure_name
            )
            return False
    
    async def get_warehouse_usage(
        self,
        warehouse_name: Optional[str] = None,
        hours_back: int = 24
    ) -> Optional[Dict[str, Any]]:
        """Get warehouse usage statistics."""
        try:
            if not self._ensure_connected():
                return None
            
            target_warehouse = warehouse_name or self.warehouse
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            usage_query = f"""
            SELECT 
                WAREHOUSE_NAME,
                SUM(CREDITS_USED) as total_credits,
                AVG(CREDITS_USED_COMPUTE) as avg_compute_credits,
                COUNT(*) as query_count,
                AVG(EXECUTION_TIME) as avg_execution_time_ms
            FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
            WHERE WAREHOUSE_NAME = '{target_warehouse}'
            AND START_TIME >= '{start_time.isoformat()}'
            AND START_TIME <= '{end_time.isoformat()}'
            GROUP BY WAREHOUSE_NAME
            """
            
            result = await self.execute_query(usage_query)
            
            if result is not None and not result.empty:
                usage_data = result.iloc[0].to_dict()
                usage_data["period_hours"] = hours_back
                usage_data["query_start_time"] = start_time.isoformat()
                usage_data["query_end_time"] = end_time.isoformat()
                return usage_data
            else:
                return {
                    "warehouse_name": target_warehouse,
                    "total_credits": 0,
                    "avg_compute_credits": 0,
                    "query_count": 0,
                    "avg_execution_time_ms": 0,
                    "period_hours": hours_back
                }
                
        except Exception as e:
            self.logger.error("Failed to get warehouse usage", error=str(e))
            return None
    
    # Helper methods
    
    def _ensure_connected(self) -> bool:
        """Ensure connection is established."""
        if not self._connection:
            self.logger.error("Not connected to Snowflake")
            return False
        return True
    
    async def _write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write DataFrame to Snowflake table."""
        try:
            # Use pandas write_pandas utility
            success, nchunks, nrows, _ = write_pandas(
                self._connection,
                df,
                table_name,
                auto_create_table=options.get("auto_create_table", True) if options else True,
                overwrite=options.get("overwrite", False) if options else False,
                chunk_size=options.get("chunk_size", 16384) if options else 16384
            )
            
            if success:
                self.logger.info(
                    "DataFrame written successfully",
                    table_name=table_name,
                    rows=nrows,
                    chunks=nchunks
                )
            
            return success
            
        except Exception as e:
            self.logger.error("DataFrame write failed", error=str(e), table_name=table_name)
            return False
    
    async def _write_records(
        self,
        records: List[Dict[str, Any]],
        table_name: str,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Write records to Snowflake table."""
        try:
            # Convert records to DataFrame
            df = pd.DataFrame(records)
            return await self._write_dataframe(df, table_name, options)
            
        except Exception as e:
            self.logger.error("Records write failed", error=str(e), table_name=table_name)
            return False
    
    def _build_select_query(
        self,
        table_name: str,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build SELECT query with options."""
        query = f"SELECT * FROM {table_name}"
        
        if options:
            # Add WHERE clause
            if "where" in options:
                query += f" WHERE {options['where']}"
            
            # Add ORDER BY clause
            if "order_by" in options:
                query += f" ORDER BY {options['order_by']}"
            
            # Add LIMIT clause
            if "limit" in options:
                query += f" LIMIT {options['limit']}"
        
        return query
    
    async def _handle_api_error(self, error: Exception, operation: str) -> None:
        """Handle API errors with consistent logging and metrics."""
        self.logger.error(
            f"Snowflake API error in {operation}",
            error=str(error),
            error_type=type(error).__name__
        )
        
        # Update metrics
        await self._update_metrics(False, 0.0)
        
        # Check if we need to reconnect
        if "connection" in str(error).lower() or "timeout" in str(error).lower():
            await self._set_status(IntegrationStatus.ERROR)
    
    async def _validate_platform_config(self, config: IntegrationConfig) -> bool:
        """Validate Snowflake-specific configuration."""
        required_fields = ["account", "user", "password", "warehouse", "database"]
        
        for field in required_fields:
            if not config.credentials.get(field):
                self.logger.error(f"Snowflake configuration missing required field: {field}")
                return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SnowflakeIntegration(account={self.account}, "
            f"warehouse={self.warehouse}, "
            f"database={self.database}, "
            f"connected={self._connection is not None})"
        )