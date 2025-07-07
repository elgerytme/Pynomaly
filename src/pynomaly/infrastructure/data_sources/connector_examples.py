"""Example configurations and utilities for enterprise data connectors."""

from __future__ import annotations

from typing import Dict, Any
from pydantic import SecretStr

from .enterprise_connectors import ConnectionConfig, ConnectorType, QueryOptimization


def get_snowflake_config_example() -> ConnectionConfig:
    """Example Snowflake configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.SNOWFLAKE,
        connection_string="myaccount.snowflakecomputing.com",
        username="myuser",
        password=SecretStr("mypassword"),
        database="ANALYTICS_DB",
        schema="PUBLIC",
        warehouse="COMPUTE_WH",
        role="ANALYST_ROLE",
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            enable_predicate_pushdown=True,
            batch_size=50000,
            parallel_degree=4,
            cache_results=True
        )
    )


def get_databricks_config_example() -> ConnectionConfig:
    """Example Databricks configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.DATABRICKS,
        connection_string="https://myworkspace.cloud.databricks.com",
        password=SecretStr("dapi1234567890abcdef"),  # Databricks token
        database="default",
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            enable_vectorized_execution=True,
            batch_size=100000,
            parallel_degree=8
        )
    )


def get_bigquery_config_example() -> ConnectionConfig:
    """Example BigQuery configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.BIGQUERY,
        connection_string="my-project-id",
        schema="analytics_dataset",
        ssl_config={
            "service_account_path": "/path/to/service-account.json"
        },
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            use_partition_pruning=True,
            cache_results=True,
            batch_size=1000000
        )
    )


def get_redshift_config_example() -> ConnectionConfig:
    """Example Redshift configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.REDSHIFT,
        connection_string="myredshift.cluster.amazonaws.com:5439",
        username="analyst",
        password=SecretStr("mypassword"),
        database="analytics",
        schema="public",
        ssl_config={"sslmode": "require"},
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            enable_predicate_pushdown=True,
            batch_size=25000,
            parallel_degree=4
        )
    )


def get_azure_synapse_config_example() -> ConnectionConfig:
    """Example Azure Synapse configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.AZURE_SYNAPSE,
        connection_string="mysynapse.sql.azuresynapse.net",
        username="sqladmin",
        password=SecretStr("mypassword"),
        database="analytics",
        schema="dbo",
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            cache_results=True,
            batch_size=50000,
            parallel_degree=6
        )
    )


def get_azure_data_lake_config_example() -> ConnectionConfig:
    """Example Azure Data Lake configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.AZURE_DATA_LAKE,
        connection_string="mystorageaccount",  # Storage account name
        ssl_config={
            "account_key": "myaccountkey123456789=="
            # OR "sas_token": "sv=2020-08-04&ss=bfqt&srt=..."
        },
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            enable_vectorized_execution=True,
            batch_size=100000
        )
    )


def get_aws_s3_config_example() -> ConnectionConfig:
    """Example AWS S3/Athena configuration."""
    return ConnectionConfig(
        connector_type=ConnectorType.AWS_S3,
        connection_string="s3://my-data-bucket/",
        username="AKIAI...",  # AWS Access Key ID
        password=SecretStr("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),  # AWS Secret
        database="analytics_db",
        region="us-east-1",
        ssl_config={
            "results_bucket": "athena-query-results"
        },
        optimization=QueryOptimization(
            enable_pushdown=True,
            use_column_pruning=True,
            use_partition_pruning=True,
            batch_size=1000000
        )
    )


# Configuration templates for different use cases
ENTERPRISE_CONFIG_TEMPLATES = {
    "financial_services": {
        "snowflake": {
            "warehouse": "FINANCE_WH",
            "role": "FINANCE_ANALYST",
            "optimization": {
                "cache_results": True,
                "batch_size": 100000,
                "parallel_degree": 8
            }
        },
        "redshift": {
            "schema": "finance",
            "optimization": {
                "enable_pushdown": True,
                "batch_size": 50000
            }
        }
    },
    "healthcare": {
        "bigquery": {
            "optimization": {
                "cache_results": False,  # Privacy considerations
                "use_partition_pruning": True
            }
        },
        "azure_synapse": {
            "optimization": {
                "cache_results": False,
                "parallel_degree": 4
            }
        }
    },
    "retail": {
        "databricks": {
            "optimization": {
                "enable_vectorized_execution": True,
                "batch_size": 200000,
                "parallel_degree": 12
            }
        }
    },
    "manufacturing": {
        "aws_s3": {
            "optimization": {
                "use_partition_pruning": True,
                "batch_size": 500000
            }
        }
    }
}


def get_optimized_config_for_use_case(
    connector_type: ConnectorType,
    use_case: str,
    base_config: ConnectionConfig
) -> ConnectionConfig:
    """Get optimized configuration for specific use case."""
    
    template = ENTERPRISE_CONFIG_TEMPLATES.get(use_case, {}).get(connector_type.value, {})
    
    if template:
        # Apply template optimizations
        if "optimization" in template:
            for key, value in template["optimization"].items():
                setattr(base_config.optimization, key, value)
        
        # Apply other template settings
        for key, value in template.items():
            if key != "optimization" and hasattr(base_config, key):
                setattr(base_config, key, value)
    
    return base_config


# Performance tuning recommendations by data size
PERFORMANCE_RECOMMENDATIONS = {
    "small": {  # < 1GB
        "batch_size": 10000,
        "parallel_degree": 2,
        "enable_pushdown": True,
        "cache_results": True
    },
    "medium": {  # 1GB - 100GB
        "batch_size": 50000,
        "parallel_degree": 4,
        "enable_pushdown": True,
        "cache_results": True,
        "use_column_pruning": True
    },
    "large": {  # 100GB - 10TB
        "batch_size": 100000,
        "parallel_degree": 8,
        "enable_pushdown": True,
        "cache_results": True,
        "use_column_pruning": True,
        "use_partition_pruning": True,
        "enable_vectorized_execution": True
    },
    "xlarge": {  # > 10TB
        "batch_size": 500000,
        "parallel_degree": 16,
        "enable_pushdown": True,
        "cache_results": True,
        "use_column_pruning": True,
        "use_partition_pruning": True,
        "enable_vectorized_execution": True,
        "use_materialized_views": True
    }
}


def get_performance_optimized_config(
    base_config: ConnectionConfig,
    data_size_category: str
) -> ConnectionConfig:
    """Apply performance optimizations based on data size."""
    
    recommendations = PERFORMANCE_RECOMMENDATIONS.get(data_size_category, PERFORMANCE_RECOMMENDATIONS["medium"])
    
    for key, value in recommendations.items():
        if hasattr(base_config.optimization, key):
            setattr(base_config.optimization, key, value)
    
    return base_config


# Validation utilities
def validate_connection_config(config: ConnectionConfig) -> Dict[str, Any]:
    """Validate enterprise connection configuration."""
    
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    # Check required fields
    if not config.connection_string:
        validation_results["errors"].append("Connection string is required")
        validation_results["valid"] = False
    
    # Connector-specific validations
    if config.connector_type == ConnectorType.SNOWFLAKE:
        if not config.warehouse:
            validation_results["warnings"].append("Snowflake warehouse not specified")
        if not config.role:
            validation_results["warnings"].append("Snowflake role not specified")
    
    elif config.connector_type == ConnectorType.BIGQUERY:
        if not config.ssl_config.get("service_account_path"):
            validation_results["warnings"].append("BigQuery service account not specified")
    
    elif config.connector_type == ConnectorType.REDSHIFT:
        if not config.ssl_config.get("sslmode"):
            validation_results["recommendations"].append("Consider enabling SSL for Redshift")
    
    # Performance recommendations
    if config.optimization.batch_size > 1000000:
        validation_results["warnings"].append("Large batch size may cause memory issues")
    
    if config.optimization.parallel_degree > 16:
        validation_results["warnings"].append("High parallelism may overwhelm the system")
    
    if not config.optimization.enable_pushdown:
        validation_results["recommendations"].append("Enable pushdown optimization for better performance")
    
    return validation_results


# Security utilities
def mask_sensitive_config(config: ConnectionConfig) -> Dict[str, Any]:
    """Mask sensitive information in configuration for logging."""
    
    config_dict = config.__dict__.copy()
    
    # Mask password
    if config_dict.get("password"):
        config_dict["password"] = "***MASKED***"
    
    # Mask SSL configuration
    if config_dict.get("ssl_config"):
        ssl_config = config_dict["ssl_config"].copy()
        for key in ["account_key", "sas_token", "service_account_path"]:
            if key in ssl_config:
                ssl_config[key] = "***MASKED***"
        config_dict["ssl_config"] = ssl_config
    
    return config_dict