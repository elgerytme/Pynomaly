#!/usr/bin/env python3
"""Command-line interface for data profiling."""

import click
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from application.use_cases.profile_dataset import ProfileDatasetUseCase
from application.services.performance_optimizer import PerformanceOptimizer
from infrastructure.adapters.data_source_adapter import DataSourceFactory
from infrastructure.adapters.cloud_storage_adapter import get_cloud_storage_adapter
from infrastructure.adapters.streaming_adapter import get_streaming_adapter, StreamingDataProfiler
from infrastructure.adapters.nosql_adapter import get_nosql_adapter, NoSQLProfiler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """Data Profiling CLI - Comprehensive data analysis tool."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--strategy', '-s', default='full', 
              type=click.Choice(['full', 'sample', 'adaptive']),
              help='Profiling strategy')
@click.option('--sample-size', '-n', type=int, 
              help='Sample size for sampling strategies')
@click.option('--sample-percentage', '-p', type=float,
              help='Sample percentage (0-100)')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for results (JSON)')
@click.option('--format', '-f', type=click.Choice(['json', 'table', 'summary']),
              default='summary', help='Output format')
@click.option('--optimize', is_flag=True,
              help='Enable performance optimization')
def profile_file(file_path, strategy, sample_size, sample_percentage, 
                output, format, optimize):
    """Profile a file dataset."""
    try:
        click.echo(f"🔍 Profiling file: {file_path}")
        
        use_case = ProfileDatasetUseCase()
        
        # Execute profiling based on strategy
        if strategy == "sample" and sample_size:
            click.echo(f"📊 Using sample strategy with {sample_size} rows")
            profile = use_case.execute_sample(file_path, sample_size)
        elif strategy == "sample" and sample_percentage:
            click.echo(f"📊 Using sample strategy with {sample_percentage}% of data")
            profile = use_case.execute_percentage_sample(file_path, sample_percentage)
        else:
            if optimize:
                click.echo("⚡ Performance optimization enabled")
            profile = use_case.execute(file_path, strategy)
        
        # Get summary
        summary = use_case.get_profiling_summary(profile)
        
        # Output results
        if output:
            with open(output, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            click.echo(f"✅ Results saved to: {output}")
        
        # Display results based on format
        if format == 'json':
            click.echo(json.dumps(summary, indent=2, default=str))
        elif format == 'table':
            _display_table_format(summary)
        else:
            _display_summary_format(summary)
        
        click.echo(f"✅ Profiling completed in {summary.get('execution_time_seconds', 0):.2f} seconds")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--db-type', '-t', required=True,
              type=click.Choice(['postgresql', 'mysql', 'sqlite']),
              help='Database type')
@click.option('--host', '-h', default='localhost', help='Database host')
@click.option('--port', '-p', type=int, help='Database port')
@click.option('--database', '-d', required=True, help='Database name')
@click.option('--username', '-u', help='Database username')
@click.option('--password', '-w', help='Database password')
@click.option('--table', help='Table name to profile')
@click.option('--query', help='Custom SQL query')
@click.option('--sample-size', '-n', type=int, help='Sample size limit')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--format', '-f', type=click.Choice(['json', 'table', 'summary']),
              default='summary', help='Output format')
def profile_database(db_type, host, port, database, username, password, 
                    table, query, sample_size, output, format):
    """Profile a database table or query."""
    try:
        # Build connection parameters
        connection_params = {
            'host': host,
            'database': database
        }
        
        if port:
            connection_params['port'] = port
        if username:
            connection_params['username'] = username
        if password:
            connection_params['password'] = password
        
        click.echo(f"🔍 Connecting to {db_type} database: {database}")
        
        # Create data source
        source = DataSourceFactory.create_database_source(db_type, connection_params)
        
        # Load data
        load_kwargs = {}
        if table:
            load_kwargs['table_name'] = table
            click.echo(f"📊 Profiling table: {table}")
        elif query:
            load_kwargs['query'] = query
            click.echo(f"📊 Profiling query: {query}")
        else:
            raise click.ClickException("Either --table or --query must be provided")
        
        if sample_size:
            load_kwargs['limit'] = sample_size
            click.echo(f"📈 Using sample size: {sample_size}")
        
        df = source.load_data(**load_kwargs)
        click.echo(f"📦 Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Create temporary file for profiling
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_path = temp_file.name
        
        try:
            use_case = ProfileDatasetUseCase()
            profile = use_case.execute(temp_path, 'full')
            summary = use_case.get_profiling_summary(profile)
            
            # Add database-specific information
            summary['source_info'] = {
                'type': 'database',
                'db_type': db_type,
                'host': host,
                'database': database,
                'table': table,
                'query': query
            }
            
            # Output results
            if output:
                with open(output, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                click.echo(f"✅ Results saved to: {output}")
            
            # Display results
            if format == 'json':
                click.echo(json.dumps(summary, indent=2, default=str))
            elif format == 'table':
                _display_table_format(summary)
            else:
                _display_summary_format(summary)
            
            click.echo(f"✅ Database profiling completed")
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--provider', '-p', required=True,
              type=click.Choice(['s3', 'azure', 'gcs']),
              help='Cloud storage provider')
@click.option('--bucket', '-b', required=True, help='Bucket/container name')
@click.option('--object-key', '-k', required=True, help='Object key/path')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='JSON config file with credentials')
@click.option('--region', help='AWS region (for S3)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def profile_cloud(provider, bucket, object_key, config, region, output):
    """Profile data from cloud storage."""
    try:
        # Load configuration
        if config:
            with open(config, 'r') as f:
                connection_config = json.load(f)
        else:
            connection_config = {}
        
        # Add provider-specific configuration
        if provider == 's3':
            connection_config['bucket_name'] = bucket
            if region:
                connection_config['region_name'] = region
        elif provider == 'azure':
            connection_config['container_name'] = bucket
        elif provider == 'gcs':
            connection_config['bucket_name'] = bucket
        
        click.echo(f"☁️  Connecting to {provider.upper()}: {bucket}/{object_key}")
        
        # Create adapter and load data
        adapter = get_cloud_storage_adapter(provider, connection_config)
        
        with adapter:
            df = adapter.load_object(object_key)
            click.echo(f"📦 Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Create temporary file for profiling
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name
            
            try:
                use_case = ProfileDatasetUseCase()
                profile = use_case.execute(temp_path, 'full')
                summary = use_case.get_profiling_summary(profile)
                
                # Add cloud-specific information
                summary['source_info'] = {
                    'type': 'cloud_storage',
                    'provider': provider,
                    'bucket': bucket,
                    'object_key': object_key
                }
                
                # Output results
                if output:
                    with open(output, 'w') as f:
                        json.dump(summary, f, indent=2, default=str)
                    click.echo(f"✅ Results saved to: {output}")
                
                _display_summary_format(summary)
                click.echo(f"✅ Cloud storage profiling completed")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--provider', '-p', required=True,
              type=click.Choice(['kafka', 'kinesis']),
              help='Streaming provider')
@click.option('--stream', '-s', required=True, help='Stream/topic name')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='JSON config file with connection details')
@click.option('--sample-size', '-n', default=1000, type=int,
              help='Number of messages to sample')
@click.option('--timeout', '-t', default=30, type=int,
              help='Timeout in seconds')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def profile_stream(provider, stream, config, sample_size, timeout, output):
    """Profile streaming data."""
    try:
        # Load configuration
        if config:
            with open(config, 'r') as f:
                connection_config = json.load(f)
        else:
            connection_config = {}
        
        click.echo(f"🌊 Connecting to {provider.upper()} stream: {stream}")
        
        # Create adapter and profiler
        adapter = get_streaming_adapter(provider, connection_config)
        profiler = StreamingDataProfiler(adapter)
        
        with adapter:
            click.echo(f"📡 Sampling {sample_size} messages (timeout: {timeout}s)")
            
            profile_result = profiler.profile_stream(
                stream,
                sample_size=sample_size,
                timeout_seconds=timeout
            )
            
            if profile_result['success']:
                click.echo(f"📦 Processed {profile_result['sample_size']} messages")
                
                # Output results
                if output:
                    with open(output, 'w') as f:
                        json.dump(profile_result, f, indent=2, default=str)
                    click.echo(f"✅ Results saved to: {output}")
                
                _display_streaming_summary(profile_result)
                click.echo(f"✅ Streaming data profiling completed")
            else:
                click.echo(f"❌ Profiling failed: {profile_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--db-type', '-t', required=True,
              type=click.Choice(['mongodb', 'cassandra', 'dynamodb']),
              help='NoSQL database type')
@click.option('--collection', '-c', required=True, help='Collection/table name')
@click.option('--config', type=click.Path(exists=True), required=True,
              help='JSON config file with connection details')
@click.option('--sample-size', '-n', type=int, help='Sample size limit')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
def profile_nosql(db_type, collection, config, sample_size, output):
    """Profile NoSQL database collection."""
    try:
        # Load configuration
        with open(config, 'r') as f:
            connection_config = json.load(f)
        
        click.echo(f"🗃️  Connecting to {db_type.upper()}: {collection}")
        
        # Create adapter and profiler
        adapter = get_nosql_adapter(db_type, connection_config)
        profiler = NoSQLProfiler(adapter)
        
        with adapter:
            # Load collection data
            load_kwargs = {}
            if sample_size:
                load_kwargs['limit'] = sample_size
                click.echo(f"📈 Using sample size: {sample_size}")
            
            df = adapter.load_collection(collection, **load_kwargs)
            click.echo(f"📦 Loaded {len(df)} documents, {len(df.columns)} fields")
            
            # Get collection info
            collection_info = adapter.get_collection_info(collection)
            
            # Create temporary file for profiling
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name
            
            try:
                use_case = ProfileDatasetUseCase()
                profile = use_case.execute(temp_path, 'full')
                summary = use_case.get_profiling_summary(profile)
                
                # Add NoSQL-specific information
                summary['source_info'] = {
                    'type': 'nosql',
                    'db_type': db_type,
                    'collection': collection,
                    'collection_info': collection_info
                }
                
                # Output results
                if output:
                    with open(output, 'w') as f:
                        json.dump(summary, f, indent=2, default=str)
                    click.echo(f"✅ Results saved to: {output}")
                
                _display_summary_format(summary)
                click.echo(f"✅ NoSQL profiling completed")
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def system_info():
    """Display system resource information."""
    try:
        optimizer = PerformanceOptimizer()
        resources = optimizer.check_system_resources()
        
        click.echo("💻 System Resources:")
        click.echo(f"  Memory: {resources['memory']['available_gb']:.1f}GB available "
                  f"/ {resources['memory']['total_gb']:.1f}GB total "
                  f"({resources['memory']['used_percent']:.1f}% used)")
        click.echo(f"  CPU: {resources['cpu']['count']} cores, "
                  f"{resources['cpu']['usage_percent']:.1f}% usage")
        
        if resources['recommendations']:
            click.echo("\n⚠️  Recommendations:")
            for rec in resources['recommendations']:
                click.echo(f"  • {rec}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_formats():
    """List supported file formats and data sources."""
    formats = {
        "File Formats": {
            "csv": "Comma-Separated Values",
            "json": "JavaScript Object Notation",
            "jsonl": "JSON Lines",
            "parquet": "Apache Parquet",
            "xlsx": "Excel (newer format)",
            "xls": "Excel (legacy format)",
            "tsv": "Tab-Separated Values",
            "feather": "Apache Arrow Feather"
        },
        "Databases": {
            "postgresql": "PostgreSQL",
            "mysql": "MySQL",
            "sqlite": "SQLite"
        },
        "NoSQL Databases": {
            "mongodb": "MongoDB",
            "cassandra": "Apache Cassandra",
            "dynamodb": "AWS DynamoDB"
        },
        "Cloud Storage": {
            "s3": "Amazon S3",
            "azure": "Azure Blob Storage",
            "gcs": "Google Cloud Storage"
        },
        "Streaming": {
            "kafka": "Apache Kafka",
            "kinesis": "AWS Kinesis"
        }
    }
    
    for category, items in formats.items():
        click.echo(f"\n📋 {category}:")
        for key, description in items.items():
            click.echo(f"  {key:<12} - {description}")


def _display_summary_format(summary: Dict[str, Any]) -> None:
    """Display profiling summary in a readable format."""
    click.echo(f"\n📊 Profiling Summary:")
    click.echo(f"  Profile ID: {summary.get('profile_id', 'N/A')}")
    click.echo(f"  Status: {summary.get('status', 'N/A')}")
    click.echo(f"  Execution Time: {summary.get('execution_time_seconds', 0):.2f}s")
    click.echo(f"  Memory Usage: {summary.get('memory_usage_mb', 0):.1f}MB")
    
    click.echo(f"\n📈 Dataset Overview:")
    click.echo(f"  Total Rows: {summary.get('total_rows', 0):,}")
    click.echo(f"  Total Columns: {summary.get('total_columns', 0)}")
    click.echo(f"  Overall Quality Score: {summary.get('overall_quality_score', 0):.3f}")
    
    if summary.get('data_types'):
        click.echo(f"\n🏷️  Data Types:")
        for dtype, count in summary['data_types'].items():
            click.echo(f"  {dtype}: {count} columns")
    
    if summary.get('quality_issues'):
        issues = summary['quality_issues']
        total_issues = sum(issues.values())
        if total_issues > 0:
            click.echo(f"\n⚠️  Quality Issues ({total_issues} total):")
            for severity, count in issues.items():
                if count > 0:
                    click.echo(f"  {severity.title()}: {count}")
    
    if summary.get('patterns_discovered', 0) > 0:
        click.echo(f"\n🔍 Patterns Discovered: {summary['patterns_discovered']}")
    
    if summary.get('recommendations'):
        click.echo(f"\n💡 Recommendations:")
        for rec in summary['recommendations']:
            click.echo(f"  • {rec}")


def _display_table_format(summary: Dict[str, Any]) -> None:
    """Display profiling summary in table format."""
    from tabulate import tabulate
    
    # Basic metrics table
    basic_data = [
        ["Profile ID", summary.get('profile_id', 'N/A')],
        ["Status", summary.get('status', 'N/A')],
        ["Execution Time", f"{summary.get('execution_time_seconds', 0):.2f}s"],
        ["Memory Usage", f"{summary.get('memory_usage_mb', 0):.1f}MB"],
        ["Total Rows", f"{summary.get('total_rows', 0):,}"],
        ["Total Columns", summary.get('total_columns', 0)],
        ["Quality Score", f"{summary.get('overall_quality_score', 0):.3f}"],
        ["Patterns Found", summary.get('patterns_discovered', 0)]
    ]
    
    click.echo("\n📊 Profiling Summary:")
    click.echo(tabulate(basic_data, headers=["Metric", "Value"], tablefmt="grid"))
    
    # Data types table
    if summary.get('data_types'):
        dtype_data = [[dtype, count] for dtype, count in summary['data_types'].items()]
        click.echo("\n🏷️  Data Types:")
        click.echo(tabulate(dtype_data, headers=["Type", "Count"], tablefmt="grid"))


def _display_streaming_summary(result: Dict[str, Any]) -> None:
    """Display streaming profiling summary."""
    if not result.get('success'):
        click.echo(f"❌ Profiling failed: {result.get('error', 'Unknown error')}")
        return
    
    stream_info = result.get('stream_info', {})
    profile = result.get('profile', {})
    
    click.echo(f"\n🌊 Stream Information:")
    for key, value in stream_info.items():
        if key != 'error':
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")
    
    if profile:
        click.echo(f"\n📊 Data Profile:")
        click.echo(f"  Messages Analyzed: {profile.get('row_count', 0)}")
        click.echo(f"  Fields Found: {profile.get('column_count', 0)}")
        click.echo(f"  Data Size: {profile.get('memory_usage_mb', 0):.1f}MB")
        
        if profile.get('message_frequency'):
            freq = profile['message_frequency']
            click.echo(f"  Message Rate: {freq.get('messages_per_second', 0):.2f} msg/s")
        
        if profile.get('data_freshness'):
            freshness = profile['data_freshness']
            click.echo(f"  Average Age: {freshness.get('average_age_seconds', 0):.1f}s")


if __name__ == '__main__':
    cli()