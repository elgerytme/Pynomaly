"""NoSQL database adapters for data profiling."""

import pandas as pd
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import logging
import json

logger = logging.getLogger(__name__)


class NoSQLAdapter(ABC):
    """Abstract base class for NoSQL database adapters."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self._client = None
        self._database = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    def load_collection(self, collection_name: str, 
                       query: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from a collection/table."""
        pass
    
    @abstractmethod
    def get_collection_names(self) -> List[str]:
        """Get list of available collections/tables."""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a specific collection."""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class MongoDBAdapter(NoSQLAdapter):
    """MongoDB database adapter."""
    
    def connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            import pymongo
            from pymongo import MongoClient
            
            # Extract connection parameters
            host = self.connection_params.get('host', 'localhost')
            port = self.connection_params.get('port', 27017)
            database_name = self.connection_params.get('database')
            username = self.connection_params.get('username')
            password = self.connection_params.get('password')
            auth_database = self.connection_params.get('auth_database', 'admin')
            
            if not database_name:
                raise ValueError("Database name is required for MongoDB")
            
            # Build connection string
            if username and password:
                uri = f"mongodb://{username}:{password}@{host}:{port}/{auth_database}"
            else:
                uri = f"mongodb://{host}:{port}"
            
            # Additional connection options
            connection_options = {
                'serverSelectionTimeoutMS': self.connection_params.get('timeout_ms', 5000),
                'maxPoolSize': self.connection_params.get('max_pool_size', 10)
            }
            
            self._client = MongoClient(uri, **connection_options)
            self._database = self._client[database_name]
            
            # Test connection
            self._client.admin.command('ismaster')
            logger.info(f"Connected to MongoDB database: {database_name}")
            
        except ImportError:
            raise ImportError("pymongo is required for MongoDB connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("Disconnected from MongoDB")
    
    def load_collection(self, collection_name: str, 
                       query: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from MongoDB collection."""
        if not self._database:
            raise RuntimeError("Not connected to MongoDB")
        
        try:
            collection = self._database[collection_name]
            
            # Build query
            find_query = query or {}
            
            # Execute query
            if limit:
                cursor = collection.find(find_query).limit(limit)
            else:
                cursor = collection.find(find_query)
            
            # Convert to list of dictionaries
            documents = list(cursor)
            
            if not documents:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.json_normalize(documents)
            
            # Handle ObjectId conversion
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            
            logger.info(f"Loaded {len(df)} documents from MongoDB collection {collection_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load MongoDB collection {collection_name}: {e}")
            raise
    
    def get_collection_names(self) -> List[str]:
        """Get list of available collections."""
        if not self._database:
            raise RuntimeError("Not connected to MongoDB")
        
        try:
            return self._database.list_collection_names()
        except Exception as e:
            logger.error(f"Failed to list MongoDB collections: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about MongoDB collection."""
        if not self._database:
            raise RuntimeError("Not connected to MongoDB")
        
        try:
            collection = self._database[collection_name]
            
            # Get collection stats
            stats = self._database.command("collStats", collection_name)
            
            # Get sample document for schema inference
            sample_doc = collection.find_one()
            
            # Analyze field types
            field_info = {}
            if sample_doc:
                field_info = self._analyze_document_fields(sample_doc)
            
            return {
                'collection_name': collection_name,
                'document_count': stats.get('count', 0),
                'size_bytes': stats.get('size', 0),
                'storage_size_bytes': stats.get('storageSize', 0),
                'average_object_size': stats.get('avgObjSize', 0),
                'indexes': len(stats.get('indexSizes', {})),
                'field_info': field_info,
                'sample_document': sample_doc
            }
            
        except Exception as e:
            logger.error(f"Failed to get MongoDB collection info for {collection_name}: {e}")
            return {'error': str(e)}
    
    def _analyze_document_fields(self, document: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
        """Analyze field types in a MongoDB document."""
        field_types = {}
        
        for key, value in document.items():
            field_name = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Nested document
                field_types[field_name] = "object"
                field_types.update(self._analyze_document_fields(value, field_name))
            elif isinstance(value, list):
                field_types[field_name] = "array"
                if value and isinstance(value[0], dict):
                    field_types.update(self._analyze_document_fields(value[0], f"{field_name}[0]"))
            else:
                field_types[field_name] = type(value).__name__
        
        return field_types


class CassandraAdapter(NoSQLAdapter):
    """Apache Cassandra database adapter."""
    
    def connect(self) -> None:
        """Establish Cassandra connection."""
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            
            # Extract connection parameters
            hosts = self.connection_params.get('hosts', ['localhost'])
            port = self.connection_params.get('port', 9042)
            keyspace = self.connection_params.get('keyspace')
            username = self.connection_params.get('username')
            password = self.connection_params.get('password')
            
            if not keyspace:
                raise ValueError("Keyspace is required for Cassandra")
            
            # Build cluster configuration
            cluster_kwargs = {'port': port}
            
            if username and password:
                auth_provider = PlainTextAuthProvider(username=username, password=password)
                cluster_kwargs['auth_provider'] = auth_provider
            
            cluster = Cluster(hosts, **cluster_kwargs)
            self._client = cluster.connect()
            self._client.set_keyspace(keyspace)
            
            logger.info(f"Connected to Cassandra keyspace: {keyspace}")
            
        except ImportError:
            raise ImportError("cassandra-driver is required for Cassandra connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to Cassandra: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Cassandra connection."""
        if self._client:
            self._client.shutdown()
            self._client = None
            logger.info("Disconnected from Cassandra")
    
    def load_collection(self, table_name: str, 
                       query: Optional[str] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from Cassandra table."""
        if not self._client:
            raise RuntimeError("Not connected to Cassandra")
        
        try:
            # Build CQL query
            if query:
                cql_query = query
            else:
                cql_query = f"SELECT * FROM {table_name}"
            
            if limit and "LIMIT" not in cql_query.upper():
                cql_query += f" LIMIT {limit}"
            
            # Execute query
            rows = self._client.execute(cql_query)
            
            # Convert to DataFrame
            if rows:
                data = []
                for row in rows:
                    data.append(dict(row._asdict()))
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} rows from Cassandra table {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Cassandra table {table_name}: {e}")
            raise
    
    def get_collection_names(self) -> List[str]:
        """Get list of available tables."""
        if not self._client:
            raise RuntimeError("Not connected to Cassandra")
        
        try:
            # Query system tables to get table names
            query = "SELECT table_name FROM system_schema.tables WHERE keyspace_name = %s"
            keyspace = self.connection_params.get('keyspace')
            rows = self._client.execute(query, [keyspace])
            
            return [row.table_name for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to list Cassandra tables: {e}")
            raise
    
    def get_collection_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about Cassandra table."""
        if not self._client:
            raise RuntimeError("Not connected to Cassandra")
        
        try:
            keyspace = self.connection_params.get('keyspace')
            
            # Get table schema
            schema_query = """
                SELECT column_name, type, kind 
                FROM system_schema.columns 
                WHERE keyspace_name = %s AND table_name = %s
            """
            schema_rows = self._client.execute(schema_query, [keyspace, table_name])
            
            columns = []
            partition_keys = []
            clustering_keys = []
            
            for row in schema_rows:
                column_info = {
                    'name': row.column_name,
                    'type': row.type,
                    'kind': row.kind
                }
                columns.append(column_info)
                
                if row.kind == 'partition_key':
                    partition_keys.append(row.column_name)
                elif row.kind == 'clustering':
                    clustering_keys.append(row.column_name)
            
            # Get approximate row count (this is expensive in Cassandra)
            try:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                count_result = self._client.execute(count_query)
                row_count = count_result[0].count
            except Exception:
                row_count = None  # COUNT(*) might be disabled or too expensive
            
            return {
                'table_name': table_name,
                'keyspace': keyspace,
                'columns': columns,
                'partition_keys': partition_keys,
                'clustering_keys': clustering_keys,
                'column_count': len(columns),
                'estimated_row_count': row_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get Cassandra table info for {table_name}: {e}")
            return {'error': str(e)}


class DynamoDBAdapter(NoSQLAdapter):
    """AWS DynamoDB adapter."""
    
    def connect(self) -> None:
        """Establish DynamoDB connection."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Extract connection parameters
            aws_access_key_id = self.connection_params.get('aws_access_key_id')
            aws_secret_access_key = self.connection_params.get('aws_secret_access_key')
            region_name = self.connection_params.get('region_name', 'us-east-1')
            endpoint_url = self.connection_params.get('endpoint_url')  # For local DynamoDB
            
            client_kwargs = {'region_name': region_name}
            
            if aws_access_key_id and aws_secret_access_key:
                client_kwargs.update({
                    'aws_access_key_id': aws_access_key_id,
                    'aws_secret_access_key': aws_secret_access_key
                })
            
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url
            
            self._client = boto3.client('dynamodb', **client_kwargs)
            self._resource = boto3.resource('dynamodb', **client_kwargs)
            
            # Test connection
            self._client.list_tables(Limit=1)
            logger.info("Connected to DynamoDB successfully")
            
        except ImportError:
            raise ImportError("boto3 is required for DynamoDB connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to DynamoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close DynamoDB connection."""
        self._client = None
        self._resource = None
        logger.info("Disconnected from DynamoDB")
    
    def load_collection(self, table_name: str, 
                       query: Optional[Dict[str, Any]] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from DynamoDB table."""
        if not self._resource:
            raise RuntimeError("Not connected to DynamoDB")
        
        try:
            table = self._resource.Table(table_name)
            
            # Scan table (DynamoDB doesn't have simple SELECT *)
            scan_kwargs = {}
            if limit:
                scan_kwargs['Limit'] = limit
            
            # Add filter expression if query provided
            if query:
                # This is a simplified implementation
                # Real implementation would need proper expression building
                logger.warning("Query filtering not fully implemented for DynamoDB")
            
            response = table.scan(**scan_kwargs)
            items = response.get('Items', [])
            
            # Handle pagination if needed (simplified)
            while 'LastEvaluatedKey' in response and len(items) < (limit or float('inf')):
                if limit and len(items) >= limit:
                    break
                
                scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
                response = table.scan(**scan_kwargs)
                items.extend(response.get('Items', []))
            
            # Convert to DataFrame
            if items:
                df = pd.json_normalize(items)
            else:
                df = pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} items from DynamoDB table {table_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load DynamoDB table {table_name}: {e}")
            raise
    
    def get_collection_names(self) -> List[str]:
        """Get list of available tables."""
        if not self._client:
            raise RuntimeError("Not connected to DynamoDB")
        
        try:
            response = self._client.list_tables()
            return response.get('TableNames', [])
            
        except Exception as e:
            logger.error(f"Failed to list DynamoDB tables: {e}")
            raise
    
    def get_collection_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about DynamoDB table."""
        if not self._client:
            raise RuntimeError("Not connected to DynamoDB")
        
        try:
            response = self._client.describe_table(TableName=table_name)
            table_desc = response['Table']
            
            # Extract key information
            key_schema = table_desc.get('KeySchema', [])
            attribute_definitions = table_desc.get('AttributeDefinitions', [])
            
            partition_key = None
            sort_key = None
            
            for key in key_schema:
                if key['KeyType'] == 'HASH':
                    partition_key = key['AttributeName']
                elif key['KeyType'] == 'RANGE':
                    sort_key = key['AttributeName']
            
            return {
                'table_name': table_name,
                'table_status': table_desc.get('TableStatus'),
                'item_count': table_desc.get('ItemCount'),
                'table_size_bytes': table_desc.get('TableSizeBytes'),
                'partition_key': partition_key,
                'sort_key': sort_key,
                'attribute_definitions': attribute_definitions,
                'provisioned_throughput': table_desc.get('ProvisionedThroughput'),
                'billing_mode': table_desc.get('BillingModeSummary', {}).get('BillingMode'),
                'global_secondary_indexes': len(table_desc.get('GlobalSecondaryIndexes', [])),
                'local_secondary_indexes': len(table_desc.get('LocalSecondaryIndexes', []))
            }
            
        except Exception as e:
            logger.error(f"Failed to get DynamoDB table info for {table_name}: {e}")
            return {'error': str(e)}


class NoSQLProfiler:
    """Helper class for profiling NoSQL sources."""
    
    def __init__(self, adapter: NoSQLAdapter):
        self.adapter = adapter
    
    def profile_collections(self, collection_names: Optional[List[str]] = None,
                          sample_size: int = 1000) -> Dict[str, Any]:
        """Profile collections in the NoSQL database."""
        if collection_names is None:
            collection_names = self.adapter.get_collection_names()
        
        profiles = {}
        
        for collection_name in collection_names:
            try:
                # Get collection info
                collection_info = self.adapter.get_collection_info(collection_name)
                
                # Load sample data
                df = self.adapter.load_collection(collection_name, limit=sample_size)
                
                # Perform basic profiling
                profile = {
                    'collection_info': collection_info,
                    'sample_size': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict() if len(df) > 0 else {},
                    'null_counts': df.isnull().sum().to_dict() if len(df) > 0 else {},
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024) if len(df) > 0 else 0,
                    'success': True
                }
                
                profiles[collection_name] = profile
                
            except Exception as e:
                logger.error(f"Failed to profile collection {collection_name}: {e}")
                profiles[collection_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return profiles
    
    def estimate_database_size(self) -> Dict[str, Any]:
        """Estimate total database size and collection count."""
        try:
            collections = self.adapter.get_collection_names()
            total_size = 0
            total_documents = 0
            
            for collection_name in collections:
                try:
                    info = self.adapter.get_collection_info(collection_name)
                    if isinstance(info, dict) and 'error' not in info:
                        # Extract size information based on adapter type
                        if 'size_bytes' in info:  # MongoDB
                            total_size += info.get('size_bytes', 0)
                            total_documents += info.get('document_count', 0)
                        elif 'table_size_bytes' in info:  # DynamoDB
                            total_size += info.get('table_size_bytes', 0)
                            total_documents += info.get('item_count', 0)
                except Exception:
                    continue
            
            return {
                'total_collections': len(collections),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'total_documents': total_documents,
                'average_collection_size_mb': (total_size / (1024 * 1024)) / len(collections) if collections else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate database size: {e}")
            return {'error': str(e)}


def get_nosql_adapter(db_type: str, connection_params: Dict[str, Any]) -> NoSQLAdapter:
    """Factory function to get the appropriate NoSQL adapter."""
    adapters = {
        'mongodb': MongoDBAdapter,
        'mongo': MongoDBAdapter,
        'cassandra': CassandraAdapter,
        'dynamodb': DynamoDBAdapter,
        'dynamo': DynamoDBAdapter
    }
    
    db_type_lower = db_type.lower()
    if db_type_lower not in adapters:
        raise ValueError(f"Unsupported NoSQL database type: {db_type}. Supported: {list(adapters.keys())}")
    
    return adapters[db_type_lower](connection_params)