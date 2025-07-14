import pandas as pd
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self._connection = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    def load_table(self, table_name: str, schema: Optional[str] = None, 
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from a database table."""
        pass
    
    @abstractmethod
    def load_query(self, query: str) -> pd.DataFrame:
        """Load data using a custom SQL query."""
        pass
    
    @abstractmethod
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of available tables."""
        pass
    
    @abstractmethod
    def get_schema_names(self) -> List[str]:
        """Get list of available schemas."""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.engine = None
    
    def connect(self) -> None:
        """Establish PostgreSQL connection."""
        try:
            import psycopg2
            from sqlalchemy import create_engine
            
            # Build connection string
            host = self.connection_params.get('host', 'localhost')
            port = self.connection_params.get('port', 5432)
            database = self.connection_params.get('database')
            username = self.connection_params.get('username')
            password = self.connection_params.get('password')
            
            if not all([database, username, password]):
                raise ValueError("Missing required connection parameters: database, username, password")
            
            connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info(f"Connected to PostgreSQL database: {database}")
            
        except ImportError:
            raise ImportError("psycopg2 and sqlalchemy are required for PostgreSQL connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Disconnected from PostgreSQL")
    
    def load_table(self, table_name: str, schema: Optional[str] = None, 
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from a PostgreSQL table."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        query = f'SELECT * FROM {schema + "." if schema else ""}{table_name}'
        if limit:
            query += f' LIMIT {limit}'
        
        return pd.read_sql(query, self.engine)
    
    def load_query(self, query: str) -> pd.DataFrame:
        """Load data using a custom SQL query."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        return pd.read_sql(query, self.engine)
    
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of available tables."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        if schema:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            result = pd.read_sql(query, self.engine, params=[schema])
        else:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            result = pd.read_sql(query, self.engine)
        
        return result['table_name'].tolist()
    
    def get_schema_names(self) -> List[str]:
        """Get list of available schemas."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        query = """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name
        """
        result = pd.read_sql(query, self.engine)
        return result['schema_name'].tolist()


class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.engine = None
    
    def connect(self) -> None:
        """Establish MySQL connection."""
        try:
            import pymysql
            from sqlalchemy import create_engine
            
            # Build connection string
            host = self.connection_params.get('host', 'localhost')
            port = self.connection_params.get('port', 3306)
            database = self.connection_params.get('database')
            username = self.connection_params.get('username')
            password = self.connection_params.get('password')
            
            if not all([database, username, password]):
                raise ValueError("Missing required connection parameters: database, username, password")
            
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info(f"Connected to MySQL database: {database}")
            
        except ImportError:
            raise ImportError("pymysql and sqlalchemy are required for MySQL connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close MySQL connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Disconnected from MySQL")
    
    def load_table(self, table_name: str, schema: Optional[str] = None, 
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from a MySQL table."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        query = f'SELECT * FROM {schema + "." if schema else ""}{table_name}'
        if limit:
            query += f' LIMIT {limit}'
        
        return pd.read_sql(query, self.engine)
    
    def load_query(self, query: str) -> pd.DataFrame:
        """Load data using a custom SQL query."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        return pd.read_sql(query, self.engine)
    
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of available tables."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        if schema:
            query = "SHOW TABLES FROM " + schema
        else:
            query = "SHOW TABLES"
        
        result = pd.read_sql(query, self.engine)
        # MySQL SHOW TABLES returns different column names
        column_name = result.columns[0]
        return result[column_name].tolist()
    
    def get_schema_names(self) -> List[str]:
        """Get list of available schemas (databases in MySQL)."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        query = "SHOW DATABASES"
        result = pd.read_sql(query, self.engine)
        # Filter out system databases
        databases = result['Database'].tolist()
        return [db for db in databases if db not in ['information_schema', 'performance_schema', 'mysql', 'sys']]


class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter."""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params)
        self.engine = None
    
    def connect(self) -> None:
        """Establish SQLite connection."""
        try:
            from sqlalchemy import create_engine
            
            # SQLite connection string
            database_path = self.connection_params.get('database_path')
            if not database_path:
                raise ValueError("Missing required parameter: database_path")
            
            connection_string = f"sqlite:///{database_path}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            logger.info(f"Connected to SQLite database: {database_path}")
            
        except ImportError:
            raise ImportError("sqlalchemy is required for SQLite connectivity")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close SQLite connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            logger.info("Disconnected from SQLite")
    
    def load_table(self, table_name: str, schema: Optional[str] = None, 
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from a SQLite table."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        query = f'SELECT * FROM {table_name}'
        if limit:
            query += f' LIMIT {limit}'
        
        return pd.read_sql(query, self.engine)
    
    def load_query(self, query: str) -> pd.DataFrame:
        """Load data using a custom SQL query."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        return pd.read_sql(query, self.engine)
    
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of available tables."""
        if not self.engine:
            raise RuntimeError("Not connected to database")
        
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        result = pd.read_sql(query, self.engine)
        return result['name'].tolist()
    
    def get_schema_names(self) -> List[str]:
        """Get list of available schemas (not applicable for SQLite)."""
        return []  # SQLite doesn't have schemas


def get_database_adapter(db_type: str, connection_params: Dict[str, Any]) -> DatabaseAdapter:
    """Factory function to get the appropriate database adapter."""
    adapters = {
        'postgresql': PostgreSQLAdapter,
        'postgres': PostgreSQLAdapter,
        'mysql': MySQLAdapter,
        'sqlite': SQLiteAdapter
    }
    
    db_type_lower = db_type.lower()
    if db_type_lower not in adapters:
        raise ValueError(f"Unsupported database type: {db_type}. Supported types: {list(adapters.keys())}")
    
    return adapters[db_type_lower](connection_params)


class DatabaseProfiler:
    """Helper class for profiling database sources."""
    
    def __init__(self, adapter: DatabaseAdapter):
        self.adapter = adapter
    
    def profile_table(self, table_name: str, schema: Optional[str] = None, 
                     sample_size: Optional[int] = None) -> pd.DataFrame:
        """Profile a specific database table."""
        return self.adapter.load_table(table_name, schema, sample_size)
    
    def profile_query(self, query: str) -> pd.DataFrame:
        """Profile results from a custom query."""
        return self.adapter.load_query(query)
    
    def get_table_info(self, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get information about available tables."""
        tables = self.adapter.get_table_names(schema)
        schemas = self.adapter.get_schema_names()
        
        return {
            'schemas': schemas,
            'tables': tables,
            'total_tables': len(tables)
        }
    
    def estimate_table_size(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Estimate table size and row count."""
        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {schema + '.' if schema else ''}{table_name}"
            count_result = self.adapter.load_query(count_query)
            row_count = int(count_result['row_count'].iloc[0])
            
            # Sample a few rows to estimate memory usage
            sample_df = self.adapter.load_table(table_name, schema, limit=100)
            avg_row_size = sample_df.memory_usage(deep=True).sum() / len(sample_df) if len(sample_df) > 0 else 0
            estimated_size_mb = (row_count * avg_row_size) / (1024 * 1024)
            
            return {
                'row_count': row_count,
                'column_count': len(sample_df.columns),
                'estimated_size_mb': estimated_size_mb,
                'sample_columns': list(sample_df.columns)
            }
        except Exception as e:
            logger.warning(f"Failed to estimate size for table {table_name}: {e}")
            return {
                'row_count': 0,
                'column_count': 0,
                'estimated_size_mb': 0,
                'error': str(e)
            }