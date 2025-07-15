import pandas as pd
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class FileAdapter:
    """Base class for file adapters."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = options or {}
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load data from the given path into a pandas DataFrame."""
        raise NotImplementedError("load must be implemented by subclasses")
    
    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get information about the file."""
        import os
        try:
            stat = os.stat(path)
            return {
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'last_modified': stat.st_mtime,
                'exists': True
            }
        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }

class CSVAdapter(FileAdapter):
    """CSV file adapter with enhanced options."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with configurable options."""
        try:
            # Default options
            options = {
                'encoding': 'utf-8',
                'low_memory': False,
                'parse_dates': True,
                'infer_datetime_format': True
            }
            options.update(self.options)
            options.update(kwargs)
            
            logger.info(f"Loading CSV file: {path}")
            df = pd.read_csv(path, **options)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    logger.warning(f"Retrying with encoding: {encoding}")
                    options['encoding'] = encoding
                    return pd.read_csv(path, **options)
                except UnicodeDecodeError:
                    continue
            raise
        except Exception as e:
            logger.error(f"Failed to load CSV file {path}: {e}")
            raise

class JSONAdapter(FileAdapter):
    """JSON file adapter with enhanced options."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load JSON file with automatic format detection."""
        try:
            options = self.options.copy()
            options.update(kwargs)
            
            logger.info(f"Loading JSON file: {path}")
            
            # Try JSONL format first (lines=True)
            try:
                df = pd.read_json(path, lines=True, **options)
                logger.info(f"Loaded JSONL format: {len(df)} rows and {len(df.columns)} columns")
                return df
            except ValueError:
                # Try regular JSON format
                df = pd.read_json(path, **options)
                logger.info(f"Loaded JSON format: {len(df)} rows and {len(df.columns)} columns")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load JSON file {path}: {e}")
            raise

class ParquetAdapter(FileAdapter):
    """Parquet file adapter with enhanced options."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            options = self.options.copy()
            options.update(kwargs)
            
            logger.info(f"Loading Parquet file: {path}")
            df = pd.read_parquet(path, **options)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except ImportError:
            raise ImportError("pyarrow or fastparquet is required for Parquet support")
        except Exception as e:
            logger.error(f"Failed to load Parquet file {path}: {e}")
            raise

class ExcelAdapter(FileAdapter):
    """Excel file adapter."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        try:
            options = self.options.copy()
            options.update(kwargs)
            
            logger.info(f"Loading Excel file: {path}")
            
            # If sheet_name not specified, load the first sheet
            if 'sheet_name' not in options:
                options['sheet_name'] = 0
            
            df = pd.read_excel(path, **options)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except ImportError:
            raise ImportError("openpyxl or xlrd is required for Excel support")
        except Exception as e:
            logger.error(f"Failed to load Excel file {path}: {e}")
            raise
    
    def get_sheet_names(self, path: str) -> list:
        """Get list of sheet names in Excel file."""
        try:
            excel_file = pd.ExcelFile(path)
            return excel_file.sheet_names
        except Exception as e:
            logger.error(f"Failed to get sheet names from {path}: {e}")
            return []


class TSVAdapter(FileAdapter):
    """TSV (Tab-Separated Values) file adapter."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load TSV file."""
        options = {
            'sep': '\t',
            'encoding': 'utf-8',
            'low_memory': False
        }
        options.update(self.options)
        options.update(kwargs)
        
        csv_adapter = CSVAdapter(options)
        return csv_adapter.load(path)


class FeatherAdapter(FileAdapter):
    """Feather file adapter."""
    
    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load Feather file."""
        try:
            options = self.options.copy()
            options.update(kwargs)
            
            logger.info(f"Loading Feather file: {path}")
            df = pd.read_feather(path, **options)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except ImportError:
            raise ImportError("pyarrow is required for Feather support")
        except Exception as e:
            logger.error(f"Failed to load Feather file {path}: {e}")
            raise


def get_file_adapter(path: str, options: Optional[Dict[str, Any]] = None) -> FileAdapter:
    """Return an appropriate FileAdapter based on file extension."""
    lower_path = path.lower()
    
    if lower_path.endswith('.csv'):
        return CSVAdapter(options)
    elif lower_path.endswith(('.json', '.jsonl')):
        return JSONAdapter(options)
    elif lower_path.endswith('.parquet'):
        return ParquetAdapter(options)
    elif lower_path.endswith(('.xlsx', '.xls')):
        return ExcelAdapter(options)
    elif lower_path.endswith('.tsv'):
        return TSVAdapter(options)
    elif lower_path.endswith('.feather'):
        return FeatherAdapter(options)
    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats: .csv, .json, .jsonl, .parquet, .xlsx, .xls, .tsv, .feather")


def detect_file_format(path: str) -> str:
    """Detect file format from file extension."""
    lower_path = path.lower()
    
    if lower_path.endswith('.csv'):
        return 'csv'
    elif lower_path.endswith('.json'):
        return 'json'
    elif lower_path.endswith('.jsonl'):
        return 'jsonl'
    elif lower_path.endswith('.parquet'):
        return 'parquet'
    elif lower_path.endswith('.xlsx'):
        return 'xlsx'
    elif lower_path.endswith('.xls'):
        return 'xls'
    elif lower_path.endswith('.tsv'):
        return 'tsv'
    elif lower_path.endswith('.feather'):
        return 'feather'
    else:
        return 'unknown'


def get_supported_formats() -> Dict[str, str]:
    """Get list of supported file formats."""
    return {
        'csv': 'Comma-Separated Values',
        'json': 'JavaScript Object Notation',
        'jsonl': 'JSON Lines',
        'parquet': 'Apache Parquet',
        'xlsx': 'Excel (newer format)',
        'xls': 'Excel (legacy format)',
        'tsv': 'Tab-Separated Values',
        'feather': 'Apache Arrow Feather'
    }