import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import json
import logging
from pathlib import Path
import chardet

logger = logging.getLogger(__name__)


class FileAdapter:
    """Enhanced adapter for various file formats."""
    
    def __init__(self, file_path: str, **kwargs):
        self.file_path = Path(file_path)
        self.kwargs = kwargs
        self.detected_encoding = None
        self.detected_separator = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file with automatic format detection."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        file_extension = self.file_path.suffix.lower()
        
        # Route to appropriate loader
        if file_extension == '.csv':
            return self._load_csv()
        elif file_extension in ['.json', '.jsonl']:
            return self._load_json()
        elif file_extension in ['.parquet', '.pq']:
            return self._load_parquet()
        elif file_extension in ['.xlsx', '.xls']:
            return self._load_excel()
        elif file_extension in ['.avro']:
            return self._load_avro()
        elif file_extension in ['.orc']:
            return self._load_orc()
        elif file_extension in ['.feather']:
            return self._load_feather()
        elif file_extension in ['.h5', '.hdf5']:
            return self._load_hdf5()
        elif file_extension in ['.pkl', '.pickle']:
            return self._load_pickle()
        elif file_extension in ['.tsv', '.tab']:
            return self._load_tsv()
        elif file_extension in ['.txt']:
            return self._load_text()
        else:
            # Try to auto-detect format
            return self._auto_detect_and_load()
    
    def _detect_encoding(self, sample_size: int = 10000) -> str:
        """Detect file encoding."""
        if self.detected_encoding:
            return self.detected_encoding
        
        try:
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(sample_size)
            
            result = chardet.detect(raw_data)
            self.detected_encoding = result['encoding']
            
            if self.detected_encoding is None:
                self.detected_encoding = 'utf-8'
            
            logger.info(f"Detected encoding: {self.detected_encoding} (confidence: {result['confidence']:.2f})")
            return self.detected_encoding
            
        except Exception as e:
            logger.warning(f"Could not detect encoding: {e}")
            self.detected_encoding = 'utf-8'
            return self.detected_encoding
    
    def _detect_csv_separator(self, sample_lines: int = 10) -> str:
        """Detect CSV separator."""
        if self.detected_separator:
            return self.detected_separator
        
        try:
            encoding = self._detect_encoding()
            
            with open(self.file_path, 'r', encoding=encoding) as f:
                sample_text = ''
                for _ in range(sample_lines):
                    line = f.readline()
                    if not line:
                        break
                    sample_text += line
            
            # Count potential separators
            separators = [',', ';', '\t', '|', ':', ' ']
            separator_counts = {}
            
            for sep in separators:
                # Count occurrences per line to find consistent separator
                lines = sample_text.strip().split('\n')
                if len(lines) > 1:
                    counts = [line.count(sep) for line in lines if line.strip()]
                    if counts and len(set(counts)) == 1 and counts[0] > 0:
                        separator_counts[sep] = counts[0]
            
            if separator_counts:
                self.detected_separator = max(separator_counts, key=separator_counts.get)
            else:
                self.detected_separator = ','
            
            logger.info(f"Detected CSV separator: '{self.detected_separator}'")
            return self.detected_separator
            
        except Exception as e:
            logger.warning(f"Could not detect CSV separator: {e}")
            self.detected_separator = ','
            return self.detected_separator
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CSV file with intelligent parameter detection."""
        try:
            encoding = self._detect_encoding()
            separator = self._detect_csv_separator()
            
            # Try to load with detected parameters
            df = pd.read_csv(
                self.file_path,
                encoding=encoding,
                sep=separator,
                low_memory=False,
                **self.kwargs
            )
            
            logger.info(f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load CSV with detected parameters: {e}")
            
            # Fallback to pandas auto-detection
            try:
                df = pd.read_csv(self.file_path, **self.kwargs)
                logger.info(f"Successfully loaded CSV with fallback: {len(df)} rows, {len(df.columns)} columns")
                return df
            except Exception as e:
                logger.error(f"Failed to load CSV file: {e}")
                raise
    
    def _load_json(self) -> pd.DataFrame:
        """Load JSON file (single object or JSON Lines)."""
        try:
            encoding = self._detect_encoding()
            
            with open(self.file_path, 'r', encoding=encoding) as f:
                # Try to detect if it's JSON Lines format
                first_line = f.readline().strip()
                f.seek(0)
                
                if first_line.startswith('{') and not first_line.endswith('}'):
                    # Likely JSON Lines format
                    data = []
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
                    df = pd.DataFrame(data)
                else:
                    # Regular JSON
                    data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])
                    else:
                        raise ValueError("Unsupported JSON structure")
            
            logger.info(f"Successfully loaded JSON: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise
    
    def _load_parquet(self) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            df = pd.read_parquet(self.file_path, **self.kwargs)
            logger.info(f"Successfully loaded Parquet: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet file: {e}")
            raise
    
    def _load_excel(self) -> pd.DataFrame:
        """Load Excel file."""
        try:
            # Try to detect sheets
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names
            
            # Use first sheet by default or specified sheet
            sheet_name = self.kwargs.get('sheet_name', sheet_names[0])
            
            df = pd.read_excel(self.file_path, sheet_name=sheet_name, **self.kwargs)
            logger.info(f"Successfully loaded Excel: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise
    
    def _load_avro(self) -> pd.DataFrame:
        """Load Avro file."""
        try:
            import fastavro
            
            records = []
            with open(self.file_path, 'rb') as f:
                reader = fastavro.reader(f)
                for record in reader:
                    records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"Successfully loaded Avro: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except ImportError:
            logger.error("fastavro package is required to load Avro files")
            raise
        except Exception as e:
            logger.error(f"Failed to load Avro file: {e}")
            raise
    
    def _load_orc(self) -> pd.DataFrame:
        """Load ORC file."""
        try:
            import pyorc
            
            records = []
            with open(self.file_path, 'rb') as f:
                reader = pyorc.Reader(f)
                for row in reader:
                    records.append(row)
            
            df = pd.DataFrame(records)
            logger.info(f"Successfully loaded ORC: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except ImportError:
            logger.error("pyorc package is required to load ORC files")
            raise
        except Exception as e:
            logger.error(f"Failed to load ORC file: {e}")
            raise
    
    def _load_feather(self) -> pd.DataFrame:
        """Load Feather file."""
        try:
            df = pd.read_feather(self.file_path, **self.kwargs)
            logger.info(f"Successfully loaded Feather: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load Feather file: {e}")
            raise
    
    def _load_hdf5(self) -> pd.DataFrame:
        """Load HDF5 file."""
        try:
            # Try to find the key if not specified
            key = self.kwargs.get('key')
            if not key:
                store = pd.HDFStore(self.file_path, mode='r')
                keys = store.keys()
                if keys:
                    key = keys[0]
                store.close()
            
            df = pd.read_hdf(self.file_path, key=key, **self.kwargs)
            logger.info(f"Successfully loaded HDF5: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load HDF5 file: {e}")
            raise
    
    def _load_pickle(self) -> pd.DataFrame:
        """Load Pickle file."""
        try:
            df = pd.read_pickle(self.file_path, **self.kwargs)
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Pickle file does not contain a DataFrame")
            
            logger.info(f"Successfully loaded Pickle: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load Pickle file: {e}")
            raise
    
    def _load_tsv(self) -> pd.DataFrame:
        """Load TSV (Tab-Separated Values) file."""
        try:
            encoding = self._detect_encoding()
            
            df = pd.read_csv(
                self.file_path,
                sep='\t',
                encoding=encoding,
                low_memory=False,
                **self.kwargs
            )
            
            logger.info(f"Successfully loaded TSV: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load TSV file: {e}")
            raise
    
    def _load_text(self) -> pd.DataFrame:
        """Load text file as single column DataFrame."""
        try:
            encoding = self._detect_encoding()
            
            with open(self.file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            df = pd.DataFrame({'text': [line.strip() for line in lines]})
            logger.info(f"Successfully loaded text file: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load text file: {e}")
            raise
    
    def _auto_detect_and_load(self) -> pd.DataFrame:
        """Auto-detect file format and load accordingly."""
        try:
            # Try common formats in order of likelihood
            formats_to_try = [
                ('CSV', self._load_csv),
                ('JSON', self._load_json),
                ('TSV', self._load_tsv),
                ('Text', self._load_text)
            ]
            
            last_error = None
            for format_name, loader in formats_to_try:
                try:
                    logger.info(f"Trying to load as {format_name}")
                    return loader()
                except Exception as e:
                    last_error = e
                    continue
            
            # If all formats fail, raise the last error
            raise last_error
            
        except Exception as e:
            logger.error(f"Failed to auto-detect and load file: {e}")
            raise
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get metadata about the file."""
        try:
            stat = self.file_path.stat()
            
            return {
                'file_path': str(self.file_path),
                'file_name': self.file_path.name,
                'file_extension': self.file_path.suffix,
                'file_size_bytes': stat.st_size,
                'file_size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'detected_encoding': self.detected_encoding,
                'detected_separator': self.detected_separator
            }
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {}
    
    def sample_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Load a sample of the data for profiling."""
        try:
            # For large files, we want to sample efficiently
            file_extension = self.file_path.suffix.lower()
            
            if file_extension == '.csv':
                return self._sample_csv(n_rows)
            elif file_extension in ['.parquet', '.pq']:
                return self._sample_parquet(n_rows)
            else:
                # For other formats, load full data and sample
                df = self.load_data()
                if len(df) > n_rows:
                    return df.sample(n=n_rows, random_state=42)
                return df
        except Exception as e:
            logger.error(f"Failed to sample data: {e}")
            raise
    
    def _sample_csv(self, n_rows: int) -> pd.DataFrame:
        """Sample CSV file efficiently."""
        try:
            encoding = self._detect_encoding()
            separator = self._detect_csv_separator()
            
            # First, try to load only n_rows
            df = pd.read_csv(
                self.file_path,
                encoding=encoding,
                sep=separator,
                nrows=n_rows,
                low_memory=False,
                **self.kwargs
            )
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to sample CSV efficiently: {e}")
            # Fallback to full load and sample
            df = self._load_csv()
            if len(df) > n_rows:
                return df.sample(n=n_rows, random_state=42)
            return df
    
    def _sample_parquet(self, n_rows: int) -> pd.DataFrame:
        """Sample Parquet file efficiently."""
        try:
            # Try to read only first n_rows
            df = pd.read_parquet(self.file_path, **self.kwargs)
            
            if len(df) > n_rows:
                return df.sample(n=n_rows, random_state=42)
            return df
            
        except Exception as e:
            logger.warning(f"Failed to sample Parquet efficiently: {e}")
            raise


class MultiFileAdapter:
    """Adapter for loading multiple files as a single dataset."""
    
    def __init__(self, file_paths: List[str], **kwargs):
        self.file_paths = [Path(p) for p in file_paths]
        self.kwargs = kwargs
    
    def load_data(self) -> pd.DataFrame:
        """Load and concatenate data from multiple files."""
        dataframes = []
        
        for file_path in self.file_paths:
            try:
                adapter = FileAdapter(file_path, **self.kwargs)
                df = adapter.load_data()
                
                # Add source file column
                df['_source_file'] = file_path.name
                dataframes.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {e}")
                # Continue with other files
                continue
        
        if not dataframes:
            raise ValueError("No files could be loaded successfully")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        logger.info(f"Successfully loaded {len(self.file_paths)} files: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        return combined_df
    
    def get_files_info(self) -> List[Dict[str, Any]]:
        """Get metadata about all files."""
        files_info = []
        
        for file_path in self.file_paths:
            try:
                adapter = FileAdapter(file_path)
                info = adapter.get_file_info()
                files_info.append(info)
            except Exception as e:
                logger.error(f"Failed to get info for file {file_path}: {e}")
                files_info.append({'file_path': str(file_path), 'error': str(e)})
        
        return files_info