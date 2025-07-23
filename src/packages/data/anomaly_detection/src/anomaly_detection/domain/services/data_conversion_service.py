"""Data conversion service for format transformations."""

import asyncio
import gzip
import bz2
import lzma
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


class DataConversionService:
    """Service for converting data between different formats."""
    
    def __init__(self):
        self.supported_formats = {
            'csv': self._to_csv,
            'json': self._to_json,
            'parquet': self._to_parquet,
            'excel': self._to_excel,
            'hdf5': self._to_hdf5,
            'pickle': self._to_pickle
        }
        
        self.compression_handlers = {
            'gzip': gzip,
            'bz2': bz2,
            'xz': lzma
        }
    
    async def convert_file(
        self,
        input_file: Path,
        output_format: str,
        output_dir: Path,
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        preserve_dtypes: bool = True,
        conversion_options: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Convert a single file to target format."""
        
        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        if compression and compression not in self.compression_handlers:
            raise ValueError(f"Unsupported compression: {compression}")
        
        try:
            logger.info("Starting file conversion",
                       input_file=str(input_file),
                       output_format=output_format,
                       compression=compression)
            
            # Load input data
            data = await self._load_input_data(input_file, preserve_dtypes)
            
            # Generate output filename
            output_filename = f"{input_file.stem}.{output_format}"
            if compression:
                output_filename += f".{compression}"
            
            output_file = output_dir / output_filename
            
            # Convert data
            converter = self.supported_formats[output_format]
            await converter(
                data, 
                output_file, 
                compression=compression,
                chunk_size=chunk_size,
                options=conversion_options or {}
            )
            
            logger.info("File conversion completed",
                       input_file=str(input_file),
                       output_file=str(output_file),
                       input_size_mb=self._get_file_size_mb(input_file),
                       output_size_mb=self._get_file_size_mb(output_file))
            
            return output_file
            
        except Exception as e:
            logger.error("File conversion failed",
                        input_file=str(input_file),
                        error=str(e))
            raise
    
    async def _load_input_data(self, input_file: Path, preserve_dtypes: bool) -> pd.DataFrame:
        """Load data from input file."""
        file_extension = input_file.suffix.lower()
        
        try:
            if file_extension == '.csv':
                dtype_arg = None if preserve_dtypes else str
                return pd.read_csv(input_file, dtype=dtype_arg)
            
            elif file_extension == '.json':
                return pd.read_json(input_file)
            
            elif file_extension == '.parquet':
                return pd.read_parquet(input_file)
            
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(input_file)
            
            elif file_extension == '.h5' or file_extension == '.hdf5':
                # Try to read the first key from HDF5
                with pd.HDFStore(input_file, 'r') as store:
                    keys = store.keys()
                    if not keys:
                        raise ValueError("No data found in HDF5 file")
                    return pd.read_hdf(input_file, key=keys[0])
            
            elif file_extension == '.pkl' or file_extension == '.pickle':
                return pd.read_pickle(input_file)
            
            else:
                raise ValueError(f"Unsupported input format: {file_extension}")
                
        except Exception as e:
            raise ValueError(f"Failed to load input data: {str(e)}")
    
    async def _to_csv(
        self, 
        data: pd.DataFrame, 
        output_file: Path, 
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        options: Dict[str, Any] = {}
    ) -> None:
        """Convert to CSV format."""
        csv_options = {
            'index': False,
            'compression': compression,
            **options
        }
        
        if len(data) > chunk_size:
            # Write in chunks for large datasets
            with open(output_file, 'w', newline='') as f:
                for i, chunk in enumerate(self._chunked_dataframe(data, chunk_size)):
                    chunk.to_csv(f, header=(i == 0), **csv_options)
        else:
            data.to_csv(output_file, **csv_options)
    
    async def _to_json(
        self, 
        data: pd.DataFrame, 
        output_file: Path, 
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        options: Dict[str, Any] = {}
    ) -> None:
        """Convert to JSON format."""
        json_options = {
            'orient': 'records',
            'date_format': 'iso',
            'compression': compression,
            **options
        }
        
        if compression and compression != 'gzip':
            # pandas json writer only supports gzip compression
            # Handle other compression manually
            json_str = data.to_json(**{k: v for k, v in json_options.items() if k != 'compression'})
            await self._write_compressed(json_str.encode(), output_file, compression)
        else:
            data.to_json(output_file, **json_options)
    
    async def _to_parquet(
        self, 
        data: pd.DataFrame, 
        output_file: Path, 
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        options: Dict[str, Any] = {}
    ) -> None:
        """Convert to Parquet format."""
        parquet_options = {
            'compression': compression or 'snappy',  # Default to snappy for parquet
            'index': False,
            **options
        }
        
        data.to_parquet(output_file, **parquet_options)
    
    async def _to_excel(
        self, 
        data: pd.DataFrame, 
        output_file: Path, 
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        options: Dict[str, Any] = {}
    ) -> None:
        """Convert to Excel format."""
        excel_options = {
            'index': False,
            'engine': 'openpyxl',
            **options
        }
        
        # Excel doesn't support compression directly, but we can compress the file afterward
        temp_file = output_file if not compression else output_file.with_suffix('.xlsx')
        
        data.to_excel(temp_file, **excel_options)
        
        if compression:
            await self._compress_file(temp_file, output_file, compression)
            temp_file.unlink()  # Remove temporary file
    
    async def _to_hdf5(
        self, 
        data: pd.DataFrame, 
        output_file: Path, 
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        options: Dict[str, Any] = {}
    ) -> None:
        """Convert to HDF5 format."""
        hdf5_options = {
            'key': 'data',
            'mode': 'w',
            'format': 'table',
            'complib': 'blosc' if not compression else compression,
            'complevel': 9,
            **options
        }
        
        data.to_hdf(output_file, **hdf5_options)
    
    async def _to_pickle(
        self, 
        data: pd.DataFrame, 
        output_file: Path, 
        compression: Optional[str] = None,
        chunk_size: int = 10000,
        options: Dict[str, Any] = {}
    ) -> None:
        """Convert to Pickle format."""
        pickle_options = {
            'compression': compression,
            **options
        }
        
        data.to_pickle(output_file, **pickle_options)
    
    def _chunked_dataframe(self, df: pd.DataFrame, chunk_size: int):
        """Generator to yield chunks of a DataFrame."""
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
    
    async def _write_compressed(self, data: bytes, output_file: Path, compression: str) -> None:
        """Write compressed data to file."""
        compressor = self.compression_handlers[compression]
        
        if compression == 'gzip':
            with gzip.open(output_file, 'wb') as f:
                f.write(data)
        elif compression == 'bz2':
            with bz2.open(output_file, 'wb') as f:
                f.write(data)
        elif compression == 'xz':
            with lzma.open(output_file, 'wb') as f:
                f.write(data)
    
    async def _compress_file(self, input_file: Path, output_file: Path, compression: str) -> None:
        """Compress an existing file."""
        with open(input_file, 'rb') as f_in:
            data = f_in.read()
        
        await self._write_compressed(data, output_file, compression)
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            return 0.0
    
    async def batch_convert(
        self,
        input_files: list[Path],
        output_format: str,
        output_dir: Path,
        **conversion_options
    ) -> Dict[str, Any]:
        """Convert multiple files in batch."""
        
        logger.info("Starting batch conversion",
                   files=len(input_files),
                   output_format=output_format)
        
        results = {
            'successful': [],
            'failed': [],
            'summary': {
                'total_files': len(input_files),
                'successful_count': 0,
                'failed_count': 0,
                'total_input_size_mb': 0,
                'total_output_size_mb': 0,
                'start_time': datetime.utcnow().isoformat()
            }
        }
        
        # Calculate total input size
        for file_path in input_files:
            results['summary']['total_input_size_mb'] += self._get_file_size_mb(file_path)
        
        # Convert files
        for file_path in input_files:
            try:
                output_file = await self.convert_file(
                    file_path,
                    output_format,
                    output_dir,
                    **conversion_options
                )
                
                results['successful'].append({
                    'input_file': str(file_path),
                    'output_file': str(output_file),
                    'input_size_mb': self._get_file_size_mb(file_path),
                    'output_size_mb': self._get_file_size_mb(output_file)
                })
                
                results['summary']['successful_count'] += 1
                results['summary']['total_output_size_mb'] += self._get_file_size_mb(output_file)
                
            except Exception as e:
                results['failed'].append({
                    'input_file': str(file_path),
                    'error': str(e)
                })
                results['summary']['failed_count'] += 1
        
        results['summary']['end_time'] = datetime.utcnow().isoformat()
        results['summary']['compression_ratio'] = (
            results['summary']['total_output_size_mb'] / results['summary']['total_input_size_mb']
            if results['summary']['total_input_size_mb'] > 0 else 0
        )
        
        logger.info("Batch conversion completed",
                   successful=results['summary']['successful_count'],
                   failed=results['summary']['failed_count'],
                   compression_ratio=results['summary']['compression_ratio'])
        
        return results
    
    def get_supported_formats(self) -> Dict[str, str]:
        """Get supported output formats and their descriptions."""
        return {
            'csv': 'Comma-separated values - widely compatible text format',
            'json': 'JavaScript Object Notation - structured text format',
            'parquet': 'Apache Parquet - efficient columnar storage format',
            'excel': 'Microsoft Excel format (.xlsx)',
            'hdf5': 'Hierarchical Data Format - scientific data format',
            'pickle': 'Python pickle format - preserves all data types'
        }
    
    def get_supported_compressions(self) -> Dict[str, str]:
        """Get supported compression methods and their descriptions."""
        return {
            'gzip': 'GZIP compression - good balance of speed and compression',
            'bz2': 'BZIP2 compression - higher compression ratio, slower',
            'xz': 'XZ compression - highest compression ratio, slowest'
        }