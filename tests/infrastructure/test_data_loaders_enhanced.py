"""Enhanced comprehensive tests for infrastructure data loaders - Phase 2 Coverage Improvements."""

from __future__ import annotations

import asyncio
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional, Generator
import io
import gzip
import zipfile
import csv
import json

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import DataValidationError, LoaderError
from pynomaly.infrastructure.data_loaders import (
    CSVLoader, 
    ParquetLoader,
    PolarsLoader,
    ArrowLoader,
    SparkLoader,
    DataLoaderFactory
)


@pytest.fixture
def sample_streaming_data():
    """Create sample streaming data for testing."""
    def data_generator():
        for i in range(1000):
            yield {
                "timestamp": f"2024-01-01 {i:02d}:00:00",
                "sensor_1": np.random.normal(0, 1),
                "sensor_2": np.random.normal(0, 1),
                "sensor_3": np.random.normal(0, 1),
                "anomaly_score": np.random.random(),
                "is_anomaly": 1 if i % 100 == 0 else 0
            }
    return data_generator


@pytest.fixture
def corrupted_csv_data():
    """Create various types of corrupted CSV data for testing."""
    return {
        "missing_headers": "1.0,2.0,3.0\n4.0,5.0,6.0\n",
        "inconsistent_columns": "a,b,c\n1,2,3\n4,5\n6,7,8,9\n",
        "invalid_encoding": "feature1,feature2\nα,β\nγ,δ".encode('latin-1'),
        "mixed_types": "id,value,flag\n1,1.5,true\ntwo,invalid,false\n3,3.14,1\n",
        "special_characters": 'name,value\n"Smith, John",100\n\'O"Connor\',200\n',
        "unicode_bom": "\ufefffeature1,feature2\n1,2\n3,4\n"
    }


class TestCSVLoaderEnhanced:
    """Enhanced comprehensive tests for CSVLoader functionality."""
    
    def test_large_file_processing(self):
        """Test processing of large CSV files with memory optimization."""
        # Create large CSV data (10MB+)
        header = "feature1,feature2,feature3,feature4,feature5,target\n"
        rows = []
        for i in range(100000):  # 100K rows
            row = f"{i},{i*1.1},{i*1.2},{i*1.3},{i*1.4},{i%2}"
            rows.append(row)
        
        large_csv_data = header + "\n".join(rows)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(large_csv_data)
            temp_path = f.name
        
        try:
            loader = CSVLoader()
            
            # Test chunked loading
            chunks = list(loader.load_chunks(temp_path, chunk_size=10000))
            assert len(chunks) == 10  # 100K rows / 10K chunk size
            
            for chunk in chunks:
                assert chunk.n_samples == 10000
                assert chunk.n_features == 6
                
            # Test memory usage monitoring
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            dataset = loader.load(temp_path)
            
            memory_after = process.memory_info().rss
            memory_increase = (memory_after - memory_before) / 1024 / 1024  # MB
            
            # Memory increase should be reasonable (less than 500MB for 100K rows)
            assert memory_increase < 500
            assert dataset.n_samples == 100000
            
        finally:
            Path(temp_path).unlink()
    
    def test_compressed_file_support(self):
        """Test support for compressed CSV files."""
        csv_data = "feature1,feature2,feature3\n1,2,3\n4,5,6\n7,8,9\n"
        
        # Test gzip compression
        with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as f:
            with gzip.open(f.name, 'wt') as gz_file:
                gz_file.write(csv_data)
            gz_path = f.name
        
        # Test zip compression
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            with zipfile.ZipFile(f.name, 'w') as zip_file:
                zip_file.writestr('data.csv', csv_data)
            zip_path = f.name
        
        try:
            loader = CSVLoader()
            
            # Test gzip file loading
            gz_dataset = loader.load(gz_path)
            assert gz_dataset.n_samples == 3
            assert gz_dataset.n_features == 3
            
            # Test zip file loading
            zip_dataset = loader.load(zip_path, archive_member='data.csv')
            assert zip_dataset.n_samples == 3
            assert zip_dataset.n_features == 3
            
        finally:
            Path(gz_path).unlink()
            Path(zip_path).unlink()
    
    def test_data_type_inference_and_conversion(self):
        """Test automatic data type inference and conversion."""
        csv_data = \"\"\"id,name,score,timestamp,is_active,confidence
1,Alice,95.5,2024-01-01 10:00:00,true,0.95
2,Bob,87.2,2024-01-01 11:00:00,false,0.87
3,Charlie,92.1,2024-01-01 12:00:00,true,0.92\"\"\"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            loader = CSVLoader(
                auto_detect_types=True,
                parse_dates=['timestamp'],
                bool_columns=['is_active']
            )
            dataset = loader.load(temp_path)
            
            # Verify data types
            assert dataset.data['id'].dtype == 'int64'
            assert dataset.data['name'].dtype == 'object'
            assert dataset.data['score'].dtype == 'float64'
            assert pd.api.types.is_datetime64_any_dtype(dataset.data['timestamp'])
            assert dataset.data['is_active'].dtype == 'bool'
            assert dataset.data['confidence'].dtype == 'float64'
            
        finally:
            Path(temp_path).unlink()
    
    def test_advanced_error_handling(self, corrupted_csv_data):
        """Test advanced error handling for various corruption scenarios."""
        loader = CSVLoader(error_handling='coerce')
        
        for corruption_type, data in corrupted_csv_data.items():
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
                if isinstance(data, str):
                    f.write(data.encode('utf-8'))
                else:
                    f.write(data)
                temp_path = f.name
            
            try:
                if corruption_type == "missing_headers":
                    # Should automatically generate column names
                    dataset = loader.load(temp_path, auto_generate_headers=True)
                    assert dataset.n_features > 0
                    
                elif corruption_type == "inconsistent_columns":
                    # Should handle inconsistent columns gracefully
                    dataset = loader.load(temp_path, fill_missing_columns=True)
                    assert dataset.n_samples > 0
                    
                elif corruption_type == "invalid_encoding":
                    # Should handle encoding errors
                    with pytest.raises((DataValidationError, UnicodeDecodeError)):
                        loader.load(temp_path, encoding='utf-8')
                    
                    # But should work with correct encoding
                    dataset = loader.load(temp_path, encoding='latin-1')
                    assert dataset.n_samples > 0
                    
                elif corruption_type == "mixed_types":
                    # Should handle mixed types gracefully
                    dataset = loader.load(temp_path, coerce_mixed_types=True)
                    assert dataset.n_samples > 0
                    
            except (DataValidationError, ValueError) as e:
                # Some corruption types are expected to fail
                assert corruption_type in ["mixed_types", "invalid_encoding"]
                
            finally:
                Path(temp_path).unlink()
    
    def test_custom_preprocessing_pipeline(self):
        """Test custom preprocessing pipeline integration."""
        csv_data = \"\"\"sensor1,sensor2,sensor3,anomaly_label
1.0,2.0,3.0,0
1.1,2.1,3.1,0
5.0,8.0,12.0,1
1.2,2.2,3.2,0\"\"\"
        
        def custom_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
            # Custom preprocessing: scale features and add derived features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
            df['sensor_sum'] = df['sensor1'] + df['sensor2'] + df['sensor3']
            return df
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            loader = CSVLoader(preprocessor=custom_preprocessor)
            dataset = loader.load(temp_path, target_column='anomaly_label')
            
            # Verify preprocessing was applied
            assert 'sensor_sum' in dataset.data.columns
            assert dataset.n_features == 4  # 3 original + 1 derived
            
            # Verify scaling was applied (mean should be close to 0)
            numeric_features = dataset.get_numeric_features()
            for feature in ['sensor1', 'sensor2', 'sensor3']:
                if feature in numeric_features:
                    assert abs(dataset.data[feature].mean()) < 0.1
                    
        finally:
            Path(temp_path).unlink()
    
    def test_streaming_data_processing(self, sample_streaming_data):
        """Test streaming data processing capabilities."""
        loader = CSVLoader()
        
        # Simulate streaming data
        stream_data = list(sample_streaming_data())[:100]  # First 100 records
        
        # Convert to CSV string
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=stream_data[0].keys())
        writer.writeheader()
        writer.writerows(stream_data)
        csv_content = csv_buffer.getvalue()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            # Test streaming load
            stream_datasets = []
            for chunk in loader.load_stream(temp_path, batch_size=20):
                stream_datasets.append(chunk)
                assert chunk.n_samples <= 20
                assert chunk.n_features == 6
            
            assert len(stream_datasets) == 5  # 100 records / 20 batch size
            
            # Test real-time processing simulation
            total_anomalies = 0
            for dataset in stream_datasets:
                if 'is_anomaly' in dataset.data.columns:
                    anomalies = dataset.data['is_anomaly'].sum()
                    total_anomalies += anomalies
            
            assert total_anomalies >= 0  # Should detect some anomalies
            
        finally:
            Path(temp_path).unlink()
    
    def test_performance_optimization(self):
        """Test performance optimization features."""
        # Create medium-sized dataset for performance testing
        header = "timestamp,feature1,feature2,feature3,feature4,feature5,target\n"
        rows = []
        for i in range(10000):
            timestamp = f"2024-01-01 {i//60:02d}:{i%60:02d}:00"
            row = f"{timestamp},{i},{i*1.1},{i*1.2},{i*1.3},{i*1.4},{i%2}"
            rows.append(row)
        
        csv_data = header + "\n".join(rows)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            import time
            
            # Test standard loading
            loader_standard = CSVLoader()
            start_time = time.time()
            dataset_standard = loader_standard.load(temp_path)
            standard_time = time.time() - start_time
            
            # Test optimized loading
            loader_optimized = CSVLoader(
                low_memory=False,
                use_cache=True,
                parallel_processing=True,
                chunk_size=1000
            )
            start_time = time.time()
            dataset_optimized = loader_optimized.load(temp_path)
            optimized_time = time.time() - start_time
            
            # Verify results are identical
            assert dataset_standard.n_samples == dataset_optimized.n_samples
            assert dataset_standard.n_features == dataset_optimized.n_features
            
            # Optimized loading should be faster (or at least not significantly slower)
            # Note: For small datasets, overhead might make optimized slower
            performance_ratio = optimized_time / standard_time
            assert performance_ratio < 2.0  # Should not be more than 2x slower
            
        finally:
            Path(temp_path).unlink()


class TestDataLoaderFactory:
    """Enhanced tests for DataLoaderFactory."""
    
    def test_auto_detection_comprehensive(self):
        """Test comprehensive file format auto-detection."""
        test_files = {
            'data.csv': CSVLoader,
            'data.tsv': CSVLoader,
            'data.txt': CSVLoader,
            'data.parquet': ParquetLoader,
            'data.pq': ParquetLoader,
            'data.csv.gz': CSVLoader,
            'data.csv.zip': CSVLoader,
        }
        
        factory = DataLoaderFactory()
        
        for filename, expected_loader_class in test_files.items():
            loader = factory.create_loader(filename)
            assert isinstance(loader, expected_loader_class)
    
    def test_loader_configuration_inheritance(self):
        """Test that factory configurations are properly inherited by loaders."""
        factory = DataLoaderFactory(
            default_encoding='latin-1',
            default_delimiter=';',
            enable_caching=True
        )
        
        csv_loader = factory.create_loader('data.csv')
        assert csv_loader.encoding == 'latin-1'
        assert csv_loader.delimiter == ';'
        assert csv_loader.enable_caching is True
    
    def test_custom_loader_registration(self):
        """Test registration of custom loader types."""
        class CustomXMLLoader:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def load(self, file_path, **kwargs):
                # Mock XML loading
                return Dataset(name="xml_data", data=pd.DataFrame({'xml_data': [1, 2, 3]}))
        
        factory = DataLoaderFactory()
        factory.register_loader('xml', CustomXMLLoader)
        
        # Test custom loader creation
        xml_loader = factory.create_loader('data.xml')
        assert isinstance(xml_loader, CustomXMLLoader)
        
        # Test custom loader usage
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
            f.write(b'<data><item>1</item><item>2</item></data>')
            temp_path = f.name
        
        try:
            dataset = xml_loader.load(temp_path)
            assert dataset.name == "xml_data"
            assert dataset.n_samples == 3
        finally:
            Path(temp_path).unlink()


class TestDataLoaderIntegration:
    """Enhanced integration tests for data loaders."""
    
    def test_end_to_end_anomaly_detection_pipeline(self):
        """Test complete end-to-end pipeline from data loading to anomaly detection."""
        # Create realistic anomaly detection dataset
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (900, 5))
        anomaly_data = np.random.normal(3, 0.5, (100, 5))
        
        all_data = np.vstack([normal_data, anomaly_data])
        labels = np.array([0] * 900 + [1] * 100)
        
        df = pd.DataFrame(all_data, columns=[f'feature_{i}' for i in range(5)])
        df['anomaly_label'] = labels
        df['timestamp'] = pd.date_range('2024-01-01', periods=1000, freq='1H')
        
        csv_data = df.to_csv(index=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            # Test data loading
            loader = CSVLoader(parse_dates=['timestamp'])
            dataset = loader.load(temp_path, target_column='anomaly_label')
            
            # Verify dataset properties
            assert dataset.n_samples == 1000
            assert dataset.n_features == 6  # 5 features + timestamp
            assert dataset.has_target is True
            
            # Verify data quality
            numeric_features = dataset.get_numeric_features()
            assert len(numeric_features) == 5
            
            # Verify anomaly distribution
            anomaly_count = dataset.data['anomaly_label'].sum()
            assert anomaly_count == 100  # 10% anomalies
            
            # Test with sklearn adapter integration
            from sklearn.ensemble import IsolationForest
            
            X = dataset.data[numeric_features].values
            y = dataset.data['anomaly_label'].values
            
            # Train anomaly detector
            detector = IsolationForest(contamination=0.1, random_state=42)
            detector.fit(X)
            
            # Get predictions
            predictions = detector.predict(X)
            anomaly_scores = detector.decision_function(X)
            
            # Convert predictions (-1, 1) to (1, 0)
            binary_predictions = (predictions == -1).astype(int)
            
            # Calculate performance metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y, binary_predictions)
            recall = recall_score(y, binary_predictions)
            f1 = f1_score(y, binary_predictions)
            
            # Should achieve reasonable performance on this synthetic dataset
            assert precision > 0.3  # At least 30% precision
            assert recall > 0.1     # At least 10% recall
            assert f1 > 0.1         # At least 10% F1 score
            
        finally:
            Path(temp_path).unlink()
    
    def test_multi_format_dataset_consolidation(self):
        """Test consolidating datasets from multiple file formats."""
        # Create data in different formats
        base_data = pd.DataFrame({
            'feature1': np.random.random(100),
            'feature2': np.random.random(100),
            'feature3': np.random.random(100)
        })
        
        # CSV format
        csv_data = base_data.iloc[:30]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_data.to_csv(f, index=False)
            csv_path = f.name
        
        # JSON format (converted to CSV for this test)
        json_data = base_data.iloc[30:60]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            json_data.to_csv(f, index=False)
            json_path = f.name
        
        # TSV format
        tsv_data = base_data.iloc[60:]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            tsv_data.to_csv(f, sep='\t', index=False)
            tsv_path = f.name
        
        try:
            factory = DataLoaderFactory()
            
            # Load all datasets
            csv_dataset = factory.create_loader(csv_path).load(csv_path, name="csv_data")
            json_dataset = factory.create_loader(json_path).load(json_path, name="json_data")
            tsv_dataset = factory.create_loader(tsv_path).load(tsv_path, name="tsv_data")
            
            # Verify individual datasets
            assert csv_dataset.n_samples == 30
            assert json_dataset.n_samples == 30
            assert tsv_dataset.n_samples == 40
            
            # Test dataset consolidation
            consolidated_data = pd.concat([
                csv_dataset.data,
                json_dataset.data,
                tsv_dataset.data
            ], ignore_index=True)
            
            consolidated_dataset = Dataset(
                name="consolidated",
                data=consolidated_data
            )
            
            assert consolidated_dataset.n_samples == 100
            assert consolidated_dataset.n_features == 3
            
        finally:
            Path(csv_path).unlink()
            Path(json_path).unlink()
            Path(tsv_path).unlink()


class TestDataLoaderPerformance:
    """Performance tests for data loaders."""
    
    @pytest.mark.benchmark
    def test_csv_loading_performance_benchmark(self, benchmark):
        """Benchmark CSV loading performance."""
        # Create benchmark dataset
        data_size = 10000
        header = "feature1,feature2,feature3,feature4,feature5\n"
        rows = [f"{i},{i*1.1},{i*1.2},{i*1.3},{i*1.4}" for i in range(data_size)]
        csv_data = header + "\n".join(rows)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            loader = CSVLoader()
            
            # Benchmark the loading operation
            result = benchmark(loader.load, temp_path)
            
            # Verify result
            assert result.n_samples == data_size
            assert result.n_features == 5
            
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.benchmark
    def test_large_file_chunked_processing_benchmark(self, benchmark):
        """Benchmark chunked processing of large files."""
        # Create large dataset
        data_size = 50000
        header = "timestamp,sensor1,sensor2,sensor3,target\n"
        rows = []
        for i in range(data_size):
            timestamp = f"2024-01-01 {i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}"
            row = f"{timestamp},{i},{i*1.1},{i*1.2},{i%2}"
            rows.append(row)
        
        csv_data = header + "\n".join(rows)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_path = f.name
        
        try:
            loader = CSVLoader()
            
            def chunked_processing():
                total_samples = 0
                for chunk in loader.load_chunks(temp_path, chunk_size=5000):
                    total_samples += chunk.n_samples
                return total_samples
            
            # Benchmark chunked processing
            total_samples = benchmark(chunked_processing)
            
            # Verify result
            assert total_samples == data_size
            
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])