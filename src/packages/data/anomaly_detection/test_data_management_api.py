#!/usr/bin/env python3
"""Test script for data management API endpoints."""

import asyncio
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

async def test_data_management_services():
    """Test the data management services directly without API calls."""
    
    print("🔍 Testing Data Management Services...")
    
    # Create test data
    print("\n1. Creating test data...")
    test_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(0, 1, 1000),
        'feature3': np.random.normal(0, 1, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'timestamp': pd.date_range('2023-01-01', periods=1000, freq='H')
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(1000, 50, replace=False)
    test_data.loc[anomaly_indices, 'feature1'] += 5  # Make some values anomalous
    
    # Save test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        print(f"✅ Test data created: {test_file}")
        print(f"   Shape: {test_data.shape}")
        print(f"   Columns: {list(test_data.columns)}")
        
        # Test Data Validation Service
        print("\n2. Testing Data Validation Service...")
        try:
            from src.anomaly_detection.domain.services.data_validation_service import DataValidationService
            
            validation_service = DataValidationService()
            validation_result = await validation_service.validate_file(test_file)
            
            print(f"✅ Validation completed")
            print(f"   Valid: {validation_result['is_valid']}")
            print(f"   Errors: {len(validation_result['errors'])}")
            print(f"   Warnings: {len(validation_result['warnings'])}")
            print(f"   Checks performed: {validation_result['checks_performed']}")
            
        except Exception as e:
            print(f"❌ Validation service failed: {e}")
        
        # Test Data Profiling Service
        print("\n3. Testing Data Profiling Service...")
        try:
            from src.anomaly_detection.domain.services.data_profiling_service import DataProfilingService
            
            profiling_service = DataProfilingService()
            profile_result = await profiling_service.profile_file(test_file)
            
            print(f"✅ Profiling completed")
            print(f"   Dataset rows: {profile_result.get('dataset_info', {}).get('row_count', 'N/A')}")
            print(f"   Dataset columns: {profile_result.get('dataset_info', {}).get('column_count', 'N/A')}")
            print(f"   Memory usage (MB): {profile_result.get('dataset_info', {}).get('memory_usage_mb', 'N/A'):.2f}")
            
            column_info = profile_result.get('column_info', {})
            print(f"   Profiled columns: {len(column_info)}")
            
        except Exception as e:
            print(f"❌ Profiling service failed: {e}")
        
        # Test Data Conversion Service
        print("\n4. Testing Data Conversion Service...")
        try:
            from src.anomaly_detection.domain.services.data_conversion_service import DataConversionService
            
            conversion_service = DataConversionService()
            output_file = await conversion_service.convert_file(
                input_file=test_file,
                output_format="json",
                output_dir=temp_path
            )
            
            print(f"✅ Conversion completed")
            print(f"   Input: {test_file}")
            print(f"   Output: {output_file}")
            print(f"   Output exists: {output_file.exists()}")
            
        except Exception as e:
            print(f"❌ Conversion service failed: {e}")
        
        # Test Data Sampling Service
        print("\n5. Testing Data Sampling Service...")
        try:
            from src.anomaly_detection.domain.services.data_sampling_service import DataSamplingService
            
            sampling_service = DataSamplingService()
            sample_result = await sampling_service.sample_file(
                file_path=test_file,
                sample_size=100,
                method="random",
                seed=42
            )
            
            print(f"✅ Sampling completed")
            print(f"   Original size: {len(test_data)}")
            print(f"   Sample size: {len(sample_result)}")
            print(f"   Sample columns: {list(sample_result.columns)}")
            
        except Exception as e:
            print(f"❌ Sampling service failed: {e}")
        
        # Test Batch Processing Service
        print("\n6. Testing Batch Processing Service...")
        try:
            from src.anomaly_detection.domain.services.batch_processing_service import BatchProcessingService
            from src.anomaly_detection.domain.services.detection_service import DetectionService
            from src.anomaly_detection.infrastructure.repositories.in_memory_model_repository import InMemoryModelRepository
            
            detection_service = DetectionService()
            model_repository = InMemoryModelRepository()
            batch_service = BatchProcessingService(detection_service, model_repository)
            
            # Create output directory for batch results
            batch_output_dir = temp_path / "batch_results"
            batch_output_dir.mkdir(exist_ok=True)
            
            batch_result = await batch_service.batch_detect_anomalies(
                file_paths=[test_file],
                output_dir=batch_output_dir,
                algorithms=["isolation_forest"],
                output_format="json"
            )
            
            print(f"✅ Batch processing completed")
            print(f"   Total files: {batch_result['total_files']}")
            print(f"   Successful files: {batch_result['successful_files']}")
            print(f"   Failed files: {batch_result['failed_files']}")
            print(f"   Output directory: {batch_result['output_directory']}")
            
        except Exception as e:
            print(f"❌ Batch processing service failed: {e}")
    
    print("\n🎉 Data Management Services Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_data_management_services())