#!/usr/bin/env python3
"""
Simple CLI Wrapper for Pynomaly
Bypasses Typer compatibility issues by providing direct function calls
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def show_help():
    """Show help information"""
    help_text = """
Pynomaly - State-of-the-art anomaly detection CLI

Usage: python pynomaly_cli.py <command> [options]

Commands:
  help               Show this help message
  version            Show version information
  detector-list      List available detectors
  dataset-info FILE  Show dataset information
  detect FILE        Run anomaly detection on dataset
  server-start       Start the API server
  test-imports       Test core system imports

Examples:
  python pynomaly_cli.py version
  python pynomaly_cli.py detector-list
  python pynomaly_cli.py dataset-info data.csv
  python pynomaly_cli.py detect data.csv
  python pynomaly_cli.py server-start
"""
    print(help_text)

def show_version():
    """Show version information"""
    try:
        from pynomaly.infrastructure.config import Settings
        settings = Settings()
        print(f"Pynomaly v{settings.app.version}")
        print(f"Python {sys.version.split()[0]}")
        print(f"Storage: {settings.storage_path}")
    except Exception as e:
        print(f"Pynomaly v0.1.0 (config error: {e})")
        print(f"Python {sys.version.split()[0]}")

def list_detectors():
    """List available detectors"""
    try:
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        print("Available Detectors:")
        print("- IsolationForest (Isolation Forest)")
        print("- LocalOutlierFactor (Local Outlier Factor)")
        print("- OneClassSVM (One-Class SVM)")
        print("- EllipticEnvelope (Elliptic Envelope)")
        print("- SGDOneClassSVM (SGD One-Class SVM)")
        print("\nNote: More detectors available through other adapters")
    except Exception as e:
        print(f"Error listing detectors: {e}")

def show_dataset_info(file_path):
    """Show dataset information"""
    try:
        import pandas as pd
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            return
        
        # Try to read the dataset
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                print(f"Unsupported file format. Supported: .csv, .json")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
        
        print(f"Dataset Information for: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nBasic statistics:")
        print(df.describe())
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

def run_detection(file_path):
    """Run anomaly detection on dataset"""
    try:
        import pandas as pd
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
        from pynomaly.domain.entities import Dataset
        from pynomaly.domain.value_objects import ContaminationRate
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            return
        
        print(f"Running anomaly detection on: {file_path}")
        
        # Load data
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            print(f"Unsupported file format. Supported: .csv, .json")
            return
        
        print(f"Data loaded: {data.shape}")
        
        # Create dataset
        dataset = Dataset(
            name=f"dataset_{os.path.basename(file_path)}",
            data=data
        )
        
        # Create adapter and run detection
        adapter = SklearnAdapter("IsolationForest", contamination_rate=ContaminationRate(0.1))
        adapter.fit(dataset)
        result = adapter.detect(dataset)
        
        # Show results
        anomaly_count = len(result.anomalies)
        print(f"\nDetection Results:")
        print(f"Total samples: {len(result.labels)}")
        print(f"Anomalies detected: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_count/len(result.labels)*100:.1f}%")
        print(f"Threshold: {result.threshold:.3f}")
        print(f"Execution time: {result.execution_time_ms:.1f}ms")
        
        if anomaly_count > 0:
            anomaly_indices = [i for i, label in enumerate(result.labels) if label == 1]
            print(f"\nAnomaly indices: {anomaly_indices}")
        
    except Exception as e:
        print(f"Error running detection: {e}")
        import traceback
        traceback.print_exc()

def start_server():
    """Start the API server"""
    try:
        import uvicorn
        from pynomaly.presentation.api import create_app
        
        print("Starting Pynomaly API server...")
        print("Server will be available at: http://127.0.0.1:8000")
        print("API documentation: http://127.0.0.1:8000/docs")
        print("Press CTRL+C to stop")
        
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()

def test_imports():
    """Test core system imports"""
    tests = [
        ("Domain entities", "from pynomaly.domain.entities import Anomaly, Dataset, Detector, DetectionResult"),
        ("Value objects", "from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate"),
        ("Configuration", "from pynomaly.infrastructure.config import Container, Settings"),
        ("Sklearn adapter", "from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter"),
        ("Detection service", "from pynomaly.application.services import DetectionService"),
        ("API app", "from pynomaly.presentation.api import create_app")
    ]
    
    print("Testing core system imports...")
    success_count = 0
    
    for test_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {test_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
    
    print(f"\nImport test results: {success_count}/{len(tests)} successful")
    
    if success_count == len(tests):
        print("🎉 All core imports working!")
    elif success_count >= len(tests) * 0.8:
        print("⚠️ Most imports working, minor issues remain")
    else:
        print("❌ Significant import issues detected")

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command in ['help', '-h', '--help']:
        show_help()
    elif command in ['version', '-v', '--version']:
        show_version()
    elif command == 'detector-list':
        list_detectors()
    elif command == 'dataset-info':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py dataset-info <file>")
        else:
            show_dataset_info(sys.argv[2])
    elif command == 'detect':
        if len(sys.argv) < 3:
            print("Error: Please provide a file path")
            print("Usage: python pynomaly_cli.py detect <file>")
        else:
            run_detection(sys.argv[2])
    elif command == 'server-start':
        start_server()
    elif command == 'test-imports':
        test_imports()
    else:
        print(f"Unknown command: {command}")
        print("Use 'python pynomaly_cli.py help' for available commands")

if __name__ == "__main__":
    main()