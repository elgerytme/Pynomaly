# Troubleshooting and FAQ Guide

This guide provides comprehensive troubleshooting information, common issues, solutions, and frequently asked questions for the Anomaly Detection package.

## Table of Contents

1. [Quick Diagnosis Tools](#quick-diagnosis-tools)
2. [Installation Issues](#installation-issues)
3. [Configuration Problems](#configuration-problems)
4. [Model Training Issues](#model-training-issues)
5. [Performance Problems](#performance-problems)
6. [API and Integration Issues](#api-and-integration-issues)
7. [Data Processing Errors](#data-processing-errors)
8. [Deployment and Infrastructure](#deployment-and-infrastructure)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Common Error Messages](#common-error-messages)
11. [Frequently Asked Questions](#frequently-asked-questions)
12. [Advanced Troubleshooting](#advanced-troubleshooting)

## Quick Diagnosis Tools

### Health Check Script

```python
#!/usr/bin/env python3
"""
Anomaly Detection System Health Check
Run this script to quickly diagnose common issues.
"""

import sys
import subprocess
import importlib
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json

class HealthChecker:
    """Comprehensive health check for anomaly detection system."""
    
    def __init__(self):
        self.results = {
            'overall_status': 'unknown',
            'checks': {},
            'recommendations': [],
            'errors': [],
            'warnings': []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        print("🔍 Starting Anomaly Detection System Health Check...")
        print("=" * 60)
        
        checks = [
            ('Python Environment', self.check_python_environment),
            ('Package Installation', self.check_package_installation),
            ('Dependencies', self.check_dependencies),
            ('Configuration', self.check_configuration),
            ('Data Access', self.check_data_access),
            ('Model Training', self.check_model_training),
            ('API Endpoints', self.check_api_endpoints),
            ('Performance', self.check_performance),
            ('Storage', self.check_storage),
            ('Network Connectivity', self.check_network)
        ]
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            print(f"\n🔧 Checking {check_name}...")
            try:
                result = check_func()
                self.results['checks'][check_name] = result
                
                if result['status'] == 'pass':
                    print(f"   ✅ {check_name}: PASS")
                    passed_checks += 1
                elif result['status'] == 'warning':
                    print(f"   ⚠️  {check_name}: WARNING - {result.get('message', '')}")
                    self.results['warnings'].append(f"{check_name}: {result.get('message', '')}")
                else:
                    print(f"   ❌ {check_name}: FAIL - {result.get('message', '')}")
                    self.results['errors'].append(f"{check_name}: {result.get('message', '')}")
                    
            except Exception as e:
                print(f"   ❌ {check_name}: ERROR - {str(e)}")
                self.results['errors'].append(f"{check_name}: {str(e)}")
                self.results['checks'][check_name] = {
                    'status': 'fail',
                    'message': str(e)
                }
        
        # Determine overall status
        if len(self.results['errors']) == 0:
            if len(self.results['warnings']) == 0:
                self.results['overall_status'] = 'healthy'
            else:
                self.results['overall_status'] = 'warning'
        else:
            self.results['overall_status'] = 'unhealthy'
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"🏥 HEALTH CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print(f"Checks Passed: {passed_checks}/{total_checks}")
        print(f"Warnings: {len(self.results['warnings'])}")
        print(f"Errors: {len(self.results['errors'])}")
        
        if self.results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        return self.results
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment."""
        python_version = sys.version_info
        
        if python_version.major != 3 or python_version.minor < 8:
            return {
                'status': 'fail',
                'message': f'Python {python_version.major}.{python_version.minor} detected. Python 3.8+ required.',
                'details': {'version': f'{python_version.major}.{python_version.minor}.{python_version.micro}'}
            }
        
        return {
            'status': 'pass',
            'message': f'Python {python_version.major}.{python_version.minor}.{python_version.micro}',
            'details': {'version': f'{python_version.major}.{python_version.minor}.{python_version.micro}'}
        }
    
    def check_package_installation(self) -> Dict[str, Any]:
        """Check if anomaly detection package is installed."""
        try:
            import anomaly_detection
            version = getattr(anomaly_detection, '__version__', 'unknown')
            return {
                'status': 'pass',
                'message': f'Package installed, version: {version}',
                'details': {'version': version}
            }
        except ImportError as e:
            return {
                'status': 'fail',
                'message': f'Package not installed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        required_packages = [
            ('numpy', '1.20.0'),
            ('pandas', '1.3.0'),
            ('scikit-learn', '1.0.0'),
            ('scipy', '1.7.0'),
            ('joblib', '1.0.0'),
            ('typer', '0.7.0'),
            ('fastapi', '0.68.0'),
            ('pydantic', '1.8.0')
        ]
        
        missing_packages = []
        version_issues = []
        
        for package_name, min_version in required_packages:
            try:
                package = importlib.import_module(package_name)
                installed_version = getattr(package, '__version__', 'unknown')
                
                # Simple version comparison (not perfect but adequate for most cases)
                if installed_version != 'unknown':
                    if self._compare_versions(installed_version, min_version) < 0:
                        version_issues.append(f'{package_name}: {installed_version} < {min_version}')
                        
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            return {
                'status': 'fail',
                'message': f'Missing packages: {", ".join(missing_packages)}',
                'details': {'missing': missing_packages, 'version_issues': version_issues}
            }
        elif version_issues:
            return {
                'status': 'warning',
                'message': f'Version issues: {", ".join(version_issues)}',
                'details': {'version_issues': version_issues}
            }
        else:
            return {
                'status': 'pass',
                'message': 'All dependencies satisfied',
                'details': {'checked_packages': len(required_packages)}
            }
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files and environment variables."""
        issues = []
        config_found = False
        
        # Check for config files
        config_paths = [
            'config.yaml',
            'config.json',
            'anomaly_detection.cfg',
            os.path.expanduser('~/.anomaly_detection/config.yaml'),
            '/etc/anomaly_detection/config.yaml'
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                config_found = True
                break
        
        # Check environment variables
        env_vars = [
            'ANOMALY_DETECTION_CONFIG',
            'DATABASE_URL',
            'REDIS_URL',
            'MODEL_STORAGE_PATH'
        ]
        
        missing_env_vars = []
        for var in env_vars:
            if not os.getenv(var):
                missing_env_vars.append(var)
        
        if not config_found and missing_env_vars:
            return {
                'status': 'warning',
                'message': 'No configuration file found and environment variables missing',
                'details': {'missing_env_vars': missing_env_vars}
            }
        
        return {
            'status': 'pass',
            'message': 'Configuration appears to be set up',
            'details': {'config_found': config_found, 'env_vars_checked': len(env_vars)}
        }
    
    def check_data_access(self) -> Dict[str, Any]:
        """Check data access capabilities."""
        try:
            # Test basic numpy operations
            test_data = np.random.randn(100, 5)
            
            # Test pandas operations
            import pandas as pd
            df = pd.DataFrame(test_data)
            
            # Test basic statistics
            mean_values = df.mean()
            std_values = df.std()
            
            return {
                'status': 'pass',
                'message': 'Data processing capabilities working',
                'details': {'test_data_shape': test_data.shape}
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Data processing error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def check_model_training(self) -> Dict[str, Any]:
        """Check model training capabilities."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            
            # Generate test data
            X = np.random.randn(1000, 10)
            
            # Test Isolation Forest
            iso_forest = IsolationForest(n_estimators=10, random_state=42)
            iso_forest.fit(X)
            iso_scores = iso_forest.decision_function(X)
            
            # Test LOF
            lof = LocalOutlierFactor(n_neighbors=20)
            lof_scores = lof.fit_predict(X)
            
            return {
                'status': 'pass',
                'message': 'Model training capabilities working',
                'details': {
                    'isolation_forest_scores': len(iso_scores),
                    'lof_scores': len(lof_scores)
                }
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Model training error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def check_api_endpoints(self) -> Dict[str, Any]:
        """Check API endpoint availability."""
        try:
            import requests
            
            # Try to connect to common API endpoints
            endpoints = [
                'http://localhost:8000/health',
                'http://localhost:8000/docs',
                'http://127.0.0.1:8000/health'
            ]
            
            working_endpoints = []
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        working_endpoints.append(endpoint)
                except:
                    continue
            
            if working_endpoints:
                return {
                    'status': 'pass',
                    'message': f'API endpoints accessible: {", ".join(working_endpoints)}',
                    'details': {'working_endpoints': working_endpoints}
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'No API endpoints accessible (may not be running)',
                    'details': {'tested_endpoints': endpoints}
                }
                
        except ImportError:
            return {
                'status': 'warning',
                'message': 'requests package not available for API testing',
                'details': {}
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'API check error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def check_performance(self) -> Dict[str, Any]:
        """Check system performance."""
        try:
            import time
            
            # CPU performance test
            start_time = time.time()
            test_data = np.random.randn(10000, 50)
            matrix_mult = np.dot(test_data, test_data.T)
            cpu_time = time.time() - start_time
            
            # Memory usage test
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # Disk I/O test
            start_time = time.time()
            with tempfile.NamedTemporaryFile() as tmp_file:
                np.save(tmp_file.name, test_data)
                loaded_data = np.load(tmp_file.name + '.npy')
            io_time = time.time() - start_time
            
            performance_issues = []
            if cpu_time > 5.0:
                performance_issues.append(f'Slow CPU performance: {cpu_time:.2f}s')
            if memory_usage > 1000:  # 1GB
                performance_issues.append(f'High memory usage: {memory_usage:.1f}MB')
            if io_time > 2.0:
                performance_issues.append(f'Slow I/O: {io_time:.2f}s')
            
            if performance_issues:
                return {
                    'status': 'warning',
                    'message': f'Performance issues detected: {", ".join(performance_issues)}',
                    'details': {
                        'cpu_time': cpu_time,
                        'memory_mb': memory_usage,
                        'io_time': io_time
                    }
                }
            else:
                return {
                    'status': 'pass',
                    'message': 'Performance looks good',
                    'details': {
                        'cpu_time': cpu_time,
                        'memory_mb': memory_usage,
                        'io_time': io_time
                    }
                }
                
        except ImportError:
            return {
                'status': 'warning',
                'message': 'psutil not available for performance monitoring',
                'details': {}
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Performance check error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def check_storage(self) -> Dict[str, Any]:
        """Check storage availability and permissions."""
        try:
            # Check current directory
            current_dir = os.getcwd()
            can_write = os.access(current_dir, os.W_OK)
            
            # Check common storage locations
            storage_locations = [
                '/tmp',
                os.path.expanduser('~'),
                '/var/lib/anomaly_detection',
                './models',
                './data'
            ]
            
            writable_locations = []
            for location in storage_locations:
                if os.path.exists(location) and os.access(location, os.W_OK):
                    writable_locations.append(location)
            
            # Try to create a temporary file
            try:
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tmp_file.write(b'test')
                temp_file_creation = True
            except:
                temp_file_creation = False
            
            if not can_write or not temp_file_creation:
                return {
                    'status': 'fail',
                    'message': 'Storage access issues detected',
                    'details': {
                        'current_dir_writable': can_write,
                        'temp_file_creation': temp_file_creation,
                        'writable_locations': writable_locations
                    }
                }
            
            return {
                'status': 'pass',
                'message': 'Storage access working',
                'details': {
                    'writable_locations': len(writable_locations),
                    'temp_file_creation': temp_file_creation
                }
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Storage check error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def check_network(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            import socket
            
            # Test DNS resolution
            try:
                socket.gethostbyname('google.com')
                dns_working = True
            except:
                dns_working = False
            
            # Test local network
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('127.0.0.1', 80))
                sock.close()
                local_network = result == 0
            except:
                local_network = False
            
            # Test internet connectivity
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex(('8.8.8.8', 53))
                sock.close()
                internet_access = result == 0
            except:
                internet_access = False
            
            issues = []
            if not dns_working:
                issues.append('DNS resolution failing')
            if not internet_access:
                issues.append('No internet access')
            
            if issues:
                return {
                    'status': 'warning',
                    'message': f'Network issues: {", ".join(issues)}',
                    'details': {
                        'dns_working': dns_working,
                        'local_network': local_network,
                        'internet_access': internet_access
                    }
                }
            
            return {
                'status': 'pass',
                'message': 'Network connectivity working',
                'details': {
                    'dns_working': dns_working,
                    'internet_access': internet_access
                }
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Network check error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        def normalize(v):
            return [int(x) for x in v.split('.')]
        
        v1_parts = normalize(version1)
        v2_parts = normalize(version2)
        
        # Pad with zeros to make them the same length
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for i in range(max_len):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        
        return 0
    
    def _generate_recommendations(self):
        """Generate recommendations based on check results."""
        recommendations = []
        
        # Check for specific issues and provide recommendations
        for check_name, result in self.results['checks'].items():
            if result['status'] == 'fail':
                if check_name == 'Python Environment':
                    recommendations.append('Upgrade to Python 3.8 or higher')
                elif check_name == 'Package Installation':
                    recommendations.append('Install anomaly-detection package: pip install anomaly-detection')
                elif check_name == 'Dependencies':
                    recommendations.append('Install missing dependencies: pip install -r requirements.txt')
                elif check_name == 'Model Training':
                    recommendations.append('Check scikit-learn installation and data format')
                elif check_name == 'Storage':
                    recommendations.append('Ensure write permissions for data and model directories')
        
        # General recommendations based on warnings
        if len(self.results['warnings']) > 0:
            recommendations.append('Review warning messages and consider addressing them')
        
        # Performance recommendations
        performance_result = self.results['checks'].get('Performance', {})
        if performance_result.get('status') == 'warning':
            recommendations.append('Consider optimizing system resources or reducing data size')
        
        self.results['recommendations'] = recommendations

# Diagnostic functions
def diagnose_import_error(package_name: str):
    """Diagnose package import issues."""
    print(f"🔍 Diagnosing import error for: {package_name}")
    
    try:
        import importlib
        package = importlib.import_module(package_name)
        print(f"✅ {package_name} imported successfully")
        
        # Check package info
        if hasattr(package, '__version__'):
            print(f"   Version: {package.__version__}")
        if hasattr(package, '__file__'):
            print(f"   Location: {package.__file__}")
            
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        
        # Suggest installation
        print(f"💡 Try installing: pip install {package_name}")
        
        # Check if it's a submodule issue
        if '.' in package_name:
            parent_package = package_name.split('.')[0]
            try:
                importlib.import_module(parent_package)
                print(f"   Parent package {parent_package} is available")
                print(f"   Submodule {package_name} might not exist or be properly configured")
            except ImportError:
                print(f"   Parent package {parent_package} is also missing")

def diagnose_model_training_error(error_message: str, data_shape: tuple = None):
    """Diagnose model training errors."""
    print(f"🔍 Diagnosing model training error:")
    print(f"   Error: {error_message}")
    
    # Common error patterns and solutions
    error_solutions = {
        'memory': [
            'Reduce batch size or data size',
            'Use data chunking/streaming',
            'Increase system memory',
            'Use memory-efficient algorithms'
        ],
        'nan': [
            'Check for NaN values in input data',
            'Use data imputation techniques',
            'Validate data preprocessing steps',
            'Check feature scaling'
        ],
        'shape': [
            'Verify input data dimensions',
            'Check feature consistency',
            'Ensure proper data reshaping',
            'Validate train/test data compatibility'
        ],
        'convergence': [
            'Adjust algorithm parameters',
            'Increase max_iter parameter',
            'Check data scaling/normalization',
            'Try different algorithm'
        ]
    }
    
    error_lower = error_message.lower()
    suggestions_found = False
    
    for error_type, solutions in error_solutions.items():
        if error_type in error_lower:
            print(f"💡 Possible solutions for {error_type} error:")
            for i, solution in enumerate(solutions, 1):
                print(f"   {i}. {solution}")
            suggestions_found = True
            break
    
    if not suggestions_found:
        print("💡 General troubleshooting steps:")
        print("   1. Check input data format and quality")
        print("   2. Verify algorithm parameters")
        print("   3. Review error logs for more details")
        print("   4. Try with smaller dataset first")
    
    if data_shape:
        print(f"📊 Data shape: {data_shape}")
        if len(data_shape) != 2:
            print("   ⚠️  Warning: Data should be 2D (samples x features)")
        if data_shape[0] < 10:
            print("   ⚠️  Warning: Very few samples for training")

def diagnose_performance_issue(symptoms: Dict[str, Any]):
    """Diagnose performance issues."""
    print("🔍 Diagnosing performance issues:")
    
    high_cpu = symptoms.get('high_cpu', False)
    high_memory = symptoms.get('high_memory', False)
    slow_training = symptoms.get('slow_training', False)
    slow_prediction = symptoms.get('slow_prediction', False)
    
    recommendations = []
    
    if high_cpu:
        recommendations.extend([
            'Use parallel processing with n_jobs parameter',
            'Reduce algorithm complexity (e.g., fewer estimators)',
            'Consider using faster algorithms (e.g., HBOS instead of LOF)',
            'Implement data sampling for large datasets'
        ])
    
    if high_memory:
        recommendations.extend([
            'Process data in chunks/batches',
            'Use memory-efficient data types (float32 instead of float64)',
            'Clear unused variables and call gc.collect()',
            'Consider streaming algorithms for large datasets'
        ])
    
    if slow_training:
        recommendations.extend([
            'Reduce training data size through sampling',
            'Use simpler algorithms for initial experiments',
            'Optimize hyperparameters',
            'Consider using GPU acceleration if available'
        ])
    
    if slow_prediction:
        recommendations.extend([
            'Cache model predictions for repeated queries',
            'Use model compression techniques',
            'Implement batch prediction for multiple samples',
            'Consider model optimization (e.g., quantization)'
        ])
    
    if recommendations:
        print("💡 Recommended optimizations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print("💡 General performance optimization tips:")
        print("   1. Profile your code to identify bottlenecks")
        print("   2. Monitor system resources during execution")
        print("   3. Consider algorithm selection based on data size")
        print("   4. Implement proper data preprocessing")

# Main execution
if __name__ == "__main__":
    health_checker = HealthChecker()
    results = health_checker.run_all_checks()
    
    # Save results to file
    with open('health_check_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: health_check_results.json")
    
    # Exit with appropriate code
    if results['overall_status'] == 'unhealthy':
        sys.exit(1)
    elif results['overall_status'] == 'warning':
        sys.exit(2)
    else:
        sys.exit(0)
```

## Installation Issues

### Package Installation Problems

**Issue**: `pip install anomaly-detection` fails

**Common Causes & Solutions**:

1. **Python version incompatibility**
   ```bash
   # Check Python version
   python --version
   
   # If Python < 3.8, upgrade Python first
   # Then install with specific Python version
   python3.8 -m pip install anomaly-detection
   ```

2. **Missing system dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev gcc g++ make
   
   # CentOS/RHEL
   sudo yum install python3-devel gcc gcc-c++ make
   
   # macOS
   xcode-select --install
   ```

3. **Virtual environment issues**
   ```bash
   # Create clean virtual environment
   python -m venv anomaly_detection_env
   source anomaly_detection_env/bin/activate  # Linux/Mac
   # anomaly_detection_env\Scripts\activate  # Windows
   
   # Upgrade pip and install
   pip install --upgrade pip
   pip install anomaly-detection
   ```

4. **Conflicting packages**
   ```bash
   # Check for conflicts
   pip check
   
   # Resolve conflicts by upgrading
   pip install --upgrade scipy numpy scikit-learn
   pip install anomaly-detection
   ```

**Issue**: Import errors after installation

**Solutions**:
```bash
# Verify installation
python -c "import anomaly_detection; print(anomaly_detection.__version__)"

# Check for missing optional dependencies
pip install anomaly-detection[all]

# Reinstall with force
pip uninstall anomaly-detection
pip install --no-cache-dir anomaly-detection
```

### Docker Installation Issues

**Issue**: Docker build fails

**Solution**:
```dockerfile
# Use specific Python version
FROM python:3.11-slim

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    gcc g++ make libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
```

## Configuration Problems

### Configuration File Issues

**Issue**: Configuration file not found

**Solutions**:
```bash
# Check configuration search paths
python -c "
from anomaly_detection.infrastructure.config.settings import get_settings
settings = get_settings()
print('Config loaded successfully')
"

# Create default configuration
mkdir -p ~/.anomaly_detection
cat > ~/.anomaly_detection/config.yaml << EOF
server:
  host: 0.0.0.0
  port: 8000
  
database:
  url: sqlite:///anomaly_detection.db
  
algorithms:
  default: isolation_forest
  
logging:
  level: INFO
EOF
```

**Issue**: Environment variables not recognized

**Solutions**:
```bash
# Check current environment variables
env | grep ANOMALY

# Set required environment variables
export ANOMALY_DETECTION_CONFIG="/path/to/config.yaml"
export DATABASE_URL="postgresql://user:pass@localhost/db"
export REDIS_URL="redis://localhost:6379"

# Make permanent (add to ~/.bashrc or ~/.profile)
echo 'export ANOMALY_DETECTION_CONFIG="/path/to/config.yaml"' >> ~/.bashrc
```

### Database Connection Issues

**Issue**: Database connection fails

**Solutions**:
```python
# Test database connection
import sqlalchemy

# For PostgreSQL
engine = sqlalchemy.create_engine("postgresql://user:pass@localhost/db")
try:
    connection = engine.connect()
    print("Database connection successful")
    connection.close()
except Exception as e:
    print(f"Database connection failed: {e}")

# Common fixes:
# 1. Check database service is running
# 2. Verify credentials
# 3. Check network connectivity
# 4. Ensure database exists
```

## Model Training Issues

### Data-related Problems

**Issue**: "ValueError: Input contains NaN, infinity or a value too large"

**Solutions**:
```python
import numpy as np
import pandas as pd

# Check for problematic values
def diagnose_data(X):
    print(f"Data shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"NaN values: {np.isnan(X).sum()}")
    print(f"Infinite values: {np.isinf(X).sum()}")
    print(f"Min value: {np.min(X)}")
    print(f"Max value: {np.max(X)}")

# Clean data
def clean_data(X):
    # Remove NaN values
    X = X.dropna() if isinstance(X, pd.DataFrame) else X[~np.isnan(X).any(axis=1)]
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna() if isinstance(X, pd.DataFrame) else X[~np.isinf(X).any(axis=1)]
    
    # Cap extreme values
    for col in range(X.shape[1]):
        q99 = np.percentile(X[:, col], 99)
        q01 = np.percentile(X[:, col], 1)
        X[:, col] = np.clip(X[:, col], q01, q99)
    
    return X
```

**Issue**: "ValueError: Found array with 0 sample(s)"

**Solutions**:
```python
# Check data after filtering
if len(X) == 0:
    print("❌ No data remaining after preprocessing")
    print("💡 Check your data filtering logic")
    
# Ensure minimum sample size
MIN_SAMPLES = 50
if len(X) < MIN_SAMPLES:
    print(f"⚠️  Only {len(X)} samples available, minimum {MIN_SAMPLES} recommended")
```

### Memory Issues

**Issue**: "MemoryError" during training

**Solutions**:
```python
# Monitor memory usage
import psutil
import gc

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

# Reduce memory footprint
def train_with_memory_optimization(X):
    print(f"Initial memory: {memory_usage():.1f} MB")
    
    # Use float32 instead of float64
    X = X.astype(np.float32)
    
    # Process in chunks for large datasets
    chunk_size = 10000
    if len(X) > chunk_size:
        print("Processing in chunks due to large dataset")
        # Implement chunked processing
    
    # Clear unused variables
    gc.collect()
    
    print(f"After optimization: {memory_usage():.1f} MB")
```

### Algorithm-specific Issues

**Issue**: LOF training very slow

**Solutions**:
```python
from sklearn.neighbors import LocalOutlierFactor

# Optimize LOF parameters
lof = LocalOutlierFactor(
    n_neighbors=min(20, len(X) // 10),  # Reduce neighbors for large datasets
    algorithm='ball_tree',  # Use ball_tree for high dimensions
    leaf_size=30,  # Optimize leaf size
    n_jobs=-1  # Use all CPU cores
)

# For very large datasets, use sampling
if len(X) > 50000:
    sample_size = min(50000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    lof.fit(X_sample)
```

## Performance Problems

### Slow Training Performance

**Issue**: Model training takes too long

**Diagnostic Script**:
```python
import time
from sklearn.ensemble import IsolationForest

def benchmark_training(X, algorithms=None):
    """Benchmark different algorithms."""
    if algorithms is None:
        algorithms = {
            'IsolationForest_10': IsolationForest(n_estimators=10),
            'IsolationForest_100': IsolationForest(n_estimators=100),
            'IsolationForest_parallel': IsolationForest(n_estimators=100, n_jobs=-1),
        }
    
    results = {}
    for name, algo in algorithms.items():
        start_time = time.time()
        algo.fit(X)
        training_time = time.time() - start_time
        
        start_time = time.time()
        scores = algo.decision_function(X)
        prediction_time = time.time() - start_time
        
        results[name] = {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'total_time': training_time + prediction_time
        }
        
        print(f"{name}:")
        print(f"  Training: {training_time:.2f}s")
        print(f"  Prediction: {prediction_time:.2f}s")
        print()
    
    return results
```

**Optimization Strategies**:
```python
# 1. Reduce data size through sampling
def smart_sampling(X, max_samples=50000):
    if len(X) <= max_samples:
        return X
    
    # Stratified sampling to preserve data distribution
    from sklearn.cluster import KMeans
    
    n_clusters = min(100, len(X) // 100)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    samples_per_cluster = max_samples // n_clusters
    sampled_indices = []
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) <= samples_per_cluster:
            sampled_indices.extend(cluster_indices)
        else:
            sampled = np.random.choice(
                cluster_indices, samples_per_cluster, replace=False
            )
            sampled_indices.extend(sampled)
    
    return X[sampled_indices]

# 2. Algorithm selection based on data size
def select_algorithm_by_size(data_size):
    if data_size < 1000:
        return "local_outlier_factor"
    elif data_size < 50000:
        return "isolation_forest"
    else:
        return "hbos"  # Faster for large datasets
```

### High Memory Usage

**Issue**: Process consumes too much memory

**Memory Profiling**:
```python
from memory_profiler import profile

@profile
def memory_intensive_function(X):
    """Profile memory usage of anomaly detection."""
    from sklearn.ensemble import IsolationForest
    
    # This will show line-by-line memory usage
    model = IsolationForest(n_estimators=100)
    model.fit(X)
    scores = model.decision_function(X)
    return scores

# Run with: python -m memory_profiler script.py
```

**Memory Optimization**:
```python
def optimize_memory_usage():
    # 1. Use generators for large datasets
    def data_generator(file_path, chunk_size=1000):
        import pandas as pd
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            yield chunk.values
    
    # 2. Process in batches
    def batch_process(X, model, batch_size=5000):
        n_batches = len(X) // batch_size + (1 if len(X) % batch_size else 0)
        results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            batch = X[start_idx:end_idx]
            
            batch_scores = model.decision_function(batch)
            results.append(batch_scores)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        return np.concatenate(results)
    
    # 3. Use memory-mapped files for very large datasets
    def create_memory_mapped_data(file_path):
        import numpy as np
        data = np.load(file_path)
        
        # Save as memory-mapped file
        mm_file = file_path.replace('.npy', '_mm.dat')
        mm_array = np.memmap(
            mm_file, dtype=data.dtype, mode='w+', shape=data.shape
        )
        mm_array[:] = data[:]
        
        return mm_array
```

## API and Integration Issues

### FastAPI Issues

**Issue**: API server won't start

**Diagnostic Steps**:
```bash
# Check if port is already in use
netstat -tlnp | grep :8000
# or
lsof -i :8000

# Kill process using the port
kill -9 <PID>

# Start with different port
uvicorn anomaly_detection.main:app --host 0.0.0.0 --port 8001

# Check for import errors
python -c "from anomaly_detection.main import app; print('Import successful')"
```

**Issue**: API endpoints returning 500 errors

**Debugging**:
```python
# Enable debug mode
import uvicorn
from anomaly_detection.main import app

if __name__ == "__main__":
    uvicorn.run(
        "anomaly_detection.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        debug=True,
        log_level="debug"
    )

# Add error handling to endpoints
from fastapi import HTTPException
import traceback

@app.post("/detect")
async def detect_anomalies(data: dict):
    try:
        # Your detection logic here
        result = perform_detection(data)
        return result
    except Exception as e:
        # Log the full traceback
        print(f"Error in detection endpoint: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Database Integration Issues

**Issue**: Database queries timeout

**Solutions**:
```python
import asyncio
import asyncpg

async def test_database_performance():
    """Test database performance and identify bottlenecks."""
    
    # Connection with timeout settings
    conn = await asyncpg.connect(
        "postgresql://user:pass@localhost/db",
        command_timeout=60,
        server_settings={
            'jit': 'off',  # Disable JIT for better performance on small queries
            'statement_timeout': '30s'
        }
    )
    
    # Test query performance
    import time
    
    start_time = time.time()
    result = await conn.fetch("SELECT COUNT(*) FROM large_table")
    query_time = time.time() - start_time
    
    print(f"Query took {query_time:.2f} seconds")
    
    if query_time > 5:
        print("💡 Consider adding indexes or optimizing query")
        
        # Check for missing indexes
        missing_indexes = await conn.fetch("""
            SELECT schemaname, tablename, attname, n_distinct, correlation
            FROM pg_stats
            WHERE tablename = $1
            ORDER BY n_distinct DESC
        """, "your_table_name")
        
        print("Potential index candidates:")
        for row in missing_indexes:
            print(f"  {row['attname']}: {row['n_distinct']} distinct values")
    
    await conn.close()

# Optimize batch inserts
async def optimize_batch_inserts(data_batch):
    """Optimize large batch insertions."""
    conn = await asyncpg.connect("postgresql://user:pass@localhost/db")
    
    # Use COPY for very large batches
    if len(data_batch) > 10000:
        # Create temporary CSV
        import io
        import csv
        
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        for row in data_batch:
            writer.writerow(row)
        
        csv_buffer.seek(0)
        
        # Use COPY command
        await conn.copy_to_table(
            'your_table', 
            source=csv_buffer, 
            format='csv'
        )
    else:
        # Use batch insert for smaller batches
        await conn.executemany(
            "INSERT INTO your_table (col1, col2) VALUES ($1, $2)",
            data_batch
        )
    
    await conn.close()
```

## Data Processing Errors

### Data Format Issues

**Issue**: "Unable to convert string to float"

**Solutions**:
```python
import pandas as pd
import numpy as np

def diagnose_data_types(df):
    """Diagnose data type issues."""
    print("Data Types:")
    print(df.dtypes)
    print("\nNon-numeric columns:")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"  {col}: {df[col].unique()[:10]}")  # Show first 10 unique values
            
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col])
                print(f"    ✅ Can be converted to numeric")
            except:
                print(f"    ❌ Cannot be converted to numeric")

def clean_numeric_data(df):
    """Clean and convert data to numeric."""
    df_cleaned = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remove common non-numeric characters
            df_cleaned[col] = df_cleaned[col].astype(str)
            df_cleaned[col] = df_cleaned[col].str.replace(',', '')
            df_cleaned[col] = df_cleaned[col].str.replace('$', '')
            df_cleaned[col] = df_cleaned[col].str.replace('%', '')
            df_cleaned[col] = df_cleaned[col].str.strip()
            
            # Convert to numeric, errors='coerce' will set invalid values to NaN
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    return df_cleaned
```

**Issue**: "Data contains only one class"

**Solutions**:
```python
def check_data_distribution(y):
    """Check label distribution for supervised learning."""
    unique_classes = np.unique(y)
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Class distribution:")
    
    for class_val in unique_classes:
        count = np.sum(y == class_val)
        percentage = count / len(y) * 100
        print(f"  Class {class_val}: {count} samples ({percentage:.1f}%)")
    
    if len(unique_classes) == 1:
        print("❌ Only one class found - cannot train supervised model")
        print("💡 Consider:")
        print("   1. Using unsupervised anomaly detection")
        print("   2. Collecting more diverse data")
        print("   3. Checking data filtering logic")
        
        return False
    
    return True
```

### Feature Engineering Issues

**Issue**: Features have very different scales

**Solutions**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def analyze_feature_scales(X, feature_names=None):
    """Analyze feature scales and recommend scaling method."""
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    print("Feature Scale Analysis:")
    print("-" * 50)
    
    scaling_needed = False
    for i, name in enumerate(feature_names):
        mean_val = np.mean(X[:, i])
        std_val = np.std(X[:, i])
        min_val = np.min(X[:, i])
        max_val = np.max(X[:, i])
        range_val = max_val - min_val
        
        print(f"{name}:")
        print(f"  Range: [{min_val:.2f}, {max_val:.2f}] (span: {range_val:.2f})")
        print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        
        # Check if scaling is needed
        if range_val > 100 or abs(mean_val) > 10:
            scaling_needed = True
            print(f"  ⚠️  Large scale detected")
        
        print()
    
    if scaling_needed:
        print("💡 Scaling recommendations:")
        print("  1. StandardScaler: For normally distributed features")
        print("  2. MinMaxScaler: For bounded features")
        print("  3. RobustScaler: For features with outliers")
        
        # Demonstrate scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"\nAfter StandardScaler:")
        print(f"  Mean: {np.mean(X_scaled, axis=0)}")
        print(f"  Std: {np.std(X_scaled, axis=0)}")
        
        return X_scaled, scaler
    
    return X, None
```

## Deployment and Infrastructure

### Docker Issues

**Issue**: Container crashes on startup

**Diagnostic Steps**:
```bash
# Check container logs
docker logs <container_id>

# Run container interactively for debugging
docker run -it --entrypoint /bin/bash your-image

# Check resource limits
docker stats <container_id>

# Verify environment variables
docker exec <container_id> env | grep ANOMALY
```

**Common Solutions**:
```dockerfile
# Dockerfile best practices for debugging
FROM python:3.11-slim

# Add debugging tools
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Start application
CMD ["uvicorn", "anomaly_detection.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Issues

**Issue**: Pods failing to start

**Diagnostic Commands**:
```bash
# Check pod status
kubectl get pods -l app=anomaly-detection

# Describe pod for events
kubectl describe pod <pod-name>

# Check pod logs
kubectl logs <pod-name> -c <container-name>

# Check resource usage
kubectl top pods

# Check persistent volume claims
kubectl get pvc
```

**Common Issues & Solutions**:

1. **ImagePullBackOff**
   ```bash
   # Check image exists and is accessible
   docker pull your-registry/anomaly-detection:latest
   
   # Check image pull secrets
   kubectl get secrets
   kubectl describe secret <image-pull-secret>
   ```

2. **CrashLoopBackOff**
   ```yaml
   # Add debugging to deployment
   apiVersion: apps/v1
   kind: Deployment
   spec:
     template:
       spec:
         containers:
         - name: anomaly-detection
           image: your-image
           command: ["/bin/sh"]
           args: ["-c", "sleep 3600"]  # Keep container running for debug
   ```

3. **Resource limits**
   ```yaml
   # Adjust resource limits
   resources:
     requests:
       memory: "1Gi"
       cpu: "500m"
     limits:
       memory: "4Gi"
       cpu: "2000m"
   ```

## Common Error Messages

### Error Reference Guide

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `ModuleNotFoundError: No module named 'anomaly_detection'` | Package not installed | `pip install anomaly-detection` |
| `ValueError: Input contains NaN` | Data has missing values | Clean data or use imputation |
| `MemoryError` | Insufficient memory | Reduce data size or use chunking |
| `ConnectionError: [Errno 111] Connection refused` | Service not running | Start the API server |
| `sklearn.exceptions.NotFittedError` | Model not trained | Call `fit()` before `predict()` |
| `ValueError: Found array with 0 sample(s)` | Empty dataset after filtering | Check data filtering logic |
| `TimeoutError` | Operation timeout | Increase timeout or optimize query |
| `PermissionError: [Errno 13]` | File permission issues | Check file/directory permissions |
| `OSError: [Errno 28] No space left on device` | Disk full | Free up disk space |
| `ImportError: cannot import name 'IsolationForest'` | Version compatibility | Update scikit-learn |

### Detailed Error Solutions

**Error**: `sklearn.exceptions.NotFittedError: This IsolationForest instance is not fitted yet.`

```python
# Problem: Trying to predict before training
from sklearn.ensemble import IsolationForest

model = IsolationForest()
# This will fail:
# scores = model.decision_function(X)

# Solution: Fit the model first
model.fit(X_train)
scores = model.decision_function(X_test)

# Or check if model is fitted
def safe_predict(model, X):
    try:
        return model.decision_function(X)
    except sklearn.exceptions.NotFittedError:
        print("❌ Model not fitted. Training model first...")
        model.fit(X)
        return model.decision_function(X)
```

**Error**: `ValueError: could not convert string to float`

```python
# Problem: Non-numeric data in features
# Solution: Data type conversion
import pandas as pd

def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values created by conversion
    df = df.fillna(df.mean())  # or use other imputation methods
    
    return df
```

## Frequently Asked Questions

### General Questions

**Q: Which algorithm should I use for my data?**

A: Algorithm selection depends on your data characteristics:

- **Small datasets (< 1,000 samples)**: Local Outlier Factor (LOF)
- **Medium datasets (1,000 - 50,000 samples)**: Isolation Forest
- **Large datasets (> 50,000 samples)**: HBOS or One-Class SVM
- **High-dimensional data**: PCA + Isolation Forest
- **Time series data**: Streaming algorithms or LSTM-based approaches
- **Mixed data types**: Ensemble methods

```python
def recommend_algorithm(data_size, n_features, data_type):
    """Recommend algorithm based on data characteristics."""
    
    if data_type == 'time_series':
        return 'lstm_autoencoder'
    
    if n_features > 100:
        return 'pca_isolation_forest'
    
    if data_size < 1000:
        return 'local_outlier_factor'
    elif data_size < 50000:
        return 'isolation_forest'
    else:
        return 'hbos'

# Usage
recommendation = recommend_algorithm(
    data_size=len(X),
    n_features=X.shape[1],
    data_type='tabular'
)
print(f"Recommended algorithm: {recommendation}")
```

**Q: How do I interpret anomaly scores?**

A: Anomaly scores vary by algorithm:

- **Isolation Forest**: Negative scores indicate anomalies
- **LOF**: Scores > 1 indicate anomalies
- **One-Class SVM**: Negative scores indicate anomalies

```python
def interpret_scores(scores, algorithm_name):
    """Interpret anomaly scores based on algorithm."""
    
    if algorithm_name == 'isolation_forest':
        anomalies = scores < 0
        interpretation = "Negative scores = anomalies"
    elif algorithm_name == 'local_outlier_factor':
        anomalies = scores > 1
        interpretation = "Scores > 1 = anomalies"
    elif algorithm_name == 'one_class_svm':
        anomalies = scores < 0
        interpretation = "Negative scores = anomalies"
    else:
        # Generic threshold-based approach
        threshold = np.percentile(scores, 5)  # Bottom 5%
        anomalies = scores < threshold
        interpretation = f"Scores < {threshold:.3f} = anomalies"
    
    n_anomalies = np.sum(anomalies)
    anomaly_rate = n_anomalies / len(scores) * 100
    
    print(f"Algorithm: {algorithm_name}")
    print(f"Interpretation: {interpretation}")
    print(f"Anomalies detected: {n_anomalies} ({anomaly_rate:.1f}%)")
    
    return anomalies
```

**Q: How do I tune algorithm parameters?**

A: Use grid search or Bayesian optimization:

```python
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import adjusted_rand_score

def tune_isolation_forest(X, y_true=None):
    """Tune Isolation Forest parameters."""
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'contamination': [0.05, 0.1, 0.15, 0.2],
        'max_features': [0.5, 0.7, 1.0]
    }
    
    best_score = -np.inf
    best_params = None
    
    for params in ParameterGrid(param_grid):
        model = IsolationForest(**params, random_state=42)
        model.fit(X)
        scores = model.decision_function(X)
        
        if y_true is not None:
            # Use labeled data for evaluation
            y_pred = (scores < 0).astype(int)
            score = adjusted_rand_score(y_true, y_pred)
        else:
            # Use silhouette score or other unsupervised metric
            from sklearn.metrics import silhouette_score
            y_pred = (scores < 0).astype(int)
            if len(np.unique(y_pred)) > 1:
                score = silhouette_score(X, y_pred)
            else:
                score = -1
        
        print(f"Params: {params}, Score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_params = params
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.3f}")
    
    return best_params
```

### Performance Questions

**Q: Why is training so slow?**

A: Common causes and solutions:

1. **Large dataset**: Use sampling or chunking
2. **High dimensionality**: Apply dimensionality reduction
3. **Inefficient algorithm**: Choose faster algorithm
4. **Single-threaded execution**: Use `n_jobs=-1`

```python
# Performance optimization checklist
def optimize_training_performance(X):
    print("Training Performance Optimization")
    print("-" * 40)
    
    # 1. Check data size
    print(f"Data size: {X.shape}")
    if X.shape[0] > 50000:
        print("💡 Consider sampling large dataset")
    
    # 2. Check dimensionality
    if X.shape[1] > 100:
        print("💡 Consider dimensionality reduction")
    
    # 3. Check data types
    print(f"Data type: {X.dtype}")
    if X.dtype == np.float64:
        print("💡 Convert to float32 for memory efficiency")
        X = X.astype(np.float32)
    
    # 4. Memory usage
    memory_mb = X.nbytes / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    
    return X
```

**Q: How can I speed up predictions?**

A: Prediction optimization strategies:

```python
# 1. Batch predictions
def batch_predict(model, X, batch_size=10000):
    """Predict in batches for better performance."""
    n_samples = len(X)
    predictions = np.zeros(n_samples)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        predictions[i:end_idx] = model.decision_function(batch)
    
    return predictions

# 2. Model serialization for faster loading
import joblib

# Save model
joblib.dump(model, 'model.pkl', compress=3)

# Load model (faster than retraining)
model = joblib.load('model.pkl')

# 3. Caching predictions
from functools import lru_cache
import hashlib

def hash_array(arr):
    """Create hash of numpy array for caching."""
    return hashlib.md5(arr.tobytes()).hexdigest()

@lru_cache(maxsize=1000)
def cached_predict(model_hash, data_hash):
    """Cache predictions based on model and data hash."""
    # This is a simplified example
    # In practice, you'd need to reconstruct the data
    pass
```

### Data Questions

**Q: How much data do I need for training?**

A: Data requirements depend on algorithm and data complexity:

- **Minimum**: 100-500 samples
- **Recommended**: 1,000-10,000 samples
- **Complex data**: 10,000+ samples

```python
def assess_data_adequacy(X):
    """Assess if dataset is adequate for anomaly detection."""
    n_samples, n_features = X.shape
    
    print("Data Adequacy Assessment")
    print("-" * 30)
    
    # Sample size assessment
    if n_samples < 100:
        print("❌ Very small dataset - results may be unreliable")
        return False
    elif n_samples < 1000:
        print("⚠️  Small dataset - consider collecting more data")
        adequacy = "marginal"
    elif n_samples < 10000:
        print("✅ Adequate dataset size")
        adequacy = "good"
    else:
        print("✅ Large dataset - excellent for training")
        adequacy = "excellent"
    
    # Feature-to-sample ratio
    ratio = n_features / n_samples
    print(f"Feature-to-sample ratio: {ratio:.3f}")
    
    if ratio > 0.1:
        print("⚠️  High dimensionality relative to sample size")
        print("💡 Consider dimensionality reduction")
    
    return adequacy in ["good", "excellent"]
```

**Q: How do I handle missing values?**

A: Missing value strategies:

```python
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_values(X, strategy='auto'):
    """Handle missing values with different strategies."""
    
    missing_percentage = np.isnan(X).sum() / X.size * 100
    print(f"Missing values: {missing_percentage:.1f}%")
    
    if missing_percentage == 0:
        print("✅ No missing values")
        return X
    
    if strategy == 'auto':
        if missing_percentage < 5:
            strategy = 'drop'
        elif missing_percentage < 20:
            strategy = 'impute_mean'
        else:
            strategy = 'impute_knn'
    
    if strategy == 'drop':
        # Remove rows with any missing values
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        print(f"Dropped {np.sum(~mask)} rows with missing values")
        
    elif strategy == 'impute_mean':
        imputer = SimpleImputer(strategy='mean')
        X_clean = imputer.fit_transform(X)
        print("Imputed missing values with mean")
        
    elif strategy == 'impute_knn':
        imputer = KNNImputer(n_neighbors=5)
        X_clean = imputer.fit_transform(X)
        print("Imputed missing values with KNN")
        
    return X_clean
```

This comprehensive troubleshooting guide provides systematic approaches to diagnose and resolve common issues in anomaly detection systems, from installation problems to performance optimization and deployment challenges.