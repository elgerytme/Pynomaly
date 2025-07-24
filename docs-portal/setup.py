#!/usr/bin/env python3
"""
Documentation Portal Setup Script

This script sets up the unified documentation portal for the Anomaly Detection Platform.
It aggregates documentation from all packages and creates a cohesive user experience.

Usage:
    python setup.py --initialize  # Set up the portal structure
    python setup.py --build       # Build the documentation
    python setup.py --serve       # Serve locally for development
    python setup.py --deploy      # Deploy to production
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json


class DocumentationPortal:
    """Manages the unified documentation portal setup and deployment."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).parent.parent
        self.docs_portal_dir = self.root_dir / 'docs-portal'
        self.packages_dir = self.root_dir / 'src' / 'packages'
        
        # Documentation sources mapping
        self.doc_sources = {
            'anomaly-detection': self.packages_dir / 'data' / 'anomaly_detection' / 'docs',
            'machine-learning': self.packages_dir / 'ai' / 'machine_learning' / 'docs',
            'data-platform': self.packages_dir / 'data',
            'enterprise': self.packages_dir / 'enterprise',
            'infrastructure': self.packages_dir / 'infrastructure'
        }
    
    def initialize_portal(self):
        """Initialize the documentation portal structure."""
        print("🚀 Initializing Documentation Portal...")
        
        # Create portal directory structure
        self._create_directory_structure()
        
        # Copy and adapt existing documentation
        self._aggregate_package_documentation()
        
        # Generate navigation and indexes
        self._generate_navigation_structure()
        
        # Set up build configuration
        self._setup_build_configuration()
        
        print("✅ Documentation portal initialized successfully!")
    
    def _create_directory_structure(self):
        """Create the documentation portal directory structure."""
        dirs_to_create = [
            'docs',
            'docs/getting-started',
            'docs/packages',
            'docs/packages/anomaly-detection',
            'docs/packages/machine-learning',
            'docs/packages/data-platform',
            'docs/packages/enterprise',
            'docs/packages/infrastructure',
            'docs/guides',
            'docs/api',
            'docs/architecture',
            'docs/examples',
            'docs/resources',
            'docs/assets',
            'docs/assets/css',
            'docs/assets/js',
            'docs/assets/images',
            'overrides',
            'overrides/partials'
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.docs_portal_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
    
    def _aggregate_package_documentation(self):
        """Aggregate documentation from all packages."""
        print("📚 Aggregating package documentation...")
        
        for package_name, source_path in self.doc_sources.items():
            if not source_path.exists():
                print(f"⚠️  Source path not found: {source_path}")
                continue
            
            target_path = self.docs_portal_dir / 'docs' / 'packages' / package_name
            
            # Copy existing documentation
            if (source_path / 'docs').exists():
                self._copy_docs_with_adaptation(
                    source_path / 'docs',
                    target_path,
                    package_name
                )
            
            # Generate package index if not exists
            self._generate_package_index(package_name, target_path)
            
            print(f"✅ Aggregated documentation for {package_name}")
    
    def _copy_docs_with_adaptation(self, source: Path, target: Path, package_name: str):
        """Copy documentation with adaptation for unified portal."""
        if not source.exists():
            return
        
        for item in source.iterdir():
            if item.is_file() and item.suffix == '.md':
                # Adapt markdown files for unified portal
                self._adapt_markdown_file(item, target / item.name, package_name)
            elif item.is_dir() and item.name not in ['.git', '__pycache__']:
                # Recursively copy directories
                target_subdir = target / item.name
                target_subdir.mkdir(exist_ok=True)
                self._copy_docs_with_adaptation(item, target_subdir, package_name)
    
    def _adapt_markdown_file(self, source: Path, target: Path, package_name: str):
        """Adapt markdown files for the unified portal."""
        try:
            with open(source, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix relative links to work in unified portal
            content = self._fix_relative_links(content, package_name)
            
            # Add package context to headers
            content = self._add_package_context(content, package_name)
            
            # Update image paths
            content = self._fix_image_paths(content, package_name)
            
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            print(f"⚠️  Error adapting {source}: {e}")
    
    def _fix_relative_links(self, content: str, package_name: str) -> str:
        """Fix relative links to work in unified portal context."""
        import re
        
        # Fix markdown links
        def fix_link(match):
            link_text = match.group(1)
            link_url = match.group(2)
            
            # Skip external links
            if link_url.startswith(('http', 'https', 'mailto')):
                return match.group(0)
            
            # Fix relative links within package
            if not link_url.startswith('/'):
                if link_url.startswith('../'):
                    # Handle parent directory references
                    link_url = f"/packages/{package_name}/{link_url[3:]}"
                else:
                    link_url = f"/packages/{package_name}/{link_url}"
            
            return f"[{link_text}]({link_url})"
        
        content = re.sub(r'\\[([^\\]]+)\\]\\(([^)]+)\\)', fix_link, content)
        return content
    
    def _add_package_context(self, content: str, package_name: str) -> str:
        """Add package context information to documentation."""
        package_info = {
            'anomaly-detection': {
                'title': 'Anomaly Detection Package',
                'description': 'Core anomaly detection algorithms and workflows',
                'status': '🟢 Production Ready',
                'maturity': '⭐⭐⭐⭐⭐'
            },
            'machine-learning': {
                'title': 'Machine Learning Package',
                'description': 'Advanced ML capabilities and model management',
                'status': '🟡 Beta',
                'maturity': '⭐⭐⭐⭐'
            },
            'data-platform': {
                'title': 'Data Platform Package',
                'description': 'Data processing and engineering capabilities',
                'status': '🟡 Beta',
                'maturity': '⭐⭐⭐'
            },
            'enterprise': {
                'title': 'Enterprise Package',
                'description': 'Enterprise features and governance',
                'status': '🟢 Production Ready',
                'maturity': '⭐⭐⭐⭐⭐'
            },
            'infrastructure': {
                'title': 'Infrastructure Package',
                'description': 'Infrastructure and observability',
                'status': '🟢 Production Ready',
                'maturity': '⭐⭐⭐⭐'
            }
        }
        
        info = package_info.get(package_name, {})
        if info and not content.startswith('# '):
            # Add package header if not already present
            header = f"""# {info['title']}

{info['status']} | {info['maturity']}

{info['description']}

---

"""
            content = header + content
        
        return content
    
    def _fix_image_paths(self, content: str, package_name: str) -> str:
        """Fix image paths to work in unified portal."""
        import re
        
        def fix_image(match):
            alt_text = match.group(1)
            image_url = match.group(2)
            
            # Skip external images
            if image_url.startswith(('http', 'https')):
                return match.group(0)
            
            # Fix relative image paths
            if not image_url.startswith('/'):
                image_url = f"/assets/images/packages/{package_name}/{image_url}"
            
            return f"![{alt_text}]({image_url})"
        
        content = re.sub(r'!\\[([^\\]]*)\\]\\(([^)]+)\\)', fix_image, content)
        return content
    
    def _generate_package_index(self, package_name: str, target_dir: Path):
        """Generate package index file if it doesn't exist."""
        index_file = target_dir / 'index.md'
        
        if index_file.exists():
            return
        
        # Generate basic index content
        package_templates = {
            'anomaly-detection': self._generate_anomaly_detection_index(),
            'machine-learning': self._generate_machine_learning_index(),
            'data-platform': self._generate_data_platform_index(),
            'enterprise': self._generate_enterprise_index(),
            'infrastructure': self._generate_infrastructure_index()
        }
        
        content = package_templates.get(package_name, self._generate_generic_index(package_name))
        
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"📝 Generated index for {package_name}")
    
    def _generate_anomaly_detection_index(self) -> str:
        """Generate anomaly detection package index."""
        return """# Anomaly Detection Package

🟢 Production Ready | ⭐⭐⭐⭐⭐

The flagship package providing comprehensive anomaly detection capabilities with 20+ algorithms, ensemble methods, and real-time processing.

## Quick Start

```python
from anomaly_detection import create_detector, load_dataset

# Load sample data
data = load_dataset('timeseries_sample')

# Create detector
detector = create_detector('IsolationForest', contamination_rate=0.1)

# Detect anomalies
result = detector.detect(data)
print(f"Found {result.n_anomalies} anomalies")
```

## Key Features

- **🧮 20+ Algorithms**: Statistical, ML, and deep learning approaches
- **⚡ Real-time Processing**: High-performance streaming analytics
- **🎯 Ensemble Methods**: Combine algorithms for better accuracy
- **🔍 Explainable AI**: SHAP and LIME integration
- **🚀 Production Ready**: Enterprise deployment tools

## Available Algorithms

### Statistical Methods
- Z-Score and Modified Z-Score
- Interquartile Range (IQR)
- Seasonal Decomposition
- STL Decomposition

### Machine Learning
- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Elliptic Envelope

### Deep Learning
- Autoencoders
- LSTM Networks
- Transformer Models
- Variational Autoencoders

## Documentation Sections

- [Installation](installation.md) - Setup and configuration
- [Quick Start](quickstart.md) - Get started in 5 minutes
- [Architecture](architecture.md) - System design and components
- [Algorithms](algorithms.md) - Detailed algorithm documentation
- [API Reference](api.md) - Complete API documentation
- [CLI Tools](cli.md) - Command-line interface
- [Configuration](configuration.md) - Configuration options
- [Ensemble Methods](ensemble.md) - Combining algorithms
- [Streaming](streaming.md) - Real-time processing
- [Explainability](explainability.md) - Model interpretability
- [Performance](performance.md) - Optimization and benchmarking
- [Deployment](deployment.md) - Production deployment
- [Security](security.md) - Security considerations
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## Use Cases

- **Financial Services**: Fraud detection, market anomalies
- **Manufacturing**: Quality control, predictive maintenance
- **Healthcare**: Patient monitoring, clinical anomalies
- **Technology**: Infrastructure monitoring, security

## Next Steps

1. [Install the package](installation.md)
2. [Follow the quick start guide](quickstart.md)
3. [Explore the algorithms](algorithms.md)
4. [Try the examples](../examples/basic.md)
"""
    
    def _generate_machine_learning_index(self) -> str:
        """Generate machine learning package index."""
        return """# Machine Learning Package

🟡 Beta | ⭐⭐⭐⭐

Advanced ML capabilities including AutoML, active learning, and comprehensive model management with MLOps integration.

## Quick Start

```python
from machine_learning import AutoMLOptimizer, ModelManager

# AutoML optimization
automl = AutoMLOptimizer(
    algorithms=['isolation_forest', 'one_class_svm'],
    optimization_metric='f1_score'
)

best_config = automl.optimize(dataset, time_budget_hours=1)

# Model management
model_manager = ModelManager()
model_id = model_manager.register_model(best_config.model)
```

## Key Features

- **🤖 AutoML**: Automated model selection and hyperparameter tuning
- **📚 Active Learning**: Human-in-the-loop learning workflows
- **🔄 MLOps**: Complete model lifecycle management
- **🧪 A/B Testing**: Model comparison and validation
- **📊 Experiment Tracking**: MLflow integration

## Core Components

### AutoML Optimizer
- Automated algorithm selection
- Hyperparameter optimization
- Feature engineering automation
- Model ensemble generation

### Active Learning
- Uncertainty sampling
- Query by committee
- Expected model change
- Human feedback integration

### Model Management
- Model versioning and tracking
- Performance monitoring
- Drift detection
- Automated retraining

## Documentation Sections

- [Model Management](model-management.md) - Lifecycle management
- [Active Learning](active-learning.md) - Human-in-the-loop learning
- [AutoML](automl.md) - Automated machine learning
- [MLOps](mlops.md) - Operations and deployment

## Integration Examples

- [Cross-package Workflows](../guides/cross-package-workflows.md)
- [Production Deployment](../guides/production-deployment.md)
- [Monitoring Integration](../guides/monitoring.md)
"""
    
    def _generate_data_platform_index(self) -> str:
        """Generate data monorepo package index."""
        return """# Data Platform Package

🟡 Beta | ⭐⭐⭐

Comprehensive data processing and engineering capabilities for batch and streaming workloads with enterprise-grade data management.

## Quick Start

```python
from data_platform import DataPipeline, StreamProcessor

# Batch processing
pipeline = DataPipeline()
data = pipeline.load_from_source('s3://data-lake/sensor-data')
processed = pipeline.process(data, steps=['clean', 'normalize'])

# Stream processing
streamer = StreamProcessor('kafka://anomaly-stream')
for batch in streamer.stream():
    result = process_batch(batch)
```

## Key Features

- **🔄 ETL/ELT**: Flexible pipeline management
- **🌊 Streaming**: Real-time data processing
- **✅ Quality**: Data validation and monitoring
- **📊 Formats**: Multi-format support (JSON, CSV, Parquet, Avro)

## Core Components

### Data Pipelines
- Batch and streaming processing
- Data transformation and validation
- Schema management and evolution
- Performance optimization

### Quality Monitoring
- Data profiling and validation
- Quality metrics and reporting
- Anomaly detection in data quality
- Automated alerting

## Documentation Sections

- [Data Engineering](data-engineering.md) - Pipeline development
- [Data Architecture](data-architecture.md) - System design
- [Streaming](streaming.md) - Real-time processing
"""
    
    def _generate_enterprise_index(self) -> str:
        """Generate enterprise package index."""
        return """# Enterprise Package

🟢 Production Ready | ⭐⭐⭐⭐⭐

Enterprise-grade features including authentication, authorization, governance, and compliance for production deployments.

## Key Features

- **🔐 Authentication**: Multi-factor authentication and SSO
- **👥 Authorization**: Role-based access control
- **📋 Governance**: Policy enforcement and compliance
- **📊 Audit**: Comprehensive logging and reporting

## Documentation Sections

- [Authentication](authentication.md) - User authentication
- [Authorization](authorization.md) - Access control
- [Governance](governance.md) - Policy management
- [Compliance](compliance.md) - Regulatory compliance
"""
    
    def _generate_infrastructure_index(self) -> str:
        """Generate infrastructure package index."""
        return """# Infrastructure Package

🟢 Production Ready | ⭐⭐⭐⭐

Infrastructure concerns including monitoring, logging, and observability for production-ready deployments.

## Key Features

- **📊 Monitoring**: Prometheus and Grafana integration
- **📝 Logging**: Structured logging with ELK stack
- **🔍 Tracing**: Distributed tracing with Jaeger
- **🏥 Health**: Health checks and circuit breakers

## Documentation Sections

- [Monitoring](monitoring.md) - Metrics and alerting
- [Logging](logging.md) - Log management
- [Deployment](deployment.md) - Infrastructure deployment
"""
    
    def _generate_generic_index(self, package_name: str) -> str:
        """Generate generic package index."""
        return f"""# {package_name.title().replace('-', ' ')} Package

Documentation for the {package_name} package.

## Overview

This package provides core functionality for the Anomaly Detection Platform.

## Getting Started

Detailed documentation coming soon.

## Contributing

Contributions are welcome! Please see the main project README for contribution guidelines.
"""
    
    def _generate_navigation_structure(self):
        """Generate navigation structure and update mkdocs.yml."""
        print("🧭 Generating navigation structure...")
        
        # The navigation is already defined in mkdocs.yml
        # Here we could dynamically update it based on available content
        pass
    
    def _setup_build_configuration(self):
        """Set up build configuration and requirements."""
        print("⚙️  Setting up build configuration...")
        
        # Create requirements.txt for the docs portal
        requirements = [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.0.0',
            'mkdocs-git-revision-date-localized-plugin>=1.2.0',
            'mkdocs-minify-plugin>=0.7.0',
            'mkdocs-social>=0.1.0',
            'mkdocstrings[python]>=0.23.0',
            'pymdown-extensions>=10.0.0',
            'pillow>=10.0.0',
            'cairosvg>=2.5.0'
        ]
        
        requirements_file = self.docs_portal_dir / 'requirements.txt'
        with open(requirements_file, 'w') as f:
            f.write('\\n'.join(requirements))
        
        # Create build script
        build_script = self.docs_portal_dir / 'build.sh'
        with open(build_script, 'w') as f:
            f.write('''#!/bin/bash
set -e

echo "🏗️  Building documentation portal..."

# Install dependencies
pip install -r requirements.txt

# Build documentation
mkdocs build --clean

echo "✅ Documentation built successfully!"
echo "📁 Output available in: site/"
''')
        
        # Make build script executable
        os.chmod(build_script, 0o755)
        
        print("✅ Build configuration complete!")
    
    def build_documentation(self):
        """Build the documentation using MkDocs."""
        print("🏗️  Building documentation...")
        
        os.chdir(self.docs_portal_dir)
        
        try:
            # Install requirements
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            
            # Build docs
            subprocess.run(['mkdocs', 'build', '--clean'], check=True)
            
            print("✅ Documentation built successfully!")
            print(f"📁 Output available in: {self.docs_portal_dir / 'site'}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Build failed: {e}")
            return False
        
        return True
    
    def serve_documentation(self, host: str = '127.0.0.1', port: int = 8000):
        """Serve documentation locally for development."""
        print(f"🌐 Serving documentation at http://{host}:{port}")
        
        os.chdir(self.docs_portal_dir)
        
        try:
            subprocess.run([
                'mkdocs', 'serve', '--dev-addr', f'{host}:{port}'
            ])
        except KeyboardInterrupt:
            print("\\n🛑 Development server stopped")
    
    def deploy_documentation(self, method: str = 'github-pages'):
        """Deploy documentation to production."""
        print(f"🚀 Deploying documentation using {method}...")
        
        os.chdir(self.docs_portal_dir)
        
        if method == 'github-pages':
            try:
                subprocess.run(['mkdocs', 'gh-deploy', '--force'], check=True)
                print("✅ Documentation deployed to GitHub Pages!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Deployment failed: {e}")
                return False
        
        elif method == 'docker':
            self._deploy_with_docker()
        
        else:
            print(f"❌ Unknown deployment method: {method}")
            return False
        
        return True
    
    def _deploy_with_docker(self):
        """Deploy documentation using Docker."""
        dockerfile_content = '''FROM nginx:alpine

COPY site/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
'''
        
        dockerfile_path = self.docs_portal_dir / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        try:
            # Build Docker image
            subprocess.run([
                'docker', 'build', '-t', 'anomaly-detection-docs', '.'
            ], check=True)
            
            print("✅ Docker image built successfully!")
            print("🐳 Run with: docker run -p 8080:80 anomaly-detection-docs")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Documentation Portal Setup')
    parser.add_argument('--initialize', action='store_true',
                       help='Initialize the documentation portal')
    parser.add_argument('--build', action='store_true',
                       help='Build the documentation')
    parser.add_argument('--serve', action='store_true',
                       help='Serve documentation locally')
    parser.add_argument('--deploy', action='store_true',
                       help='Deploy documentation to production')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host for development server')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for development server')
    parser.add_argument('--deploy-method', default='github-pages',
                       choices=['github-pages', 'docker'],
                       help='Deployment method')
    
    args = parser.parse_args()
    
    if not any([args.initialize, args.build, args.serve, args.deploy]):
        parser.print_help()
        return
    
    portal = DocumentationPortal()
    
    if args.initialize:
        portal.initialize_portal()
    
    if args.build:
        if not portal.build_documentation():
            sys.exit(1)
    
    if args.serve:
        portal.serve_documentation(args.host, args.port)
    
    if args.deploy:
        if not portal.deploy_documentation(args.deploy_method):
            sys.exit(1)


if __name__ == '__main__':
    main()