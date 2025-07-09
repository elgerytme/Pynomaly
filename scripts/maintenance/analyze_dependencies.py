#!/usr/bin/env python3
"""
Dependency Analysis Script for Pynomaly
Analyzes dependency complexity and provides optimization recommendations.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import subprocess
import sys


def find_requirement_files() -> List[Path]:
    """Find all requirement files in the project."""
    project_root = Path(".")
    requirement_files = []
    
    patterns = [
        "requirements*.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "poetry.lock"
    ]
    
    for pattern in patterns:
        requirement_files.extend(project_root.glob(pattern))
        requirement_files.extend(project_root.glob(f"**/{pattern}"))
    
    # Filter out virtual environments and build directories
    filtered_files = []
    exclude_patterns = [
        "venv", ".venv", "env", ".env", "virtualenv",
        "build", "dist", "site-packages", "node_modules",
        ".tox", ".pytest_cache", "__pycache__"
    ]
    
    for file in requirement_files:
        if not any(pattern in str(file) for pattern in exclude_patterns):
            filtered_files.append(file)
    
    return filtered_files


def parse_requirements_txt(file_path: Path) -> Dict[str, str]:
    """Parse requirements.txt file and extract dependencies."""
    dependencies = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Extract package name and version
                    if '==' in line:
                        name, version = line.split('==', 1)
                        dependencies[name.strip()] = version.strip()
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        dependencies[name.strip()] = f">={version.strip()}"
                    elif '<=' in line:
                        name, version = line.split('<=', 1)
                        dependencies[name.strip()] = f"<={version.strip()}"
                    elif '~=' in line:
                        name, version = line.split('~=', 1)
                        dependencies[name.strip()] = f"~={version.strip()}"
                    elif line.count('==') == 0 and line.count('>=') == 0:
                        # Plain package name
                        dependencies[line.strip()] = "*"
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return dependencies


def parse_pyproject_toml(file_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse pyproject.toml file and extract dependencies."""
    dependencies = {
        "main": {},
        "optional": {},
        "dev": {}
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract main dependencies
        main_deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if main_deps_match:
            deps_str = main_deps_match.group(1)
            for line in deps_str.split('\n'):
                line = line.strip().strip('"').strip("'").strip(',')
                if line and not line.startswith('#'):
                    if '>=' in line:
                        name, version = line.split('>=', 1)
                        dependencies["main"][name.strip()] = f">={version.strip().rstrip(',')}"
                    elif '==' in line:
                        name, version = line.split('==', 1)
                        dependencies["main"][name.strip()] = version.strip().rstrip(',')
                    else:
                        dependencies["main"][line.strip()] = "*"
        
        # Extract optional dependencies
        optional_deps_match = re.search(r'\[project\.optional-dependencies\](.*?)(?=\n\[|\Z)', content, re.DOTALL)
        if optional_deps_match:
            optional_content = optional_deps_match.group(1)
            current_group = None
            
            for line in optional_content.split('\n'):
                line = line.strip()
                if line.endswith('= ['):
                    current_group = line.split('=')[0].strip()
                    dependencies["optional"][current_group] = {}
                elif current_group and line and not line.startswith('#'):
                    line = line.strip('"').strip("'").strip(',')
                    if '>=' in line:
                        name, version = line.split('>=', 1)
                        dependencies["optional"][current_group][name.strip()] = f">={version.strip().rstrip(',')}"
                    elif '==' in line:
                        name, version = line.split('==', 1)
                        dependencies["optional"][current_group][name.strip()] = version.strip().rstrip(',')
                    elif line and not line in [']', '[']:
                        dependencies["optional"][current_group][line.strip()] = "*"
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return dependencies


def analyze_dependency_overlap(files_data: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Analyze overlapping dependencies across files."""
    package_to_files = defaultdict(set)
    
    for file_path, data in files_data.items():
        if isinstance(data, dict):
            if "main" in data:
                for package in data["main"].keys():
                    package_to_files[package].add(file_path)
            else:
                for package in data.keys():
                    package_to_files[package].add(file_path)
    
    overlapping_packages = {}
    for package, files in package_to_files.items():
        if len(files) > 1:
            overlapping_packages[package] = list(files)
    
    return overlapping_packages


def identify_heavy_dependencies() -> List[str]:
    """Identify dependencies that are typically large or heavy."""
    heavy_deps = [
        "tensorflow", "torch", "pytorch", "transformers", "datasets",
        "jupyter", "jupyterlab", "notebook", "scipy", "numpy",
        "pandas", "matplotlib", "seaborn", "plotly", "bokeh",
        "scikit-learn", "xgboost", "lightgbm", "catboost",
        "dask", "ray", "spark", "pyspark", "opencv-python",
        "pillow", "spacy", "nltk", "gensim", "keras",
        "streamlit", "dash", "voila", "airflow", "prefect",
        "mlflow", "wandb", "tensorboard", "boto3", "azure-storage-blob",
        "google-cloud-storage", "kubernetes", "docker"
    ]
    return heavy_deps


def analyze_version_conflicts(files_data: Dict[str, Dict]) -> List[Dict]:
    """Analyze version conflicts between different requirement files."""
    conflicts = []
    package_versions = defaultdict(dict)
    
    for file_path, data in files_data.items():
        if isinstance(data, dict):
            if "main" in data:
                for package, version in data["main"].items():
                    package_versions[package][file_path] = version
            else:
                for package, version in data.items():
                    package_versions[package][file_path] = version
    
    for package, file_versions in package_versions.items():
        if len(file_versions) > 1:
            unique_versions = set(file_versions.values())
            if len(unique_versions) > 1:
                conflicts.append({
                    "package": package,
                    "versions": dict(file_versions),
                    "conflict_count": len(unique_versions)
                })
    
    return conflicts


def generate_optimization_recommendations(
    files_data: Dict[str, Dict],
    overlapping_packages: Dict[str, List[str]],
    version_conflicts: List[Dict],
    heavy_deps: List[str]
) -> List[str]:
    """Generate optimization recommendations."""
    recommendations = []
    
    # Recommendation 1: Consolidate overlapping packages
    if overlapping_packages:
        recommendations.append(
            f"🔄 **Consolidate Dependencies**: {len(overlapping_packages)} packages "
            f"are duplicated across multiple files. Consider consolidating into "
            f"pyproject.toml optional dependencies."
        )
    
    # Recommendation 2: Address version conflicts
    if version_conflicts:
        recommendations.append(
            f"⚠️ **Resolve Version Conflicts**: {len(version_conflicts)} packages "
            f"have conflicting versions across files. This can cause installation issues."
        )
    
    # Recommendation 3: Optimize heavy dependencies
    total_deps = sum(len(data.get("main", data)) for data in files_data.values() if isinstance(data, dict))
    heavy_found = []
    for file_path, data in files_data.items():
        if isinstance(data, dict):
            deps = data.get("main", data)
            heavy_found.extend([dep for dep in deps.keys() if dep in heavy_deps])
    
    if heavy_found:
        recommendations.append(
            f"🔧 **Optimize Heavy Dependencies**: {len(set(heavy_found))} heavy dependencies "
            f"found. Consider making them optional or grouping them appropriately."
        )
    
    # Recommendation 4: Reduce total dependency count
    if total_deps > 100:
        recommendations.append(
            f"📉 **Reduce Dependency Count**: {total_deps} total dependencies found. "
            f"Consider removing unused dependencies and using lighter alternatives."
        )
    
    # Recommendation 5: Use dependency groups
    req_txt_files = [f for f in files_data.keys() if f.endswith('.txt')]
    if len(req_txt_files) > 3:
        recommendations.append(
            f"📋 **Simplify File Structure**: {len(req_txt_files)} requirements.txt files "
            f"found. Consider consolidating into pyproject.toml with optional dependencies."
        )
    
    return recommendations


def create_consolidated_pyproject() -> str:
    """Create a consolidated pyproject.toml structure."""
    return """
# Consolidated pyproject.toml structure recommendation

[project]
name = "pynomaly"
dependencies = [
    # Core dependencies only
    "pyod>=2.0.5",
    "numpy>=1.26.0,<2.2.0",
    "pandas>=2.2.3",
    "polars>=1.19.0",
    "pydantic>=2.10.4",
    "structlog>=24.4.0",
    "dependency-injector>=4.42.0",
    "networkx>=3.0",
]

[project.optional-dependencies]
# Lightweight ML
minimal = ["scikit-learn>=1.6.0", "scipy>=1.15.0"]

# Standard ML with common tools
standard = ["scikit-learn>=1.6.0", "scipy>=1.15.0", "pyarrow>=18.1.0"]

# API and server
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "httpx>=0.28.1",
    "python-multipart>=0.0.20",
    "pydantic-settings>=2.8.0"
]

# CLI tools
cli = ["typer[all]>=0.15.1", "rich>=13.9.4"]

# Authentication
auth = ["pyjwt>=2.10.1", "passlib[bcrypt]>=1.7.4"]

# Database
database = ["sqlalchemy>=2.0.36", "psycopg2-binary>=2.9.10", "alembic>=1.13.0"]

# Caching
caching = ["redis>=5.2.1"]

# Monitoring
monitoring = [
    "prometheus-client>=0.21.1",
    "opentelemetry-api>=1.29.0",
    "opentelemetry-sdk>=1.29.0",
    "psutil>=6.1.1"
]

# Production (combines essential production dependencies)
production = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "redis>=5.2.1",
    "sqlalchemy>=2.0.36",
    "psycopg2-binary>=2.9.10",
    "prometheus-client>=0.21.1",
    "pyjwt>=2.10.1",
    "passlib[bcrypt]>=1.7.4",
    "pydantic-settings>=2.8.0"
]

# Development
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "black>=24.0.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0"
]

# Heavy ML dependencies (optional)
ml-heavy = [
    "torch>=2.5.1",
    "tensorflow>=2.18.0,<2.20.0",
    "transformers>=4.35.0"
]

# Analytics and visualization
analytics = [
    "plotly>=5.17.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "jupyter>=1.0.0",
    "streamlit>=1.28.0"
]

# All dependencies (use sparingly)
all = [
    "pynomaly[production,dev,analytics,ml-heavy]"
]
"""


def main():
    """Main function to analyze dependencies."""
    print("🔍 Analyzing Pynomaly dependency structure...")
    
    # Find all requirement files
    requirement_files = find_requirement_files()
    print(f"📁 Found {len(requirement_files)} dependency files:")
    for file in requirement_files:
        print(f"  - {file}")
    
    # Parse all dependency files
    files_data = {}
    for file in requirement_files:
        if file.name.endswith('.txt'):
            files_data[str(file)] = parse_requirements_txt(file)
        elif file.name == 'pyproject.toml':
            files_data[str(file)] = parse_pyproject_toml(file)
    
    # Analyze overlapping dependencies
    overlapping_packages = analyze_dependency_overlap(files_data)
    
    # Identify heavy dependencies
    heavy_deps = identify_heavy_dependencies()
    
    # Analyze version conflicts
    version_conflicts = analyze_version_conflicts(files_data)
    
    # Generate recommendations
    recommendations = generate_optimization_recommendations(
        files_data, overlapping_packages, version_conflicts, heavy_deps
    )
    
    # Create analysis report
    report = {
        "analysis_date": "2025-07-09",
        "total_files": len(requirement_files),
        "total_unique_packages": len(set(
            pkg for data in files_data.values() 
            if isinstance(data, dict)
            for pkg in (data.get("main", data).keys() if "main" in data else data.keys())
        )),
        "overlapping_packages": len(overlapping_packages),
        "version_conflicts": len(version_conflicts),
        "heavy_dependencies_found": len([
            pkg for data in files_data.values() 
            if isinstance(data, dict)
            for pkg in (data.get("main", data).keys() if "main" in data else data.keys())
            if pkg in heavy_deps
        ]),
        "recommendations": recommendations,
        "files_analyzed": list(files_data.keys()),
        "top_overlapping_packages": dict(list(overlapping_packages.items())[:10]),
        "version_conflicts_detail": version_conflicts[:5],
        "consolidated_structure": create_consolidated_pyproject()
    }
    
    # Save analysis report
    os.makedirs("reports", exist_ok=True)
    with open("reports/dependency_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    markdown_report = f"""# Dependency Analysis Report

**Analysis Date:** {report['analysis_date']}

## Summary

- **Total Files:** {report['total_files']}
- **Total Unique Packages:** {report['total_unique_packages']}
- **Overlapping Packages:** {report['overlapping_packages']}
- **Version Conflicts:** {report['version_conflicts']}
- **Heavy Dependencies:** {report['heavy_dependencies_found']}

## Key Findings

"""
    
    for i, rec in enumerate(recommendations, 1):
        markdown_report += f"{i}. {rec}\n"
    
    markdown_report += f"""

## Files Analyzed

"""
    
    for file in report['files_analyzed']:
        markdown_report += f"- `{file}`\n"
    
    markdown_report += f"""

## Top Overlapping Packages

"""
    
    for pkg, files in report['top_overlapping_packages'].items():
        markdown_report += f"- **{pkg}**: {len(files)} files ({', '.join(files)})\n"
    
    markdown_report += f"""

## Version Conflicts

"""
    
    for conflict in report['version_conflicts_detail']:
        markdown_report += f"- **{conflict['package']}**: {conflict['conflict_count']} different versions\n"
        for file, version in conflict['versions'].items():
            markdown_report += f"  - `{file}`: {version}\n"
    
    markdown_report += f"""

## Recommendations

### 1. Consolidate to pyproject.toml

Replace multiple requirements.txt files with a single pyproject.toml with well-organized optional dependencies.

### 2. Remove Redundant Files

The following files can be removed after consolidation:
- `requirements-prod.txt`
- `requirements-analytics.txt`
- `requirements-mlops.txt`
- `requirements-enterprise.txt`

### 3. Optimize Dependencies

- Group related dependencies together
- Make heavy dependencies optional
- Use version ranges instead of exact pins where appropriate
- Remove unused dependencies

### 4. Simplify Installation

Provide clear installation instructions for different use cases:
```bash
# Minimal installation
pip install pynomaly[minimal]

# API server
pip install pynomaly[api,production]

# Development
pip install pynomaly[dev]

# Full analytics
pip install pynomaly[analytics,ml-heavy]
```

{report['consolidated_structure']}

## Next Steps

1. **Audit Dependencies**: Review all dependencies for necessity
2. **Consolidate Files**: Move to single pyproject.toml
3. **Test Installation**: Verify all combinations work
4. **Update Documentation**: Update installation instructions
5. **CI/CD Updates**: Update workflows to use new structure
"""
    
    with open("reports/dependency_analysis_report.md", "w") as f:
        f.write(markdown_report)
    
    # Print summary
    print(f"\n📊 **Dependency Analysis Complete**")
    print(f"  - Total files analyzed: {report['total_files']}")
    print(f"  - Total unique packages: {report['total_unique_packages']}")
    print(f"  - Overlapping packages: {report['overlapping_packages']}")
    print(f"  - Version conflicts: {report['version_conflicts']}")
    print(f"  - Heavy dependencies: {report['heavy_dependencies_found']}")
    print(f"\n📋 **Key Recommendations:**")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n📄 **Reports Generated:**")
    print(f"  - JSON: reports/dependency_analysis_report.json")
    print(f"  - Markdown: reports/dependency_analysis_report.md")
    
    return report


if __name__ == "__main__":
    main()