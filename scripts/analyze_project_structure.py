#!/usr/bin/env python3
"""Analyze project structure and identify stray files."""

import os
import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

def analyze_project_structure() -> Dict:
    """Analyze the project structure and categorize files."""
    
    project_root = Path.cwd()
    
    # Define what should be in the root directory
    ALLOWED_ROOT_FILES = {
        # Essential project files
        'README.md', 'LICENSE', 'CHANGELOG.md', 'TODO.md', 'CLAUDE.md',
        'CONTRIBUTING.md', 'MANIFEST.in', 'Makefile',
        
        # Python package configuration
        'pyproject.toml', 'setup.py', 'setup.cfg',
        
        # Requirements files
        'requirements.txt', 'requirements-minimal.txt', 
        'requirements-server.txt', 'requirements-production.txt', 'requirements-test.txt',
        
        # Node.js/Frontend
        'package.json', 'package-lock.json',
        
        # IDE/Editor
        'Pynomaly.code-workspace',
        
        # Git
        '.gitignore', '.gitattributes',
        
        # CI/CD
        '.pre-commit-config.yaml'
    }
    
    # Define allowed root directories
    ALLOWED_ROOT_DIRS = {
        'src', 'tests', 'docs', 'examples', 'scripts', 'deploy', 'config',
        'reports', 'storage', 'templates', 'analytics', 'screenshots',
        '.github', '.git', 'node_modules'
    }
    
    # Scan root directory
    root_items = list(project_root.iterdir())
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'project_root': str(project_root),
        'total_root_items': len(root_items),
        'stray_files': [],
        'stray_directories': [],
        'allowed_files': [],
        'allowed_directories': [],
        'problematic_items': [],
        'recommendations': []
    }
    
    for item in root_items:
        item_name = item.name
        
        # Skip hidden files/directories for now
        if item_name.startswith('.') and item_name not in {'.gitignore', '.gitattributes', '.github', '.git'}:
            continue
            
        if item.is_file():
            if item_name in ALLOWED_ROOT_FILES:
                analysis['allowed_files'].append({
                    'name': item_name,
                    'size': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            else:
                # Categorize stray files
                category = categorize_stray_file(item)
                analysis['stray_files'].append({
                    'name': item_name,
                    'category': category,
                    'size': item.stat().st_size,
                    'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'recommended_location': get_recommended_location(item_name, category)
                })
                
        elif item.is_dir():
            if item_name in ALLOWED_ROOT_DIRS:
                analysis['allowed_directories'].append({
                    'name': item_name,
                    'item_count': len(list(item.iterdir())) if item.exists() else 0
                })
            else:
                # Categorize stray directories
                category = categorize_stray_directory(item)
                analysis['stray_directories'].append({
                    'name': item_name,
                    'category': category,
                    'item_count': len(list(item.iterdir())) if item.exists() else 0,
                    'recommended_location': get_recommended_location(item_name, category)
                })
    
    # Generate recommendations
    analysis['recommendations'] = generate_recommendations(analysis)
    
    return analysis

def categorize_stray_file(file_path: Path) -> str:
    """Categorize a stray file based on its name and extension."""
    name = file_path.name.lower()
    
    # Testing files
    if any(x in name for x in ['test_', 'testing', '_test']):
        return 'testing'
    
    # Documentation
    if name.endswith(('.md', '.rst', '.txt')) and any(x in name for x in ['readme', 'doc', 'guide', 'manual']):
        return 'documentation'
    
    # Reports
    if any(x in name for x in ['report', 'summary', 'analysis']):
        return 'reports'
    
    # Scripts
    if name.endswith(('.py', '.sh', '.ps1', '.bat')) and not name.startswith('test_'):
        return 'scripts'
    
    # Temporary/cache files
    if any(x in name for x in ['temp', 'tmp', 'cache', 'backup']):
        return 'temporary'
    
    # Build artifacts
    if any(x in name for x in ['build', 'dist', '.egg-info']):
        return 'build_artifacts'
    
    # Configuration
    if name.endswith(('.json', '.yaml', '.yml', '.ini', '.toml', '.cfg')) and 'config' in name:
        return 'configuration'
    
    # Version numbers (like "2.0", "=0.2.0.1")
    if name.replace('.', '').replace('=', '').isdigit() or name.startswith('='):
        return 'version_artifacts'
    
    # PowerShell/Windows specific
    if name.endswith('.ps1'):
        return 'scripts'
    
    return 'miscellaneous'

def categorize_stray_directory(dir_path: Path) -> str:
    """Categorize a stray directory based on its name and contents."""
    name = dir_path.name.lower()
    
    # Testing directories
    if any(x in name for x in ['test_', 'testing', '_test']):
        return 'testing'
    
    # Temporary/environment directories
    if any(x in name for x in ['temp', 'tmp', 'env', 'venv', '.venv']):
        return 'temporary'
    
    # Specific testing environments
    if 'test_env' in name or 'test_venv' in name:
        return 'testing'
    
    # Virtual environments
    if name in ['venv', '.venv', 'env', '.env']:
        return 'environment'
    
    # Package installation artifacts
    if name == 'node_modules' or '.egg-info' in name:
        return 'build_artifacts'
    
    # Storage/runtime
    if any(x in name for x in ['storage', 'data', 'logs']):
        return 'runtime_data'
    
    return 'miscellaneous'

def get_recommended_location(item_name: str, category: str) -> str:
    """Get recommended location for a stray item."""
    
    location_map = {
        'testing': 'tests/',
        'documentation': 'docs/',
        'reports': 'reports/',
        'scripts': 'scripts/',
        'temporary': 'DELETE (temporary files)',
        'build_artifacts': 'DELETE (build artifacts)',
        'version_artifacts': 'DELETE (version artifacts)',
        'configuration': 'config/',
        'environment': 'DELETE (virtual environments)',
        'runtime_data': 'storage/',
        'miscellaneous': 'REVIEW (manual classification needed)'
    }
    
    return location_map.get(category, 'REVIEW')

def generate_recommendations(analysis: Dict) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # Count items by category
    file_categories = {}
    dir_categories = {}
    
    for file_info in analysis['stray_files']:
        cat = file_info['category']
        file_categories[cat] = file_categories.get(cat, 0) + 1
    
    for dir_info in analysis['stray_directories']:
        cat = dir_info['category']
        dir_categories[cat] = dir_categories.get(cat, 0) + 1
    
    # Generate specific recommendations
    if file_categories.get('testing', 0) > 0:
        recommendations.append(f"Move {file_categories['testing']} testing files to tests/ directory")
    
    if file_categories.get('scripts', 0) > 0:
        recommendations.append(f"Move {file_categories['scripts']} script files to scripts/ directory")
    
    if file_categories.get('reports', 0) > 0:
        recommendations.append(f"Move {file_categories['reports']} report files to reports/ directory")
    
    if file_categories.get('temporary', 0) > 0:
        recommendations.append(f"DELETE {file_categories['temporary']} temporary files")
    
    if file_categories.get('build_artifacts', 0) > 0:
        recommendations.append(f"DELETE {file_categories['build_artifacts']} build artifact files")
    
    if file_categories.get('version_artifacts', 0) > 0:
        recommendations.append(f"DELETE {file_categories['version_artifacts']} version artifact files")
    
    if dir_categories.get('testing', 0) > 0:
        recommendations.append(f"Move/consolidate {dir_categories['testing']} testing directories")
    
    if dir_categories.get('temporary', 0) > 0:
        recommendations.append(f"DELETE {dir_categories['temporary']} temporary directories")
    
    if dir_categories.get('environment', 0) > 0:
        recommendations.append(f"DELETE {dir_categories['environment']} virtual environment directories")
    
    # General recommendations
    total_stray = len(analysis['stray_files']) + len(analysis['stray_directories'])
    if total_stray > 10:
        recommendations.append("Consider implementing automated file organization rules")
        recommendations.append("Add comprehensive .gitignore patterns")
        recommendations.append("Create pre-commit hooks for file organization validation")
    
    return recommendations

def main():
    """Main function to run the analysis."""
    print("🔍 Analyzing Pynomaly project structure...")
    
    analysis = analyze_project_structure()
    
    # Save detailed analysis to file
    output_file = Path('reports/project_structure_analysis.json')
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print(f"\n📊 Project Structure Analysis Summary")
    print(f"=" * 50)
    print(f"Total root items: {analysis['total_root_items']}")
    print(f"Allowed files: {len(analysis['allowed_files'])}")
    print(f"Allowed directories: {len(analysis['allowed_directories'])}")
    print(f"Stray files: {len(analysis['stray_files'])}")
    print(f"Stray directories: {len(analysis['stray_directories'])}")
    
    if analysis['stray_files']:
        print(f"\n🗂️ Stray Files by Category:")
        categories = {}
        for file_info in analysis['stray_files']:
            cat = file_info['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count} files")
    
    if analysis['stray_directories']:
        print(f"\n📁 Stray Directories by Category:")
        categories = {}
        for dir_info in analysis['stray_directories']:
            cat = dir_info['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for category, count in sorted(categories.items()):
            print(f"  {category}: {count} directories")
    
    if analysis['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n📄 Detailed analysis saved to: {output_file}")
    
    return analysis

if __name__ == '__main__':
    main()