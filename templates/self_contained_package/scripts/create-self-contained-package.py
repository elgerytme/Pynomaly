#!/usr/bin/env python3
"""
{package_name} - Self-Contained Package Generator
================================================
Intelligent package generator that customizes templates based on requirements
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import subprocess
import re


@dataclass
class PackageConfig:
    """Configuration for generating self-contained package"""
    package_name: str
    package_description: str
    author_name: str
    author_email: str
    github_username: str
    repository_name: str
    docker_registry: str = "docker.io"
    package_version: str = "0.1.0"
    python_version: str = "3.11"
    license: str = "MIT"
    
    # Feature flags
    use_database: bool = True
    database_type: str = "postgresql"
    use_cache: bool = True
    cache_type: str = "redis"
    use_monitoring: bool = True
    use_kubernetes: bool = True
    use_docker: bool = True
    use_async: bool = True
    use_web_framework: bool = True
    web_framework: str = "fastapi"
    
    # Optional components
    use_celery: bool = False
    use_message_queue: bool = False
    message_queue_type: str = "rabbitmq"
    use_elasticsearch: bool = False
    use_mlflow: bool = False
    use_jupyter: bool = False


class PackageGenerator:
    """Generates self-contained packages from templates"""
    
    def __init__(self, templates_dir: Path, output_dir: Path):
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        self.template_vars: Dict[str, Any] = {}
    
    def generate_package(self, config: PackageConfig) -> None:
        """Generate a complete self-contained package"""
        print(f"🚀 Generating self-contained package: {config.package_name}")
        
        # Prepare template variables
        self._prepare_template_vars(config)
        
        # Create output directory
        package_dir = self.output_dir / config.package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and process templates
        self._process_templates(package_dir, config)
        
        # Create package structure
        self._create_package_structure(package_dir, config)
        
        # Generate configuration files
        self._generate_config_files(package_dir, config)
        
        # Initialize git repository
        self._initialize_git_repo(package_dir, config)
        
        # Install dependencies (optional)
        if self._ask_yes_no("Install dependencies now?", default=True):
            self._install_dependencies(package_dir, config)
        
        print(f"✅ Package '{config.package_name}' generated successfully!")
        print(f"📁 Location: {package_dir.absolute()}")
        print(f"🔧 Next steps:")
        print(f"   cd {package_dir}")
        print(f"   make setup-dev")
        print(f"   make test")
    
    def _prepare_template_vars(self, config: PackageConfig) -> None:
        """Prepare template variables for substitution"""
        self.template_vars = asdict(config)
        
        # Add computed variables
        self.template_vars.update({
            'package_name_upper': config.package_name.upper(),
            'package_name_title': config.package_name.title(),
            'package_slug': re.sub(r'[^a-zA-Z0-9]', '-', config.package_name).lower(),
            'python_module': config.package_name.replace('-', '_'),
            'current_year': str(__import__('datetime').date.today().year),
        })
    
    def _process_templates(self, package_dir: Path, config: PackageConfig) -> None:
        """Process all template files"""
        print("📋 Processing templates...")
        
        template_files = self._find_template_files()
        
        for template_file in template_files:
            output_file = self._get_output_path(template_file, package_dir)
            
            # Skip files based on configuration
            if self._should_skip_file(template_file, config):
                continue
            
            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process template
            if template_file.suffix == '.template':
                self._process_template_file(template_file, output_file)
            else:
                shutil.copy2(template_file, output_file)
        
        print(f"   Processed {len(template_files)} template files")
    
    def _find_template_files(self) -> List[Path]:
        """Find all template files"""
        template_files = []
        for root, dirs, files in os.walk(self.templates_dir):
            for file in files:
                template_files.append(Path(root) / file)
        return template_files
    
    def _get_output_path(self, template_file: Path, package_dir: Path) -> Path:
        """Get output path for template file"""
        relative_path = template_file.relative_to(self.templates_dir)
        
        # Remove .template extension
        if relative_path.suffix == '.template':
            relative_path = relative_path.with_suffix('')
        
        # Replace template placeholders in path
        path_str = str(relative_path)
        for key, value in self.template_vars.items():
            path_str = path_str.replace(f'{{{key}}}', str(value))
        
        return package_dir / path_str
    
    def _should_skip_file(self, template_file: Path, config: PackageConfig) -> bool:
        """Determine if a file should be skipped based on configuration"""
        path_str = str(template_file)
        
        # Skip database files if not using database
        if not config.use_database and ('database' in path_str or 'postgres' in path_str):
            return True
        
        # Skip cache files if not using cache
        if not config.use_cache and ('cache' in path_str or 'redis' in path_str):
            return True
        
        # Skip monitoring files if not using monitoring
        if not config.use_monitoring and 'monitoring' in path_str:
            return True
        
        # Skip Kubernetes files if not using Kubernetes
        if not config.use_kubernetes and ('k8s' in path_str or 'kubernetes' in path_str):
            return True
        
        # Skip Docker files if not using Docker
        if not config.use_docker and ('docker' in path_str or 'Dockerfile' in path_str):
            return True
        
        # Skip Celery files if not using Celery
        if not config.use_celery and 'celery' in path_str:
            return True
        
        # Skip message queue files if not using message queue
        if not config.use_message_queue and ('rabbitmq' in path_str or 'queue' in path_str):
            return True
        
        return False
    
    def _process_template_file(self, template_file: Path, output_file: Path) -> None:
        """Process a single template file"""
        try:
            content = template_file.read_text(encoding='utf-8')
            
            # Replace template variables
            for key, value in self.template_vars.items():
                content = content.replace(f'{{{key}}}', str(value))
            
            output_file.write_text(content, encoding='utf-8')
            
            # Preserve file permissions
            if template_file.suffix == '.sh' or template_file.name.startswith('health-check'):
                output_file.chmod(0o755)
                
        except Exception as e:
            print(f"❌ Error processing {template_file}: {e}")
    
    def _create_package_structure(self, package_dir: Path, config: PackageConfig) -> None:
        """Create basic package structure"""
        print("📁 Creating package structure...")
        
        # Core directories
        directories = [
            f"src/{config.package_name}",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "tests/performance",
            "tests/security",
            "docs",
            "config",
            "scripts",
            "data",
            "logs",
            "reports",
        ]
        
        # Optional directories based on configuration
        if config.use_monitoring:
            directories.extend([
                "monitoring/prometheus/rules",
                "monitoring/grafana/dashboards",
                "monitoring/grafana/provisioning",
            ])
        
        if config.use_kubernetes:
            directories.extend([
                "k8s/staging",
                "k8s/production",
            ])
        
        if config.use_docker:
            directories.append("docker")
        
        # Create directories
        for directory in directories:
            (package_dir / directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_files = [
            f"src/{config.package_name}/__init__.py",
            "tests/__init__.py",
            "tests/unit/__init__.py",
            "tests/integration/__init__.py",
        ]
        
        for init_file in init_files:
            init_path = package_dir / init_file
            if not init_path.exists():
                init_path.write_text('"""Package initialization."""\n')
    
    def _generate_config_files(self, package_dir: Path, config: PackageConfig) -> None:
        """Generate configuration files"""
        print("⚙️ Generating configuration files...")
        
        # Create .env file
        env_file = package_dir / ".env"
        env_content = self._generate_env_content(config)
        env_file.write_text(env_content)
        
        # Create .gitignore
        gitignore_file = package_dir / ".gitignore"
        gitignore_content = self._generate_gitignore_content()
        gitignore_file.write_text(gitignore_content)
        
        # Create LICENSE file
        license_file = package_dir / "LICENSE"
        license_content = self._generate_license_content(config)
        license_file.write_text(license_content)
    
    def _generate_env_content(self, config: PackageConfig) -> str:
        """Generate .env file content"""
        content = [
            "# Development environment variables",
            f"PACKAGE_NAME={config.package_name}",
            f"PACKAGE_VERSION={config.package_version}",
            "ENVIRONMENT=development",
            "DEBUG=true",
            "LOG_LEVEL=DEBUG",
            "",
        ]
        
        if config.use_database:
            content.extend([
                "# Database configuration",
                f"DATABASE_URL={config.database_type}://user:password@localhost:5432/{config.package_name}_dev",
                "",
            ])
        
        if config.use_cache:
            content.extend([
                "# Cache configuration",
                f"REDIS_URL=redis://localhost:6379/0",
                "",
            ])
        
        content.extend([
            "# Security",
            "SECRET_KEY=dev-secret-key-change-in-production",
            "JWT_SECRET=dev-jwt-secret-change-in-production",
        ])
        
        return '\n'.join(content)
    
    def _generate_gitignore_content(self) -> str:
        """Generate .gitignore file content"""
        return """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/
reports/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Local data
data/
logs/
*.log

# Docker
.dockerignore
        """.strip()
    
    def _generate_license_content(self, config: PackageConfig) -> str:
        """Generate LICENSE file content"""
        if config.license.upper() == "MIT":
            year = __import__('datetime').date.today().year
            return f"""MIT License

Copyright (c) {year} {config.author_name}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        return "# License\n\nPlease specify your license terms."
    
    def _initialize_git_repo(self, package_dir: Path, config: PackageConfig) -> None:
        """Initialize git repository"""
        if self._ask_yes_no("Initialize git repository?", default=True):
            print("🔧 Initializing git repository...")
            try:
                subprocess.run(['git', 'init'], cwd=package_dir, check=True, capture_output=True)
                subprocess.run(['git', 'add', '.'], cwd=package_dir, check=True, capture_output=True)
                subprocess.run(['git', 'commit', '-m', f'Initial commit for {config.package_name}'], 
                             cwd=package_dir, check=True, capture_output=True)
                print("   ✅ Git repository initialized")
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️ Could not initialize git repository: {e}")
    
    def _install_dependencies(self, package_dir: Path, config: PackageConfig) -> None:
        """Install package dependencies"""
        print("📦 Installing dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[dev]'], 
                         cwd=package_dir, check=True, capture_output=True)
            print("   ✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Could not install dependencies: {e}")
    
    def _ask_yes_no(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question"""
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{question} [{default_str}]: ").strip().lower()
            if not response:
                return default
            if response in ('y', 'yes'):
                return True
            if response in ('n', 'no'):
                return False
            print("Please enter 'y' or 'n'")


def create_interactive_config() -> PackageConfig:
    """Create configuration interactively"""
    print("🎯 Creating self-contained package configuration")
    print("=" * 50)
    
    # Required fields
    package_name = input("Package name: ").strip()
    while not package_name or not re.match(r'^[a-z][a-z0-9\-_]*$', package_name):
        print("Package name must start with a letter and contain only lowercase letters, numbers, hyphens, and underscores")
        package_name = input("Package name: ").strip()
    
    package_description = input("Package description: ").strip()
    author_name = input("Author name: ").strip()
    author_email = input("Author email: ").strip()
    github_username = input("GitHub username: ").strip()
    repository_name = input("Repository name (or press Enter for package name): ").strip()
    if not repository_name:
        repository_name = package_name
    
    # Optional fields with defaults
    print("\n🔧 Optional configuration (press Enter for defaults)")
    docker_registry = input("Docker registry [docker.io]: ").strip() or "docker.io"
    
    # Feature selection
    print("\n🎛️ Feature selection")
    use_database = input("Use database? [Y/n]: ").strip().lower() not in ('n', 'no')
    database_type = "postgresql"
    if use_database:
        database_type = input("Database type [postgresql]: ").strip() or "postgresql"
    
    use_cache = input("Use cache (Redis)? [Y/n]: ").strip().lower() not in ('n', 'no')
    use_monitoring = input("Use monitoring (Prometheus/Grafana)? [Y/n]: ").strip().lower() not in ('n', 'no')
    use_kubernetes = input("Use Kubernetes deployment? [Y/n]: ").strip().lower() not in ('n', 'no')
    use_docker = input("Use Docker containerization? [Y/n]: ").strip().lower() not in ('n', 'no')
    
    # Advanced features
    print("\n🚀 Advanced features (optional)")
    use_celery = input("Use Celery for background tasks? [y/N]: ").strip().lower() in ('y', 'yes')
    use_message_queue = input("Use message queue? [y/N]: ").strip().lower() in ('y', 'yes')
    use_elasticsearch = input("Use Elasticsearch? [y/N]: ").strip().lower() in ('y', 'yes')
    
    return PackageConfig(
        package_name=package_name,
        package_description=package_description,
        author_name=author_name,
        author_email=author_email,
        github_username=github_username,
        repository_name=repository_name,
        docker_registry=docker_registry,
        use_database=use_database,
        database_type=database_type,
        use_cache=use_cache,
        use_monitoring=use_monitoring,
        use_kubernetes=use_kubernetes,
        use_docker=use_docker,
        use_celery=use_celery,
        use_message_queue=use_message_queue,
        use_elasticsearch=use_elasticsearch,
    )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate self-contained Python packages")
    parser.add_argument('--config', type=str, help="Configuration file (JSON)")
    parser.add_argument('--templates', type=str, default="templates/self_contained_package", 
                       help="Templates directory")
    parser.add_argument('--output', type=str, default=".", help="Output directory")
    parser.add_argument('--interactive', action='store_true', help="Interactive configuration")
    
    args = parser.parse_args()
    
    # Determine script location
    script_dir = Path(__file__).parent.parent
    templates_dir = script_dir / args.templates
    output_dir = Path(args.output)
    
    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        return 1
    
    # Get configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = PackageConfig(**config_data)
    elif args.interactive:
        config = create_interactive_config()
    else:
        print("❌ Must provide either --config or --interactive")
        return 1
    
    # Generate package
    generator = PackageGenerator(templates_dir, output_dir)
    generator.generate_package(config)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())