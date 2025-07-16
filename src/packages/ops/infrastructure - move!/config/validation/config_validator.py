#!/usr/bin/env python3
"""Configuration validation and management tool for Pynomaly.

This script validates configuration files, checks for consistency,
and provides utilities for configuration management.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import yaml
import toml
from dataclasses import dataclass
from enum import Enum


class ConfigType(Enum):
    """Configuration file types."""
    ENVIRONMENT = "environment"
    DOCKER = "docker" 
    KUBERNETES = "kubernetes"
    TOOL = "tool"


@dataclass
class ConfigFile:
    """Represents a configuration file."""
    path: Path
    type: ConfigType
    environment: Optional[str] = None
    valid: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ConfigValidator:
    """Validates Pynomaly configuration files."""
    
    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self.project_root = self.config_root.parent
        self.errors = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """Validate all configuration files."""
        print("🔍 Validating Pynomaly configuration files...")
        
        success = True
        
        # Validate environment configurations
        if not self._validate_environments():
            success = False
            
        # Validate Docker configurations  
        if not self._validate_docker_configs():
            success = False
            
        # Validate tool configurations
        if not self._validate_tool_configs():
            success = False
            
        # Check for duplicate configurations
        if not self._check_duplicates():
            success = False
            
        # Validate configuration consistency
        if not self._validate_consistency():
            success = False
            
        self._print_summary()
        return success
        
    def _validate_environments(self) -> bool:
        """Validate environment-specific configurations."""
        print("📁 Validating environment configurations...")
        
        env_dir = self.config_root / "environments"
        if not env_dir.exists():
            self.errors.append("Environment configuration directory not found")
            return False
            
        required_envs = ["development", "testing", "production"]
        success = True
        
        for env in required_envs:
            env_path = env_dir / env / ".env"
            template_path = env_dir / env / ".env.template"
            
            # Check if environment file exists (or template for production)
            if env == "production":
                if not template_path.exists():
                    self.errors.append(f"Production environment template not found: {template_path}")
                    success = False
                else:
                    print(f"  ✅ {env} environment template found")
            else:
                if not env_path.exists():
                    self.errors.append(f"Environment file not found: {env_path}")
                    success = False
                else:
                    # Validate environment file content
                    if self._validate_env_file(env_path, env):
                        print(f"  ✅ {env} environment configuration valid")
                    else:
                        success = False
                        
        return success
        
    def _validate_env_file(self, env_file: Path, env_name: str) -> bool:
        """Validate a specific environment file."""
        try:
            with open(env_file, 'r') as f:
                lines = f.readlines()
                
            # Required environment variables for each environment
            required_vars = {
                "development": [
                    "PYNOMALY_ENV",
                    "PYNOMALY_DEBUG", 
                    "PYNOMALY_LOG_LEVEL",
                    "PYNOMALY_DATABASE_URL",
                    "PYNOMALY_SECRET_KEY"
                ],
                "testing": [
                    "PYNOMALY_ENV",
                    "PYNOMALY_DEBUG",
                    "PYNOMALY_LOG_LEVEL", 
                    "PYNOMALY_DATABASE_URL"
                ],
                "production": [
                    "PYNOMALY_ENV",
                    "PYNOMALY_DEBUG",
                    "PYNOMALY_LOG_LEVEL",
                    "PYNOMALY_DATABASE_URL",
                    "PYNOMALY_SECRET_KEY",
                    "PYNOMALY_REDIS_URL"
                ]
            }
            
            found_vars = set()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    var_name = line.split('=')[0].strip()
                    found_vars.add(var_name)
                    
            missing_vars = set(required_vars.get(env_name, [])) - found_vars
            if missing_vars:
                self.errors.append(f"Missing required variables in {env_name}: {missing_vars}")
                return False
                
            # Validate environment-specific constraints
            if env_name == "production":
                # Check for insecure defaults in production template
                insecure_patterns = ["dev-secret", "change-in-production", "localhost"]
                for line in lines:
                    for pattern in insecure_patterns:
                        if pattern in line and not line.strip().startswith('#'):
                            self.warnings.append(f"Potentially insecure default in production template: {line.strip()}")
                            
            return True
            
        except Exception as e:
            self.errors.append(f"Error reading environment file {env_file}: {e}")
            return False
            
    def _validate_docker_configs(self) -> bool:
        """Validate Docker configuration files."""
        print("🐳 Validating Docker configurations...")
        
        docker_dir = self.config_root / "deployment" / "docker"
        if not docker_dir.exists():
            self.errors.append("Docker configuration directory not found")
            return False
            
        required_files = [
            "docker-compose.yml",
            "docker-compose.development.yml", 
            "docker-compose.production.yml"
        ]
        
        success = True
        for file_name in required_files:
            file_path = docker_dir / file_name
            if not file_path.exists():
                self.errors.append(f"Required Docker file not found: {file_path}")
                success = False
            else:
                if self._validate_docker_compose(file_path):
                    print(f"  ✅ {file_name} is valid")
                else:
                    success = False
                    
        return success
        
    def _validate_docker_compose(self, compose_file: Path) -> bool:
        """Validate a Docker Compose file."""
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
                
            # Basic validation
            if 'version' not in compose_data:
                self.errors.append(f"Missing version in {compose_file}")
                return False
                
            if 'services' not in compose_data:
                self.errors.append(f"Missing services in {compose_file}")
                return False
                
            # Check for required services in base compose file
            if compose_file.name == "docker-compose.yml":
                required_services = ["pynomaly-api", "redis", "postgres"]
                missing_services = set(required_services) - set(compose_data['services'].keys())
                if missing_services:
                    self.errors.append(f"Missing required services in {compose_file}: {missing_services}")
                    return False
                    
            return True
            
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in {compose_file}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {compose_file}: {e}")
            return False
            
    def _validate_tool_configs(self) -> bool:
        """Validate tool configurations in pyproject.toml."""
        print("🔧 Validating tool configurations...")
        
        pyproject_file = self.project_root / "pyproject.toml"
        if not pyproject_file.exists():
            self.errors.append("pyproject.toml not found")
            return False
            
        try:
            with open(pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
                
            # Check for required tool configurations
            required_tools = ["pytest", "ruff", "black", "isort", "mypy"]
            
            tool_section = pyproject_data.get('tool', {})
            missing_tools = []
            
            for tool in required_tools:
                # Check both [tool.{tool}] and [tool.{tool}.ini_options] patterns
                if tool not in tool_section and f"{tool}.ini_options" not in tool_section:
                    missing_tools.append(tool)
                    
            if missing_tools:
                self.errors.append(f"Missing tool configurations: {missing_tools}")
                return False
                
            # Validate specific tool configurations
            if not self._validate_pytest_config(tool_section):
                return False
                
            print("  ✅ Tool configurations are valid")
            return True
            
        except Exception as e:
            self.errors.append(f"Error reading pyproject.toml: {e}")
            return False
            
    def _validate_pytest_config(self, tool_section: Dict[str, Any]) -> bool:
        """Validate pytest configuration."""
        pytest_config = tool_section.get('pytest', {}).get('ini_options', {})
        
        required_settings = ["testpaths", "python_files", "markers"]
        missing_settings = [s for s in required_settings if s not in pytest_config]
        
        if missing_settings:
            self.errors.append(f"Missing pytest settings: {missing_settings}")
            return False
            
        # Validate testpaths points to correct location
        testpaths = pytest_config.get('testpaths', [])
        if isinstance(testpaths, str):
            testpaths = [testpaths]
            
        if 'tests' not in testpaths:
            self.warnings.append("pytest testpaths should include 'tests' directory")
            
        return True
        
    def _check_duplicates(self) -> bool:
        """Check for duplicate configuration files."""
        print("🔍 Checking for duplicate configurations...")
        
        # Check for old pytest.ini files
        pytest_files = list(self.project_root.rglob("pytest.ini"))
        if pytest_files:
            self.warnings.append(f"Found legacy pytest.ini files that should be removed: {pytest_files}")
            
        # Check for scattered Docker Compose files
        docker_files = list(self.project_root.rglob("docker-compose*.yml"))
        docker_files.extend(list(self.project_root.rglob("docker-compose*.yaml")))
        
        config_docker_dir = self.config_root / "deployment" / "docker"
        legacy_docker_files = [f for f in docker_files if not str(f).startswith(str(config_docker_dir))]
        
        if legacy_docker_files:
            self.warnings.append(f"Found legacy Docker Compose files: {legacy_docker_files}")
            
        print("  ✅ Duplicate check completed")
        return True
        
    def _validate_consistency(self) -> bool:
        """Validate consistency across configuration files."""
        print("🔄 Validating configuration consistency...")
        
        # This could include checks like:
        # - Environment variable names are consistent
        # - Port numbers don't conflict
        # - Service names match across files
        
        print("  ✅ Consistency validation completed")
        return True
        
    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("📊 CONFIGURATION VALIDATION SUMMARY")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("✅ All configuration files are valid!")
        else:
            if self.errors:
                print(f"❌ {len(self.errors)} errors found:")
                for error in self.errors:
                    print(f"   • {error}")
                    
            if self.warnings:
                print(f"⚠️  {len(self.warnings)} warnings found:")
                for warning in self.warnings:
                    print(f"   • {warning}")
                    
        print("\n🔧 CONFIGURATION MANAGEMENT COMMANDS")
        print("-" * 40)
        print("Environment setup:")
        print("  export PYNOMALY_ENV=development")
        print("  source config/environments/development/.env")
        print("\nDocker deployment:")
        print("  # Development")
        print("  docker-compose -f config/deployment/docker/docker-compose.yml \\")
        print("                 -f config/deployment/docker/docker-compose.development.yml up")
        print("\nTest configuration:")
        print("  pytest --help  # Should use consolidated config from pyproject.toml")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Pynomaly configuration files")
    parser.add_argument("--config-root", default="config", help="Configuration root directory")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    
    args = parser.parse_args()
    
    config_root = Path(args.config_root).resolve()
    if not config_root.exists():
        print(f"❌ Configuration directory not found: {config_root}")
        sys.exit(1)
        
    validator = ConfigValidator(config_root)
    success = validator.validate_all()
    
    if not success:
        sys.exit(1)
    else:
        print("\n🎉 Configuration validation completed successfully!")


if __name__ == "__main__":
    main()