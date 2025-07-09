"""
Strategies module for mutation testing.

This module provides helpers to map component names to source & test paths,
defaulting to configuration in advanced_testing_config.json.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class ComponentPathMapper:
    """Maps component names to source and test paths based on configuration."""
    
    def __init__(self, config_file: str = "advanced_testing_config.json"):
        """
        Initialize the mapper with configuration file.
        
        Args:
            config_file: Path to the configuration file
            
        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration file contains invalid JSON
        """
        self.config_file = config_file
        self.components: Dict[str, Dict[str, Any]] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.components = config.get("mutation_testing", {}).get("components", {})
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {self.config_file}")
    
    def get_component_paths(self, component_name: str) -> Tuple[List[str], List[str]]:
        """
        Get source and test paths for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Tuple of (source_paths, test_paths)
            
        Raises:
            KeyError: If component is not found in configuration
        """
        if component_name not in self.components:
            raise KeyError(f"Component '{component_name}' not found in configuration")
        
        component_config = self.components[component_name]
        source_paths = component_config.get("source_paths", [])
        test_paths = component_config.get("test_paths", [])
        
        return source_paths, test_paths
    
    def get_all_components(self) -> List[str]:
        """Get list of all available components."""
        return list(self.components.keys())
    
    def validate_paths(self, component_name: str) -> None:
        """
        Validate that paths for a component exist.
        
        Args:
            component_name: Name of the component
            
        Raises:
            FileNotFoundError: If any path does not exist
        """
        source_paths, test_paths = self.get_component_paths(component_name)
        
        all_paths = source_paths + test_paths
        for path_str in all_paths:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path_str}")


class MutationStrategy:
    """Creates mutation testing strategies for components."""
    
    def __init__(self, path_mapper: ComponentPathMapper = None):
        """
        Initialize the mutation strategy.
        
        Args:
            path_mapper: Optional ComponentPathMapper instance
        """
        self.path_mapper = path_mapper or ComponentPathMapper()
    
    def create_strategy_for_component(self, component_name: str) -> Dict[str, Any]:
        """
        Create a mutation strategy for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary containing strategy configuration
        """
        source_paths, test_paths = self.path_mapper.get_component_paths(component_name)
        
        strategy = {
            "component": component_name,
            "source_paths": source_paths,
            "test_paths": test_paths,
            "mutation_targets": self.get_mutation_targets(component_name)
        }
        
        return strategy
    
    def create_strategy_for_all_components(self) -> List[Dict[str, Any]]:
        """
        Create mutation strategies for all available components.
        
        Returns:
            List of strategy configurations
        """
        strategies = []
        for component_name in self.path_mapper.get_all_components():
            strategy = self.create_strategy_for_component(component_name)
            strategies.append(strategy)
        
        return strategies
    
    def get_mutation_targets(self, component_name: str) -> List[str]:
        """
        Get list of Python files that should be mutated for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            List of Python file paths to mutate
        """
        source_paths, _ = self.path_mapper.get_component_paths(component_name)
        
        targets = []
        for source_path in source_paths:
            path = Path(source_path)
            if path.exists():
                # Find all Python files in the source path
                python_files = list(path.rglob("*.py"))
                targets.extend([str(f) for f in python_files])
        
        return targets
