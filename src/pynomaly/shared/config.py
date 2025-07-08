"""Basic configuration module."""
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    """Basic configuration class."""
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.settings.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.settings[key] = value
