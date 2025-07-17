"""
Email value object for authentication domain
"""
from dataclasses import dataclass
import re
from typing import Optional

@dataclass(frozen=True)
class Email:
    """Email value object with validation"""
    
    value: str
    
    def __post_init__(self):
        if not self._is_valid_email(self.value):
            raise ValueError(f"Invalid email format: {self.value}")
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @property
    def domain(self) -> str:
        """Get email domain"""
        return self.value.split('@')[1]
    
    @property
    def local_part(self) -> str:
        """Get email local part"""
        return self.value.split('@')[0]
    
    def is_corporate_email(self, corporate_domains: list[str]) -> bool:
        """Check if email is from corporate domain"""
        return self.domain in corporate_domains