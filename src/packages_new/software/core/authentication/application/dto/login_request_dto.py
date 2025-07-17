"""
Login request DTO
"""
from dataclasses import dataclass

@dataclass
class LoginRequestDto:
    """Data transfer object for login request"""
    
    identifier: str  # Email or username
    password: str
    remember_me: bool = False
    
    def __post_init__(self):
        if not self.identifier:
            raise ValueError("Identifier is required")
        
        if not self.password:
            raise ValueError("Password is required")