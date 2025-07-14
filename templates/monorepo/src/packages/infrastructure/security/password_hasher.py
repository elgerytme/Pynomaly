"""Password hashing implementation."""

from passlib.context import CryptContext


class BCryptPasswordHasher:
    """BCrypt password hasher implementation."""
    
    def __init__(self) -> None:
        self._context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using BCrypt."""
        return self._context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self._context.verify(plain_password, hashed_password)
    
    def needs_update(self, hashed_password: str) -> bool:
        """Check if password hash needs updating."""
        return self._context.needs_update(hashed_password)