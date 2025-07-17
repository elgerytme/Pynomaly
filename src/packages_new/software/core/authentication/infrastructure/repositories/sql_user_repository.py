"""
SQL implementation of UserRepository
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ...domain.entities.user import User
from ...domain.repositories.user_repository import UserRepository
from ...domain.value_objects.email import Email
from .models.user_model import UserModel

class SqlUserRepository(UserRepository):
    """SQL database implementation of UserRepository"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save(self, user: User) -> User:
        """Save user to database"""
        try:
            # Check if user already exists
            user_model = self.session.query(UserModel).filter_by(id=user.id).first()
            
            if user_model is None:
                # Create new user
                user_model = UserModel()
                user_model.id = user.id
            
            # Update model from entity
            user_model.email = user.email
            user_model.username = user.username
            user_model.password_hash = user.password_hash
            user_model.created_at = user.created_at
            user_model.last_login = user.last_login
            user_model.is_active = user.is_active
            user_model.is_verified = user.is_verified
            user_model.failed_login_attempts = user.failed_login_attempts
            user_model.locked_until = user.locked_until
            
            self.session.add(user_model)
            self.session.commit()
            
            return user
            
        except IntegrityError:
            self.session.rollback()
            raise ValueError("User with this email or username already exists")
    
    def find_by_id(self, user_id: UUID) -> Optional[User]:
        """Find user by ID"""
        user_model = self.session.query(UserModel).filter_by(id=user_id).first()
        
        if user_model is None:
            return None
        
        return self._model_to_entity(user_model)
    
    def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email"""
        user_model = self.session.query(UserModel).filter_by(email=email.value).first()
        
        if user_model is None:
            return None
        
        return self._model_to_entity(user_model)
    
    def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        user_model = self.session.query(UserModel).filter_by(username=username).first()
        
        if user_model is None:
            return None
        
        return self._model_to_entity(user_model)
    
    def find_active_users(self) -> List[User]:
        """Find all active users"""
        user_models = self.session.query(UserModel).filter_by(is_active=True).all()
        
        return [self._model_to_entity(model) for model in user_models]
    
    def find_unverified_users(self) -> List[User]:
        """Find all unverified users"""
        user_models = self.session.query(UserModel).filter_by(is_verified=False).all()
        
        return [self._model_to_entity(model) for model in user_models]
    
    def delete(self, user_id: UUID) -> bool:
        """Delete a user"""
        try:
            user_model = self.session.query(UserModel).filter_by(id=user_id).first()
            
            if user_model is None:
                return False
            
            self.session.delete(user_model)
            self.session.commit()
            
            return True
            
        except Exception:
            self.session.rollback()
            return False
    
    def exists_by_email(self, email: Email) -> bool:
        """Check if user exists by email"""
        count = self.session.query(UserModel).filter_by(email=email.value).count()
        return count > 0
    
    def exists_by_username(self, username: str) -> bool:
        """Check if user exists by username"""
        count = self.session.query(UserModel).filter_by(username=username).count()
        return count > 0
    
    def _model_to_entity(self, model: UserModel) -> User:
        """Convert database model to domain entity"""
        return User(
            id=model.id,
            email=model.email,
            username=model.username,
            password_hash=model.password_hash,
            created_at=model.created_at,
            last_login=model.last_login,
            is_active=model.is_active,
            is_verified=model.is_verified,
            failed_login_attempts=model.failed_login_attempts,
            locked_until=model.locked_until
        )