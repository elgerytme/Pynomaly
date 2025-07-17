"""
Processing Repository Interface

Defines the contract for persisting and retrieving processing-related data.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.pattern_analysis_request import PatternAnalysisRequest


class PatternAnalysisRepository(ABC):
    """
    Abstract repository for processing-related data persistence.
    
    This interface defines the contract that infrastructure adapters
    must implement to provide data persistence capabilities for
    processing requests and results.
    """
    
    @abstractmethod
    async def save_request(self, request: PatternAnalysisRequest) -> None:
        """
        Save a processing request to persistent storage.
        
        Args:
            request: The processing request to save.
            
        Raises:
            RepositoryError: If the save operation fails.
        """
        pass
    
    @abstractmethod
    async def get_request_by_id(self, request_id: UUID) -> Optional[PatternAnalysisRequest]:
        """
        Retrieve a processing request by its unique identifier.
        
        Args:
            request_id: The unique identifier of the request.
            
        Returns:
            Optional[PatternAnalysisRequest]: The request if found, None otherwise.
            
        Raises:
            RepositoryError: If the retrieval operation fails.
        """
        pass
    
    @abstractmethod
    async def get_requests_by_user(self, user_id: str) -> List[PatternAnalysisRequest]:
        """
        Retrieve all processing requests for a specific user.
        
        Args:
            user_id: The user identifier.
            
        Returns:
            List[PatternAnalysisRequest]: List of requests for the user.
            
        Raises:
            RepositoryError: If the retrieval operation fails.
        """
        pass
    
    @abstractmethod
    async def update_request_status(self, request_id: UUID, status: str) -> None:
        """
        Update the status of a processing request.
        
        Args:
            request_id: The unique identifier of the request.
            status: The new status to set.
            
        Raises:
            RepositoryError: If the update operation fails.
        """
        pass
    
    @abstractmethod
    async def delete_request(self, request_id: UUID) -> bool:
        """
        Delete a processing request from storage.
        
        Args:
            request_id: The unique identifier of the request.
            
        Returns:
            bool: True if the request was deleted, False if not found.
            
        Raises:
            RepositoryError: If the delete operation fails.
        """
        pass
    
    @abstractmethod
    async def list_requests(
        self, 
        limit: int = 100, 
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[PatternAnalysisRequest]:
        """
        List processing requests with pagination and filtering.
        
        Args:
            limit: Maximum number of requests to return.
            offset: Number of requests to skip.
            status_filter: Optional status filter.
            
        Returns:
            List[PatternAnalysisRequest]: List of processing requests.
            
        Raises:
            RepositoryError: If the list operation fails.
        """
        pass
    
    @abstractmethod
    async def count_requests(self, status_filter: Optional[str] = None) -> int:
        """
        Count the total number of processing requests.
        
        Args:
            status_filter: Optional status filter.
            
        Returns:
            int: Total number of requests matching the filter.
            
        Raises:
            RepositoryError: If the count operation fails.
        """
        pass