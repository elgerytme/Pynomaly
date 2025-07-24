"""
BaseDataDomainService

Base class for all data domain services.
"""

from data.abstractions.base_service import BaseService

class BaseDataDomainService(BaseService):
    """
    Base class for all data domain services.
    """
    def __init__(self):
        super().__init__()
