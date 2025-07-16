"""Dependency Injection Container

Container for managing dependencies and configuration.
"""

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pynomaly_mlops.application.services.model_registry_service import ModelRegistryService
from pynomaly_mlops.domain.repositories.model_repository import ModelRepository
from pynomaly_mlops.domain.services.model_promotion_service import (
    ModelPromotionService, DefaultModelPromotionService
)
from pynomaly_mlops.infrastructure.config.database import create_async_engine
from pynomaly_mlops.infrastructure.config.settings import MLOpsSettings
from pynomaly_mlops.infrastructure.persistence.repositories import SqlAlchemyModelRepository
from pynomaly_mlops.infrastructure.storage.artifact_storage import (
    ArtifactStorageService, S3ArtifactStorage, LocalArtifactStorage
)


class MLOpsContainer:
    """Dependency injection container for MLOps platform."""
    
    def __init__(self, settings: Optional[MLOpsSettings] = None):
        """Initialize container with settings.
        
        Args:
            settings: MLOps platform settings
        """
        self.settings = settings or MLOpsSettings.from_env()
        self._session_maker: Optional[async_sessionmaker] = None
        self._artifact_storage: Optional[ArtifactStorageService] = None
        self._model_repository: Optional[ModelRepository] = None
        self._promotion_service: Optional[ModelPromotionService] = None
        self._model_registry_service: Optional[ModelRegistryService] = None
    
    @property
    def session_maker(self) -> async_sessionmaker:
        """Get SQLAlchemy session maker.
        
        Returns:
            Async session maker
        """
        if self._session_maker is None:
            engine = create_async_engine(self.settings.database)
            self._session_maker = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
        
        return self._session_maker
    
    def get_session(self) -> AsyncSession:
        """Get new database session.
        
        Returns:
            Database session
        """
        return self.session_maker()
    
    @property
    def artifact_storage(self) -> ArtifactStorageService:
        """Get artifact storage service.
        
        Returns:
            Artifact storage service
        """
        if self._artifact_storage is None:
            storage_config = self.settings.storage
            
            if storage_config.backend == 's3':
                if not storage_config.s3_bucket:
                    raise ValueError("S3 bucket name is required for S3 storage")
                
                self._artifact_storage = S3ArtifactStorage(
                    bucket_name=storage_config.s3_bucket,
                    region_name=storage_config.s3_region,
                    endpoint_url=storage_config.s3_endpoint_url,
                    aws_access_key_id=storage_config.s3_access_key_id,
                    aws_secret_access_key=storage_config.s3_secret_access_key,
                    prefix=storage_config.s3_prefix
                )
            else:
                # Default to local storage
                self._artifact_storage = LocalArtifactStorage(
                    base_path=storage_config.local_path
                )
        
        return self._artifact_storage
    
    def get_model_repository(self, session: AsyncSession) -> ModelRepository:
        """Get model repository.
        
        Args:
            session: Database session
            
        Returns:
            Model repository
        """
        return SqlAlchemyModelRepository(session)
    
    @property
    def promotion_service(self) -> ModelPromotionService:
        """Get model promotion service.
        
        Returns:
            Model promotion service
        """
        if self._promotion_service is None:
            self._promotion_service = DefaultModelPromotionService()
        
        return self._promotion_service
    
    def get_model_registry_service(self, session: AsyncSession) -> ModelRegistryService:
        """Get model registry service.
        
        Args:
            session: Database session
            
        Returns:
            Model registry service
        """
        return ModelRegistryService(
            model_repository=self.get_model_repository(session),
            artifact_storage=self.artifact_storage,
            promotion_service=self.promotion_service
        )
    
    async def health_check(self) -> dict:
        """Perform health check on all services.
        
        Returns:
            Health check results
        """
        results = {
            'database': False,
            'storage': False,
            'overall': False
        }
        
        # Test database connection
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                results['database'] = True
        except Exception as e:
            results['database_error'] = str(e)
        
        # Test storage
        try:
            # For S3, try to list bucket
            if hasattr(self.artifact_storage, 's3_client'):
                self.artifact_storage.s3_client.head_bucket(
                    Bucket=self.artifact_storage.bucket_name
                )
            results['storage'] = True
        except Exception as e:
            results['storage_error'] = str(e)
        
        results['overall'] = results['database'] and results['storage']
        
        return results
    
    async def initialize(self):
        """Initialize the container and create database tables.
        
        This should be called once at application startup.
        """
        from pynomaly_mlops.infrastructure.persistence.models import Base
        
        # Create database tables
        engine = create_async_engine(self.settings.database)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize storage
        _ = self.artifact_storage  # This will initialize the storage backend
    
    async def cleanup(self):
        """Cleanup resources.
        
        This should be called at application shutdown.
        """
        if self._session_maker:
            # Close all sessions
            await self._session_maker.close_all()
        
        # Additional cleanup if needed
        pass