# Storage Infrastructure Audit Report

## Executive Summary

This report documents the current storage infrastructure in the Pynomaly system, focusing on storage-related abstractions in the **Infrastructure layer**, current class diagram, extension points, and how `ModelStorageInfo.storage_backend` is consumed.

## Current Storage Architecture

### 1. Core Storage Abstractions

#### 1.1 Domain Value Objects
- **`ModelStorageInfo`** (`src/pynomaly/domain/value_objects/model_storage_info.py`)
  - Central value object encapsulating storage information
  - Contains `storage_backend` field of type `StorageBackend` enum
  - Supports multiple storage backends and serialization formats
  - Provides storage-specific factory methods

#### 1.2 Storage Backend Enumeration
```python
class StorageBackend(Enum):
    LOCAL_FILESYSTEM = "local_filesystem"
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"
    MLFLOW = "mlflow"
    HUGGINGFACE_HUB = "huggingface_hub"
    DATABASE = "database"
    REDIS = "redis"
```

#### 1.3 Serialization Format Support
```python
class SerializationFormat(Enum):
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    TENSORFLOW_SAVEDMODEL = "tensorflow_savedmodel"
    PYTORCH_STATE_DICT = "pytorch_state_dict"
    HUGGINGFACE = "huggingface"
    MLFLOW_MODEL = "mlflow_model"
    SCIKIT_LEARN_PICKLE = "scikit_learn_pickle"
    JAX_PARAMS = "jax_params"
```

### 2. Repository Layer Abstractions

#### 2.1 Base Repository Protocol
- **`RepositoryProtocol[T]`** (`src/pynomaly/shared/protocols/repository_protocol.py`)
  - Generic base protocol for all repository implementations
  - Defines standard CRUD operations: `save`, `find_by_id`, `find_all`, `delete`, `exists`, `count`

#### 2.2 Specific Repository Protocols
- **`DetectorRepositoryProtocol`** - For detector storage with model artifacts
- **`DatasetRepositoryProtocol`** - For dataset storage with data persistence
- **`DetectionResultRepositoryProtocol`** - For detection result storage
- **`ModelRepositoryProtocol`** - For model entity storage
- **`ModelVersionRepositoryProtocol`** - For model version storage
- **`ExperimentRepositoryProtocol`** - For experiment storage
- Additional protocols for pipelines, alerts, etc.

### 3. Current Repository Implementations

#### 3.1 Memory-Based Repositories
- **`MemoryDetectorRepository`** (`src/pynomaly/infrastructure/repositories/memory_repository.py`)
- **`MemoryDatasetRepository`** 
- **`MemoryDetectionResultRepository`**
- **`FileSystemDetectorRepository`** - Pickle-based file storage

#### 3.2 File-Based Repositories
- **`FileDetectorRepository`** (`src/pynomaly/infrastructure/repositories/file_repositories.py`)
- **`FileDatasetRepository`**
- **`FileResultRepository`**
- **`ConfigurationRepository`** (`src/pynomaly/infrastructure/persistence/configuration_repository.py`)

#### 3.3 Training-Specific Repositories
- **`TrainingRepository`** (`src/pynomaly/infrastructure/persistence/training_repository.py`)
  - Supports multiple backends: memory, file, SQL (placeholder)
  - Provides backend selection via `storage_type` parameter
  - Includes optimization trial storage

#### 3.4 Repository Factory Pattern
- **`RepositoryFactory`** (`src/pynomaly/infrastructure/repositories/repository_factory.py`)
  - Creates repository services with specified storage backends
  - Supports "memory" and "filesystem" storage types
  - Environment-based configuration support

### 4. How `ModelStorageInfo.storage_backend` is Consumed

#### 4.1 In Domain Entities
- **`ModelVersion`** entity contains `storage_info: ModelStorageInfo` field
- Used to track where and how model versions are stored
- Provides storage metadata and validation

#### 4.2 In Application Services
- **`ModelManagementService`** accepts `ModelStorageInfo` for model version creation
- **`EnhancedModelPersistenceService`** uses storage info for persistence operations
- **`DeploymentOrchestrationService`** consumes storage info for deployment

#### 4.3 Current Usage Pattern
```python
# Model version creation with storage info
model_version = ModelVersion(
    model_id=model_id,
    version=version,
    storage_info=ModelStorageInfo(
        storage_backend=StorageBackend.LOCAL_FILESYSTEM,
        storage_path="/path/to/model.pkl",
        format=SerializationFormat.PICKLE,
        size_bytes=1024,
        checksum="abc123..."
    )
)
```

## Current Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INFRASTRUCTURE LAYER                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │  RepositoryFactory  │    │  RepositoryService  │                    │
│  │                     │    │                     │                    │
│  │ + create_repository │───▶│ + repositories      │                    │
│  │   _service()        │    │                     │                    │
│  └─────────────────────┘    └─────────────────────┘                    │
│                                        │                               │
│                                        ▼                               │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │ Memory Repositories │    │  File Repositories  │                    │
│  │                     │    │                     │                    │
│  │ • MemoryDetector    │    │ • FileDetector      │                    │
│  │ • MemoryDataset     │    │ • FileDataset       │                    │
│  │ • MemoryResult      │    │ • FileResult        │                    │
│  └─────────────────────┘    └─────────────────────┘                    │
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │ TrainingRepository  │    │ConfigurationRepository│                   │
│  │                     │    │                     │                    │
│  │ • Multi-backend     │    │ • File-based        │                    │
│  │ • Memory/File/SQL   │    │ • JSON storage      │                    │
│  └─────────────────────┘    └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            DOMAIN LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │ ModelStorageInfo    │    │  StorageBackend     │                    │
│  │                     │    │                     │                    │
│  │ • storage_backend   │───▶│ • LOCAL_FILESYSTEM  │                    │
│  │ • storage_path      │    │ • AWS_S3           │                    │
│  │ • format           │    │ • AZURE_BLOB       │                    │
│  │ • size_bytes       │    │ • GCP_STORAGE      │                    │
│  │ • checksum         │    │ • MLFLOW           │                    │
│  └─────────────────────┘    └─────────────────────┘                    │
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐                    │
│  │   ModelVersion      │    │ Repository          │                    │
│  │                     │    │ Protocols           │                    │
│  │ • storage_info      │    │                     │                    │
│  │ • model_id          │    │ • RepositoryProtocol│                    │
│  │ • version           │    │ • DetectorRepo...   │                    │
│  └─────────────────────┘    │ • DatasetRepo...    │                    │
│                              │ • ModelRepo...      │                    │
│                              └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Extension Points

### 1. Repository Protocol Extensions
- All repository protocols can be extended with new implementations
- Base `RepositoryProtocol[T]` provides consistent interface
- Specialized protocols add domain-specific methods

### 2. Storage Backend Extensions
- `StorageBackend` enum can be extended with new storage types
- `ModelStorageInfo` factory methods can be added for new backends
- Serialization formats can be extended via `SerializationFormat` enum

### 3. Factory Pattern Extensions
- `RepositoryFactory` can be extended to support new storage backends
- Environment-based configuration allows runtime backend selection

## Missing Abstractions

### 1. **Critical Gap: Abstract Storage Adapter Layer**

#### Missing: `AbstractStorageAdapter`
- **No unified interface** for storage operations across different backends
- **No abstract base class** for storage adapters
- **No pluggable storage backend system**

#### Missing: `CloudStorageAdapter` Base Class
- **No common interface** for cloud storage providers (AWS S3, Azure Blob, GCP Storage)
- **No shared functionality** for cloud storage operations
- **No consistent error handling** across cloud providers

#### Missing: Storage-Specific Adapters
- **No `S3StorageAdapter`** - AWS S3 implementation
- **No `AzureBlobStorageAdapter`** - Azure Blob Storage implementation  
- **No `GcpStorageAdapter`** - Google Cloud Storage implementation
- **No `MLflowStorageAdapter`** - MLflow artifact storage implementation

### 2. **Storage Operation Abstractions**

#### Missing: `StorageOperations` Interface
```python
# Proposed interface
class StorageOperations(Protocol):
    async def save(self, data: bytes, path: str) -> ModelStorageInfo
    async def load(self, storage_info: ModelStorageInfo) -> bytes
    async def delete(self, storage_info: ModelStorageInfo) -> bool
    async def exists(self, storage_info: ModelStorageInfo) -> bool
    async def list_objects(self, prefix: str) -> List[str]
```

#### Missing: `StorageAdapterFactory`
- **No factory** to create storage adapters based on `StorageBackend`
- **No registration system** for storage adapters
- **No dependency injection** for storage adapters

### 3. **Configuration and Management**

#### Missing: `StorageConfiguration` Value Object
- **No centralized configuration** for storage backends
- **No connection pooling** configuration
- **No retry/timeout** configuration

#### Missing: `StorageHealthCheck` Service
- **No health monitoring** for storage backends
- **No connectivity validation**
- **No performance metrics** collection

### 4. **Advanced Features**

#### Missing: `StorageMiddleware` Support
- **No encryption middleware** for storage operations
- **No compression middleware**
- **No caching middleware**

#### Missing: `StorageTransaction` Support
- **No transactional storage** operations
- **No rollback capabilities**
- **No consistency guarantees**

## Recommendations

### 1. **Priority 1: Implement Core Storage Abstractions**

#### Create `AbstractStorageAdapter`
```python
@abstractmethod
class AbstractStorageAdapter(Protocol):
    async def save(self, data: bytes, path: str) -> ModelStorageInfo
    async def load(self, storage_info: ModelStorageInfo) -> bytes
    async def delete(self, storage_info: ModelStorageInfo) -> bool
    async def exists(self, storage_info: ModelStorageInfo) -> bool
    async def get_metadata(self, storage_info: ModelStorageInfo) -> Dict[str, Any]
```

#### Create `CloudStorageAdapter` Base Class
```python
class CloudStorageAdapter(AbstractStorageAdapter):
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self.client = self._create_client()
    
    @abstractmethod
    def _create_client(self) -> Any:
        pass
    
    async def save(self, data: bytes, path: str) -> ModelStorageInfo:
        # Common cloud storage save logic
        pass
```

### 2. **Priority 2: Implement Storage Adapter Factory**

#### Create `StorageAdapterFactory`
```python
class StorageAdapterFactory:
    _adapters: Dict[StorageBackend, Type[AbstractStorageAdapter]] = {}
    
    @classmethod
    def register(cls, backend: StorageBackend, adapter_class: Type[AbstractStorageAdapter]):
        cls._adapters[backend] = adapter_class
    
    @classmethod
    def create(cls, storage_info: ModelStorageInfo) -> AbstractStorageAdapter:
        adapter_class = cls._adapters.get(storage_info.storage_backend)
        if not adapter_class:
            raise ValueError(f"No adapter registered for {storage_info.storage_backend}")
        return adapter_class(storage_info)
```

### 3. **Priority 3: Update Repository Implementations**

#### Modify Existing Repositories
- Update repositories to use storage adapters instead of direct file operations
- Implement storage backend selection in repository constructors
- Add proper error handling and retry logic

### 4. **Priority 4: Add Storage Management Services**

#### Create `StorageService`
```python
class StorageService:
    def __init__(self, adapter_factory: StorageAdapterFactory):
        self.adapter_factory = adapter_factory
    
    async def save_model(self, model_data: bytes, storage_backend: StorageBackend) -> ModelStorageInfo:
        adapter = self.adapter_factory.create_for_backend(storage_backend)
        return await adapter.save(model_data, self._generate_path())
```

## Conclusion

The current storage infrastructure has a solid foundation with:
- **Good domain modeling** via `ModelStorageInfo` and storage enums
- **Consistent repository patterns** using protocols
- **Multiple storage backend support** (in design)

However, there are significant gaps in the **Infrastructure layer**:
- **Missing storage adapter abstractions** for pluggable backends
- **No unified storage operations interface**
- **Limited cloud storage implementation**
- **No storage adapter factory or registry**

Implementing the recommended abstractions would provide a robust, extensible storage infrastructure that can support multiple storage backends while maintaining consistency and reducing code duplication.
