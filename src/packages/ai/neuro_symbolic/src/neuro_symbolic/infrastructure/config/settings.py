"""Configuration management for neuro-symbolic AI."""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import yaml
from pathlib import Path
import warnings

try:
    from pydantic import BaseSettings, validator
    HAS_PYDANTIC = True
except ImportError:
    try:
        from pydantic_settings import BaseSettings
        from pydantic import validator
        HAS_PYDANTIC = True
    except ImportError:
        HAS_PYDANTIC = False
        warnings.warn("Pydantic not available. Configuration validation will be limited.")


class DeviceType(Enum):
    """Supported computation devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    AUTO = "auto"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class NeuralConfig:
    """Configuration for neural network components."""
    device: DeviceType = DeviceType.AUTO
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_value: Optional[float] = 1.0
    weight_decay: float = 0.01
    scheduler_type: str = "cosine"  # cosine, linear, exponential
    warmup_steps: int = 100
    dropout_rate: float = 0.1
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    # Model-specific settings
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    attention_heads: int = 8
    transformer_layers: int = 6
    embedding_dim: int = 512
    max_sequence_length: int = 512


@dataclass
class SymbolicConfig:
    """Configuration for symbolic reasoning components."""
    reasoning_timeout: float = 30.0  # seconds
    max_inference_depth: int = 10
    enable_caching: bool = True
    cache_size: int = 1000
    proof_search_strategy: str = "breadth_first"  # breadth_first, depth_first, best_first
    
    # Logic engine settings
    use_z3_solver: bool = True
    z3_timeout: int = 10000  # milliseconds
    use_sympy: bool = True
    enable_rdfs_inference: bool = True
    
    # Rule settings
    max_rules: int = 10000
    rule_confidence_threshold: float = 0.5
    enable_rule_learning: bool = False


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_multiprocessing: bool = True
    num_workers: int = -1  # -1 for auto-detection
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Memory management
    max_memory_usage_gb: float = 8.0
    garbage_collection_threshold: int = 1000
    enable_memory_profiling: bool = False
    
    # Caching
    enable_result_caching: bool = True
    cache_backend: str = "memory"  # memory, redis, filesystem
    cache_ttl: int = 3600  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_performance_stats: bool = True


@dataclass
class StorageConfig:
    """Configuration for data storage and persistence."""
    base_path: str = "./neuro_symbolic_data"
    models_path: str = "models"
 
    logs_path: str = "logs"
    cache_path: str = "cache"
    
    # Model persistence
    save_format: str = "pytorch"  # pytorch, onnx, tensorflow
    compression: bool = True
    versioning_enabled: bool = True
    max_versions: int = 5
    
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Database settings (for production deployments)
    database_url: Optional[str] = None
    connection_pool_size: int = 10
    connection_timeout: float = 30.0


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_authentication: bool = False
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_expiration_hours: int = 24
    
    # Input validation
    max_input_size_mb: float = 100.0
    allowed_file_extensions: List[str] = field(
        default_factory=lambda: ['.json', '.yaml', '.yml', '.rdf', '.owl', '.ttl']
    )
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Data privacy
    enable_data_anonymization: bool = False
    log_sensitive_data: bool = False


if HAS_PYDANTIC:
    class NeuroSymbolicSettings(BaseSettings):
        """Pydantic-based settings with environment variable support."""
        
        # Neural settings
        neural_device: str = "auto"
        neural_batch_size: int = 32
        neural_learning_rate: float = 0.001
        neural_epochs: int = 100
        
        # Symbolic settings
        symbolic_reasoning_timeout: float = 30.0
        symbolic_max_inference_depth: int = 10
        symbolic_enable_caching: bool = True
        
        # Performance settings
        performance_enable_multiprocessing: bool = True
        performance_num_workers: int = -1
        performance_max_memory_gb: float = 8.0
        
        # Storage settings
        storage_base_path: str = "./neuro_symbolic_data"
        storage_models_path: str = "models"
        
        # Security settings
        security_enable_auth: bool = False
        security_api_key: Optional[str] = None
        security_max_input_size_mb: float = 100.0
        
        # Logging
        log_level: str = "INFO"
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        class Config:
            env_prefix = "NEURO_SYMBOLIC_"
            case_sensitive = False
            
        @validator('neural_device')
        def validate_device(cls, v):
            valid_devices = [e.value for e in DeviceType]
            if v not in valid_devices:
                raise ValueError(f"Device must be one of: {valid_devices}")
            return v
            
        @validator('log_level')
        def validate_log_level(cls, v):
            valid_levels = [e.value for e in LogLevel]
            if v.upper() not in valid_levels:
                raise ValueError(f"Log level must be one of: {valid_levels}")
            return v.upper()
            
        @validator('neural_learning_rate')
        def validate_learning_rate(cls, v):
            if not 0 < v <= 1:
                raise ValueError("Learning rate must be between 0 and 1")
            return v


@dataclass
class NeuroSymbolicConfig:
    """Main configuration class for neuro-symbolic AI."""
    
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Global settings
    debug_mode: bool = False
    log_level: LogLevel = LogLevel.INFO
    random_seed: Optional[int] = 42
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "NeuroSymbolicConfig":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        try:
            with open(config_path, 'r') as f:
                if suffix in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {suffix}")
            
            return cls.from_dict(data)
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuroSymbolicConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Update neural config
        if 'neural' in data:
            neural_data = data['neural']
            for key, value in neural_data.items():
                if hasattr(config.neural, key):
                    setattr(config.neural, key, value)
        
        # Update symbolic config
        if 'symbolic' in data:
            symbolic_data = data['symbolic']
            for key, value in symbolic_data.items():
                if hasattr(config.symbolic, key):
                    setattr(config.symbolic, key, value)
        
        # Update performance config
        if 'performance' in data:
            perf_data = data['performance']
            for key, value in perf_data.items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)
        
        # Update storage config
        if 'storage' in data:
            storage_data = data['storage']
            for key, value in storage_data.items():
                if hasattr(config.storage, key):
                    setattr(config.storage, key, value)
        
        # Update security config
        if 'security' in data:
            security_data = data['security']
            for key, value in security_data.items():
                if hasattr(config.security, key):
                    setattr(config.security, key, value)
        
        # Update global settings
        for key in ['debug_mode', 'log_level', 'random_seed']:
            if key in data:
                value = data[key]
                if key == 'log_level' and isinstance(value, str):
                    value = LogLevel(value.upper())
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_env(cls) -> "NeuroSymbolicConfig":
        """Load configuration from environment variables."""
        if not HAS_PYDANTIC:
            warnings.warn("Pydantic not available. Using default configuration.")
            return cls()
        
        settings = NeuroSymbolicSettings()
        config = cls()
        
        # Map pydantic settings to dataclass config
        config.neural.device = DeviceType(settings.neural_device)
        config.neural.batch_size = settings.neural_batch_size
        config.neural.learning_rate = settings.neural_learning_rate
        config.neural.epochs = settings.neural_epochs
        
        config.symbolic.reasoning_timeout = settings.symbolic_reasoning_timeout
        config.symbolic.max_inference_depth = settings.symbolic_max_inference_depth
        config.symbolic.enable_caching = settings.symbolic_enable_caching
        
        config.performance.enable_multiprocessing = settings.performance_enable_multiprocessing
        config.performance.num_workers = settings.performance_num_workers
        config.performance.max_memory_usage_gb = settings.performance_max_memory_gb
        
        config.storage.base_path = settings.storage_base_path
        config.storage.models_path = settings.storage_models_path
        
        config.security.enable_authentication = settings.security_enable_auth
        config.security.api_key = settings.security_api_key
        config.security.max_input_size_mb = settings.security_max_input_size_mb
        
        config.log_level = LogLevel(settings.log_level)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'neural': {
                'device': self.neural.device.value,
                'batch_size': self.neural.batch_size,
                'learning_rate': self.neural.learning_rate,
                'epochs': self.neural.epochs,
                'early_stopping_patience': self.neural.early_stopping_patience,
                'gradient_clip_value': self.neural.gradient_clip_value,
                'weight_decay': self.neural.weight_decay,
                'scheduler_type': self.neural.scheduler_type,
                'warmup_steps': self.neural.warmup_steps,
                'dropout_rate': self.neural.dropout_rate,
                'mixed_precision': self.neural.mixed_precision,
                'compile_model': self.neural.compile_model,
                'hidden_dims': self.neural.hidden_dims,
                'attention_heads': self.neural.attention_heads,
                'transformer_layers': self.neural.transformer_layers,
                'embedding_dim': self.neural.embedding_dim,
                'max_sequence_length': self.neural.max_sequence_length
            },
            'symbolic': {
                'reasoning_timeout': self.symbolic.reasoning_timeout,
                'max_inference_depth': self.symbolic.max_inference_depth,
                'enable_caching': self.symbolic.enable_caching,
                'cache_size': self.symbolic.cache_size,
                'proof_search_strategy': self.symbolic.proof_search_strategy,
                'use_z3_solver': self.symbolic.use_z3_solver,
                'z3_timeout': self.symbolic.z3_timeout,
                'use_sympy': self.symbolic.use_sympy,
                'enable_rdfs_inference': self.symbolic.enable_rdfs_inference,
                'max_rules': self.symbolic.max_rules,
                'rule_confidence_threshold': self.symbolic.rule_confidence_threshold,
                'enable_rule_learning': self.symbolic.enable_rule_learning
            },
            'performance': {
                'enable_multiprocessing': self.performance.enable_multiprocessing,
                'num_workers': self.performance.num_workers,
                'prefetch_factor': self.performance.prefetch_factor,
                'pin_memory': self.performance.pin_memory,
                'max_memory_usage_gb': self.performance.max_memory_usage_gb,
                'garbage_collection_threshold': self.performance.garbage_collection_threshold,
                'enable_memory_profiling': self.performance.enable_memory_profiling,
                'enable_result_caching': self.performance.enable_result_caching,
                'cache_backend': self.performance.cache_backend,
                'cache_ttl': self.performance.cache_ttl,
                'enable_metrics': self.performance.enable_metrics,
                'metrics_port': self.performance.metrics_port,
                'log_performance_stats': self.performance.log_performance_stats
            },
            'storage': {
                'base_path': self.storage.base_path,
                'models_path': self.storage.models_path,
                'logs_path': self.storage.logs_path,
                'cache_path': self.storage.cache_path,
                'save_format': self.storage.save_format,
                'compression': self.storage.compression,
                'versioning_enabled': self.storage.versioning_enabled,
                'max_versions': self.storage.max_versions,
                'backup_enabled': self.storage.backup_enabled,
                'backup_interval_hours': self.storage.backup_interval_hours,
                'database_url': self.storage.database_url,
                'connection_pool_size': self.storage.connection_pool_size,
                'connection_timeout': self.storage.connection_timeout
            },
            'security': {
                'enable_authentication': self.security.enable_authentication,
                'api_key': self.security.api_key,
                'jwt_secret': self.security.jwt_secret,
                'jwt_expiration_hours': self.security.jwt_expiration_hours,
                'max_input_size_mb': self.security.max_input_size_mb,
                'allowed_file_extensions': self.security.allowed_file_extensions,
                'enable_rate_limiting': self.security.enable_rate_limiting,
                'max_requests_per_minute': self.security.max_requests_per_minute,
                'enable_data_anonymization': self.security.enable_data_anonymization,
                'log_sensitive_data': self.security.log_sensitive_data
            },
            'debug_mode': self.debug_mode,
            'log_level': self.log_level.value,
            'random_seed': self.random_seed
        }
    
    def save_to_file(self, config_path: Union[str, Path], format: str = 'yaml') -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if format.lower() in ['yml', 'yaml']:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate neural config
        if self.neural.learning_rate <= 0 or self.neural.learning_rate > 1:
            issues.append("Learning rate must be between 0 and 1")
        
        if self.neural.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        if self.neural.epochs <= 0:
            issues.append("Epochs must be positive")
        
        # Validate symbolic config
        if self.symbolic.reasoning_timeout <= 0:
            issues.append("Reasoning timeout must be positive")
        
        if self.symbolic.max_inference_depth <= 0:
            issues.append("Max inference depth must be positive")
        
        # Validate storage paths
        try:
            Path(self.storage.base_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create base storage path: {e}")
        
        # Validate performance settings
        if self.performance.max_memory_usage_gb <= 0:
            issues.append("Max memory usage must be positive")
        
        return issues
    
    def get_device(self) -> str:
        """Get the appropriate device for computation."""
        if self.neural.device == DeviceType.AUTO:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        else:
            return self.neural.device.value
    
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        import logging
        
        logging.basicConfig(
            level=getattr(logging, self.log_level.value),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(self.storage.base_path) / self.storage.logs_path / "neuro_symbolic.log"
                )
            ]
        )
        
        if self.debug_mode:
            logging.getLogger().setLevel(logging.DEBUG)


# Global configuration instance
_config: Optional[NeuroSymbolicConfig] = None


def get_config() -> NeuroSymbolicConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = NeuroSymbolicConfig.from_env()
    return _config


def set_config(config: NeuroSymbolicConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration to default."""
    global _config
    _config = None