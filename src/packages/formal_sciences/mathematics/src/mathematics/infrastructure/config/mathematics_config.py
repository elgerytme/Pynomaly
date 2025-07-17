"""Configuration for mathematics package."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class MathematicsConfig:
    """Configuration for mathematics package."""
    
    # Numerical computation settings
    default_dtype: np.dtype = np.float64
    numerical_tolerance: float = 1e-10
    max_iterations: int = 1000
    
    # Cache settings
    enable_function_cache: bool = True
    function_cache_size: int = 1000
    enable_matrix_cache: bool = True
    matrix_cache_size: int = 100
    
    # Computation settings
    enable_parallel_computation: bool = True
    max_threads: int = 4
    
    # Symbolic computation settings
    enable_symbolic_computation: bool = True
    symbolic_simplification: bool = True
    
    # Matrix computation settings
    matrix_decomposition_method: str = "auto"  # auto, lu, qr, svd
    eigenvalue_method: str = "auto"  # auto, numpy, scipy
    
    # Integration settings
    integration_method: str = "quad"  # quad, simpson, trapz
    integration_tolerance: float = 1e-6
    integration_max_subdivisions: int = 50
    
    # Optimization settings
    optimization_method: str = "minimize"  # minimize, fsolve
    optimization_tolerance: float = 1e-6
    optimization_max_iterations: int = 1000
    
    # Plotting settings (if visualization is enabled)
    enable_plotting: bool = False
    plot_backend: str = "matplotlib"  # matplotlib, plotly
    plot_dpi: int = 100
    plot_figsize: tuple = (8, 6)
    
    # Debug settings
    debug_mode: bool = False
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Memory settings
    max_memory_usage: int = 1024 * 1024 * 1024  # 1GB in bytes
    garbage_collection_threshold: int = 1000
    
    @classmethod
    def development(cls) -> "MathematicsConfig":
        """Create development configuration."""
        return cls(
            debug_mode=True,
            log_level="DEBUG",
            enable_function_cache=True,
            function_cache_size=500,
            enable_matrix_cache=True,
            matrix_cache_size=50,
            numerical_tolerance=1e-8,
        )
    
    @classmethod
    def production(cls) -> "MathematicsConfig":
        """Create production configuration."""
        return cls(
            debug_mode=False,
            log_level="WARNING",
            enable_function_cache=True,
            function_cache_size=2000,
            enable_matrix_cache=True,
            matrix_cache_size=200,
            numerical_tolerance=1e-12,
            enable_parallel_computation=True,
            max_threads=8,
        )
    
    @classmethod
    def testing(cls) -> "MathematicsConfig":
        """Create testing configuration."""
        return cls(
            debug_mode=True,
            log_level="DEBUG",
            enable_function_cache=False,
            enable_matrix_cache=False,
            numerical_tolerance=1e-6,
            enable_parallel_computation=False,
            max_threads=1,
        )
    
    @classmethod
    def high_performance(cls) -> "MathematicsConfig":
        """Create high-performance configuration."""
        return cls(
            enable_parallel_computation=True,
            max_threads=16,
            enable_function_cache=True,
            function_cache_size=5000,
            enable_matrix_cache=True,
            matrix_cache_size=500,
            numerical_tolerance=1e-10,
            max_iterations=2000,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "default_dtype": str(self.default_dtype),
            "numerical_tolerance": self.numerical_tolerance,
            "max_iterations": self.max_iterations,
            "enable_function_cache": self.enable_function_cache,
            "function_cache_size": self.function_cache_size,
            "enable_matrix_cache": self.enable_matrix_cache,
            "matrix_cache_size": self.matrix_cache_size,
            "enable_parallel_computation": self.enable_parallel_computation,
            "max_threads": self.max_threads,
            "enable_symbolic_computation": self.enable_symbolic_computation,
            "symbolic_simplification": self.symbolic_simplification,
            "matrix_decomposition_method": self.matrix_decomposition_method,
            "eigenvalue_method": self.eigenvalue_method,
            "integration_method": self.integration_method,
            "integration_tolerance": self.integration_tolerance,
            "integration_max_subdivisions": self.integration_max_subdivisions,
            "optimization_method": self.optimization_method,
            "optimization_tolerance": self.optimization_tolerance,
            "optimization_max_iterations": self.optimization_max_iterations,
            "enable_plotting": self.enable_plotting,
            "plot_backend": self.plot_backend,
            "plot_dpi": self.plot_dpi,
            "plot_figsize": self.plot_figsize,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "max_memory_usage": self.max_memory_usage,
            "garbage_collection_threshold": self.garbage_collection_threshold,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MathematicsConfig":
        """Create configuration from dictionary."""
        # Convert dtype string back to numpy dtype
        if "default_dtype" in config_dict:
            config_dict["default_dtype"] = np.dtype(config_dict["default_dtype"])
        
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.numerical_tolerance <= 0:
            raise ValueError("Numerical tolerance must be positive")
        
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        
        if self.function_cache_size <= 0:
            raise ValueError("Function cache size must be positive")
        
        if self.matrix_cache_size <= 0:
            raise ValueError("Matrix cache size must be positive")
        
        if self.max_threads <= 0:
            raise ValueError("Max threads must be positive")
        
        if self.integration_tolerance <= 0:
            raise ValueError("Integration tolerance must be positive")
        
        if self.optimization_tolerance <= 0:
            raise ValueError("Optimization tolerance must be positive")
        
        if self.max_memory_usage <= 0:
            raise ValueError("Max memory usage must be positive")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")
        
        valid_decomposition_methods = ["auto", "lu", "qr", "svd"]
        if self.matrix_decomposition_method not in valid_decomposition_methods:
            raise ValueError(f"Matrix decomposition method must be one of {valid_decomposition_methods}")
        
        valid_eigenvalue_methods = ["auto", "numpy", "scipy"]
        if self.eigenvalue_method not in valid_eigenvalue_methods:
            raise ValueError(f"Eigenvalue method must be one of {valid_eigenvalue_methods}")
        
        valid_integration_methods = ["quad", "simpson", "trapz"]
        if self.integration_method not in valid_integration_methods:
            raise ValueError(f"Integration method must be one of {valid_integration_methods}")
        
        valid_optimization_methods = ["minimize", "fsolve"]
        if self.optimization_method not in valid_optimization_methods:
            raise ValueError(f"Optimization method must be one of {valid_optimization_methods}")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()