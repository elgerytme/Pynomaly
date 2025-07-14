"""Scaling Configuration Value Object

Immutable value object for deployment scaling configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ScalingConfig:
    """Scaling configuration value object for deployments.
    
    Defines auto-scaling behavior and resource allocation for
    deployed model services.
    """
    
    # Replica Configuration
    min_replicas: int = 1
    max_replicas: int = 10
    
    # CPU-based Scaling
    target_cpu_utilization: float = 70.0  # Percentage (0-100)
    cpu_scale_up_threshold: float = 80.0   # Percentage (0-100)
    cpu_scale_down_threshold: float = 30.0 # Percentage (0-100)
    
    # Memory-based Scaling
    target_memory_utilization: float = 80.0 # Percentage (0-100)
    memory_scale_up_threshold: float = 90.0  # Percentage (0-100)
    memory_scale_down_threshold: float = 40.0 # Percentage (0-100)
    
    # Request-based Scaling
    target_requests_per_second: Optional[float] = None
    max_requests_per_replica: Optional[int] = None
    
    # Scaling Behavior
    scale_up_cooldown_seconds: int = 300    # 5 minutes
    scale_down_cooldown_seconds: int = 600  # 10 minutes
    scale_up_pods_per_period: int = 2       # Max pods to add per scaling event
    scale_down_pods_per_period: int = 1     # Max pods to remove per scaling event
    
    # Stability Configuration
    stabilization_window_seconds: int = 120  # Window for scaling decisions
    scale_up_percentage: int = 100           # Max % increase per scaling event
    scale_down_percentage: int = 10          # Max % decrease per scaling event
    
    # Advanced Options
    enable_predictive_scaling: bool = False
    enable_vertical_scaling: bool = False
    scaling_disabled: bool = False
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Validate replica counts
        if self.min_replicas < 0:
            raise ValueError("min_replicas must be non-negative")
        
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        
        # Validate CPU utilization thresholds
        cpu_thresholds = [
            ("target_cpu_utilization", self.target_cpu_utilization),
            ("cpu_scale_up_threshold", self.cpu_scale_up_threshold),
            ("cpu_scale_down_threshold", self.cpu_scale_down_threshold),
        ]
        
        for name, value in cpu_thresholds:
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100, got {value}")
        
        # Validate memory utilization thresholds
        memory_thresholds = [
            ("target_memory_utilization", self.target_memory_utilization),
            ("memory_scale_up_threshold", self.memory_scale_up_threshold),
            ("memory_scale_down_threshold", self.memory_scale_down_threshold),
        ]
        
        for name, value in memory_thresholds:
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be between 0 and 100, got {value}")
        
        # Validate request-based metrics
        if self.target_requests_per_second is not None and self.target_requests_per_second <= 0:
            raise ValueError("target_requests_per_second must be positive")
        
        if self.max_requests_per_replica is not None and self.max_requests_per_replica <= 0:
            raise ValueError("max_requests_per_replica must be positive")
        
        # Validate cooldown periods
        if self.scale_up_cooldown_seconds < 0:
            raise ValueError("scale_up_cooldown_seconds must be non-negative")
        
        if self.scale_down_cooldown_seconds < 0:
            raise ValueError("scale_down_cooldown_seconds must be non-negative")
        
        # Validate scaling pods per period
        if self.scale_up_pods_per_period <= 0:
            raise ValueError("scale_up_pods_per_period must be positive")
        
        if self.scale_down_pods_per_period <= 0:
            raise ValueError("scale_down_pods_per_period must be positive")
        
        # Validate stabilization window
        if self.stabilization_window_seconds < 0:
            raise ValueError("stabilization_window_seconds must be non-negative")
        
        # Validate scaling percentages
        if not 1 <= self.scale_up_percentage <= 1000:
            raise ValueError("scale_up_percentage must be between 1 and 1000")
        
        if not 1 <= self.scale_down_percentage <= 100:
            raise ValueError("scale_down_percentage must be between 1 and 100")
        
        # Validate threshold relationships
        if self.cpu_scale_down_threshold >= self.cpu_scale_up_threshold:
            raise ValueError("cpu_scale_down_threshold must be < cpu_scale_up_threshold")
        
        if self.memory_scale_down_threshold >= self.memory_scale_up_threshold:
            raise ValueError("memory_scale_down_threshold must be < memory_scale_up_threshold")
    
    @classmethod
    def minimal(cls) -> "ScalingConfig":
        """Create minimal scaling configuration for development.
        
        Returns:
            ScalingConfig with minimal settings
        """
        return cls(
            min_replicas=1,
            max_replicas=2,
            target_cpu_utilization=80.0,
            target_memory_utilization=85.0,
            scale_up_cooldown_seconds=60,
            scale_down_cooldown_seconds=120,
        )
    
    @classmethod
    def production(cls) -> "ScalingConfig":
        """Create production scaling configuration.
        
        Returns:
            ScalingConfig with production-ready settings
        """
        return cls(
            min_replicas=3,
            max_replicas=50,
            target_cpu_utilization=60.0,
            target_memory_utilization=70.0,
            cpu_scale_up_threshold=70.0,
            cpu_scale_down_threshold=30.0,
            memory_scale_up_threshold=80.0,
            memory_scale_down_threshold=35.0,
            scale_up_cooldown_seconds=180,
            scale_down_cooldown_seconds=300,
            scale_up_pods_per_period=4,
            scale_down_pods_per_period=2,
            stabilization_window_seconds=180,
            enable_predictive_scaling=True,
        )
    
    @classmethod
    def high_throughput(cls) -> "ScalingConfig":
        """Create high-throughput scaling configuration.
        
        Returns:
            ScalingConfig optimized for high request volumes
        """
        return cls(
            min_replicas=5,
            max_replicas=100,
            target_cpu_utilization=50.0,
            target_memory_utilization=60.0,
            cpu_scale_up_threshold=60.0,
            cpu_scale_down_threshold=25.0,
            memory_scale_up_threshold=70.0,
            memory_scale_down_threshold=30.0,
            scale_up_cooldown_seconds=120,
            scale_down_cooldown_seconds=240,
            scale_up_pods_per_period=8,
            scale_down_pods_per_period=2,
            stabilization_window_seconds=90,
            scale_up_percentage=200,
            enable_predictive_scaling=True,
        )
    
    @classmethod
    def cost_optimized(cls) -> "ScalingConfig":
        """Create cost-optimized scaling configuration.
        
        Returns:
            ScalingConfig optimized for cost efficiency
        """
        return cls(
            min_replicas=1,
            max_replicas=10,
            target_cpu_utilization=85.0,
            target_memory_utilization=90.0,
            cpu_scale_up_threshold=90.0,
            cpu_scale_down_threshold=40.0,
            memory_scale_up_threshold=95.0,
            memory_scale_down_threshold=50.0,
            scale_up_cooldown_seconds=600,  # 10 minutes
            scale_down_cooldown_seconds=900, # 15 minutes
            scale_up_pods_per_period=1,
            scale_down_pods_per_period=1,
            stabilization_window_seconds=300,
            scale_down_percentage=5,  # Conservative scale down
        )
    
    def with_min_replicas(self, min_replicas: int) -> "ScalingConfig":
        """Create new config with different min_replicas.
        
        Args:
            min_replicas: New minimum replica count
            
        Returns:
            New ScalingConfig instance
        """
        return ScalingConfig(
            min_replicas=min_replicas,
            max_replicas=max(self.max_replicas, min_replicas),
            target_cpu_utilization=self.target_cpu_utilization,
            cpu_scale_up_threshold=self.cpu_scale_up_threshold,
            cpu_scale_down_threshold=self.cpu_scale_down_threshold,
            target_memory_utilization=self.target_memory_utilization,
            memory_scale_up_threshold=self.memory_scale_up_threshold,
            memory_scale_down_threshold=self.memory_scale_down_threshold,
            target_requests_per_second=self.target_requests_per_second,
            max_requests_per_replica=self.max_requests_per_replica,
            scale_up_cooldown_seconds=self.scale_up_cooldown_seconds,
            scale_down_cooldown_seconds=self.scale_down_cooldown_seconds,
            scale_up_pods_per_period=self.scale_up_pods_per_period,
            scale_down_pods_per_period=self.scale_down_pods_per_period,
            stabilization_window_seconds=self.stabilization_window_seconds,
            scale_up_percentage=self.scale_up_percentage,
            scale_down_percentage=self.scale_down_percentage,
            enable_predictive_scaling=self.enable_predictive_scaling,
            enable_vertical_scaling=self.enable_vertical_scaling,
            scaling_disabled=self.scaling_disabled,
        )
    
    def with_max_replicas(self, max_replicas: int) -> "ScalingConfig":
        """Create new config with different max_replicas.
        
        Args:
            max_replicas: New maximum replica count
            
        Returns:
            New ScalingConfig instance
        """
        return ScalingConfig(
            min_replicas=min(self.min_replicas, max_replicas),
            max_replicas=max_replicas,
            target_cpu_utilization=self.target_cpu_utilization,
            cpu_scale_up_threshold=self.cpu_scale_up_threshold,
            cpu_scale_down_threshold=self.cpu_scale_down_threshold,
            target_memory_utilization=self.target_memory_utilization,
            memory_scale_up_threshold=self.memory_scale_up_threshold,
            memory_scale_down_threshold=self.memory_scale_down_threshold,
            target_requests_per_second=self.target_requests_per_second,
            max_requests_per_replica=self.max_requests_per_replica,
            scale_up_cooldown_seconds=self.scale_up_cooldown_seconds,
            scale_down_cooldown_seconds=self.scale_down_cooldown_seconds,
            scale_up_pods_per_period=self.scale_up_pods_per_period,
            scale_down_pods_per_period=self.scale_down_pods_per_period,
            stabilization_window_seconds=self.stabilization_window_seconds,
            scale_up_percentage=self.scale_up_percentage,
            scale_down_percentage=self.scale_down_percentage,
            enable_predictive_scaling=self.enable_predictive_scaling,
            enable_vertical_scaling=self.enable_vertical_scaling,
            scaling_disabled=self.scaling_disabled,
        )
    
    def disable_scaling(self) -> "ScalingConfig":
        """Create new config with scaling disabled.
        
        Returns:
            New ScalingConfig instance with scaling disabled
        """
        return ScalingConfig(
            min_replicas=self.min_replicas,
            max_replicas=self.min_replicas,  # Same as min to prevent scaling
            target_cpu_utilization=self.target_cpu_utilization,
            cpu_scale_up_threshold=self.cpu_scale_up_threshold,
            cpu_scale_down_threshold=self.cpu_scale_down_threshold,
            target_memory_utilization=self.target_memory_utilization,
            memory_scale_up_threshold=self.memory_scale_up_threshold,
            memory_scale_down_threshold=self.memory_scale_down_threshold,
            target_requests_per_second=self.target_requests_per_second,
            max_requests_per_replica=self.max_requests_per_replica,
            scale_up_cooldown_seconds=self.scale_up_cooldown_seconds,
            scale_down_cooldown_seconds=self.scale_down_cooldown_seconds,
            scale_up_pods_per_period=self.scale_up_pods_per_period,
            scale_down_pods_per_period=self.scale_down_pods_per_period,
            stabilization_window_seconds=self.stabilization_window_seconds,
            scale_up_percentage=self.scale_up_percentage,
            scale_down_percentage=self.scale_down_percentage,
            enable_predictive_scaling=self.enable_predictive_scaling,
            enable_vertical_scaling=self.enable_vertical_scaling,
            scaling_disabled=True,
        )
    
    def calculate_target_replicas(
        self, 
        current_replicas: int,
        cpu_utilization: float,
        memory_utilization: float,
        requests_per_second: Optional[float] = None
    ) -> int:
        """Calculate target number of replicas based on current metrics.
        
        Args:
            current_replicas: Current number of replicas
            cpu_utilization: Current CPU utilization percentage (0-100)
            memory_utilization: Current memory utilization percentage (0-100)
            requests_per_second: Current requests per second (optional)
            
        Returns:
            Target number of replicas
        """
        if self.scaling_disabled:
            return current_replicas
        
        # Calculate based on CPU
        cpu_target = max(1, int(current_replicas * cpu_utilization / self.target_cpu_utilization))
        
        # Calculate based on memory
        memory_target = max(1, int(current_replicas * memory_utilization / self.target_memory_utilization))
        
        # Calculate based on requests (if configured)
        request_target = current_replicas
        if (requests_per_second is not None and 
            self.target_requests_per_second is not None and
            self.target_requests_per_second > 0):
            request_target = max(1, int(requests_per_second / self.target_requests_per_second))
        
        # Take the maximum of all targets (most conservative)
        target_replicas = max(cpu_target, memory_target, request_target)
        
        # Apply constraints
        target_replicas = max(self.min_replicas, min(self.max_replicas, target_replicas))
        
        return target_replicas
    
    @property
    def can_scale_up(self) -> bool:
        """Check if scaling up is possible."""
        return not self.scaling_disabled and self.max_replicas > self.min_replicas
    
    @property
    def can_scale_down(self) -> bool:
        """Check if scaling down is possible."""
        return not self.scaling_disabled and self.max_replicas > self.min_replicas
    
    @property
    def replica_range(self) -> int:
        """Get the range of possible replicas."""
        return self.max_replicas - self.min_replicas
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary containing all configuration values
        """
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_utilization": self.target_cpu_utilization,
            "cpu_scale_up_threshold": self.cpu_scale_up_threshold,
            "cpu_scale_down_threshold": self.cpu_scale_down_threshold,
            "target_memory_utilization": self.target_memory_utilization,
            "memory_scale_up_threshold": self.memory_scale_up_threshold,
            "memory_scale_down_threshold": self.memory_scale_down_threshold,
            "target_requests_per_second": self.target_requests_per_second,
            "max_requests_per_replica": self.max_requests_per_replica,
            "scale_up_cooldown_seconds": self.scale_up_cooldown_seconds,
            "scale_down_cooldown_seconds": self.scale_down_cooldown_seconds,
            "scale_up_pods_per_period": self.scale_up_pods_per_period,
            "scale_down_pods_per_period": self.scale_down_pods_per_period,
            "stabilization_window_seconds": self.stabilization_window_seconds,
            "scale_up_percentage": self.scale_up_percentage,
            "scale_down_percentage": self.scale_down_percentage,
            "enable_predictive_scaling": self.enable_predictive_scaling,
            "enable_vertical_scaling": self.enable_vertical_scaling,
            "scaling_disabled": self.scaling_disabled,
            "can_scale_up": self.can_scale_up,
            "can_scale_down": self.can_scale_down,
            "replica_range": self.replica_range,
        }