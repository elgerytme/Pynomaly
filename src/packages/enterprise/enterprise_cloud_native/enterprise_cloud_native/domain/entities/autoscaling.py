"""
Auto-scaling domain entities for cloud-native applications.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ScalingType(str, Enum):
    """Types of auto-scaling."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    CLUSTER = "cluster"
    CUSTOM = "custom"


class ScalingDirection(str, Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class MetricType(str, Enum):
    """Scaling metric types."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"
    EXTERNAL_METRIC = "external_metric"


class ScalingPolicyType(str, Enum):
    """Scaling policy types."""
    TARGET_TRACKING = "target_tracking"
    STEP_SCALING = "step_scaling"
    SIMPLE_SCALING = "simple_scaling"
    PREDICTIVE = "predictive"


class ScalingEvent(str, Enum):
    """Scaling event types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    COOLDOWN = "cooldown"
    METRIC_THRESHOLD_BREACH = "metric_threshold_breach"
    POLICY_TRIGGERED = "policy_triggered"
    MANUAL_OVERRIDE = "manual_override"


class HorizontalPodAutoscaler(BaseModel):
    """
    Horizontal Pod Autoscaler configuration and status.
    
    Manages automatic scaling of pod replicas based on
    observed CPU utilization, memory usage, or custom metrics.
    """
    
    id: UUID = Field(default_factory=uuid4, description="HPA identifier")
    
    # HPA identification
    name: str = Field(..., description="HPA name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Target resource
    target_resource_type: str = Field(..., description="Target resource type (Deployment, StatefulSet)")
    target_resource_name: str = Field(..., description="Target resource name")
    
    # Scaling configuration
    min_replicas: int = Field(..., ge=1, description="Minimum replicas")
    max_replicas: int = Field(..., ge=1, description="Maximum replicas")
    
    # Metrics configuration
    metrics: List[Dict[str, Any]] = Field(..., description="Scaling metrics")
    
    # Scaling behavior
    scale_up_behavior: Optional[Dict[str, Any]] = Field(None, description="Scale up behavior")
    scale_down_behavior: Optional[Dict[str, Any]] = Field(None, description="Scale down behavior")
    
    # Current status
    current_replicas: int = Field(default=1, description="Current replica count")
    desired_replicas: int = Field(default=1, description="Desired replica count")
    last_scale_time: Optional[datetime] = Field(None, description="Last scaling action time")
    
    # Conditions and status
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="HPA conditions")
    current_metrics: List[Dict[str, Any]] = Field(default_factory=list, description="Current metric values")
    
    # Scaling history
    scaling_events: List[Dict[str, Any]] = Field(default_factory=list, description="Recent scaling events")
    
    # Advanced configuration
    stabilization_window_seconds: int = Field(default=300, description="Stabilization window")
    tolerance: float = Field(default=0.1, ge=0.0, le=1.0, description="Scaling tolerance")
    
    # Status
    status: str = Field(default="active", description="HPA status")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('max_replicas')
    def max_replicas_greater_than_min(cls, v, values):
        """Ensure max replicas >= min replicas."""
        if 'min_replicas' in values and v < values['min_replicas']:
            raise ValueError('max_replicas must be >= min_replicas')
        return v
    
    def is_scaling_needed(self, metric_values: Dict[str, float]) -> tuple[bool, ScalingDirection, int]:
        """Determine if scaling is needed based on current metrics."""
        scaling_needed = False
        direction = ScalingDirection.NONE
        target_replicas = self.current_replicas
        
        # Simple CPU-based scaling logic (can be extended for multiple metrics)
        cpu_metric = next((m for m in self.metrics if m.get("type") == "Resource" and m.get("resource", {}).get("name") == "cpu"), None)
        
        if cpu_metric and "cpu" in metric_values:
            target_utilization = cpu_metric.get("resource", {}).get("target", {}).get("averageUtilization", 80)
            current_utilization = metric_values["cpu"]
            
            # Calculate desired replicas based on CPU utilization
            if current_utilization > 0:
                desired = int((current_utilization / target_utilization) * self.current_replicas)
                desired = max(self.min_replicas, min(self.max_replicas, desired))
                
                if desired != self.current_replicas:
                    scaling_needed = True
                    target_replicas = desired
                    direction = ScalingDirection.UP if desired > self.current_replicas else ScalingDirection.DOWN
        
        return scaling_needed, direction, target_replicas
    
    def can_scale(self) -> bool:
        """Check if scaling action is allowed (cooldown period)."""
        if not self.last_scale_time:
            return True
        
        time_since_scale = datetime.utcnow() - self.last_scale_time
        return time_since_scale >= timedelta(seconds=self.stabilization_window_seconds)
    
    def record_scaling_event(self, event_type: ScalingEvent, old_replicas: int, new_replicas: int, reason: str) -> None:
        """Record a scaling event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type.value,
            "old_replicas": old_replicas,
            "new_replicas": new_replicas,
            "reason": reason
        }
        
        self.scaling_events.append(event)
        
        # Keep only last 50 events
        if len(self.scaling_events) > 50:
            self.scaling_events = self.scaling_events[-50:]
        
        self.last_scale_time = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_current_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        """Update current metric values."""
        self.current_metrics = metrics
        self.updated_at = datetime.utcnow()
    
    def update_replica_counts(self, current: int, desired: int) -> None:
        """Update replica counts."""
        self.current_replicas = current
        self.desired_replicas = desired
        self.updated_at = datetime.utcnow()


class VerticalPodAutoscaler(BaseModel):
    """
    Vertical Pod Autoscaler configuration and status.
    
    Manages automatic adjustment of pod resource requests
    and limits based on actual resource usage patterns.
    """
    
    id: UUID = Field(default_factory=uuid4, description="VPA identifier")
    
    # VPA identification
    name: str = Field(..., description="VPA name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Target resource
    target_resource_type: str = Field(..., description="Target resource type")
    target_resource_name: str = Field(..., description="Target resource name")
    
    # VPA mode
    update_mode: str = Field(default="Auto", description="Update mode (Off, Initial, Auto)")
    
    # Resource policy
    resource_policy: Optional[Dict[str, Any]] = Field(None, description="Resource policy")
    container_policies: List[Dict[str, Any]] = Field(default_factory=list, description="Container-specific policies")
    
    # Current recommendations
    recommendations: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="Resource recommendations")
    
    # Status
    status: str = Field(default="active", description="VPA status")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="VPA conditions")
    
    # History
    recommendation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recommendation history")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_auto_mode(self) -> bool:
        """Check if VPA is in auto mode."""
        return self.update_mode == "Auto"
    
    def update_recommendations(self, container_recommendations: Dict[str, Dict[str, str]]) -> None:
        """Update resource recommendations."""
        old_recommendations = self.recommendations.copy()
        self.recommendations = container_recommendations
        
        # Record in history if recommendations changed
        if old_recommendations != container_recommendations:
            self.recommendation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "old_recommendations": old_recommendations,
                "new_recommendations": container_recommendations
            })
            
            # Keep only last 20 entries
            if len(self.recommendation_history) > 20:
                self.recommendation_history = self.recommendation_history[-20:]
        
        self.updated_at = datetime.utcnow()


class ClusterAutoscaler(BaseModel):
    """
    Cluster Autoscaler configuration for node scaling.
    
    Manages automatic scaling of cluster nodes based on
    pod scheduling requirements and resource utilization.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Cluster autoscaler identifier")
    
    # Autoscaler identification
    name: str = Field(..., description="Autoscaler name")
    cluster_id: UUID = Field(..., description="Target cluster ID")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Node group configuration
    node_groups: List[Dict[str, Any]] = Field(..., description="Auto-scaling node groups")
    
    # Scaling configuration
    scale_down_enabled: bool = Field(default=True, description="Enable scale down")
    scale_down_delay_after_add: str = Field(default="10m", description="Scale down delay after add")
    scale_down_unneeded_time: str = Field(default="10m", description="Scale down unneeded time")
    scale_down_utilization_threshold: float = Field(default=0.5, description="Scale down threshold")
    
    # Scaling limits
    max_node_provision_time: str = Field(default="15m", description="Max node provision time")
    max_nodes_total: Optional[int] = Field(None, description="Maximum total nodes")
    
    # Current status
    current_nodes: int = Field(default=0, description="Current node count")
    desired_nodes: int = Field(default=0, description="Desired node count")
    unschedulable_pods: int = Field(default=0, description="Unschedulable pods count")
    
    # Activity tracking
    recent_scaling_events: List[Dict[str, Any]] = Field(default_factory=list, description="Recent scaling events")
    last_scale_up_time: Optional[datetime] = Field(None, description="Last scale up time")
    last_scale_down_time: Optional[datetime] = Field(None, description="Last scale down time")
    
    # Status
    status: str = Field(default="active", description="Autoscaler status")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def needs_scale_up(self) -> bool:
        """Check if cluster needs to scale up."""
        return self.unschedulable_pods > 0
    
    def needs_scale_down(self) -> bool:
        """Check if cluster can scale down."""
        if not self.scale_down_enabled:
            return False
        
        # Simple logic: scale down if utilization is low
        if self.current_nodes > 0:
            utilization = (self.current_nodes - self.get_idle_nodes()) / self.current_nodes
            return utilization < self.scale_down_utilization_threshold
        
        return False
    
    def get_idle_nodes(self) -> int:
        """Get count of idle nodes (placeholder)."""
        # This would be implemented with actual node utilization data
        return max(0, self.current_nodes - self.desired_nodes)
    
    def record_scaling_event(self, event_type: str, old_count: int, new_count: int, reason: str) -> None:
        """Record cluster scaling event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "old_node_count": old_count,
            "new_node_count": new_count,
            "reason": reason
        }
        
        self.recent_scaling_events.append(event)
        
        # Keep only last 100 events
        if len(self.recent_scaling_events) > 100:
            self.recent_scaling_events = self.recent_scaling_events[-100:]
        
        if event_type == "scale_up":
            self.last_scale_up_time = datetime.utcnow()
        elif event_type == "scale_down":
            self.last_scale_down_time = datetime.utcnow()
        
        self.updated_at = datetime.utcnow()


class PredictiveScalingPolicy(BaseModel):
    """
    Predictive scaling policy using ML models.
    
    Implements predictive auto-scaling based on historical
    patterns and forecasted demand using machine learning.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Policy identifier")
    
    # Policy identification
    name: str = Field(..., description="Policy name")
    target_resource: str = Field(..., description="Target resource")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Prediction configuration
    prediction_model_type: str = Field(default="linear", description="ML model type")
    prediction_horizon_minutes: int = Field(default=60, description="Prediction horizon")
    historical_data_days: int = Field(default=7, description="Historical data window")
    
    # Model configuration
    model_config: Dict[str, Any] = Field(default_factory=dict, description="ML model configuration")
    feature_columns: List[str] = Field(default_factory=list, description="Feature columns")
    target_metric: str = Field(..., description="Target metric to predict")
    
    # Training configuration
    retrain_interval_hours: int = Field(default=24, description="Model retrain interval")
    last_trained_at: Optional[datetime] = Field(None, description="Last training time")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy score")
    
    # Scaling thresholds
    scale_up_threshold: float = Field(default=0.8, description="Scale up threshold")
    scale_down_threshold: float = Field(default=0.3, description="Scale down threshold")
    confidence_threshold: float = Field(default=0.7, description="Prediction confidence threshold")
    
    # Current predictions
    current_prediction: Optional[Dict[str, Any]] = Field(None, description="Current prediction")
    prediction_confidence: Optional[float] = Field(None, description="Prediction confidence")
    next_scaling_action: Optional[str] = Field(None, description="Predicted next scaling action")
    
    # Status
    status: str = Field(default="training", description="Policy status")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def needs_retraining(self) -> bool:
        """Check if model needs retraining."""
        if not self.last_trained_at:
            return True
        
        time_since_training = datetime.utcnow() - self.last_trained_at
        return time_since_training >= timedelta(hours=self.retrain_interval_hours)
    
    def is_prediction_confident(self) -> bool:
        """Check if current prediction is confident enough."""
        return (
            self.prediction_confidence is not None and
            self.prediction_confidence >= self.confidence_threshold
        )
    
    def update_prediction(self, prediction: Dict[str, Any], confidence: float) -> None:
        """Update current prediction."""
        self.current_prediction = prediction
        self.prediction_confidence = confidence
        
        # Determine next scaling action based on prediction
        predicted_value = prediction.get("value", 0)
        if predicted_value > self.scale_up_threshold:
            self.next_scaling_action = "scale_up"
        elif predicted_value < self.scale_down_threshold:
            self.next_scaling_action = "scale_down"
        else:
            self.next_scaling_action = "none"
        
        self.updated_at = datetime.utcnow()
    
    def record_training(self, accuracy: float) -> None:
        """Record model training completion."""
        self.last_trained_at = datetime.utcnow()
        self.model_accuracy = accuracy
        self.status = "active" if accuracy > 0.6 else "poor_accuracy"
        self.updated_at = datetime.utcnow()


class AutoScalingProfile(BaseModel):
    """
    Auto-scaling profile for coordinated scaling policies.
    
    Combines multiple scaling policies and provides
    coordinated scaling decisions across different dimensions.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Profile identifier")
    
    # Profile identification
    name: str = Field(..., description="Profile name")
    description: str = Field(default="", description="Profile description")
    tenant_id: UUID = Field(..., description="Owning tenant")
    
    # Associated policies
    hpa_policies: List[UUID] = Field(default_factory=list, description="HPA policy IDs")
    vpa_policies: List[UUID] = Field(default_factory=list, description="VPA policy IDs")
    cluster_policies: List[UUID] = Field(default_factory=list, description="Cluster scaling policy IDs")
    predictive_policies: List[UUID] = Field(default_factory=list, description="Predictive policy IDs")
    
    # Coordination settings
    priority_order: List[str] = Field(default_factory=list, description="Policy priority order")
    conflict_resolution: str = Field(default="conservative", description="Conflict resolution strategy")
    
    # Global settings
    global_min_replicas: Optional[int] = Field(None, description="Global minimum replicas")
    global_max_replicas: Optional[int] = Field(None, description="Global maximum replicas")
    global_cooldown_seconds: int = Field(default=300, description="Global cooldown period")
    
    # Status
    status: str = Field(default="active", description="Profile status")
    last_scaling_decision: Optional[Dict[str, Any]] = Field(None, description="Last scaling decision")
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def add_policy(self, policy_type: str, policy_id: UUID) -> None:
        """Add scaling policy to profile."""
        if policy_type == "hpa" and policy_id not in self.hpa_policies:
            self.hpa_policies.append(policy_id)
        elif policy_type == "vpa" and policy_id not in self.vpa_policies:
            self.vpa_policies.append(policy_id)
        elif policy_type == "cluster" and policy_id not in self.cluster_policies:
            self.cluster_policies.append(policy_id)
        elif policy_type == "predictive" and policy_id not in self.predictive_policies:
            self.predictive_policies.append(policy_id)
        
        self.updated_at = datetime.utcnow()
    
    def get_total_policies(self) -> int:
        """Get total number of policies."""
        return (
            len(self.hpa_policies) +
            len(self.vpa_policies) +
            len(self.cluster_policies) +
            len(self.predictive_policies)
        )
    
    def record_scaling_decision(self, decision: Dict[str, Any]) -> None:
        """Record scaling decision."""
        self.last_scaling_decision = {
            **decision,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.updated_at = datetime.utcnow()