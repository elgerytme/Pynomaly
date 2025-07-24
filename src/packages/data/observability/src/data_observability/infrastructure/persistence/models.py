"""SQLAlchemy models for data observability."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .database import Base


class DataAssetModel(Base):
    """SQLAlchemy model for DataAsset."""
    
    __tablename__ = "data_assets"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    asset_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Location and format
    location: Mapped[str] = mapped_column(String(512), nullable=False)
    format: Mapped[str] = mapped_column(String(50), nullable=False)

    # Schema information
    schema_info: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Metadata
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    business_terms: Mapped[Optional[List[str]]] = mapped_column(JSON, default=list)
    
    # Ownership and classification
    owner: Mapped[Optional[str]] = mapped_column(String(255))
    steward: Mapped[Optional[str]] = mapped_column(String(255))
    classification: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Quality metrics
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    freshness_score: Mapped[Optional[float]] = mapped_column(Float)
    completeness_score: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class DataLineageGraphModel(Base):
    """SQLAlchemy model for a data lineage graph."""
    
    __tablename__ = "lineage_graphs"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    namespace: Mapped[str] = mapped_column(String(255), nullable=False, default="default", index=True)
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    nodes: Mapped[List["DataLineageNodeModel"]] = relationship(
        "DataLineageNodeModel", back_populates="lineage_graph", cascade="all, delete-orphan"
    )
    edges: Mapped[List["DataLineageEdgeModel"]] = relationship(
        "DataLineageEdgeModel", back_populates="lineage_graph", cascade="all, delete-orphan"
    )


class DataLineageNodeModel(Base):
    """SQLAlchemy model for data lineage nodes."""
    
    __tablename__ = "lineage_nodes"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    lineage_graph_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("lineage_graphs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False)
    asset_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("data_assets.id", ondelete="SET NULL"), # Use SET NULL if asset is deleted
        nullable=True,
        index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text)
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationships
    lineage_graph: Mapped["DataLineageGraphModel"] = relationship(
        "DataLineageGraphModel", back_populates="nodes"
    )
    asset: Mapped[Optional["DataAssetModel"]] = relationship("DataAssetModel")


class DataLineageEdgeModel(Base):
    """SQLAlchemy model for data lineage edges."""
    
    __tablename__ = "lineage_edges"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    lineage_graph_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("lineage_graphs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    source_node_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("lineage_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    target_node_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("lineage_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    lineage_graph: Mapped["DataLineageGraphModel"] = relationship(
        "DataLineageGraphModel", back_populates="edges"
    )
    source_node: Mapped["DataLineageNodeModel"] = relationship(
        "DataLineageNodeModel", 
        foreign_keys=[source_node_id],
        viewonly=True # Prevent SQLAlchemy from trying to manage this side of the relationship
    )
    target_node: Mapped["DataLineageNodeModel"] = relationship(
        "DataLineageNodeModel", 
        foreign_keys=[target_node_id],
        viewonly=True # Prevent SQLAlchemy from trying to manage this side of the relationship
    )


class PipelineHealthModel(Base):
    """SQLAlchemy model for pipeline health monitoring."""
    
    __tablename__ = "pipeline_health"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    pipeline_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True, unique=True)
    pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Health metrics
    overall_health_score: Mapped[float] = mapped_column(Float, nullable=False)
    execution_success_rate: Mapped[float] = mapped_column(Float)
    data_quality_score: Mapped[float] = mapped_column(Float)
    performance_score: Mapped[float] = mapped_column(Float)
    availability_percentage: Mapped[float] = mapped_column(Float, default=100.0)
    
    # Status information
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    last_run_status: Mapped[Optional[str]] = mapped_column(String(50))
    last_successful_run: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_failed_run: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_execution: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    execution_duration: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metrics
    total_runs: Mapped[int] = mapped_column(default=0)
    successful_runs: Mapped[int] = mapped_column(default=0)
    failed_runs: Mapped[int] = mapped_column(default=0)
    avg_execution_time: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict) # Keep for generic metadata
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )

    # Relationships
    metrics: Mapped[List["PipelineMetricModel"]] = relationship(
        "PipelineMetricModel", back_populates="pipeline_health", cascade="all, delete-orphan"
    )
    alerts: Mapped[List["PipelineAlertModel"]] = relationship(
        "PipelineAlertModel", back_populates="pipeline_health", cascade="all, delete-orphan"
    )


class PipelineMetricModel(Base):
    """SQLAlchemy model for pipeline metrics."""
    
    __tablename__ = "pipeline_metrics"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    pipeline_health_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("pipeline_health.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    labels: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    source: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Relationship
    pipeline_health: Mapped["PipelineHealthModel"] = relationship(
        "PipelineHealthModel", back_populates="metrics"
    )


class PipelineAlertModel(Base):
    """SQLAlchemy model for pipeline alerts."""
    
    __tablename__ = "pipeline_alerts"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    pipeline_health_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("pipeline_health.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    metric_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Alert details
    triggered_by: Mapped[Optional[str]] = mapped_column(String(255))
    current_value: Mapped[Optional[float]] = mapped_column(Float)
    threshold_value: Mapped[Optional[float]] = mapped_column(Float)
    
    # Action tracking
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(255))
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationship
    pipeline_health: Mapped["PipelineHealthModel"] = relationship(
        "PipelineHealthModel", back_populates="alerts"
    )