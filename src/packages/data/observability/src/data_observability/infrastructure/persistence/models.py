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


class DataLineageNodeModel(Base):
    """SQLAlchemy model for data lineage nodes."""
    
    __tablename__ = "lineage_nodes"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    asset_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("data_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    node_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationship
    asset: Mapped["DataAssetModel"] = relationship("DataAssetModel")


class DataLineageEdgeModel(Base):
    """SQLAlchemy model for data lineage edges."""
    
    __tablename__ = "lineage_edges"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
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
    source_node: Mapped["DataLineageNodeModel"] = relationship(
        "DataLineageNodeModel", 
        foreign_keys=[source_node_id]
    )
    target_node: Mapped["DataLineageNodeModel"] = relationship(
        "DataLineageNodeModel", 
        foreign_keys=[target_node_id]
    )


class PipelineHealthModel(Base):
    """SQLAlchemy model for pipeline health monitoring."""
    
    __tablename__ = "pipeline_health"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    pipeline_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    pipeline_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Health metrics
    overall_health_score: Mapped[float] = mapped_column(Float, nullable=False)
    execution_success_rate: Mapped[float] = mapped_column(Float)
    data_quality_score: Mapped[float] = mapped_column(Float)
    performance_score: Mapped[float] = mapped_column(Float)
    
    # Status information
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    last_run_status: Mapped[Optional[str]] = mapped_column(String(50))
    last_successful_run: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_failed_run: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Metrics
    total_runs: Mapped[int] = mapped_column(default=0)
    successful_runs: Mapped[int] = mapped_column(default=0)
    failed_runs: Mapped[int] = mapped_column(default=0)
    avg_execution_time: Mapped[Optional[float]] = mapped_column(Float)
    
    # Metadata
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )


class QualityPredictionModel(Base):
    """SQLAlchemy model for quality predictions."""
    
    __tablename__ = "quality_predictions"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    asset_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("data_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Prediction results
    predicted_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Model information
    model_version: Mapped[str] = mapped_column(String(50))
    model_features: Mapped[Optional[Dict]] = mapped_column(JSON)
    
    # Validation
    actual_score: Mapped[Optional[float]] = mapped_column(Float)
    validation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Metadata
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    prediction_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Relationship
    asset: Mapped["DataAssetModel"] = relationship("DataAssetModel")


class QualityAlertModel(Base):
    """SQLAlchemy model for quality alerts."""
    
    __tablename__ = "quality_alerts"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    asset_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("data_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Alert information
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(255))
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Resolution
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_by: Mapped[Optional[str]] = mapped_column(String(255))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    metadata: Mapped[Optional[Dict]] = mapped_column(JSON, default=dict)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        onupdate=func.now()
    )
    
    # Relationship
    asset: Mapped["DataAssetModel"] = relationship("DataAssetModel")