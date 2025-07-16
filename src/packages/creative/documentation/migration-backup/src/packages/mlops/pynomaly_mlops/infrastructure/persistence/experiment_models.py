"""SQLAlchemy models for Experiment entities."""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, String, DateTime, Float, JSON, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from .models import Base


class ExperimentORM(Base):
    """SQLAlchemy model for Experiment entity."""
    
    __tablename__ = "experiments"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="active")
    
    # Relationships
    runs = relationship("ExperimentRunORM", back_populates="experiment", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<ExperimentORM(id={self.id}, name='{self.name}', status='{self.status}')>"


class ExperimentRunORM(Base):
    """SQLAlchemy model for ExperimentRun entity."""
    
    __tablename__ = "experiment_runs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(PGUUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    parameters = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict) 
    artifacts = Column(JSON, nullable=False, default=dict)
    tags = Column(JSON, nullable=False, default=dict)
    status = Column(String(50), nullable=False, default="running")
    start_time = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(255), nullable=False)
    parent_run_id = Column(PGUUID(as_uuid=True), ForeignKey("experiment_runs.id"), nullable=True)
    source_version = Column(String(255), nullable=True)
    entry_point = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationships
    experiment = relationship("ExperimentORM", back_populates="runs")
    parent_run = relationship("ExperimentRunORM", remote_side=[id])
    child_runs = relationship("ExperimentRunORM", back_populates="parent_run")
    
    def __repr__(self) -> str:
        return f"<ExperimentRunORM(id={self.id}, experiment_id={self.experiment_id}, status='{self.status}')>"


class ExperimentMetricORM(Base):
    """SQLAlchemy model for time-series experiment metrics."""
    
    __tablename__ = "experiment_metrics"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(PGUUID(as_uuid=True), ForeignKey("experiment_runs.id"), nullable=False, index=True)
    key = Column(String(255), nullable=False, index=True)
    value = Column(Float, nullable=False)
    step = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        {"schema": None}  # Can be customized for different schemas
    )
    
    def __repr__(self) -> str:
        return f"<ExperimentMetricORM(run_id={self.run_id}, key='{self.key}', value={self.value}, step={self.step})>"


class ExperimentComparisonORM(Base):
    """SQLAlchemy model for saved experiment comparisons."""
    
    __tablename__ = "experiment_comparisons"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    run_ids = Column(JSON, nullable=False)  # List of run IDs being compared
    comparison_config = Column(JSON, nullable=False, default=dict)  # Configuration for comparison
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    created_by = Column(String(255), nullable=False)
    is_public = Column(Boolean, nullable=False, default=False)
    
    def __repr__(self) -> str:
        return f"<ExperimentComparisonORM(id={self.id}, name='{self.name}', run_count={len(self.run_ids or [])})>"