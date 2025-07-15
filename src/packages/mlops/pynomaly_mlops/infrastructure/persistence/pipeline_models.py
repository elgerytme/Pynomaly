"""SQLAlchemy Models for Pipeline Entities

ORM models for pipeline and pipeline run persistence with comprehensive
relationship management and indexing for performance.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, Boolean, JSON,
    ForeignKey, Index, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from pynomaly_mlops.domain.entities.pipeline import (
    PipelineStatus, StepStatus, StepType
)

Base = declarative_base()


class PipelineORM(Base):
    """SQLAlchemy model for Pipeline entities."""
    
    __tablename__ = "pipelines"
    
    # Primary key
    id = Column(PostgreSQL_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Basic information
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    
    # Pipeline state
    status = Column(SQLEnum(PipelineStatus), nullable=False, default=PipelineStatus.DRAFT)
    
    # Execution tracking
    current_run_id = Column(PostgreSQL_UUID(as_uuid=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=func.now())
    created_by = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=False, default=list)
    
    # Configuration
    max_parallel_steps = Column(Integer, nullable=False, default=5)
    global_timeout_minutes = Column(Integer, nullable=False, default=480)
    
    # Schedule configuration (stored as JSON)
    schedule_config = Column(JSON, nullable=True)
    
    # Relationships
    steps = relationship("PipelineStepORM", back_populates="pipeline", cascade="all, delete-orphan")
    runs = relationship("PipelineRunORM", back_populates="pipeline", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_pipeline_name', 'name'),
        Index('idx_pipeline_status', 'status'),
        Index('idx_pipeline_created_at', 'created_at'),
        Index('idx_pipeline_tags', 'tags', postgresql_using='gin'),
        UniqueConstraint('name', 'version', name='uq_pipeline_name_version'),
    )
    
    @validates('tags')
    def validate_tags(self, key, value):
        """Ensure tags is a list."""
        if isinstance(value, str):
            return [value]
        elif value is None:
            return []
        return list(value) if value else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value if self.status else None,
            "current_run_id": str(self.current_run_id) if self.current_run_id else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "tags": self.tags or [],
            "max_parallel_steps": self.max_parallel_steps,
            "global_timeout_minutes": self.global_timeout_minutes,
            "schedule_config": self.schedule_config,
            "step_count": len(self.steps) if self.steps else 0
        }


class PipelineStepORM(Base):
    """SQLAlchemy model for PipelineStep entities."""
    
    __tablename__ = "pipeline_steps"
    
    # Primary key
    id = Column(PostgreSQL_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to pipeline
    pipeline_id = Column(PostgreSQL_UUID(as_uuid=True), ForeignKey('pipelines.id'), nullable=False)
    
    # Basic information
    name = Column(String(100), nullable=False)
    step_type = Column(SQLEnum(StepType), nullable=False)
    description = Column(Text, nullable=True)
    
    # Execution configuration
    command = Column(Text, nullable=False)
    working_directory = Column(String(500), nullable=True)
    environment_variables = Column(JSON, nullable=False, default=dict)
    parameters = Column(JSON, nullable=False, default=dict)
    
    # Dependencies (stored as JSON array of UUIDs)
    depends_on = Column(JSON, nullable=False, default=list)
    
    # Resource requirements (stored as JSON)
    resource_requirements = Column(JSON, nullable=False, default=dict)
    retry_policy = Column(JSON, nullable=False, default=dict)
    
    # Runtime state
    status = Column(SQLEnum(StepStatus), nullable=False, default=StepStatus.PENDING)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    
    # Execution results
    exit_code = Column(Integer, nullable=True)
    stdout = Column(Text, nullable=True)
    stderr = Column(Text, nullable=True)
    artifacts = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=func.now())
    created_by = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=False, default=list)
    
    # Relationships
    pipeline = relationship("PipelineORM", back_populates="steps")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_step_pipeline_id', 'pipeline_id'),
        Index('idx_step_status', 'status'),
        Index('idx_step_type', 'step_type'),
        Index('idx_step_created_at', 'created_at'),
    )
    
    @validates('depends_on')
    def validate_depends_on(self, key, value):
        """Ensure depends_on is a list of UUID strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value) if value else []
    
    @validates('tags')
    def validate_tags(self, key, value):
        """Ensure tags is a list."""
        if isinstance(value, str):
            return [value]
        elif value is None:
            return []
        return list(value) if value else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "pipeline_id": str(self.pipeline_id),
            "name": self.name,
            "step_type": self.step_type.value if self.step_type else None,
            "description": self.description,
            "command": self.command,
            "working_directory": self.working_directory,
            "environment_variables": self.environment_variables or {},
            "parameters": self.parameters or {},
            "depends_on": self.depends_on or [],
            "resource_requirements": self.resource_requirements or {},
            "retry_policy": self.retry_policy or {},
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "attempt_count": self.attempt_count,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "artifacts": self.artifacts or {},
            "metrics": self.metrics or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "tags": self.tags or []
        }


class PipelineRunORM(Base):
    """SQLAlchemy model for PipelineRun entities."""
    
    __tablename__ = "pipeline_runs"
    
    # Primary key
    id = Column(PostgreSQL_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign key to pipeline
    pipeline_id = Column(PostgreSQL_UUID(as_uuid=True), ForeignKey('pipelines.id'), nullable=False)
    pipeline_version = Column(String(50), nullable=False)
    
    # Execution tracking
    status = Column(SQLEnum(PipelineStatus), nullable=False, default=PipelineStatus.RUNNING)
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Step execution state (stored as JSON)
    step_runs = Column(JSON, nullable=False, default=dict)
    
    # Execution context
    triggered_by = Column(String(100), nullable=True)
    trigger_type = Column(String(50), nullable=False, default="manual")
    parameters = Column(JSON, nullable=False, default=dict)
    
    # Results
    artifacts = Column(JSON, nullable=False, default=dict)
    metrics = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    pipeline = relationship("PipelineORM", back_populates="runs")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_run_pipeline_id', 'pipeline_id'),
        Index('idx_run_status', 'status'),
        Index('idx_run_started_at', 'started_at'),
        Index('idx_run_trigger_type', 'trigger_type'),
        Index('idx_run_triggered_by', 'triggered_by'),
    )
    
    @validates('parameters')
    def validate_parameters(self, key, value):
        """Ensure parameters is a dict."""
        return value if isinstance(value, dict) else {}
    
    @validates('artifacts')
    def validate_artifacts(self, key, value):
        """Ensure artifacts is a dict."""
        return value if isinstance(value, dict) else {}
    
    @validates('metrics')
    def validate_metrics(self, key, value):
        """Ensure metrics is a dict."""
        return value if isinstance(value, dict) else {}
    
    @validates('step_runs')
    def validate_step_runs(self, key, value):
        """Ensure step_runs is a dict."""
        return value if isinstance(value, dict) else {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "pipeline_id": str(self.pipeline_id),
            "pipeline_version": self.pipeline_version,
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "step_runs": self.step_runs or {},
            "triggered_by": self.triggered_by,
            "trigger_type": self.trigger_type,
            "parameters": self.parameters or {},
            "artifacts": self.artifacts or {},
            "metrics": self.metrics or {},
            "execution_duration": (
                (self.completed_at - self.started_at).total_seconds()
                if self.started_at and self.completed_at else None
            )
        }


class PipelineLineageORM(Base):
    """SQLAlchemy model for Pipeline lineage tracking."""
    
    __tablename__ = "pipeline_lineage"
    
    # Primary key
    id = Column(PostgreSQL_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Lineage relationship
    parent_pipeline_id = Column(PostgreSQL_UUID(as_uuid=True), ForeignKey('pipelines.id'), nullable=False)
    child_pipeline_id = Column(PostgreSQL_UUID(as_uuid=True), ForeignKey('pipelines.id'), nullable=False)
    
    # Relationship type
    relationship_type = Column(String(50), nullable=False)  # triggers, depends_on, produces_for, etc.
    
    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    
    # Relationships
    parent_pipeline = relationship("PipelineORM", foreign_keys=[parent_pipeline_id])
    child_pipeline = relationship("PipelineORM", foreign_keys=[child_pipeline_id])
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_lineage_parent', 'parent_pipeline_id'),
        Index('idx_lineage_child', 'child_pipeline_id'),
        Index('idx_lineage_type', 'relationship_type'),
        UniqueConstraint('parent_pipeline_id', 'child_pipeline_id', 'relationship_type', 
                        name='uq_pipeline_lineage'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "parent_pipeline_id": str(self.parent_pipeline_id),
            "child_pipeline_id": str(self.child_pipeline_id),
            "relationship_type": self.relationship_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "description": self.description
        }