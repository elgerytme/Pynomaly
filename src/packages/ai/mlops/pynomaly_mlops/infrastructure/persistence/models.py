"""SQLAlchemy ORM Models

Database models for the MLOps domain entities.
"""

from datetime import datetime
from typing import Dict, Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, JSON, Text, Integer, Float, Boolean, 
    ForeignKey, Table, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

Base = declarative_base()


# Association tables for many-to-many relationships
model_tags_table = Table(
    'model_tags',
    Base.metadata,
    Column('model_id', PGUUID(as_uuid=True), ForeignKey('models.id'), primary_key=True),
    Column('tag_name', String(50), primary_key=True),
    Index('idx_model_tags_model_id', 'model_id'),
    Index('idx_model_tags_tag_name', 'tag_name')
)

experiment_tags_table = Table(
    'experiment_tags',
    Base.metadata,
    Column('experiment_id', PGUUID(as_uuid=True), ForeignKey('experiments.id'), primary_key=True),
    Column('tag_name', String(50), primary_key=True)
)


class ModelORM(Base):
    """SQLAlchemy model for Model entity."""
    
    __tablename__ = 'models'
    
    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    
    # Semantic version components
    version_major = Column(Integer, nullable=False)
    version_minor = Column(Integer, nullable=False)
    version_patch = Column(Integer, nullable=False)
    version_prerelease = Column(String(50), nullable=True)
    version_build = Column(String(50), nullable=True)
    
    # Model metadata
    model_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default='development', index=True)
    description = Column(Text, nullable=True)
    
    # Lifecycle tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False, index=True)
    
    # Model storage
    artifact_uri = Column(String(500), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    checksum = Column(String(64), nullable=True)
    
    # Performance metrics (JSON field)
    metrics = Column(JSON, nullable=True)
    validation_metrics = Column(JSON, nullable=True)
    
    # Model configuration
    hyperparameters = Column(JSON, nullable=True)
    feature_schema = Column(JSON, nullable=True)
    training_config = Column(JSON, nullable=True)
    
    # Deployment tracking
    deployment_count = Column(Integer, default=0, nullable=False)
    current_stage = Column(String(20), nullable=True)
    
    # Lineage
    parent_model_id = Column(PGUUID(as_uuid=True), ForeignKey('models.id'), nullable=True)
    experiment_id = Column(PGUUID(as_uuid=True), ForeignKey('experiments.id'), nullable=True)
    
    # Relationships
    parent_model = relationship("ModelORM", remote_side=[id], backref="child_models")
    experiment = relationship("ExperimentORM", backref="models")
    deployments = relationship("DeploymentORM", back_populates="model")
    
    # Many-to-many relationships
    tags = relationship(
        "String",
        secondary=model_tags_table,
        passive_deletes=True
    )
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('name', 'version_major', 'version_minor', 'version_patch', 
                        'version_prerelease', 'version_build', name='uq_model_version'),
        Index('idx_models_name_version', 'name', 'version_major', 'version_minor', 'version_patch'),
        Index('idx_models_created_at', 'created_at'),
        Index('idx_models_status_type', 'status', 'model_type'),
    )
    
    def __repr__(self):
        version = f"{self.version_major}.{self.version_minor}.{self.version_patch}"
        if self.version_prerelease:
            version += f"-{self.version_prerelease}"
        if self.version_build:
            version += f"+{self.version_build}"
        return f"<ModelORM(name='{self.name}', version='{version}', status='{self.status}')>"


class ExperimentORM(Base):
    """SQLAlchemy model for Experiment entity."""
    
    __tablename__ = 'experiments'
    
    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Experiment metadata
    status = Column(String(20), nullable=False, default='active', index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False, index=True)
    
    # Experiment configuration
    objective = Column(String(100), nullable=True)
    configuration = Column(JSON, nullable=True)
    
    # Tracking
    run_count = Column(Integer, default=0, nullable=False)
    best_run_id = Column(PGUUID(as_uuid=True), ForeignKey('experiment_runs.id'), nullable=True)
    
    # Relationships
    runs = relationship("ExperimentRunORM", back_populates="experiment", 
                       cascade="all, delete-orphan")
    best_run = relationship("ExperimentRunORM", foreign_keys=[best_run_id], post_update=True)
    
    # Many-to-many relationships  
    tags = relationship(
        "String",
        secondary=experiment_tags_table,
        passive_deletes=True
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_experiments_created_at', 'created_at'),
        Index('idx_experiments_status_name', 'status', 'name'),
    )
    
    def __repr__(self):
        return f"<ExperimentORM(name='{self.name}', status='{self.status}', runs={self.run_count})>"


class ExperimentRunORM(Base):
    """SQLAlchemy model for ExperimentRun entity."""
    
    __tablename__ = 'experiment_runs'
    
    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(PGUUID(as_uuid=True), ForeignKey('experiments.id'), nullable=False)
    name = Column(String(255), nullable=True)
    
    # Run metadata
    status = Column(String(20), nullable=False, default='running', index=True)
    start_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(255), nullable=False)
    
    # Run data
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    artifacts = Column(JSON, nullable=True)
    logs = Column(Text, nullable=True)
    
    # Model reference
    model_id = Column(PGUUID(as_uuid=True), ForeignKey('models.id'), nullable=True)
    
    # Relationships
    experiment = relationship("ExperimentORM", back_populates="runs")
    model = relationship("ModelORM")
    
    # Indexes
    __table_args__ = (
        Index('idx_experiment_runs_experiment_id', 'experiment_id'),
        Index('idx_experiment_runs_start_time', 'start_time'),
        Index('idx_experiment_runs_status', 'status'),
    )
    
    def __repr__(self):
        return f"<ExperimentRunORM(id='{self.id}', status='{self.status}')>"


class PipelineORM(Base):
    """SQLAlchemy model for Pipeline entity."""
    
    __tablename__ = 'pipelines'
    
    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Pipeline metadata
    status = Column(String(20), nullable=False, default='draft', index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False)
    
    # Pipeline configuration
    configuration = Column(JSON, nullable=True)
    schedule = Column(String(100), nullable=True)
    timeout_seconds = Column(Integer, nullable=True)
    
    # Execution tracking
    execution_count = Column(Integer, default=0, nullable=False)
    last_execution_id = Column(String(255), nullable=True)
    last_execution_status = Column(String(20), nullable=True)
    last_execution_time = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    steps = relationship("PipelineStepORM", back_populates="pipeline", 
                        cascade="all, delete-orphan", order_by="PipelineStepORM.order")
    
    # Indexes
    __table_args__ = (
        Index('idx_pipelines_created_at', 'created_at'),
        Index('idx_pipelines_status_name', 'status', 'name'),
    )
    
    def __repr__(self):
        return f"<PipelineORM(name='{self.name}', status='{self.status}', steps={len(self.steps)})>"


class PipelineStepORM(Base):
    """SQLAlchemy model for PipelineStep entity."""
    
    __tablename__ = 'pipeline_steps'
    
    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    pipeline_id = Column(PGUUID(as_uuid=True), ForeignKey('pipelines.id'), nullable=False)
    name = Column(String(255), nullable=False)
    
    # Step configuration
    step_type = Column(String(50), nullable=False, index=True)
    order = Column(Integer, nullable=False)
    configuration = Column(JSON, nullable=True)
    dependencies = Column(JSON, nullable=True)  # List of step names this depends on
    
    # Execution settings
    timeout_seconds = Column(Integer, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Status tracking
    enabled = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    pipeline = relationship("PipelineORM", back_populates="steps")
    
    # Indexes
    __table_args__ = (
        Index('idx_pipeline_steps_pipeline_id', 'pipeline_id'),
        Index('idx_pipeline_steps_order', 'pipeline_id', 'order'),
        UniqueConstraint('pipeline_id', 'name', name='uq_pipeline_step_name'),
    )
    
    def __repr__(self):
        return f"<PipelineStepORM(name='{self.name}', type='{self.step_type}', order={self.order})>"


class DeploymentORM(Base):
    """SQLAlchemy model for Deployment entity."""
    
    __tablename__ = 'deployments'
    
    # Primary fields
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_id = Column(PGUUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    name = Column(String(255), nullable=False, index=True)
    
    # Deployment configuration
    environment = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default='pending', index=True)
    endpoint_url = Column(String(500), nullable=True)
    
    # Infrastructure configuration
    scaling_config = Column(JSON, nullable=True)
    health_check_config = Column(JSON, nullable=True)
    
    # Deployment metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String(255), nullable=False)
    
    # Deployment strategy
    deployment_strategy = Column(String(50), nullable=False, default='rolling')
    rollback_config = Column(JSON, nullable=True)
    
    # Monitoring
    health_status = Column(String(20), nullable=True, index=True)
    last_health_check = Column(DateTime(timezone=True), nullable=True)
    performance_metrics = Column(JSON, nullable=True)
    
    # Relationships
    model = relationship("ModelORM", back_populates="deployments")
    
    # Indexes
    __table_args__ = (
        Index('idx_deployments_model_id', 'model_id'),
        Index('idx_deployments_environment', 'environment'),
        Index('idx_deployments_created_at', 'created_at'),
        Index('idx_deployments_status_env', 'status', 'environment'),
        UniqueConstraint('name', 'environment', name='uq_deployment_name_env'),
    )
    
    def __repr__(self):
        return f"<DeploymentORM(name='{self.name}', environment='{self.environment}', status='{self.status}')>"