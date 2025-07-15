"""SQLAlchemy Pipeline Repository Implementation

Production-ready pipeline repository with comprehensive search, filtering,
and lineage tracking capabilities.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from uuid import UUID

from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload, joinedload

from pynomaly_mlops.domain.entities.pipeline import (
    Pipeline, PipelineStep, PipelineRun, PipelineStatus, StepStatus, StepType,
    PipelineSchedule, ResourceRequirements, RetryPolicy
)
from pynomaly_mlops.domain.repositories.pipeline_repository import (
    PipelineRepository, PipelineRunRepository
)
from pynomaly_mlops.infrastructure.persistence.pipeline_models import (
    PipelineORM, PipelineStepORM, PipelineRunORM, PipelineLineageORM
)


class SqlAlchemyPipelineRepository(PipelineRepository):
    """SQLAlchemy implementation of pipeline repository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, pipeline: Pipeline) -> Pipeline:
        """Save a pipeline to the database."""
        # Check if pipeline exists
        stmt = select(PipelineORM).where(PipelineORM.id == pipeline.id)
        result = await self.session.execute(stmt)
        pipeline_orm = result.scalar_one_or_none()
        
        if pipeline_orm:
            # Update existing pipeline
            pipeline_orm = self._update_pipeline_orm(pipeline_orm, pipeline)
        else:
            # Create new pipeline
            pipeline_orm = self._create_pipeline_orm(pipeline)
            self.session.add(pipeline_orm)
        
        # Handle steps
        await self._save_pipeline_steps(pipeline, pipeline_orm)
        
        await self.session.commit()
        await self.session.refresh(pipeline_orm)
        
        return self._orm_to_pipeline(pipeline_orm)
    
    async def find_by_id(self, pipeline_id: UUID) -> Optional[Pipeline]:
        """Find a pipeline by its ID."""
        stmt = (
            select(PipelineORM)
            .options(selectinload(PipelineORM.steps))
            .where(PipelineORM.id == pipeline_id)
        )
        result = await self.session.execute(stmt)
        pipeline_orm = result.scalar_one_or_none()
        
        if pipeline_orm:
            return self._orm_to_pipeline(pipeline_orm)
        return None
    
    async def find_by_name(self, name: str) -> Optional[Pipeline]:
        """Find a pipeline by its name."""
        stmt = (
            select(PipelineORM)
            .options(selectinload(PipelineORM.steps))
            .where(PipelineORM.name == name)
            .order_by(desc(PipelineORM.created_at))
        )
        result = await self.session.execute(stmt)
        pipeline_orm = result.scalar_one_or_none()
        
        if pipeline_orm:
            return self._orm_to_pipeline(pipeline_orm)
        return None
    
    async def find_all(self) -> List[Pipeline]:
        """Find all pipelines."""
        stmt = (
            select(PipelineORM)
            .options(selectinload(PipelineORM.steps))
            .order_by(desc(PipelineORM.created_at))
        )
        result = await self.session.execute(stmt)
        pipeline_orms = result.scalars().all()
        
        return [self._orm_to_pipeline(orm) for orm in pipeline_orms]
    
    async def find_by_status(self, status: PipelineStatus) -> List[Pipeline]:
        """Find pipelines by status."""
        stmt = (
            select(PipelineORM)
            .options(selectinload(PipelineORM.steps))
            .where(PipelineORM.status == status)
            .order_by(desc(PipelineORM.created_at))
        )
        result = await self.session.execute(stmt)
        pipeline_orms = result.scalars().all()
        
        return [self._orm_to_pipeline(orm) for orm in pipeline_orms]
    
    async def find_by_tags(self, tags: List[str]) -> List[Pipeline]:
        """Find pipelines by tags."""
        # PostgreSQL JSON contains query
        conditions = [PipelineORM.tags.op('@>')([tag]) for tag in tags]
        
        stmt = (
            select(PipelineORM)
            .options(selectinload(PipelineORM.steps))
            .where(or_(*conditions))
            .order_by(desc(PipelineORM.created_at))
        )
        result = await self.session.execute(stmt)
        pipeline_orms = result.scalars().all()
        
        return [self._orm_to_pipeline(orm) for orm in pipeline_orms]
    
    async def search(
        self,
        name_pattern: Optional[str] = None,
        status: Optional[PipelineStatus] = None,
        tags: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Pipeline]:
        """Search pipelines with multiple criteria."""
        stmt = select(PipelineORM).options(selectinload(PipelineORM.steps))
        
        # Apply filters
        conditions = []
        
        if name_pattern:
            conditions.append(PipelineORM.name.ilike(f"%{name_pattern}%"))
        
        if status:
            conditions.append(PipelineORM.status == status)
        
        if tags:
            tag_conditions = [PipelineORM.tags.op('@>')([tag]) for tag in tags]
            conditions.append(or_(*tag_conditions))
        
        if created_after:
            conditions.append(PipelineORM.created_at >= created_after)
        
        if created_before:
            conditions.append(PipelineORM.created_at <= created_before)
        
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        # Apply ordering, limit, and offset
        stmt = stmt.order_by(desc(PipelineORM.created_at))
        
        if offset:
            stmt = stmt.offset(offset)
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        pipeline_orms = result.scalars().all()
        
        return [self._orm_to_pipeline(orm) for orm in pipeline_orms]
    
    async def delete(self, pipeline_id: UUID) -> bool:
        """Delete a pipeline from the database."""
        stmt = select(PipelineORM).where(PipelineORM.id == pipeline_id)
        result = await self.session.execute(stmt)
        pipeline_orm = result.scalar_one_or_none()
        
        if pipeline_orm:
            await self.session.delete(pipeline_orm)
            await self.session.commit()
            return True
        
        return False
    
    async def get_pipeline_history(
        self, 
        pipeline_id: UUID, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get execution history for a pipeline."""
        stmt = (
            select(PipelineRunORM)
            .where(PipelineRunORM.pipeline_id == pipeline_id)
            .order_by(desc(PipelineRunORM.started_at))
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        run_orms = result.scalars().all()
        
        return [orm.to_dict() for orm in run_orms]
    
    def _create_pipeline_orm(self, pipeline: Pipeline) -> PipelineORM:
        """Create a new PipelineORM from Pipeline entity."""
        schedule_config = None
        if pipeline.schedule:
            schedule_config = {
                "enabled": pipeline.schedule.enabled,
                "cron_expression": pipeline.schedule.cron_expression,
                "timezone": pipeline.schedule.timezone,
                "max_concurrent_runs": pipeline.schedule.max_concurrent_runs
            }
        
        return PipelineORM(
            id=pipeline.id,
            name=pipeline.name,
            description=pipeline.description,
            version=pipeline.version,
            status=pipeline.status,
            current_run_id=pipeline.current_run_id,
            started_at=pipeline.started_at,
            completed_at=pipeline.completed_at,
            created_at=pipeline.created_at,
            updated_at=pipeline.updated_at,
            created_by=pipeline.created_by,
            tags=list(pipeline.tags),
            max_parallel_steps=pipeline.max_parallel_steps,
            global_timeout_minutes=pipeline.global_timeout_minutes,
            schedule_config=schedule_config
        )
    
    def _update_pipeline_orm(self, pipeline_orm: PipelineORM, pipeline: Pipeline) -> PipelineORM:
        """Update existing PipelineORM with Pipeline entity data."""
        pipeline_orm.name = pipeline.name
        pipeline_orm.description = pipeline.description
        pipeline_orm.version = pipeline.version
        pipeline_orm.status = pipeline.status
        pipeline_orm.current_run_id = pipeline.current_run_id
        pipeline_orm.started_at = pipeline.started_at
        pipeline_orm.completed_at = pipeline.completed_at
        pipeline_orm.updated_at = pipeline.updated_at
        pipeline_orm.created_by = pipeline.created_by
        pipeline_orm.tags = list(pipeline.tags)
        pipeline_orm.max_parallel_steps = pipeline.max_parallel_steps
        pipeline_orm.global_timeout_minutes = pipeline.global_timeout_minutes
        
        # Update schedule config
        schedule_config = None
        if pipeline.schedule:
            schedule_config = {
                "enabled": pipeline.schedule.enabled,
                "cron_expression": pipeline.schedule.cron_expression,
                "timezone": pipeline.schedule.timezone,
                "max_concurrent_runs": pipeline.schedule.max_concurrent_runs
            }
        pipeline_orm.schedule_config = schedule_config
        
        return pipeline_orm
    
    async def _save_pipeline_steps(self, pipeline: Pipeline, pipeline_orm: PipelineORM) -> None:
        """Save pipeline steps."""
        # Get existing steps
        existing_step_ids = {step.id for step in pipeline_orm.steps}
        current_step_ids = set(pipeline.steps.keys())
        
        # Remove deleted steps
        steps_to_remove = existing_step_ids - current_step_ids
        for step_orm in pipeline_orm.steps:
            if step_orm.id in steps_to_remove:
                await self.session.delete(step_orm)
        
        # Add or update steps
        for step_id, step in pipeline.steps.items():
            step_orm = next((s for s in pipeline_orm.steps if s.id == step_id), None)
            
            if step_orm:
                # Update existing step
                self._update_step_orm(step_orm, step)
            else:
                # Create new step
                step_orm = self._create_step_orm(step, pipeline_orm.id)
                self.session.add(step_orm)
    
    def _create_step_orm(self, step: PipelineStep, pipeline_id: UUID) -> PipelineStepORM:
        """Create a new PipelineStepORM from PipelineStep entity."""
        return PipelineStepORM(
            id=step.id,
            pipeline_id=pipeline_id,
            name=step.name,
            step_type=step.step_type,
            description=step.description,
            command=step.command,
            working_directory=step.working_directory,
            environment_variables=step.environment_variables,
            parameters=step.parameters,
            depends_on=[str(dep_id) for dep_id in step.depends_on],
            resource_requirements=step.resource_requirements.dict(),
            retry_policy=step.retry_policy.dict(),
            status=step.status,
            started_at=step.started_at,
            completed_at=step.completed_at,
            attempt_count=step.attempt_count,
            exit_code=step.exit_code,
            stdout=step.stdout,
            stderr=step.stderr,
            artifacts=step.artifacts,
            metrics=step.metrics,
            created_at=step.created_at,
            updated_at=step.updated_at,
            created_by=step.created_by,
            tags=list(step.tags)
        )
    
    def _update_step_orm(self, step_orm: PipelineStepORM, step: PipelineStep) -> None:
        """Update existing PipelineStepORM with PipelineStep entity data."""
        step_orm.name = step.name
        step_orm.step_type = step.step_type
        step_orm.description = step.description
        step_orm.command = step.command
        step_orm.working_directory = step.working_directory
        step_orm.environment_variables = step.environment_variables
        step_orm.parameters = step.parameters
        step_orm.depends_on = [str(dep_id) for dep_id in step.depends_on]
        step_orm.resource_requirements = step.resource_requirements.dict()
        step_orm.retry_policy = step.retry_policy.dict()
        step_orm.status = step.status
        step_orm.started_at = step.started_at
        step_orm.completed_at = step.completed_at
        step_orm.attempt_count = step.attempt_count
        step_orm.exit_code = step.exit_code
        step_orm.stdout = step.stdout
        step_orm.stderr = step.stderr
        step_orm.artifacts = step.artifacts
        step_orm.metrics = step.metrics
        step_orm.updated_at = step.updated_at
        step_orm.created_by = step.created_by
        step_orm.tags = list(step.tags)
    
    def _orm_to_pipeline(self, pipeline_orm: PipelineORM) -> Pipeline:
        """Convert PipelineORM to Pipeline entity."""
        # Create schedule if configured
        schedule = None
        if pipeline_orm.schedule_config:
            config = pipeline_orm.schedule_config
            schedule = PipelineSchedule(
                enabled=config.get("enabled", False),
                cron_expression=config.get("cron_expression"),
                timezone=config.get("timezone", "UTC"),
                max_concurrent_runs=config.get("max_concurrent_runs", 1)
            )
        
        # Convert steps
        steps = {}
        for step_orm in pipeline_orm.steps:
            step = self._orm_to_step(step_orm)
            steps[step.id] = step
        
        return Pipeline(
            id=pipeline_orm.id,
            name=pipeline_orm.name,
            description=pipeline_orm.description,
            version=pipeline_orm.version,
            steps=steps,
            schedule=schedule,
            status=pipeline_orm.status,
            current_run_id=pipeline_orm.current_run_id,
            started_at=pipeline_orm.started_at,
            completed_at=pipeline_orm.completed_at,
            created_at=pipeline_orm.created_at,
            updated_at=pipeline_orm.updated_at,
            created_by=pipeline_orm.created_by,
            tags=set(pipeline_orm.tags or []),
            max_parallel_steps=pipeline_orm.max_parallel_steps,
            global_timeout_minutes=pipeline_orm.global_timeout_minutes
        )
    
    def _orm_to_step(self, step_orm: PipelineStepORM) -> PipelineStep:
        """Convert PipelineStepORM to PipelineStep entity."""
        # Parse resource requirements
        resource_req_data = step_orm.resource_requirements or {}
        resource_requirements = ResourceRequirements(**resource_req_data)
        
        # Parse retry policy
        retry_policy_data = step_orm.retry_policy or {}
        retry_policy = RetryPolicy(**retry_policy_data)
        
        # Parse depends_on UUIDs
        depends_on = set()
        if step_orm.depends_on:
            for dep_str in step_orm.depends_on:
                try:
                    depends_on.add(UUID(dep_str))
                except ValueError:
                    # Skip invalid UUIDs
                    pass
        
        return PipelineStep(
            id=step_orm.id,
            name=step_orm.name,
            step_type=step_orm.step_type,
            description=step_orm.description,
            command=step_orm.command,
            working_directory=step_orm.working_directory,
            environment_variables=step_orm.environment_variables or {},
            parameters=step_orm.parameters or {},
            depends_on=depends_on,
            resource_requirements=resource_requirements,
            retry_policy=retry_policy,
            status=step_orm.status,
            started_at=step_orm.started_at,
            completed_at=step_orm.completed_at,
            attempt_count=step_orm.attempt_count,
            exit_code=step_orm.exit_code,
            stdout=step_orm.stdout,
            stderr=step_orm.stderr,
            artifacts=step_orm.artifacts or {},
            metrics=step_orm.metrics or {},
            created_at=step_orm.created_at,
            updated_at=step_orm.updated_at,
            created_by=step_orm.created_by,
            tags=set(step_orm.tags or [])
        )


class SqlAlchemyPipelineRunRepository(PipelineRunRepository):
    """SQLAlchemy implementation of pipeline run repository."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, pipeline_run: PipelineRun) -> PipelineRun:
        """Save a pipeline run to the database."""
        # Check if run exists
        stmt = select(PipelineRunORM).where(PipelineRunORM.id == pipeline_run.id)
        result = await self.session.execute(stmt)
        run_orm = result.scalar_one_or_none()
        
        if run_orm:
            # Update existing run
            run_orm = self._update_run_orm(run_orm, pipeline_run)
        else:
            # Create new run
            run_orm = self._create_run_orm(pipeline_run)
            self.session.add(run_orm)
        
        await self.session.commit()
        await self.session.refresh(run_orm)
        
        return self._orm_to_run(run_orm)
    
    async def find_by_id(self, run_id: UUID) -> Optional[PipelineRun]:
        """Find a pipeline run by its ID."""
        stmt = select(PipelineRunORM).where(PipelineRunORM.id == run_id)
        result = await self.session.execute(stmt)
        run_orm = result.scalar_one_or_none()
        
        if run_orm:
            return self._orm_to_run(run_orm)
        return None
    
    async def find_by_pipeline_id(
        self, 
        pipeline_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PipelineRun]:
        """Find pipeline runs by pipeline ID."""
        stmt = (
            select(PipelineRunORM)
            .where(PipelineRunORM.pipeline_id == pipeline_id)
            .order_by(desc(PipelineRunORM.started_at))
        )
        
        if offset:
            stmt = stmt.offset(offset)
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        run_orms = result.scalars().all()
        
        return [self._orm_to_run(orm) for orm in run_orms]
    
    async def find_by_status(
        self, 
        status: PipelineStatus,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PipelineRun]:
        """Find pipeline runs by status."""
        stmt = (
            select(PipelineRunORM)
            .where(PipelineRunORM.status == status)
            .order_by(desc(PipelineRunORM.started_at))
        )
        
        if offset:
            stmt = stmt.offset(offset)
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        run_orms = result.scalars().all()
        
        return [self._orm_to_run(orm) for orm in run_orms]
    
    async def find_latest_for_pipeline(self, pipeline_id: UUID) -> Optional[PipelineRun]:
        """Find the latest run for a specific pipeline."""
        stmt = (
            select(PipelineRunORM)
            .where(PipelineRunORM.pipeline_id == pipeline_id)
            .order_by(desc(PipelineRunORM.started_at))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        run_orm = result.scalar_one_or_none()
        
        if run_orm:
            return self._orm_to_run(run_orm)
        return None
    
    async def find_running_runs(self) -> List[PipelineRun]:
        """Find all currently running pipeline runs."""
        stmt = (
            select(PipelineRunORM)
            .where(PipelineRunORM.status == PipelineStatus.RUNNING)
            .order_by(desc(PipelineRunORM.started_at))
        )
        result = await self.session.execute(stmt)
        run_orms = result.scalars().all()
        
        return [self._orm_to_run(orm) for orm in run_orms]
    
    async def search(
        self,
        pipeline_id: Optional[UUID] = None,
        status: Optional[PipelineStatus] = None,
        triggered_by: Optional[str] = None,
        trigger_type: Optional[str] = None,
        started_after: Optional[datetime] = None,
        started_before: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[PipelineRun]:
        """Search pipeline runs with multiple criteria."""
        stmt = select(PipelineRunORM)
        
        # Apply filters
        conditions = []
        
        if pipeline_id:
            conditions.append(PipelineRunORM.pipeline_id == pipeline_id)
        
        if status:
            conditions.append(PipelineRunORM.status == status)
        
        if triggered_by:
            conditions.append(PipelineRunORM.triggered_by == triggered_by)
        
        if trigger_type:
            conditions.append(PipelineRunORM.trigger_type == trigger_type)
        
        if started_after:
            conditions.append(PipelineRunORM.started_at >= started_after)
        
        if started_before:
            conditions.append(PipelineRunORM.started_at <= started_before)
        
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        # Apply ordering, limit, and offset
        stmt = stmt.order_by(desc(PipelineRunORM.started_at))
        
        if offset:
            stmt = stmt.offset(offset)
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        run_orms = result.scalars().all()
        
        return [self._orm_to_run(orm) for orm in run_orms]
    
    async def delete(self, run_id: UUID) -> bool:
        """Delete a pipeline run from the database."""
        stmt = select(PipelineRunORM).where(PipelineRunORM.id == run_id)
        result = await self.session.execute(stmt)
        run_orm = result.scalar_one_or_none()
        
        if run_orm:
            await self.session.delete(run_orm)
            await self.session.commit()
            return True
        
        return False
    
    async def get_run_statistics(
        self, 
        pipeline_id: Optional[UUID] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get pipeline run statistics."""
        from datetime import timedelta
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Base query
        base_stmt = select(PipelineRunORM).where(PipelineRunORM.started_at >= cutoff_date)
        
        if pipeline_id:
            base_stmt = base_stmt.where(PipelineRunORM.pipeline_id == pipeline_id)
        
        # Total runs
        total_stmt = select(func.count(PipelineRunORM.id)).select_from(base_stmt.subquery())
        total_result = await self.session.execute(total_stmt)
        total_runs = total_result.scalar()
        
        # Runs by status
        status_stmt = (
            select(PipelineRunORM.status, func.count(PipelineRunORM.id))
            .select_from(base_stmt.subquery())
            .group_by(PipelineRunORM.status)
        )
        status_result = await self.session.execute(status_stmt)
        status_counts = dict(status_result.fetchall())
        
        # Average duration for completed runs
        duration_stmt = (
            select(func.avg(
                func.extract('epoch', PipelineRunORM.completed_at - PipelineRunORM.started_at)
            ))
            .select_from(base_stmt.subquery())
            .where(and_(
                PipelineRunORM.status == PipelineStatus.COMPLETED,
                PipelineRunORM.completed_at.isnot(None)
            ))
        )
        duration_result = await self.session.execute(duration_stmt)
        avg_duration = duration_result.scalar()
        
        return {
            "total_runs": total_runs or 0,
            "status_counts": status_counts,
            "average_duration_seconds": float(avg_duration) if avg_duration else None,
            "period_days": days,
            "pipeline_id": str(pipeline_id) if pipeline_id else None
        }
    
    def _create_run_orm(self, pipeline_run: PipelineRun) -> PipelineRunORM:
        """Create a new PipelineRunORM from PipelineRun entity."""
        return PipelineRunORM(
            id=pipeline_run.id,
            pipeline_id=pipeline_run.pipeline_id,
            pipeline_version=pipeline_run.pipeline_version,
            status=pipeline_run.status,
            started_at=pipeline_run.started_at,
            completed_at=pipeline_run.completed_at,
            step_runs=pipeline_run.step_runs,
            triggered_by=pipeline_run.triggered_by,
            trigger_type=pipeline_run.trigger_type,
            parameters=pipeline_run.parameters,
            artifacts=pipeline_run.artifacts,
            metrics=pipeline_run.metrics
        )
    
    def _update_run_orm(self, run_orm: PipelineRunORM, pipeline_run: PipelineRun) -> PipelineRunORM:
        """Update existing PipelineRunORM with PipelineRun entity data."""
        run_orm.status = pipeline_run.status
        run_orm.completed_at = pipeline_run.completed_at
        run_orm.step_runs = pipeline_run.step_runs
        run_orm.triggered_by = pipeline_run.triggered_by
        run_orm.trigger_type = pipeline_run.trigger_type
        run_orm.parameters = pipeline_run.parameters
        run_orm.artifacts = pipeline_run.artifacts
        run_orm.metrics = pipeline_run.metrics
        
        return run_orm
    
    def _orm_to_run(self, run_orm: PipelineRunORM) -> PipelineRun:
        """Convert PipelineRunORM to PipelineRun entity."""
        return PipelineRun(
            id=run_orm.id,
            pipeline_id=run_orm.pipeline_id,
            pipeline_version=run_orm.pipeline_version,
            status=run_orm.status,
            started_at=run_orm.started_at,
            completed_at=run_orm.completed_at,
            step_runs=run_orm.step_runs or {},
            triggered_by=run_orm.triggered_by,
            trigger_type=run_orm.trigger_type,
            parameters=run_orm.parameters or {},
            artifacts=run_orm.artifacts or {},
            metrics=run_orm.metrics or {}
        )