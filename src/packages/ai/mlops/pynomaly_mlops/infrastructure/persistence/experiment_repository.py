"""SQLAlchemy implementation of experiment repositories."""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.future import select

from pynomaly_mlops.domain.entities.experiment import Experiment, ExperimentRun, ExperimentStatus, ExperimentRunStatus
from pynomaly_mlops.domain.repositories.experiment_repository import ExperimentRepository
from .experiment_models import ExperimentORM, ExperimentRunORM, ExperimentMetricORM, ExperimentComparisonORM
from .experiment_mappers import ExperimentMapper, ExperimentRunMapper, ExperimentMetricMapper, ExperimentComparisonMapper


class SqlAlchemyExperimentRepository(ExperimentRepository):
    """SQLAlchemy implementation of ExperimentRepository."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def save(self, experiment: Experiment) -> Experiment:
        """Save experiment to database.
        
        Args:
            experiment: Experiment to save
            
        Returns:
            Saved experiment
        """
        # Check if experiment exists
        result = await self.session.execute(
            select(ExperimentORM).where(ExperimentORM.id == experiment.id)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing experiment
            ExperimentMapper.update_orm(existing, experiment)
            orm = existing
        else:
            # Create new experiment
            orm = ExperimentMapper.to_orm(experiment)
            self.session.add(orm)
        
        await self.session.flush()
        return ExperimentMapper.to_domain(orm)
    
    async def get_by_id(self, experiment_id: UUID) -> Optional[Experiment]:
        """Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment if found, None otherwise
        """
        result = await self.session.execute(
            select(ExperimentORM)
            .options(selectinload(ExperimentORM.runs))
            .where(ExperimentORM.id == experiment_id)
        )
        orm = result.scalar_one_or_none()
        
        if not orm:
            return None
        
        experiment = ExperimentMapper.to_domain(orm)
        
        # Load runs
        experiment.runs = [
            ExperimentRunMapper.to_domain(run_orm) 
            for run_orm in orm.runs
        ]
        
        return experiment
    
    async def get_by_name(self, name: str) -> Optional[Experiment]:
        """Get experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment if found, None otherwise
        """
        result = await self.session.execute(
            select(ExperimentORM)
            .options(selectinload(ExperimentORM.runs))
            .where(ExperimentORM.name == name)
        )
        orm = result.scalar_one_or_none()
        
        if not orm:
            return None
        
        experiment = ExperimentMapper.to_domain(orm)
        
        # Load runs  
        experiment.runs = [
            ExperimentRunMapper.to_domain(run_orm)
            for run_orm in orm.runs
        ]
        
        return experiment
    
    async def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[ExperimentStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        order_by: str = "created_at",
        ascending: bool = False
    ) -> List[Experiment]:
        """List experiments with filtering and pagination.
        
        Args:
            limit: Maximum number of experiments to return
            offset: Number of experiments to skip
            status: Filter by experiment status
            created_by: Filter by creator
            tags: Filter by tags (key-value pairs)
            order_by: Field to order by
            ascending: Sort direction
            
        Returns:
            List of experiments
        """
        query = select(ExperimentORM)
        
        # Apply filters
        conditions = []
        
        if status:
            conditions.append(ExperimentORM.status == status.value)
        
        if created_by:
            conditions.append(ExperimentORM.created_by == created_by)
        
        if tags:
            for key, value in tags.items():
                conditions.append(
                    func.json_extract(ExperimentORM.tags, f"$.{key}") == value
                )
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Apply ordering
        order_field = getattr(ExperimentORM, order_by, ExperimentORM.created_at)
        if ascending:
            query = query.order_by(asc(order_field))
        else:
            query = query.order_by(desc(order_field))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        orms = result.scalars().all()
        
        return [ExperimentMapper.to_domain(orm) for orm in orms]
    
    async def delete(self, experiment_id: UUID) -> bool:
        """Delete experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            select(ExperimentORM).where(ExperimentORM.id == experiment_id)
        )
        orm = result.scalar_one_or_none()
        
        if not orm:
            return False
        
        await self.session.delete(orm)
        return True
    
    async def save_run(self, run: ExperimentRun) -> ExperimentRun:
        """Save experiment run to database.
        
        Args:
            run: ExperimentRun to save
            
        Returns:
            Saved experiment run
        """
        # Check if run exists
        result = await self.session.execute(
            select(ExperimentRunORM).where(ExperimentRunORM.id == run.id)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing run
            ExperimentRunMapper.update_orm(existing, run)
            orm = existing
        else:
            # Create new run
            orm = ExperimentRunMapper.to_orm(run)
            self.session.add(orm)
        
        await self.session.flush()
        return ExperimentRunMapper.to_domain(orm)
    
    async def get_run_by_id(self, run_id: UUID) -> Optional[ExperimentRun]:
        """Get experiment run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            ExperimentRun if found, None otherwise
        """
        result = await self.session.execute(
            select(ExperimentRunORM).where(ExperimentRunORM.id == run_id)
        )
        orm = result.scalar_one_or_none()
        
        if not orm:
            return None
        
        return ExperimentRunMapper.to_domain(orm)
    
    async def list_runs(
        self,
        experiment_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[ExperimentRunStatus] = None,
        created_by: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        order_by: str = "start_time",
        ascending: bool = False
    ) -> List[ExperimentRun]:
        """List experiment runs with filtering and pagination.
        
        Args:
            experiment_id: Filter by experiment ID
            limit: Maximum number of runs to return
            offset: Number of runs to skip
            status: Filter by run status
            created_by: Filter by creator
            tags: Filter by tags
            order_by: Field to order by
            ascending: Sort direction
            
        Returns:
            List of experiment runs
        """
        query = select(ExperimentRunORM)
        
        # Apply filters
        conditions = []
        
        if experiment_id:
            conditions.append(ExperimentRunORM.experiment_id == experiment_id)
        
        if status:
            conditions.append(ExperimentRunORM.status == status.value)
        
        if created_by:
            conditions.append(ExperimentRunORM.created_by == created_by)
        
        if tags:
            for key, value in tags.items():
                conditions.append(
                    func.json_extract(ExperimentRunORM.tags, f"$.{key}") == value
                )
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Apply ordering
        order_field = getattr(ExperimentRunORM, order_by, ExperimentRunORM.start_time)
        if ascending:
            query = query.order_by(asc(order_field))
        else:
            query = query.order_by(desc(order_field))
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        result = await self.session.execute(query)
        orms = result.scalars().all()
        
        return [ExperimentRunMapper.to_domain(orm) for orm in orms]
    
    async def delete_run(self, run_id: UUID) -> bool:
        """Delete experiment run by ID.
        
        Args:
            run_id: Run ID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            select(ExperimentRunORM).where(ExperimentRunORM.id == run_id)
        )
        orm = result.scalar_one_or_none()
        
        if not orm:
            return False
        
        await self.session.delete(orm)
        return True
    
    async def log_metric(
        self, 
        run_id: UUID, 
        key: str, 
        value: float, 
        step: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a metric for an experiment run.
        
        Args:
            run_id: Run ID
            key: Metric name
            value: Metric value
            step: Optional step number
            timestamp: Optional timestamp
        """
        metric_orm = ExperimentMetricMapper.to_orm(
            run_id=run_id,
            key=key,
            value=value,
            step=step,
            timestamp=timestamp
        )
        
        self.session.add(metric_orm)
        await self.session.flush()
        
        # Also update the run's metrics dictionary for quick access
        result = await self.session.execute(
            select(ExperimentRunORM).where(ExperimentRunORM.id == run_id)
        )
        run_orm = result.scalar_one_or_none()
        
        if run_orm:
            if not run_orm.metrics:
                run_orm.metrics = {}
            run_orm.metrics[key] = value
    
    async def get_metrics(
        self, 
        run_id: UUID, 
        keys: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get time-series metrics for a run.
        
        Args:
            run_id: Run ID
            keys: Optional list of metric keys to filter
            
        Returns:
            Dictionary mapping metric keys to lists of metric data
        """
        query = select(ExperimentMetricORM).where(ExperimentMetricORM.run_id == run_id)
        
        if keys:
            query = query.where(ExperimentMetricORM.key.in_(keys))
        
        query = query.order_by(ExperimentMetricORM.timestamp, ExperimentMetricORM.step)
        
        result = await self.session.execute(query)
        metric_orms = result.scalars().all()
        
        metrics: Dict[str, List[Dict[str, Any]]] = {}
        
        for metric_orm in metric_orms:
            key = metric_orm.key
            if key not in metrics:
                metrics[key] = []
            
            metrics[key].append(ExperimentMetricMapper.to_domain(metric_orm))
        
        return metrics
    
    async def compare_runs(
        self, 
        run_ids: List[UUID], 
        metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple experiment runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_keys: Optional list of metrics to include in comparison
            
        Returns:
            Comparison data structure
        """
        # Get runs
        result = await self.session.execute(
            select(ExperimentRunORM).where(ExperimentRunORM.id.in_(run_ids))
        )
        run_orms = result.scalars().all()
        
        runs = [ExperimentRunMapper.to_domain(orm) for orm in run_orms]
        
        # Get metrics for each run
        comparison = {
            "runs": {},
            "metrics_comparison": {},
            "parameters_comparison": {}
        }
        
        for run in runs:
            run_data = {
                "id": str(run.id),
                "name": run.name,
                "status": run.status.value,
                "start_time": run.start_time.isoformat(),
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "parameters": run.parameters,
                "metrics": run.metrics,
                "tags": run.tags
            }
            
            comparison["runs"][str(run.id)] = run_data
            
            # Aggregate parameters for comparison
            for param_key, param_value in run.parameters.items():
                if param_key not in comparison["parameters_comparison"]:
                    comparison["parameters_comparison"][param_key] = {}
                comparison["parameters_comparison"][param_key][str(run.id)] = param_value
            
            # Aggregate metrics for comparison
            for metric_key, metric_value in run.metrics.items():
                if metric_keys and metric_key not in metric_keys:
                    continue
                
                if metric_key not in comparison["metrics_comparison"]:
                    comparison["metrics_comparison"][metric_key] = {}
                comparison["metrics_comparison"][metric_key][str(run.id)] = metric_value
        
        return comparison
    
    async def search_experiments(
        self, 
        query: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Experiment]:
        """Search experiments by name, description, or tags.
        
        Args:
            query: Search query
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            List of matching experiments
        """
        search_query = select(ExperimentORM).where(
            or_(
                ExperimentORM.name.ilike(f"%{query}%"),
                ExperimentORM.description.ilike(f"%{query}%"),
                func.json_extract(ExperimentORM.tags, "$").ilike(f"%{query}%")
            )
        ).order_by(desc(ExperimentORM.updated_at)).offset(offset).limit(limit)
        
        result = await self.session.execute(search_query)
        orms = result.scalars().all()
        
        return [ExperimentMapper.to_domain(orm) for orm in orms]