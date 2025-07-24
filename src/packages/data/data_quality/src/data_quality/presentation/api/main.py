
from typing import List
from uuid import UUID

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from src.packages.data.data_quality.src.data_quality.application.use_cases.create_data_profile import CreateDataProfileUseCase
from src.packages.data.data_quality.src.data_quality.application.use_cases.manage_data_quality_rules import ManageDataQualityRulesUseCase
from src.packages.data.data_quality.src.data_quality.application.use_cases.run_data_quality_check import RunDataQualityCheckUseCase
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_rule import DataQualityRule
from src.packages.data.data_quality.src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter
from src.packages.data.data_quality.src.data_quality.infrastructure.database.database import get_db
from src.packages.data.data_quality.src.data_quality.infrastructure.repositories.sqlalchemy_data_profile_repository import SQLAlchemyDataProfileRepository
from src.packages.data.data_quality.src.data_quality.infrastructure.repositories.sqlalchemy_data_quality_check_repository import SQLAlchemyDataQualityCheckRepository
from src.packages.data.data_quality.src.data_quality.infrastructure.repositories.sqlalchemy_data_quality_rule_repository import SQLAlchemyDataQualityRuleRepository
from src.packages.data.data_quality.src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.packages.data.data_quality.src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.packages.data.data_quality.src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.packages.data.data_quality.src.data_quality.application.services.rule_evaluator import RuleEvaluator
from src.packages.data.data_quality.src.data_quality.presentation.api.models import (
    DataProfileCreate, DataProfileResponse,
    DataQualityRuleCreate, DataQualityRuleResponse,
    DataQualityCheckCreate, DataQualityCheckResponse,
    RunCheckRequest
)

app = FastAPI(title="Data Quality API", version="0.1.0")


# Dependency for repositories
def get_data_profile_repository(db: Session = Depends(get_db)):
    return SQLAlchemyDataProfileRepository(db)

def get_data_quality_check_repository(db: Session = Depends(get_db)):
    return SQLAlchemyDataQualityCheckRepository(db)

def get_data_quality_rule_repository(db: Session = Depends(get_db)):
    return SQLAlchemyDataQualityRuleRepository(db)


# Dependency for services
def get_data_profiling_service(repo: SQLAlchemyDataProfileRepository = Depends(get_data_profile_repository)):
    return DataProfilingService(repo)

def get_data_quality_check_service(check_repo: SQLAlchemyDataQualityCheckRepository = Depends(get_data_quality_check_repository),
                                   rule_repo: SQLAlchemyDataQualityRuleRepository = Depends(get_data_quality_rule_repository),
                                   adapter: PandasCSVAdapter = Depends(PandasCSVAdapter),
                                   evaluator: RuleEvaluator = Depends(RuleEvaluator)):
    return DataQualityCheckService(check_repo, rule_repo, adapter, evaluator)

def get_data_quality_rule_service(repo: SQLAlchemyDataQualityRuleRepository = Depends(get_data_quality_rule_repository)):
    return DataQualityRuleService(repo)


# Endpoints
@app.post("/profiles", response_model=DataProfileResponse, status_code=201)
async def create_data_profile_api(
    profile_create: DataProfileCreate,
    profiling_service: DataProfilingService = Depends(get_data_profiling_service),
    adapter: PandasCSVAdapter = Depends(PandasCSVAdapter)
):
    """Create a new data profile."""
    try:
        profile = CreateDataProfileUseCase(profiling_service).execute(
            dataset_name=profile_create.dataset_name,
            data_source_adapter=adapter,
            source_config={"file_path": profile_create.file_path}
        )
        return profile
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Data source error: {e}")


@app.get("/profiles/{profile_id}", response_model=DataProfileResponse)
async def get_data_profile_api(
    profile_id: UUID,
    profiling_service: DataProfilingService = Depends(get_data_profiling_service)
):
    """Get a data profile by ID."""
    profile = profiling_service.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="DataProfile not found")
    return profile


@app.post("/rules", response_model=DataQualityRuleResponse, status_code=201)
async def create_data_quality_rule_api(
    rule_create: DataQualityRuleCreate,
    rule_service: DataQualityRuleService = Depends(get_data_quality_rule_service)
):
    """Create a new data quality rule."""
    try:
        rule = DataQualityRule(
            name=rule_create.name,
            description=rule_create.description,
            rule_type=rule_create.rule_type,
            severity=rule_create.severity,
            dataset_name=rule_create.dataset_name,
            table_name=rule_create.table_name,
            schema_name=rule_create.schema_name,
            conditions=[cond.dict() for cond in rule_create.conditions], # Convert Pydantic models to dicts
            logical_operator=rule_create.logical_operator,
            expression=rule_create.expression,
            is_active=rule_create.is_active,
            is_blocking=rule_create.is_blocking,
            auto_fix=rule_create.auto_fix,
            fix_action=rule_create.fix_action,
            violation_threshold=rule_create.violation_threshold,
            sample_size=rule_create.sample_size,
            created_by=rule_create.created_by,
            updated_by=rule_create.updated_by,
            config=rule_create.config,
            tags=rule_create.tags,
            depends_on=rule_create.depends_on,
        )
        created_rule = rule_service.create_rule(rule)
        return created_rule
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/rules/{rule_id}", response_model=DataQualityRuleResponse)
async def get_data_quality_rule_api(
    rule_id: UUID,
    rule_service: DataQualityRuleService = Depends(get_data_quality_rule_service)
):
    """Get a data quality rule by ID."""
    rule = rule_service.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="DataQualityRule not found")
    return rule


@app.post("/checks", response_model=DataQualityCheckResponse, status_code=201)
async def create_data_quality_check_api(
    check_create: DataQualityCheckCreate,
    check_service: DataQualityCheckService = Depends(get_data_quality_check_service),
    rule_service: DataQualityRuleService = Depends(get_data_quality_rule_service)
):
    """Create a new data quality check."""
    try:
        # Ensure the rule exists
        rule = rule_service.get_rule(check_create.rule_id)
        if not rule:
            raise HTTPException(status_code=400, detail=f"Rule with ID {check_create.rule_id} not found.")

        check = DataQualityCheck(
            name=check_create.name,
            description=check_create.description,
            check_type=check_create.check_type,
            rule_id=check_create.rule_id,
            dataset_name=check_create.dataset_name,
            column_name=check_create.column_name,
            schema_name=check_create.schema_name,
            table_name=check_create.table_name,
            query=check_create.query,
            expression=check_create.expression,
            expected_value=check_create.expected_value,
            threshold=check_create.threshold,
            tolerance=check_create.tolerance,
            is_active=check_create.is_active,
            schedule_cron=check_create.schedule_cron,
            timeout_seconds=check_create.timeout_seconds,
            retry_attempts=check_create.retry_attempts,
            created_by=check_create.created_by,
            updated_by=check_create.updated_by,
            config=check_create.config,
            environment_vars=check_create.environment_vars,
            tags=check_create.tags,
            depends_on=check_create.depends_on,
            blocks=check_create.blocks,
        )
        created_check = check_service.data_quality_check_repository.save(check) # Save the check first
        return created_check
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/checks/{check_id}/run", response_model=DataQualityCheckResponse)
async def run_data_quality_check_api(
    check_id: UUID,
    run_request: RunCheckRequest,
    check_service: DataQualityCheckService = Depends(get_data_quality_check_service)
):
    """Run a data quality check."""
    try:
        updated_check = check_service.run_check(check_id, {"file_path": run_request.file_path})
        return updated_check
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Data source error: {e}")


@app.get("/checks/{check_id}", response_model=DataQualityCheckResponse)
async def get_data_quality_check_api(
    check_id: UUID,
    check_service: DataQualityCheckService = Depends(get_data_quality_check_service)
):
    """Get a data quality check by ID."""
    check = check_service.data_quality_check_repository.get_by_id(check_id)
    if not check:
        raise HTTPException(status_code=404, detail="DataQualityCheck not found")
    return check
