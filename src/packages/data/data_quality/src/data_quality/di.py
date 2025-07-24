
from sqlalchemy.orm import Session

from src.packages.data.data_quality.src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.packages.data.data_quality.src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.packages.data.data_quality.src.data_quality.application.services.data_quality_rule_service import DataQualityRuleService
from src.packages.data.data_quality.src.data_quality.application.services.rule_evaluator import RuleEvaluator
from src.packages.data.data_quality.src.data_quality.infrastructure.repositories.sqlalchemy_data_profile_repository import SQLAlchemyDataProfileRepository
from src.packages.data.data_quality.src.data_quality.infrastructure.repositories.sqlalchemy_data_quality_check_repository import SQLAlchemyDataQualityCheckRepository
from src.packages.data.data_quality.src.data_quality.infrastructure.repositories.sqlalchemy_data_quality_rule_repository import SQLAlchemyDataQualityRuleRepository
from src.packages.data.data_quality.src.data_quality.infrastructure.database.database import SessionLocal
from src.packages.data.data_quality.src.data_quality.infrastructure.adapters.pandas_csv_adapter import PandasCSVAdapter


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Repositories
def get_data_profile_repository(db: Session = next(get_db_session())):
    return SQLAlchemyDataProfileRepository(db)

def get_data_quality_check_repository(db: Session = next(get_db_session())):
    return SQLAlchemyDataQualityCheckRepository(db)

def get_data_quality_rule_repository(db: Session = next(get_db_session())):
    return SQLAlchemyDataQualityRuleRepository(db)


# Adapters
def get_pandas_csv_adapter():
    return PandasCSVAdapter()


# Rule Evaluator
def get_rule_evaluator():
    return RuleEvaluator()


# Services
def get_data_profiling_service(data_profile_repository=get_data_profile_repository()):
    return DataProfilingService(data_profile_repository)

def get_data_quality_check_service(data_quality_check_repository=get_data_quality_check_repository(), data_quality_rule_repository=get_data_quality_rule_repository(), data_source_adapter=get_pandas_csv_adapter(), rule_evaluator=get_rule_evaluator()):
    return DataQualityCheckService(data_quality_check_repository, data_quality_rule_repository, data_source_adapter, rule_evaluator)

def get_data_quality_rule_service(data_quality_rule_repository=get_data_quality_rule_repository()):
    return DataQualityRuleService(data_quality_rule_repository)


# Expose instances for direct use (e.g., in CLI)
data_profile_repository = get_data_profile_repository()
data_quality_check_repository = get_data_quality_check_repository()
data_quality_rule_repository = get_data_quality_rule_repository()

data_profiling_service = get_data_profiling_service()
data_quality_check_service = get_data_quality_check_service()
data_quality_rule_service = get_data_quality_rule_service()
