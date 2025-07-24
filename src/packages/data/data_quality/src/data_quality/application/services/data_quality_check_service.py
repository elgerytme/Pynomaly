
from uuid import UUID
from typing import Any, Dict
import pandas as pd
from datetime import datetime

from ..ports.data_quality_check_repository import DataQualityCheckRepository
from ..ports.data_quality_rule_repository import DataQualityRuleRepository
from ...domain.entities.data_quality_check import DataQualityCheck, CheckResult, CheckStatus, CheckSeverity
from ...domain.entities.data_quality_rule import DataQualityRule
from ...infrastructure.adapters.data_source_adapter import DataSourceAdapter
from .rule_evaluator import RuleEvaluator


class DataQualityCheckService:
    """Service for data quality checks."""

    def __init__(self, data_quality_check_repository: DataQualityCheckRepository, data_quality_rule_repository: DataQualityRuleRepository, data_source_adapter: DataSourceAdapter, rule_evaluator: RuleEvaluator):
        self.data_quality_check_repository = data_quality_check_repository
        self.data_quality_rule_repository = data_quality_rule_repository
        self.data_source_adapter = data_source_adapter
        self.rule_evaluator = rule_evaluator

    def run_check(self, check_id: UUID, source_config: Dict[str, Any]) -> DataQualityCheck:
        """Run a data quality check.

        Args:
            check_id: The ID of the data quality check to run.
            source_config: Configuration for the data source adapter.

        Returns:
            The updated DataQualityCheck entity with the result.
        """
        data_quality_check = self.data_quality_check_repository.get_by_id(check_id)
        if not data_quality_check:
            raise ValueError(f"DataQualityCheck with ID {check_id} not found.")

        rule = self.data_quality_rule_repository.get_by_id(data_quality_check.rule_id)
        if not rule:
            raise ValueError(f"DataQualityRule with ID {data_quality_check.rule_id} not found for check {check_id}.")

        data_quality_check.start_profiling() # Re-using start_profiling for check execution start

        try:
            df = self.data_source_adapter.read_data(source_config)
            total_records = len(df)
            passed_records = 0
            failed_records = 0
            failed_values = []
            sample_failures = []

            for index, row in df.iterrows():
                if self.rule_evaluator.evaluate_record(rule, row.to_dict()):
                    passed_records += 1
                else:
                    failed_records += 1
                    failed_values.append(row.to_dict())
                    if len(sample_failures) < 100: # Limit sample size
                        sample_failures.append(row.to_dict())

            score = 1.0 - (failed_records / total_records) if total_records > 0 else 1.0
            passed = score >= data_quality_check.threshold

            result = CheckResult(
                check_id=data_quality_check.id,
                dataset_name=data_quality_check.dataset_name,
                column_name=data_quality_check.column_name,
                passed=passed,
                score=score,
                total_records=total_records,
                passed_records=passed_records,
                failed_records=failed_records,
                executed_at=datetime.utcnow(),
                severity=CheckSeverity.ERROR if not passed else CheckSeverity.INFO,
                message=f"Check {'passed' if passed else 'failed'}. Failed records: {failed_records}",
                failed_values=failed_values,
                sample_failures=sample_failures
            )

            data_quality_check.last_result = result
            data_quality_check.complete_profiling() # Re-using complete_profiling for check execution end

        except Exception as e:
            data_quality_check.fail_profiling(str(e)) # Re-using fail_profiling for check execution failure
            raise

        self.data_quality_check_repository.save(data_quality_check)
        return data_quality_check
