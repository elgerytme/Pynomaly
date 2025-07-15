# Test Coverage Analysis Report
==================================================

Total files missing tests: 580
High priority: 208
Medium priority: 324
Low priority: 48

## HIGH Priority Files Missing Tests (208 files)
--------------------------------------------------

### Application Services
- **src/pynomaly/application/services/__init__.py**
  - Expected test: `unit/application/services/test___init__.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/ab_testing_service.py**
  - Expected test: `unit/application/services/test_ab_testing_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/adaptive_threshold_optimizer.py**
  - Expected test: `unit/application/services/test_adaptive_threshold_optimizer.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/advanced_automl_service.py**
  - Expected test: `unit/application/services/test_advanced_automl_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/advanced_ensemble_service.py**
  - Expected test: `unit/application/services/test_advanced_ensemble_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/advanced_explainability_service.py**
  - Expected test: `unit/application/services/test_advanced_explainability_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/advanced_ml_lifecycle_service.py**
  - Expected test: `unit/application/services/test_advanced_ml_lifecycle_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/alert_management_service.py**
  - Expected test: `unit/application/services/test_alert_management_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/algorithm_adapter_registry.py**
  - Expected test: `unit/application/services/test_algorithm_adapter_registry.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/algorithm_benchmark.py**
  - Expected test: `unit/application/services/test_algorithm_benchmark.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/algorithm_optimization_service.py**
  - Expected test: `unit/application/services/test_algorithm_optimization_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/auto_retraining_service.py**
  - Expected test: `unit/application/services/test_auto_retraining_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/automated_training_service.py**
  - Expected test: `unit/application/services/test_automated_training_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/automl_configuration_integration.py**
  - Expected test: `unit/application/services/test_automl_configuration_integration.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/automl_service.py**
  - Expected test: `unit/application/services/test_automl_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/automl_with_tracking.py**
  - Expected test: `unit/application/services/test_automl_with_tracking.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/autonomous_configuration_integration.py**
  - Expected test: `unit/application/services/test_autonomous_configuration_integration.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/business_impact_scoring.py**
  - Expected test: `unit/application/services/test_business_impact_scoring.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/compliance_service.py**
  - Expected test: `unit/application/services/test_compliance_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/configuration_capture_service.py**
  - Expected test: `unit/application/services/test_configuration_capture_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/configuration_discovery_service.py**
  - Expected test: `unit/application/services/test_configuration_discovery_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/configuration_template_service.py**
  - Expected test: `unit/application/services/test_configuration_template_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/continuous_learning_service.py**
  - Expected test: `unit/application/services/test_continuous_learning_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/cost_optimization_service.py**
  - Expected test: `unit/application/services/test_cost_optimization_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/cross_domain_transfer_learning.py**
  - Expected test: `unit/application/services/test_cross_domain_transfer_learning.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/deep_learning_integration_service.py**
  - Expected test: `unit/application/services/test_deep_learning_integration_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/deployment_orchestration_service.py**
  - Expected test: `unit/application/services/test_deployment_orchestration_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/detection_service.py**
  - Expected test: `unit/application/services/test_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/drift_detection_service.py**
  - Expected test: `unit/application/services/test_drift_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/enhanced_automl_service.py**
  - Expected test: `unit/application/services/test_enhanced_automl_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/enhanced_detection_service.py**
  - Expected test: `unit/application/services/test_enhanced_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/enhanced_model_persistence_service.py**
  - Expected test: `unit/application/services/test_enhanced_model_persistence_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/ensemble_detection_service.py**
  - Expected test: `unit/application/services/test_ensemble_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/ensemble_service.py**
  - Expected test: `unit/application/services/test_ensemble_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/enterprise_dashboard_service.py**
  - Expected test: `unit/application/services/test_enterprise_dashboard_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/enterprise_integration_service.py**
  - Expected test: `unit/application/services/test_enterprise_integration_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/event_processing_service.py**
  - Expected test: `unit/application/services/test_event_processing_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/experiment_management_service.py**
  - Expected test: `unit/application/services/test_experiment_management_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/experiment_tracking_service.py**
  - Expected test: `unit/application/services/test_experiment_tracking_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/explainability_engine.py**
  - Expected test: `unit/application/services/test_explainability_engine.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/explainability_service.py**
  - Expected test: `unit/application/services/test_explainability_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/explainable_ai_service.py**
  - Expected test: `unit/application/services/test_explainable_ai_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/export_service.py**
  - Expected test: `unit/application/services/test_export_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/federated_learning_service.py**
  - Expected test: `unit/application/services/test_federated_learning_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/governance_framework_service.py**
  - Expected test: `unit/application/services/test_governance_framework_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/integration_service.py**
  - Expected test: `unit/application/services/test_integration_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/intelligent_alert_service.py**
  - Expected test: `unit/application/services/test_intelligent_alert_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/intelligent_selection_service.py**
  - Expected test: `unit/application/services/test_intelligent_selection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/interactive_investigation_dashboard.py**
  - Expected test: `unit/application/services/test_interactive_investigation_dashboard.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/memory_optimization_service.py**
  - Expected test: `unit/application/services/test_memory_optimization_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_ab_testing_service.py**
  - Expected test: `unit/application/services/test_model_ab_testing_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_drift_detection_service.py**
  - Expected test: `unit/application/services/test_model_drift_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_governance_service.py**
  - Expected test: `unit/application/services/test_model_governance_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_lineage_service.py**
  - Expected test: `unit/application/services/test_model_lineage_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_management_service.py**
  - Expected test: `unit/application/services/test_model_management_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_persistence_service.py**
  - Expected test: `unit/application/services/test_model_persistence_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/model_registry_service.py**
  - Expected test: `unit/application/services/test_model_registry_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/multi_tenant_service.py**
  - Expected test: `unit/application/services/test_multi_tenant_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/performance_benchmarking_service.py**
  - Expected test: `unit/application/services/test_performance_benchmarking_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/performance_monitoring_service.py**
  - Expected test: `unit/application/services/test_performance_monitoring_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/performance_testing_service.py**
  - Expected test: `unit/application/services/test_performance_testing_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/pipeline_orchestration_service.py**
  - Expected test: `unit/application/services/test_pipeline_orchestration_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/predictive_maintenance_analytics.py**
  - Expected test: `unit/application/services/test_predictive_maintenance_analytics.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/real_time_decision_support.py**
  - Expected test: `unit/application/services/test_real_time_decision_support.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/real_time_ensemble_optimizer.py**
  - Expected test: `unit/application/services/test_real_time_ensemble_optimizer.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/reporting_service.py**
  - Expected test: `unit/application/services/test_reporting_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/security_compliance_service.py**
  - Expected test: `unit/application/services/test_security_compliance_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/streaming_detection_service.py**
  - Expected test: `unit/application/services/test_streaming_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/streaming_pipeline_manager.py**
  - Expected test: `unit/application/services/test_streaming_pipeline_manager.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/streaming_service.py**
  - Expected test: `unit/application/services/test_streaming_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/task_tracking_service.py**
  - Expected test: `unit/application/services/test_task_tracking_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/training_automation_service.py**
  - Expected test: `unit/application/services/test_training_automation_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/training_service.py**
  - Expected test: `unit/application/services/test_training_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/unified_data_service.py**
  - Expected test: `unit/application/services/test_unified_data_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/user_management_service.py**
  - Expected test: `unit/application/services/test_user_management_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/visualization_dashboard_service.py**
  - Expected test: `unit/application/services/test_visualization_dashboard_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/visualization_service.py**
  - Expected test: `unit/application/services/test_visualization_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/web_api_configuration_integration.py**
  - Expected test: `unit/application/services/test_web_api_configuration_integration.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/application/services/workflow_simplification_service.py**
  - Expected test: `unit/application/services/test_workflow_simplification_service.py`
  - Test type: unit
  - Reason: Services contain business logic


### Data Transfer Objects
- **src/pynomaly/application/dto/active_learning_dto.py**
  - Expected test: `unit/application/dto/test_active_learning_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/automl_dto.py**
  - Expected test: `unit/application/dto/test_automl_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/configuration_dto.py**
  - Expected test: `unit/application/dto/test_configuration_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/cost_optimization_dto.py**
  - Expected test: `unit/application/dto/test_cost_optimization_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/dataset_dto.py**
  - Expected test: `unit/application/dto/test_dataset_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/detection_dto.py**
  - Expected test: `unit/application/dto/test_detection_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/detector_dto.py**
  - Expected test: `unit/application/dto/test_detector_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/ensemble_dto.py**
  - Expected test: `unit/application/dto/test_ensemble_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/experiment_dto.py**
  - Expected test: `unit/application/dto/test_experiment_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/explainability_dto.py**
  - Expected test: `unit/application/dto/test_explainability_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/optimization_dto.py**
  - Expected test: `unit/application/dto/test_optimization_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/result_dto.py**
  - Expected test: `unit/application/dto/test_result_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/selection_dto.py**
  - Expected test: `unit/application/dto/test_selection_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/streaming_dto.py**
  - Expected test: `unit/application/dto/test_streaming_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/training_dto.py**
  - Expected test: `unit/application/dto/test_training_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts

- **src/pynomaly/application/dto/uncertainty_dto.py**
  - Expected test: `unit/application/dto/test_uncertainty_dto.py`
  - Test type: unit
  - Reason: DTOs define data contracts


### Domain Entities
- **src/pynomaly/domain/entities/__init__.py**
  - Expected test: `unit/domain/entities/test___init__.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/ab_test.py**
  - Expected test: `unit/domain/entities/test_ab_test.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/ab_testing.py**
  - Expected test: `unit/domain/entities/test_ab_testing.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/active_learning_session.py**
  - Expected test: `unit/domain/entities/test_active_learning_session.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/alert.py**
  - Expected test: `unit/domain/entities/test_alert.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/anomaly.py**
  - Expected test: `unit/domain/entities/test_anomaly.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/automl.py**
  - Expected test: `unit/domain/entities/test_automl.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/compliance.py**
  - Expected test: `unit/domain/entities/test_compliance.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/continuous_learning.py**
  - Expected test: `unit/domain/entities/test_continuous_learning.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/cost_optimization.py**
  - Expected test: `unit/domain/entities/test_cost_optimization.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/dashboard.py**
  - Expected test: `unit/domain/entities/test_dashboard.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/dataset.py**
  - Expected test: `unit/domain/entities/test_dataset.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/deployment.py**
  - Expected test: `unit/domain/entities/test_deployment.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/detection_result.py**
  - Expected test: `unit/domain/entities/test_detection_result.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/detector.py**
  - Expected test: `unit/domain/entities/test_detector.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/drift_detection.py**
  - Expected test: `unit/domain/entities/test_drift_detection.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/experiment.py**
  - Expected test: `unit/domain/entities/test_experiment.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/explainability.py**
  - Expected test: `unit/domain/entities/test_explainability.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/explainable_ai.py**
  - Expected test: `unit/domain/entities/test_explainable_ai.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/governance.py**
  - Expected test: `unit/domain/entities/test_governance.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/governance_workflow.py**
  - Expected test: `unit/domain/entities/test_governance_workflow.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/integrations.py**
  - Expected test: `unit/domain/entities/test_integrations.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/lineage_record.py**
  - Expected test: `unit/domain/entities/test_lineage_record.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/model.py**
  - Expected test: `unit/domain/entities/test_model.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/model_registry.py**
  - Expected test: `unit/domain/entities/test_model_registry.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/model_version.py**
  - Expected test: `unit/domain/entities/test_model_version.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/optimization_trial.py**
  - Expected test: `unit/domain/entities/test_optimization_trial.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/pipeline.py**
  - Expected test: `unit/domain/entities/test_pipeline.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/reporting.py**
  - Expected test: `unit/domain/entities/test_reporting.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/security_compliance.py**
  - Expected test: `unit/domain/entities/test_security_compliance.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/simple_detector.py**
  - Expected test: `unit/domain/entities/test_simple_detector.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/streaming_anomaly.py**
  - Expected test: `unit/domain/entities/test_streaming_anomaly.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/streaming_session.py**
  - Expected test: `unit/domain/entities/test_streaming_session.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/tenant.py**
  - Expected test: `unit/domain/entities/test_tenant.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/training_job.py**
  - Expected test: `unit/domain/entities/test_training_job.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/training_result.py**
  - Expected test: `unit/domain/entities/test_training_result.py`
  - Test type: unit
  - Reason: Domain entities are core business objects

- **src/pynomaly/domain/entities/user.py**
  - Expected test: `unit/domain/entities/test_user.py`
  - Test type: unit
  - Reason: Domain entities are core business objects


### Domain Exceptions
- **src/pynomaly/domain/exceptions/__init__.py**
  - Expected test: `unit/domain/exceptions/test___init__.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability

- **src/pynomaly/domain/exceptions/base.py**
  - Expected test: `unit/domain/exceptions/test_base.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability

- **src/pynomaly/domain/exceptions/dataset_exceptions.py**
  - Expected test: `unit/domain/exceptions/test_dataset_exceptions.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability

- **src/pynomaly/domain/exceptions/detector_exceptions.py**
  - Expected test: `unit/domain/exceptions/test_detector_exceptions.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability

- **src/pynomaly/domain/exceptions/entity_exceptions.py**
  - Expected test: `unit/domain/exceptions/test_entity_exceptions.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability

- **src/pynomaly/domain/exceptions/result_exceptions.py**
  - Expected test: `unit/domain/exceptions/test_result_exceptions.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability


### Domain Services
- **src/pynomaly/domain/services/__init__.py**
  - Expected test: `unit/domain/services/test___init__.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/active_learning_service.py**
  - Expected test: `unit/domain/services/test_active_learning_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/advanced_detection_service.py**
  - Expected test: `unit/domain/services/test_advanced_detection_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/anomaly_scorer.py**
  - Expected test: `unit/domain/services/test_anomaly_scorer.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/automl_service.py**
  - Expected test: `unit/domain/services/test_automl_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/ensemble_aggregator.py**
  - Expected test: `unit/domain/services/test_ensemble_aggregator.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/explainability_service.py**
  - Expected test: `unit/domain/services/test_explainability_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/explainable_ai_service.py**
  - Expected test: `unit/domain/services/test_explainable_ai_service.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/feature_validator.py**
  - Expected test: `unit/domain/services/test_feature_validator.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/metrics_calculator.py**
  - Expected test: `unit/domain/services/test_metrics_calculator.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/processing_orchestrator.py**
  - Expected test: `unit/domain/services/test_processing_orchestrator.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/threshold_calculator.py**
  - Expected test: `unit/domain/services/test_threshold_calculator.py`
  - Test type: unit
  - Reason: Services contain business logic

- **src/pynomaly/domain/services/threshold_severity_classifier.py**
  - Expected test: `unit/domain/services/test_threshold_severity_classifier.py`
  - Test type: unit
  - Reason: Services contain business logic


### Infrastructure
- **src/pynomaly/infrastructure/services/hyperparameter_optimization_service.py**
  - Expected test: `unit/infrastructure/services/test_hyperparameter_optimization_service.py`
  - Test type: unit
  - Reason: Services contain business logic


### Other
- **src/pynomaly/docs_validation/core/exceptions.py**
  - Expected test: `unit/docs_validation/core/test_exceptions.py`
  - Test type: unit
  - Reason: Exception handling is critical for reliability


### Presentation Layer
- **src/pynomaly/presentation/sdk/exceptions.py**
  - Expected test: `integration/presentation/sdk/test_exceptions.py`
  - Test type: integration
  - Reason: Exception handling is critical for reliability


### Protocol Definitions
- **src/pynomaly/shared/protocols/__init__.py**
  - Expected test: `unit/shared/protocols/test___init__.py`
  - Test type: unit
  - Reason: Protocol definitions are critical interfaces

- **src/pynomaly/shared/protocols/data_loader_protocol.py**
  - Expected test: `unit/shared/protocols/test_data_loader_protocol.py`
  - Test type: unit
  - Reason: Protocol definitions are critical interfaces

- **src/pynomaly/shared/protocols/detector_protocol.py**
  - Expected test: `unit/shared/protocols/test_detector_protocol.py`
  - Test type: unit
  - Reason: Protocol definitions are critical interfaces

- **src/pynomaly/shared/protocols/export_protocol.py**
  - Expected test: `unit/shared/protocols/test_export_protocol.py`
  - Test type: unit
  - Reason: Protocol definitions are critical interfaces

- **src/pynomaly/shared/protocols/import_protocol.py**
  - Expected test: `unit/shared/protocols/test_import_protocol.py`
  - Test type: unit
  - Reason: Protocol definitions are critical interfaces

- **src/pynomaly/shared/protocols/repository_protocol.py**
  - Expected test: `unit/shared/protocols/test_repository_protocol.py`
  - Test type: unit
  - Reason: Protocol definitions are critical interfaces


### Security
- **src/pynomaly/infrastructure/security/advanced_threat_detection.py**
  - Expected test: `unit/infrastructure/security/test_advanced_threat_detection.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/apply_rbac_to_endpoints.py**
  - Expected test: `unit/infrastructure/security/test_apply_rbac_to_endpoints.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/audit_logger.py**
  - Expected test: `unit/infrastructure/security/test_audit_logger.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/audit_logging.py**
  - Expected test: `unit/infrastructure/security/test_audit_logging.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/audit_service.py**
  - Expected test: `unit/infrastructure/security/test_audit_service.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/encryption.py**
  - Expected test: `unit/infrastructure/security/test_encryption.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/input_sanitizer.py**
  - Expected test: `unit/infrastructure/security/test_input_sanitizer.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/middleware_integration.py**
  - Expected test: `unit/infrastructure/security/test_middleware_integration.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/privilege_escalation_prevention.py**
  - Expected test: `unit/infrastructure/security/test_privilege_escalation_prevention.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/rbac_middleware.py**
  - Expected test: `unit/infrastructure/security/test_rbac_middleware.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/rbac_service.py**
  - Expected test: `unit/infrastructure/security/test_rbac_service.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/security_headers.py**
  - Expected test: `unit/infrastructure/security/test_security_headers.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/security_monitor.py**
  - Expected test: `unit/infrastructure/security/test_security_monitor.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/security_service.py**
  - Expected test: `unit/infrastructure/security/test_security_service.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/sql_protection.py**
  - Expected test: `unit/infrastructure/security/test_sql_protection.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/threat_detection_config.py**
  - Expected test: `unit/infrastructure/security/test_threat_detection_config.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/user_tracking.py**
  - Expected test: `unit/infrastructure/security/test_user_tracking.py`
  - Test type: unit
  - Reason: Security modules are critical

- **src/pynomaly/infrastructure/security/validation.py**
  - Expected test: `unit/infrastructure/security/test_validation.py`
  - Test type: unit
  - Reason: Security modules are critical


### Shared Modules
- **src/pynomaly/shared/__init__.py**
  - Expected test: `unit/shared/test___init__.py`
  - Test type: unit
  - Reason: Shared modules are used across the application

- **src/pynomaly/shared/error_handling.py**
  - Expected test: `unit/shared/test_error_handling.py`
  - Test type: unit
  - Reason: Shared modules are used across the application

- **src/pynomaly/shared/exceptions.py**
  - Expected test: `unit/shared/test_exceptions.py`
  - Test type: unit
  - Reason: Shared modules are used across the application

- **src/pynomaly/shared/types.py**
  - Expected test: `unit/shared/test_types.py`
  - Test type: unit
  - Reason: Shared modules are used across the application

- **src/pynomaly/shared/utils/__init__.py**
  - Expected test: `unit/shared/utils/test___init__.py`
  - Test type: unit
  - Reason: Shared modules are used across the application


### Use Cases
- **src/pynomaly/application/use_cases/__init__.py**
  - Expected test: `unit/application/use_cases/test___init__.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/automl_optimization.py**
  - Expected test: `unit/application/use_cases/test_automl_optimization.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/automl_use_case.py**
  - Expected test: `unit/application/use_cases/test_automl_use_case.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/detect_anomalies.py**
  - Expected test: `unit/application/use_cases/test_detect_anomalies.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/drift_monitoring_use_case.py**
  - Expected test: `unit/application/use_cases/test_drift_monitoring_use_case.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/ensemble_detection_use_case.py**
  - Expected test: `unit/application/use_cases/test_ensemble_detection_use_case.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/evaluate_model.py**
  - Expected test: `unit/application/use_cases/test_evaluate_model.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/explain_anomaly.py**
  - Expected test: `unit/application/use_cases/test_explain_anomaly.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/explain_anomaly_use_case.py**
  - Expected test: `unit/application/use_cases/test_explain_anomaly_use_case.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/explainability_use_case.py**
  - Expected test: `unit/application/use_cases/test_explainability_use_case.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/manage_active_learning.py**
  - Expected test: `unit/application/use_cases/test_manage_active_learning.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/streaming_detection_use_case.py**
  - Expected test: `unit/application/use_cases/test_streaming_detection_use_case.py`
  - Test type: unit
  - Reason: Use cases define application behavior

- **src/pynomaly/application/use_cases/train_detector.py**
  - Expected test: `unit/application/use_cases/test_train_detector.py`
  - Test type: unit
  - Reason: Use cases define application behavior


### Value Objects
- **src/pynomaly/domain/value_objects/__init__.py**
  - Expected test: `unit/domain/value_objects/test___init__.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/anomaly_category.py**
  - Expected test: `unit/domain/value_objects/test_anomaly_category.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/anomaly_type.py**
  - Expected test: `unit/domain/value_objects/test_anomaly_type.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/anomaly_type_placeholder.py**
  - Expected test: `unit/domain/value_objects/test_anomaly_type_placeholder.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/confidence_interval.py**
  - Expected test: `unit/domain/value_objects/test_confidence_interval.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/contamination_rate.py**
  - Expected test: `unit/domain/value_objects/test_contamination_rate.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/hyperparameters.py**
  - Expected test: `unit/domain/value_objects/test_hyperparameters.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/model_storage_info.py**
  - Expected test: `unit/domain/value_objects/test_model_storage_info.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/performance_metrics.py**
  - Expected test: `unit/domain/value_objects/test_performance_metrics.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/semantic_version.py**
  - Expected test: `unit/domain/value_objects/test_semantic_version.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/severity_score.py**
  - Expected test: `unit/domain/value_objects/test_severity_score.py`
  - Test type: unit
  - Reason: Value objects contain business logic

- **src/pynomaly/domain/value_objects/threshold_config.py**
  - Expected test: `unit/domain/value_objects/test_threshold_config.py`
  - Test type: unit
  - Reason: Value objects contain business logic

## MEDIUM Priority Files Missing Tests (324 files)
--------------------------------------------------

### Authentication
- **src/pynomaly/infrastructure/auth/dependencies.py**
  - Expected test: `unit/infrastructure/auth/test_dependencies.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/auth/enhanced_dependencies.py**
  - Expected test: `unit/infrastructure/auth/test_enhanced_dependencies.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/auth/jwt_auth_enhanced.py**
  - Expected test: `unit/infrastructure/auth/test_jwt_auth_enhanced.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/auth/middleware.py**
  - Expected test: `unit/infrastructure/auth/test_middleware.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/auth/rate_limiting.py**
  - Expected test: `unit/infrastructure/auth/test_rate_limiting.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/auth/rbac_examples.py**
  - Expected test: `unit/infrastructure/auth/test_rbac_examples.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/auth/websocket_auth.py**
  - Expected test: `unit/infrastructure/auth/test_websocket_auth.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage


### Configuration
- **src/pynomaly/infrastructure/config/__init__.py**
  - Expected test: `unit/infrastructure/config/test___init__.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/config_manager.py**
  - Expected test: `unit/infrastructure/config/test_config_manager.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/config_templates.py**
  - Expected test: `unit/infrastructure/config/test_config_templates.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/config_validator.py**
  - Expected test: `unit/infrastructure/config/test_config_validator.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/container.py**
  - Expected test: `unit/infrastructure/config/test_container.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/container_backup.py**
  - Expected test: `unit/infrastructure/config/test_container_backup.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/database_config.py**
  - Expected test: `unit/infrastructure/config/test_database_config.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/enhanced_config_loader.py**
  - Expected test: `unit/infrastructure/config/test_enhanced_config_loader.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/feature_flags.py**
  - Expected test: `unit/infrastructure/config/test_feature_flags.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/optimization_config.py**
  - Expected test: `unit/infrastructure/config/test_optimization_config.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/service_registry.py**
  - Expected test: `unit/infrastructure/config/test_service_registry.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/settings.py**
  - Expected test: `unit/infrastructure/config/test_settings.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/simplified_container.py**
  - Expected test: `unit/infrastructure/config/test_simplified_container.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior

- **src/pynomaly/infrastructure/config/tdd_config.py**
  - Expected test: `unit/infrastructure/config/test_tdd_config.py`
  - Test type: unit
  - Reason: Configuration modules affect behavior


### Data Transfer Objects
- **src/pynomaly/application/dto/export_options.py**
  - Expected test: `unit/application/dto/test_export_options.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage


### Infrastructure
- **src/pynomaly/infrastructure/alerting/intelligent_alerting_engine.py**
  - Expected test: `unit/infrastructure/alerting/test_intelligent_alerting_engine.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/automl/advanced_optimizer.py**
  - Expected test: `unit/infrastructure/automl/test_advanced_optimizer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/batch/batch_processor.py**
  - Expected test: `unit/infrastructure/batch/test_batch_processor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/business_intelligence/reporting_service.py**
  - Expected test: `unit/infrastructure/business_intelligence/test_reporting_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/cache/cache_manager.py**
  - Expected test: `unit/infrastructure/cache/test_cache_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/cache/redis_cache.py**
  - Expected test: `unit/infrastructure/cache/test_redis_cache.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/caching/advanced_cache_service.py**
  - Expected test: `unit/infrastructure/caching/test_advanced_cache_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/caching/cache_manager.py**
  - Expected test: `unit/infrastructure/caching/test_cache_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/caching/cache_strategy.py**
  - Expected test: `unit/infrastructure/caching/test_cache_strategy.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/cicd/deployment_manager.py**
  - Expected test: `unit/infrastructure/cicd/test_deployment_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/compliance/audit_system.py**
  - Expected test: `unit/infrastructure/compliance/test_audit_system.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data/validation_pipeline.py**
  - Expected test: `unit/infrastructure/data/test_validation_pipeline.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/arrow_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_arrow_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/csv_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_csv_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/data_loader_factory.py**
  - Expected test: `unit/infrastructure/data_loaders/test_data_loader_factory.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/database_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_database_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/dvc_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_dvc_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/enhanced_parquet_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_enhanced_parquet_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/excel_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_excel_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/file_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_file_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/json_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_json_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/optimized_csv_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_optimized_csv_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/parquet_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_parquet_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/polars_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_polars_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_loaders/spark_loader.py**
  - Expected test: `unit/infrastructure/data_loaders/test_spark_loader.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_processing/advanced_data_pipeline.py**
  - Expected test: `unit/infrastructure/data_processing/test_advanced_data_pipeline.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_processing/data_validator.py**
  - Expected test: `unit/infrastructure/data_processing/test_data_validator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_processing/memory_efficient_processor.py**
  - Expected test: `unit/infrastructure/data_processing/test_memory_efficient_processor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/data_processing/streaming_processor.py**
  - Expected test: `unit/infrastructure/data_processing/test_streaming_processor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/cluster_coordinator.py**
  - Expected test: `unit/infrastructure/distributed/test_cluster_coordinator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/data_partitioner.py**
  - Expected test: `unit/infrastructure/distributed/test_data_partitioner.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/distributed_config.py**
  - Expected test: `unit/infrastructure/distributed/test_distributed_config.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/distributed_detector.py**
  - Expected test: `unit/infrastructure/distributed/test_distributed_detector.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/load_balancer.py**
  - Expected test: `unit/infrastructure/distributed/test_load_balancer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/result_aggregator.py**
  - Expected test: `unit/infrastructure/distributed/test_result_aggregator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/task_distributor.py**
  - Expected test: `unit/infrastructure/distributed/test_task_distributor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/distributed/worker_manager.py**
  - Expected test: `unit/infrastructure/distributed/test_worker_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/error_handling/error_handler.py**
  - Expected test: `unit/infrastructure/error_handling/test_error_handler.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/error_handling/error_middleware.py**
  - Expected test: `unit/infrastructure/error_handling/test_error_middleware.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/error_handling/error_reporter.py**
  - Expected test: `unit/infrastructure/error_handling/test_error_reporter.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/error_handling/error_response_formatter.py**
  - Expected test: `unit/infrastructure/error_handling/test_error_response_formatter.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/error_handling/problem_details_handler.py**
  - Expected test: `unit/infrastructure/error_handling/test_problem_details_handler.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/error_handling/recovery_strategies.py**
  - Expected test: `unit/infrastructure/error_handling/test_recovery_strategies.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/explainers/lime_explainer.py**
  - Expected test: `unit/infrastructure/explainers/test_lime_explainer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/explainers/shap_explainer.py**
  - Expected test: `unit/infrastructure/explainers/test_shap_explainer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/federated/aggregation.py**
  - Expected test: `unit/infrastructure/federated/test_aggregation.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/federated/coordinator.py**
  - Expected test: `unit/infrastructure/federated/test_coordinator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/federated/participant.py**
  - Expected test: `unit/infrastructure/federated/test_participant.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/global_scale/massive_dataset_processing.py**
  - Expected test: `unit/infrastructure/global_scale/test_massive_dataset_processing.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/global_scale/multi_region_deployment.py**
  - Expected test: `unit/infrastructure/global_scale/test_multi_region_deployment.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/lifecycle/shutdown_service.py**
  - Expected test: `unit/infrastructure/lifecycle/test_shutdown_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/log_aggregator.py**
  - Expected test: `unit/infrastructure/logging/test_log_aggregator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/log_analysis.py**
  - Expected test: `unit/infrastructure/logging/test_log_analysis.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/log_formatter.py**
  - Expected test: `unit/infrastructure/logging/test_log_formatter.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/metrics_collector.py**
  - Expected test: `unit/infrastructure/logging/test_metrics_collector.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/observability_service.py**
  - Expected test: `unit/infrastructure/logging/test_observability_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/structured_logger.py**
  - Expected test: `unit/infrastructure/logging/test_structured_logger.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/logging/tracing_manager.py**
  - Expected test: `unit/infrastructure/logging/test_tracing_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/middleware/configuration_middleware.py**
  - Expected test: `unit/infrastructure/middleware/test_configuration_middleware.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/middleware/correlation_id_middleware.py**
  - Expected test: `unit/infrastructure/middleware/test_correlation_id_middleware.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/multitenancy/data_isolation_service.py**
  - Expected test: `unit/infrastructure/multitenancy/test_data_isolation_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/multitenancy/resource_quota_service.py**
  - Expected test: `unit/infrastructure/multitenancy/test_resource_quota_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/multitenancy/tenant_manager.py**
  - Expected test: `unit/infrastructure/multitenancy/test_tenant_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/optimization/async_optimizer.py**
  - Expected test: `unit/infrastructure/optimization/test_async_optimizer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/optimization/memory_optimizer.py**
  - Expected test: `unit/infrastructure/optimization/test_memory_optimizer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/async_processor.py**
  - Expected test: `unit/infrastructure/performance/test_async_processor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/connection_pooling.py**
  - Expected test: `unit/infrastructure/performance/test_connection_pooling.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/memory_manager.py**
  - Expected test: `unit/infrastructure/performance/test_memory_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/memory_optimizer.py**
  - Expected test: `unit/infrastructure/performance/test_memory_optimizer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/performance_service.py**
  - Expected test: `unit/infrastructure/performance/test_performance_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/profiler.py**
  - Expected test: `unit/infrastructure/performance/test_profiler.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/profiling_service.py**
  - Expected test: `unit/infrastructure/performance/test_profiling_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/query_optimization.py**
  - Expected test: `unit/infrastructure/performance/test_query_optimization.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance/query_optimizer.py**
  - Expected test: `unit/infrastructure/performance/test_query_optimizer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance_v2/advanced_caching_v2.py**
  - Expected test: `unit/infrastructure/performance_v2/test_advanced_caching_v2.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance_v2/real_time_processing_enhancement.py**
  - Expected test: `unit/infrastructure/performance_v2/test_real_time_processing_enhancement.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance_v2/resource_optimization.py**
  - Expected test: `unit/infrastructure/performance_v2/test_resource_optimization.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/performance_v2/ultra_high_performance.py**
  - Expected test: `unit/infrastructure/performance_v2/test_ultra_high_performance.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/configuration_repository.py**
  - Expected test: `unit/infrastructure/persistence/test_configuration_repository.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/database.py**
  - Expected test: `unit/infrastructure/persistence/test_database.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/database_optimizations.py**
  - Expected test: `unit/infrastructure/persistence/test_database_optimizations.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/database_repositories.py**
  - Expected test: `unit/infrastructure/persistence/test_database_repositories.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/enhanced_database.py**
  - Expected test: `unit/infrastructure/persistence/test_enhanced_database.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/migrations.py**
  - Expected test: `unit/infrastructure/persistence/test_migrations.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/optimized_repositories.py**
  - Expected test: `unit/infrastructure/persistence/test_optimized_repositories.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/tdd_repository.py**
  - Expected test: `unit/infrastructure/persistence/test_tdd_repository.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/training_job_repository.py**
  - Expected test: `unit/infrastructure/persistence/test_training_job_repository.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/training_repository.py**
  - Expected test: `unit/infrastructure/persistence/test_training_repository.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/persistence/user_models.py**
  - Expected test: `unit/infrastructure/persistence/test_user_models.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/preprocessing/advanced_preprocessor.py**
  - Expected test: `unit/infrastructure/preprocessing/test_advanced_preprocessor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/preprocessing/data_cleaner.py**
  - Expected test: `unit/infrastructure/preprocessing/test_data_cleaner.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/preprocessing/data_transformer.py**
  - Expected test: `unit/infrastructure/preprocessing/test_data_transformer.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/preprocessing/preprocessing_pipeline.py**
  - Expected test: `unit/infrastructure/preprocessing/test_preprocessing_pipeline.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/quality/quality_gates.py**
  - Expected test: `unit/infrastructure/quality/test_quality_gates.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/resilience/circuit_breaker.py**
  - Expected test: `unit/infrastructure/resilience/test_circuit_breaker.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/resilience/retry.py**
  - Expected test: `unit/infrastructure/resilience/test_retry.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/resilience/service.py**
  - Expected test: `unit/infrastructure/resilience/test_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/resilience/timeout.py**
  - Expected test: `unit/infrastructure/resilience/test_timeout.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/scheduler/resource_manager.py**
  - Expected test: `unit/infrastructure/scheduler/test_resource_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/scheduler/schedule_repository.py**
  - Expected test: `unit/infrastructure/scheduler/test_schedule_repository.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/scheduler/trigger_manager.py**
  - Expected test: `unit/infrastructure/scheduler/test_trigger_manager.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/serving/model_server.py**
  - Expected test: `unit/infrastructure/serving/test_model_server.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/serving/model_server_main.py**
  - Expected test: `unit/infrastructure/serving/test_model_server_main.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/storage/s3_adapter.py**
  - Expected test: `unit/infrastructure/storage/test_s3_adapter.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/streaming/real_time_anomaly_pipeline.py**
  - Expected test: `unit/infrastructure/streaming/test_real_time_anomaly_pipeline.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/streaming/stream_processor.py**
  - Expected test: `unit/infrastructure/streaming/test_stream_processor.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/tdd/enforcement.py**
  - Expected test: `unit/infrastructure/tdd/test_enforcement.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/infrastructure/tdd/git_hooks.py**
  - Expected test: `unit/infrastructure/tdd/test_git_hooks.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage


### Infrastructure Adapters
- **src/pynomaly/infrastructure/adapters/__init__.py**
  - Expected test: `unit/infrastructure/adapters/test___init__.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/advanced_ensemble_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_advanced_ensemble_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/algorithm_factory.py**
  - Expected test: `unit/infrastructure/adapters/test_algorithm_factory.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/autosklearn2_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_autosklearn2_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/deep_learning/__init__.py**
  - Expected test: `unit/infrastructure/adapters/deep_learning/test___init__.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/deep_learning/jax_adapter.py**
  - Expected test: `unit/infrastructure/adapters/deep_learning/test_jax_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/deep_learning/jax_stub.py**
  - Expected test: `unit/infrastructure/adapters/deep_learning/test_jax_stub.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/deep_learning/pytorch_stub.py**
  - Expected test: `unit/infrastructure/adapters/deep_learning/test_pytorch_stub.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/deep_learning/tensorflow_adapter.py**
  - Expected test: `unit/infrastructure/adapters/deep_learning/test_tensorflow_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/deep_learning/tensorflow_stub.py**
  - Expected test: `unit/infrastructure/adapters/deep_learning/test_tensorflow_stub.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/drift_detection_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_drift_detection_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/enhanced_pyod_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_enhanced_pyod_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/enhanced_pyod_service.py**
  - Expected test: `unit/infrastructure/adapters/test_enhanced_pyod_service.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/enhanced_sklearn_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_enhanced_sklearn_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/ensemble_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_ensemble_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/ensemble_meta_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_ensemble_meta_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/excel_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_excel_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/jax_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_jax_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/model_trainer_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_model_trainer_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/optimized_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_optimized_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/optimized_pyod_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_optimized_pyod_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/pygod_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_pygod_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/pyod_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_pyod_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/sklearn_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_sklearn_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/tenant_repository.py**
  - Expected test: `unit/infrastructure/adapters/test_tenant_repository.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/tensorflow_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_tensorflow_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/time_series_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_time_series_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing

- **src/pynomaly/infrastructure/adapters/uncertainty_adapter.py**
  - Expected test: `unit/infrastructure/adapters/test_uncertainty_adapter.py`
  - Test type: unit
  - Reason: External integrations need testing


### Monitoring
- **src/pynomaly/infrastructure/monitoring/advanced_alerting_service.py**
  - Expected test: `unit/infrastructure/monitoring/test_advanced_alerting_service.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/alerting_service.py**
  - Expected test: `unit/infrastructure/monitoring/test_alerting_service.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/autonomous_monitor.py**
  - Expected test: `unit/infrastructure/monitoring/test_autonomous_monitor.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/cli_parameter_interceptor.py**
  - Expected test: `unit/infrastructure/monitoring/test_cli_parameter_interceptor.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/complexity_monitor.py**
  - Expected test: `unit/infrastructure/monitoring/test_complexity_monitor.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/dashboard_service.py**
  - Expected test: `unit/infrastructure/monitoring/test_dashboard_service.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/dashboards.py**
  - Expected test: `unit/infrastructure/monitoring/test_dashboards.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/distributed_tracing.py**
  - Expected test: `unit/infrastructure/monitoring/test_distributed_tracing.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/external_monitoring_service.py**
  - Expected test: `unit/infrastructure/monitoring/test_external_monitoring_service.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/health_checks.py**
  - Expected test: `unit/infrastructure/monitoring/test_health_checks.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/health_service.py**
  - Expected test: `unit/infrastructure/monitoring/test_health_service.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/metrics_service.py**
  - Expected test: `unit/infrastructure/monitoring/test_metrics_service.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/middleware.py**
  - Expected test: `unit/infrastructure/monitoring/test_middleware.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/performance_monitor.py**
  - Expected test: `unit/infrastructure/monitoring/test_performance_monitor.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/production_monitor.py**
  - Expected test: `unit/infrastructure/monitoring/test_production_monitor.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/prometheus_metrics.py**
  - Expected test: `unit/infrastructure/monitoring/test_prometheus_metrics.py`
  - Test type: unit
  - Reason: Monitoring affects operations

- **src/pynomaly/infrastructure/monitoring/telemetry.py**
  - Expected test: `unit/infrastructure/monitoring/test_telemetry.py`
  - Test type: unit
  - Reason: Monitoring affects operations


### Other
- **src/pynomaly/docs_validation/core/config.py**
  - Expected test: `unit/docs_validation/core/test_config.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/docs_validation/core/reporter.py**
  - Expected test: `unit/docs_validation/core/test_reporter.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/docs_validation/core/validator.py**
  - Expected test: `unit/docs_validation/core/test_validator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/abstractions/base_entity.py**
  - Expected test: `unit/domain/abstractions/test_base_entity.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/abstractions/base_repository.py**
  - Expected test: `unit/domain/abstractions/test_base_repository.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/abstractions/base_service.py**
  - Expected test: `unit/domain/abstractions/test_base_service.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/abstractions/base_value_object.py**
  - Expected test: `unit/domain/abstractions/test_base_value_object.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/abstractions/specification.py**
  - Expected test: `unit/domain/abstractions/test_specification.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/common/versioned.py**
  - Expected test: `unit/domain/common/test_versioned.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/active_learning.py**
  - Expected test: `unit/domain/models/test_active_learning.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/base.py**
  - Expected test: `unit/domain/models/test_base.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/causal.py**
  - Expected test: `unit/domain/models/test_causal.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/cicd.py**
  - Expected test: `unit/domain/models/test_cicd.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/detector.py**
  - Expected test: `unit/domain/models/test_detector.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/federated.py**
  - Expected test: `unit/domain/models/test_federated.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/kubernetes.py**
  - Expected test: `unit/domain/models/test_kubernetes.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/monitoring.py**
  - Expected test: `unit/domain/models/test_monitoring.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/multimodal.py**
  - Expected test: `unit/domain/models/test_multimodal.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/multitenancy.py**
  - Expected test: `unit/domain/models/test_multitenancy.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/models/nas.py**
  - Expected test: `unit/domain/models/test_nas.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/domain/repositories/user_repository.py**
  - Expected test: `unit/domain/repositories/test_user_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/research/automl/automl_v2.py**
  - Expected test: `unit/research/automl/test_automl_v2.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/research/edge/edge_deployment.py**
  - Expected test: `unit/research/edge/test_edge_deployment.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/research/explainability/explainable_ai.py**
  - Expected test: `unit/research/explainability/test_explainable_ai.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/research/quantum/quantum_algorithms.py**
  - Expected test: `unit/research/quantum/test_quantum_algorithms.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/research/synthetic/synthetic_data_generation.py**
  - Expected test: `unit/research/synthetic/test_synthetic_data_generation.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/schemas/analytics/anomaly_kpis.py**
  - Expected test: `unit/schemas/analytics/test_anomaly_kpis.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/schemas/analytics/base.py**
  - Expected test: `unit/schemas/analytics/test_base.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/schemas/analytics/system_health.py**
  - Expected test: `unit/schemas/analytics/test_system_health.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/scripts/init_database.py**
  - Expected test: `unit/scripts/test_init_database.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/testing/orchestration/orchestrator.py**
  - Expected test: `unit/testing/orchestration/test_orchestrator.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage

- **src/pynomaly/tests/test_scheduler_direct.py**
  - Expected test: `unit/tests/test_test_scheduler_direct.py`
  - Test type: unit
  - Reason: Standard module requiring test coverage


### Presentation Layer
- **src/pynomaly/presentation/api/active_learning_endpoints.py**
  - Expected test: `integration/presentation/api/test_active_learning_endpoints.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/app.py**
  - Expected test: `integration/presentation/api/test_app.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/app_runner.py**
  - Expected test: `integration/presentation/api/test_app_runner.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/auth_deps.py**
  - Expected test: `integration/presentation/api/test_auth_deps.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/dependencies/auth.py**
  - Expected test: `integration/presentation/api/dependencies/test_auth.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/dependencies/container.py**
  - Expected test: `integration/presentation/api/dependencies/test_container.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/deps.py**
  - Expected test: `integration/presentation/api/test_deps.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/docs/api_docs.py**
  - Expected test: `integration/presentation/api/docs/test_api_docs.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/docs/openapi_config.py**
  - Expected test: `integration/presentation/api/docs/test_openapi_config.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/docs/response_models.py**
  - Expected test: `integration/presentation/api/docs/test_response_models.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/docs/schema_examples.py**
  - Expected test: `integration/presentation/api/docs/test_schema_examples.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/admin.py**
  - Expected test: `integration/presentation/api/endpoints/test_admin.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/advanced_ml_lifecycle.py**
  - Expected test: `integration/presentation/api/endpoints/test_advanced_ml_lifecycle.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/auth.py**
  - Expected test: `integration/presentation/api/endpoints/test_auth.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/automl.py**
  - Expected test: `integration/presentation/api/endpoints/test_automl.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/autonomous.py**
  - Expected test: `integration/presentation/api/endpoints/test_autonomous.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/datasets.py**
  - Expected test: `integration/presentation/api/endpoints/test_datasets.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/detection.py**
  - Expected test: `integration/presentation/api/endpoints/test_detection.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/detectors.py**
  - Expected test: `integration/presentation/api/endpoints/test_detectors.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/drift_monitoring.py**
  - Expected test: `integration/presentation/api/endpoints/test_drift_monitoring.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/ensemble.py**
  - Expected test: `integration/presentation/api/endpoints/test_ensemble.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/enterprise_dashboard.py**
  - Expected test: `integration/presentation/api/endpoints/test_enterprise_dashboard.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/events.py**
  - Expected test: `integration/presentation/api/endpoints/test_events.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/experiments.py**
  - Expected test: `integration/presentation/api/endpoints/test_experiments.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/explainability.py**
  - Expected test: `integration/presentation/api/endpoints/test_explainability.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/explainable_ai.py**
  - Expected test: `integration/presentation/api/endpoints/test_explainable_ai.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/export.py**
  - Expected test: `integration/presentation/api/endpoints/test_export.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/health.py**
  - Expected test: `integration/presentation/api/endpoints/test_health.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/jwks.py**
  - Expected test: `integration/presentation/api/endpoints/test_jwks.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/model_lineage.py**
  - Expected test: `integration/presentation/api/endpoints/test_model_lineage.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/monitoring.py**
  - Expected test: `integration/presentation/api/endpoints/test_monitoring.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/performance.py**
  - Expected test: `integration/presentation/api/endpoints/test_performance.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/streaming.py**
  - Expected test: `integration/presentation/api/endpoints/test_streaming.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/streaming_pipelines.py**
  - Expected test: `integration/presentation/api/endpoints/test_streaming_pipelines.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/training.py**
  - Expected test: `integration/presentation/api/endpoints/test_training.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/version.py**
  - Expected test: `integration/presentation/api/endpoints/test_version.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/endpoints/websocket.py**
  - Expected test: `integration/presentation/api/endpoints/test_websocket.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/enhanced_automl.py**
  - Expected test: `integration/presentation/api/test_enhanced_automl.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/middleware_integration.py**
  - Expected test: `integration/presentation/api/test_middleware_integration.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/performance_metrics.py**
  - Expected test: `integration/presentation/api/test_performance_metrics.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/router_factory.py**
  - Expected test: `integration/presentation/api/test_router_factory.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/routers/compliance.py**
  - Expected test: `integration/presentation/api/routers/test_compliance.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/routers/integrations.py**
  - Expected test: `integration/presentation/api/routers/test_integrations.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/routers/reporting.py**
  - Expected test: `integration/presentation/api/routers/test_reporting.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/routers/user_management.py**
  - Expected test: `integration/presentation/api/routers/test_user_management.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/tenant_endpoints.py**
  - Expected test: `integration/presentation/api/test_tenant_endpoints.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/training_automation_endpoints.py**
  - Expected test: `integration/presentation/api/test_training_automation_endpoints.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/uncertainty_endpoints.py**
  - Expected test: `integration/presentation/api/test_uncertainty_endpoints.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/versioning.py**
  - Expected test: `integration/presentation/api/test_versioning.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/api/websocket/training_handler.py**
  - Expected test: `integration/presentation/api/websocket/test_training_handler.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/alert.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_alert.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/benchmarking.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_benchmarking.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/cost_optimization.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_cost_optimization.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/dashboard.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_dashboard.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/enhanced_automl.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_enhanced_automl.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/ensemble.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_ensemble.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/explain.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_explain.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/governance.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_governance.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/quality.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_quality.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/tenant.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_tenant.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/_click_backup/training_automation_commands.py**
  - Expected test: `integration/presentation/cli/_click_backup/test_training_automation_commands.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/alert.py**
  - Expected test: `integration/presentation/cli/test_alert.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/app.py**
  - Expected test: `integration/presentation/cli/test_app.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/automl.py**
  - Expected test: `integration/presentation/cli/test_automl.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/autonomous.py**
  - Expected test: `integration/presentation/cli/test_autonomous.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/autonomous_enhancements.py**
  - Expected test: `integration/presentation/cli/test_autonomous_enhancements.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/benchmarking.py**
  - Expected test: `integration/presentation/cli/test_benchmarking.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/config.py**
  - Expected test: `integration/presentation/cli/test_config.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/container.py**
  - Expected test: `integration/presentation/cli/test_container.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/cost_optimization.py**
  - Expected test: `integration/presentation/cli/test_cost_optimization.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/dashboard.py**
  - Expected test: `integration/presentation/cli/test_dashboard.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/datasets.py**
  - Expected test: `integration/presentation/cli/test_datasets.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/deep_learning.py**
  - Expected test: `integration/presentation/cli/test_deep_learning.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/deployment.py**
  - Expected test: `integration/presentation/cli/test_deployment.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/detection.py**
  - Expected test: `integration/presentation/cli/test_detection.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/detectors.py**
  - Expected test: `integration/presentation/cli/test_detectors.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/enhanced_automl.py**
  - Expected test: `integration/presentation/cli/test_enhanced_automl.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/ensemble.py**
  - Expected test: `integration/presentation/cli/test_ensemble.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/enterprise_dashboard.py**
  - Expected test: `integration/presentation/cli/test_enterprise_dashboard.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/explain.py**
  - Expected test: `integration/presentation/cli/test_explain.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/explainability.py**
  - Expected test: `integration/presentation/cli/test_explainability.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/export.py**
  - Expected test: `integration/presentation/cli/test_export.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/governance.py**
  - Expected test: `integration/presentation/cli/test_governance.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/performance.py**
  - Expected test: `integration/presentation/cli/test_performance.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/preprocessing.py**
  - Expected test: `integration/presentation/cli/test_preprocessing.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/quality.py**
  - Expected test: `integration/presentation/cli/test_quality.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/recommendation.py**
  - Expected test: `integration/presentation/cli/test_recommendation.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/selection.py**
  - Expected test: `integration/presentation/cli/test_selection.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/server.py**
  - Expected test: `integration/presentation/cli/test_server.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/tdd.py**
  - Expected test: `integration/presentation/cli/test_tdd.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/tenant.py**
  - Expected test: `integration/presentation/cli/test_tenant.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/training_automation_commands.py**
  - Expected test: `integration/presentation/cli/test_training_automation_commands.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/cli/validation.py**
  - Expected test: `integration/presentation/cli/test_validation.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/sdk/async_client.py**
  - Expected test: `integration/presentation/sdk/test_async_client.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/sdk/client.py**
  - Expected test: `integration/presentation/sdk/test_client.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/sdk/config.py**
  - Expected test: `integration/presentation/sdk/test_config.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/sdk/examples.py**
  - Expected test: `integration/presentation/sdk/test_examples.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/sdk/models.py**
  - Expected test: `integration/presentation/sdk/test_models.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/web/app.py**
  - Expected test: `integration/presentation/web/test_app.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/web/design_system.py**
  - Expected test: `integration/presentation/web/test_design_system.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/web/models/dashboard_models.py**
  - Expected test: `integration/presentation/web/models/test_dashboard_models.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/web/models/ui_models.py**
  - Expected test: `integration/presentation/web/models/test_ui_models.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/web/models/websocket_models.py**
  - Expected test: `integration/presentation/web/models/test_websocket_models.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage

- **src/pynomaly/presentation/web/routes/websocket_routes.py**
  - Expected test: `integration/presentation/web/routes/test_websocket_routes.py`
  - Test type: integration
  - Reason: Standard module requiring test coverage


### Repositories
- **src/pynomaly/infrastructure/repositories/__init__.py**
  - Expected test: `unit/infrastructure/repositories/test___init__.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/async_wrappers.py**
  - Expected test: `unit/infrastructure/repositories/test_async_wrappers.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/dataset_repository.py**
  - Expected test: `unit/infrastructure/repositories/test_dataset_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/detector_repository.py**
  - Expected test: `unit/infrastructure/repositories/test_detector_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/factory.py**
  - Expected test: `unit/infrastructure/repositories/test_factory.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/file_repositories.py**
  - Expected test: `unit/infrastructure/repositories/test_file_repositories.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/in_memory_repositories.py**
  - Expected test: `unit/infrastructure/repositories/test_in_memory_repositories.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/memory_repository.py**
  - Expected test: `unit/infrastructure/repositories/test_memory_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/model_performance_repository.py**
  - Expected test: `unit/infrastructure/repositories/test_model_performance_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/performance_baseline_repository.py**
  - Expected test: `unit/infrastructure/repositories/test_performance_baseline_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/repository_factory.py**
  - Expected test: `unit/infrastructure/repositories/test_repository_factory.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/repository_service.py**
  - Expected test: `unit/infrastructure/repositories/test_repository_service.py`
  - Test type: unit
  - Reason: Data access layer needs testing

- **src/pynomaly/infrastructure/repositories/sqlalchemy_user_repository.py**
  - Expected test: `unit/infrastructure/repositories/test_sqlalchemy_user_repository.py`
  - Test type: unit
  - Reason: Data access layer needs testing

## LOW Priority Files Missing Tests (48 files)
--------------------------------------------------

### Authentication
- **src/pynomaly/infrastructure/auth/__init__.py**
  - Expected test: `unit/infrastructure/auth/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules


### Data Transfer Objects
- **src/pynomaly/application/dto/__init__.py**
  - Expected test: `unit/application/dto/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules


### Infrastructure
- **src/pynomaly/infrastructure/__init__.py**
  - Expected test: `unit/infrastructure/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/alerting/__init__.py**
  - Expected test: `unit/infrastructure/alerting/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/automl/__init__.py**
  - Expected test: `unit/infrastructure/automl/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/batch/__init__.py**
  - Expected test: `unit/infrastructure/batch/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/business_intelligence/__init__.py**
  - Expected test: `unit/infrastructure/business_intelligence/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/cache/__init__.py**
  - Expected test: `unit/infrastructure/cache/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/data_loaders/__init__.py**
  - Expected test: `unit/infrastructure/data_loaders/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/data_processing/__init__.py**
  - Expected test: `unit/infrastructure/data_processing/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/distributed/__init__.py**
  - Expected test: `unit/infrastructure/distributed/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/error_handling/__init__.py**
  - Expected test: `unit/infrastructure/error_handling/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/explainers/__init__.py**
  - Expected test: `unit/infrastructure/explainers/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/federated/__init__.py**
  - Expected test: `unit/infrastructure/federated/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/lifecycle/__init__.py**
  - Expected test: `unit/infrastructure/lifecycle/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/logging/__init__.py**
  - Expected test: `unit/infrastructure/logging/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/middleware/__init__.py**
  - Expected test: `unit/infrastructure/middleware/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/multitenancy/__init__.py**
  - Expected test: `unit/infrastructure/multitenancy/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/performance/__init__.py**
  - Expected test: `unit/infrastructure/performance/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/persistence/__init__.py**
  - Expected test: `unit/infrastructure/persistence/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/preprocessing/__init__.py**
  - Expected test: `unit/infrastructure/preprocessing/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/quality/__init__.py**
  - Expected test: `unit/infrastructure/quality/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/resilience/__init__.py**
  - Expected test: `unit/infrastructure/resilience/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/storage/__init__.py**
  - Expected test: `unit/infrastructure/storage/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/streaming/__init__.py**
  - Expected test: `unit/infrastructure/streaming/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/infrastructure/tdd/__init__.py**
  - Expected test: `unit/infrastructure/tdd/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules


### Monitoring
- **src/pynomaly/infrastructure/monitoring/__init__.py**
  - Expected test: `unit/infrastructure/monitoring/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules


### Other
- **src/pynomaly/__init__.py**
  - Expected test: `unit/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/application/__init__.py**
  - Expected test: `unit/application/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/docs_validation/__init__.py**
  - Expected test: `unit/docs_validation/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/docs_validation/core/__init__.py**
  - Expected test: `unit/docs_validation/core/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/domain/__init__.py**
  - Expected test: `unit/domain/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/domain/abstractions/__init__.py**
  - Expected test: `unit/domain/abstractions/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/domain/common/__init__.py**
  - Expected test: `unit/domain/common/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/domain/models/__init__.py**
  - Expected test: `unit/domain/models/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/domain/validation/__init__.py**
  - Expected test: `unit/domain/validation/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/scripts/__init__.py**
  - Expected test: `unit/scripts/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/testing/__init__.py**
  - Expected test: `unit/testing/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules

- **src/pynomaly/testing/orchestration/__init__.py**
  - Expected test: `unit/testing/orchestration/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules


### Presentation Layer
- **src/pynomaly/presentation/__init__.py**
  - Expected test: `integration/presentation/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/api/__init__.py**
  - Expected test: `integration/presentation/api/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/api/dependencies/__init__.py**
  - Expected test: `integration/presentation/api/dependencies/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/api/docs/__init__.py**
  - Expected test: `integration/presentation/api/docs/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/api/endpoints/__init__.py**
  - Expected test: `integration/presentation/api/endpoints/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/cli/__init__.py**
  - Expected test: `integration/presentation/cli/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/sdk/__init__.py**
  - Expected test: `integration/presentation/sdk/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules

- **src/pynomaly/presentation/web/__init__.py**
  - Expected test: `integration/presentation/web/test___init__.py`
  - Test type: integration
  - Reason: Init files typically just import modules


### Security
- **src/pynomaly/infrastructure/security/__init__.py**
  - Expected test: `unit/infrastructure/security/test___init__.py`
  - Test type: unit
  - Reason: Init files typically just import modules
