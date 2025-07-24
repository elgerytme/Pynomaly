"""Refactored model registry service using hexagonal architecture."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from mlops.domain.interfaces.model_registry_operations import (
    ModelRegistryPort,
    ModelLifecyclePort,
    ModelDeploymentPort,
    ModelStoragePort,
    ModelVersioningPort,
    ModelSearchPort,
    ModelRegistrationRequest,
    ModelInfo,
    ModelStatus,
    ModelValidationResult,
    DeploymentConfig,
    DeploymentInfo,
    DeploymentStage,
    ModelSearchQuery,
    ModelFramework,
    ModelMetadata,
    ModelMetrics
)

logger = logging.getLogger(__name__)


class ModelRegistryService:
    """Clean domain service for model registry management using dependency injection."""
    
    def __init__(
        self,
        model_registry_port: ModelRegistryPort,
        model_lifecycle_port: ModelLifecyclePort,
        model_deployment_port: ModelDeploymentPort,
        model_storage_port: ModelStoragePort,
        model_versioning_port: ModelVersioningPort,
        model_search_port: ModelSearchPort
    ):
        """Initialize service with injected dependencies.
        
        Args:
            model_registry_port: Port for model registry operations
            model_lifecycle_port: Port for model lifecycle management
            model_deployment_port: Port for model deployment
            model_storage_port: Port for model storage
            model_versioning_port: Port for model versioning
            model_search_port: Port for model search
        """
        self._model_registry_port = model_registry_port
        self._model_lifecycle_port = model_lifecycle_port
        self._model_deployment_port = model_deployment_port
        self._model_storage_port = model_storage_port
        self._model_versioning_port = model_versioning_port
        self._model_search_port = model_search_port
        
        logger.info("ModelRegistryService initialized with clean architecture")
    
    async def register_model(
        self,
        model_path: str,
        model_id: str,
        algorithm: str,
        framework: ModelFramework,
        created_by: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        experiment_id: Optional[str] = None,
        validate_model: bool = True
    ) -> str:
        """Register a new model with comprehensive validation.
        
        Args:
            model_path: Path to the model file
            model_id: Unique model identifier
            algorithm: Algorithm name
            framework: ML framework used
            created_by: User registering the model
            description: Model description
            tags: Optional tags for categorization
            hyperparameters: Model hyperparameters
            metrics: Performance metrics
            experiment_id: Associated experiment ID
            validate_model: Whether to validate the model
            
        Returns:
            Model identifier (model_id:version)
            
        Raises:
            ValueError: If validation fails
        """
        # Domain validation
        await self._validate_model_registration_inputs(
            model_path, model_id, algorithm, framework, created_by
        )
        
        # Create new version
        version = await self._model_versioning_port.create_version(model_id)
        
        # Process and validate inputs
        processed_tags = self._process_model_tags(tags or [])
        validated_hyperparameters = self._validate_hyperparameters(hyperparameters or {})
        validated_metrics = self._validate_model_metrics(metrics or {})
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            algorithm=algorithm,
            framework=framework,
            created_by=created_by,
            description=description,
            tags=processed_tags,
            hyperparameters=validated_hyperparameters,
            training_data_info={"registered_at": datetime.utcnow().isoformat()},
            feature_schema={},
            model_signature={},
            dependencies=[],
            custom_metadata={}
        )
        
        # Create metrics object
        model_metrics = ModelMetrics(
            accuracy=validated_metrics.get("accuracy"),
            precision=validated_metrics.get("precision"),
            recall=validated_metrics.get("recall"),
            f1_score=validated_metrics.get("f1_score"),
            auc_roc=validated_metrics.get("auc_roc"),
            custom_metrics={
                k: v for k, v in validated_metrics.items()
                if k not in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
            }
        )
        
        # Create registration request
        request = ModelRegistrationRequest(
            model_path=model_path,
            metadata=metadata,
            metrics=model_metrics,
            experiment_id=experiment_id,
            validate_model=validate_model
        )
        
        try:
            # Register the model
            model_key = await self._model_registry_port.register_model(request)
            
            # Apply business rules for new models
            await self._apply_new_model_business_rules(model_id, version, model_metrics)
            
            logger.info(f"Registered model {model_key} by {created_by}")
            return model_key
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def promote_model(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStatus,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Promote model with comprehensive validation and business rules.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            target_stage: Target lifecycle stage
            validation_config: Optional validation configuration
            
        Returns:
            True if promotion successful
            
        Raises:
            ValueError: If validation fails
        """
        # Validate model exists
        model = await self._model_registry_port.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id}:{version} not found")
        
        # Check promotion eligibility
        eligibility = await self._model_lifecycle_port.check_promotion_eligibility(
            model_id, version, target_stage
        )
        
        if not eligibility.get("eligible", False):
            blocking_issues = eligibility.get("blocking_issues", [])
            raise ValueError(f"Model not eligible for promotion: {', '.join(blocking_issues)}")
        
        # Validate model if promoting to production
        if target_stage == ModelStatus.PRODUCTION:
            validation_result = await self._model_lifecycle_port.validate_model(
                model_id, version, validation_config
            )
            
            if not validation_result.is_valid:
                raise ValueError(f"Model validation failed: {', '.join(validation_result.errors)}")
        
        try:
            # Apply pre-promotion business rules
            await self._apply_pre_promotion_rules(model, target_stage)
            
            # Perform promotion
            success = await self._model_lifecycle_port.promote_model(
                model_id, version, target_stage
            )
            
            if success:
                # Apply post-promotion business rules
                await self._apply_post_promotion_rules(model, target_stage)
                logger.info(f"Promoted model {model_id}:{version} to {target_stage.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    async def deploy_model(
        self,
        model_id: str,
        version: str,
        stage: DeploymentStage,
        replicas: int = 1,
        resources: Optional[Dict[str, str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
        traffic_percentage: float = 100.0
    ) -> Optional[str]:
        """Deploy model with business validation and monitoring setup.
        
        Args:
            model_id: ID of the model
            version: Version of the model
            stage: Deployment stage
            replicas: Number of replicas
            resources: Resource requirements
            environment_variables: Environment variables
            traffic_percentage: Traffic percentage for canary deployments
            
        Returns:
            Deployment ID if successful
            
        Raises:
            ValueError: If validation fails
        """
        # Validate model exists and is deployable
        model = await self._model_registry_port.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id}:{version} not found")
        
        # Check deployment eligibility
        await self._check_deployment_eligibility(model, stage)
        
        # Validate deployment parameters
        validated_resources = self._validate_deployment_resources(resources or {})
        validated_env_vars = self._validate_environment_variables(environment_variables or {})
        
        # Create deployment configuration
        config = DeploymentConfig(
            stage=stage,
            replicas=max(1, replicas),
            resources=validated_resources,
            environment_variables=validated_env_vars,
            traffic_percentage=max(0.0, min(100.0, traffic_percentage))
        )
        
        try:
            # Check for existing deployments and handle conflicts
            await self._handle_deployment_conflicts(model_id, version, stage)
            
            # Deploy the model
            deployment_id = await self._model_deployment_port.deploy_model(
                model_id, version, config
            )
            
            if deployment_id:
                # Set up monitoring and health checks
                await self._setup_deployment_monitoring(deployment_id, model, config)
                
                logger.info(f"Deployed model {model_id}:{version} to {stage.value} as {deployment_id}")
            
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
    
    async def search_models(
        self,
        query: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        algorithm: Optional[str] = None,
        framework: Optional[ModelFramework] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        limit: int = 50
    ) -> List[ModelInfo]:
        """Search models with enhanced business logic.
        
        Args:
            query: Text query for searching
            status: Filter by model status
            algorithm: Filter by algorithm
            framework: Filter by framework
            tags: Filter by tags
            created_by: Filter by creator
            min_accuracy: Minimum accuracy threshold
            limit: Maximum results
            
        Returns:
            List of matching models
        """
        # Build search query
        search_query = ModelSearchQuery(
            status=status,
            algorithm=algorithm,
            framework=framework,
            tags=tags,
            created_by=created_by,
            min_accuracy=min_accuracy,
            text_query=query
        )
        
        try:
            # Perform search
            results = await self._model_search_port.search_models(
                search_query,
                sort_by="created_at",
                sort_order="desc",
                limit=limit
            )
            
            # Apply business logic to enhance results
            enhanced_results = await self._enhance_search_results(results, query)
            
            logger.info(f"Found {len(enhanced_results)} models matching search criteria")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            raise
    
    async def get_model_recommendations(
        self,
        use_case: str,
        data_characteristics: Dict[str, Any],
        performance_requirements: Dict[str, float]
    ) -> List[ModelInfo]:
        """Get model recommendations with business intelligence.
        
        Args:
            use_case: Description of the use case
            data_characteristics: Characteristics of the data
            performance_requirements: Performance requirements
            
        Returns:
            List of recommended models
        """
        try:
            # Get base recommendations
            recommendations = await self._model_search_port.recommend_models(
                use_case, data_characteristics, performance_requirements
            )
            
            # Apply business rules to rank recommendations
            ranked_recommendations = await self._rank_model_recommendations(
                recommendations, use_case, performance_requirements
            )
            
            logger.info(f"Generated {len(ranked_recommendations)} model recommendations for '{use_case}'")
            return ranked_recommendations
            
        except Exception as e:
            logger.error(f"Failed to get model recommendations: {e}")
            raise
    
    async def compare_model_versions(
        self,
        model_id: str,
        version1: str,
        version2: str,
        include_performance: bool = True,
        include_parameters: bool = True
    ) -> Dict[str, Any]:
        """Compare two model versions with business insights.
        
        Args:
            model_id: ID of the model
            version1: First version to compare
            version2: Second version to compare
            include_performance: Include performance comparison
            include_parameters: Include parameter comparison
            
        Returns:
            Detailed comparison results
        """
        # Validate both models exist
        model1 = await self._model_registry_port.get_model(model_id, version1)
        model2 = await self._model_registry_port.get_model(model_id, version2)
        
        if not model1 or not model2:
            raise ValueError(f"One or both model versions not found: {version1}, {version2}")
        
        try:
            # Get base comparison
            base_comparison = await self._model_versioning_port.compare_versions(
                model_id, version1, version2
            )
            
            # Enhance with business analysis
            enhanced_comparison = await self._enhance_version_comparison(
                model1, model2, base_comparison, include_performance, include_parameters
            )
            
            logger.info(f"Compared model versions {version1} and {version2}")
            return enhanced_comparison
            
        except Exception as e:
            logger.error(f"Failed to compare model versions: {e}")
            raise
    
    async def get_model_health_status(
        self,
        model_id: str,
        version: Optional[str] = None,
        include_deployments: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive model health status.
        
        Args:
            model_id: ID of the model
            version: Specific version (default: latest)
            include_deployments: Include deployment health
            
        Returns:
            Model health status information
        """
        # Get model information
        model = await self._model_registry_port.get_model(model_id, version or "latest")
        if not model:
            raise ValueError(f"Model {model_id}:{version or 'latest'} not found")
        
        try:
            health_status = {
                "model_id": model_id,
                "version": model.version,
                "status": model.status.value,
                "health_check_timestamp": datetime.utcnow().isoformat(),
                "overall_health": "healthy",  # Will be computed
                "registry_health": await self._check_registry_health(model),
                "storage_health": await self._check_storage_health(model),
                "performance_health": await self._check_performance_health(model)
            }
            
            # Include deployment health if requested
            if include_deployments:
                health_status["deployment_health"] = await self._check_deployment_health(model)
            
            # Compute overall health
            health_status["overall_health"] = self._compute_overall_health(health_status)
            
            logger.info(f"Retrieved health status for model {model_id}:{model.version}")
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get model health status: {e}")
            raise
    
    # Private helper methods
    
    async def _validate_model_registration_inputs(
        self,
        model_path: str,
        model_id: str,
        algorithm: str,
        framework: ModelFramework,
        created_by: str
    ) -> None:
        """Validate model registration inputs."""
        # Validate model path
        if not model_path or not model_path.strip():
            raise ValueError("Model path cannot be empty")
        
        path = Path(model_path)
        if not path.exists():
            raise ValueError(f"Model file does not exist: {model_path}")
        
        if not path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")
        
        # Validate model ID
        if not model_id or not model_id.strip():
            raise ValueError("Model ID cannot be empty")
        
        if len(model_id) > 100:
            raise ValueError("Model ID cannot exceed 100 characters")
        
        if not model_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Model ID can only contain alphanumeric characters, hyphens, and underscores")
        
        # Validate algorithm
        if not algorithm or not algorithm.strip():
            raise ValueError("Algorithm cannot be empty")
        
        # Validate framework
        if not isinstance(framework, ModelFramework):
            raise ValueError(f"Invalid framework: {framework}")
        
        # Validate created_by
        if not created_by or not created_by.strip():
            raise ValueError("Created by cannot be empty")
    
    def _process_model_tags(self, tags: List[str]) -> List[str]:
        """Process and validate model tags."""
        processed_tags = []
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                clean_tag = tag.strip().lower().replace(" ", "_")
                if len(clean_tag) <= 50 and clean_tag not in processed_tags:
                    processed_tags.append(clean_tag)
        return processed_tags[:10]  # Limit to 10 tags
    
    def _validate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters."""
        validated = {}
        for key, value in hyperparameters.items():
            if isinstance(key, str) and len(key) <= 100:
                if isinstance(value, (str, int, float, bool)):
                    validated[key] = value
                elif isinstance(value, list) and all(isinstance(x, (str, int, float)) for x in value):
                    validated[key] = value
        return validated
    
    def _validate_model_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Validate model metrics."""
        validated = {}
        for key, value in metrics.items():
            if isinstance(key, str):
                try:
                    float_value = float(value)
                    if 0 <= float_value <= 1 or key in ["training_time", "inference_time"]:
                        validated[key] = float_value
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid metric: {key}={value}")
        return validated
    
    async def _apply_new_model_business_rules(
        self,
        model_id: str,
        version: str,
        metrics: ModelMetrics
    ) -> None:
        """Apply business rules for newly registered models."""
        try:
            # Auto-tag high-performing models
            if metrics.accuracy and metrics.accuracy > 0.9:
                await self._model_registry_port.update_model_metadata(
                    model_id, version, {"auto_tags": ["high_performance"]}
                )
            
            # Auto-promote models that meet criteria
            if (metrics.accuracy and metrics.accuracy > 0.95 and
                metrics.precision and metrics.precision > 0.9 and
                metrics.recall and metrics.recall > 0.9):
                
                await self._model_lifecycle_port.promote_model(
                    model_id, version, ModelStatus.STAGING
                )
                logger.info(f"Auto-promoted model {model_id}:{version} to staging")
            
        except Exception as e:
            logger.warning(f"Failed to apply new model business rules: {e}")
    
    async def _check_deployment_eligibility(self, model: ModelInfo, stage: DeploymentStage) -> None:
        """Check if model is eligible for deployment."""
        # Production deployments require production status
        if stage == DeploymentStage.PRODUCTION and model.status != ModelStatus.PRODUCTION:
            raise ValueError(f"Model must be in production status to deploy to production stage")
        
        # Staging deployments require at least staging status
        if stage == DeploymentStage.STAGING and model.status not in [ModelStatus.STAGING, ModelStatus.PRODUCTION]:
            raise ValueError(f"Model must be in staging or production status to deploy to staging")
        
        # Check minimum performance requirements
        if model.metrics.accuracy and model.metrics.accuracy < 0.7:
            raise ValueError(f"Model accuracy too low for deployment: {model.metrics.accuracy}")
    
    def _validate_deployment_resources(self, resources: Dict[str, str]) -> Dict[str, str]:
        """Validate deployment resource requirements."""
        validated = {}
        
        # Default resources
        defaults = {
            "cpu": "100m",
            "memory": "256Mi",
            "gpu": "0"
        }
        
        for key, default_value in defaults.items():
            value = resources.get(key, default_value)
            if isinstance(value, str) and value.strip():
                validated[key] = value.strip()
            else:
                validated[key] = default_value
        
        return validated
    
    def _validate_environment_variables(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        """Validate environment variables."""
        validated = {}
        for key, value in env_vars.items():
            if isinstance(key, str) and isinstance(value, str):
                if len(key) <= 100 and len(value) <= 1000:
                    validated[key] = value
        return validated
    
    async def _apply_pre_promotion_rules(self, model: ModelInfo, target_stage: ModelStatus) -> None:
        """Apply business rules before promotion."""
        if target_stage == ModelStatus.PRODUCTION:
            # Ensure no other models are in production for the same model_id
            existing_models = await self._model_registry_port.list_models(
                ModelSearchQuery(model_ids=[model.model_id], status=ModelStatus.PRODUCTION)
            )
            
            if existing_models:
                logger.info(f"Found {len(existing_models)} existing production models for {model.model_id}")
                # They will be deprecated in post-promotion rules
    
    async def _apply_post_promotion_rules(self, model: ModelInfo, target_stage: ModelStatus) -> None:
        """Apply business rules after successful promotion."""
        if target_stage == ModelStatus.PRODUCTION:
            # Deprecate other production models of the same model_id
            existing_models = await self._model_registry_port.list_models(
                ModelSearchQuery(model_ids=[model.model_id], status=ModelStatus.PRODUCTION)
            )
            
            for existing_model in existing_models:
                if existing_model.version != model.version:
                    await self._model_lifecycle_port.deprecate_model(
                        existing_model.model_id,
                        existing_model.version,
                        f"Superseded by version {model.version}"
                    )
    
    async def _handle_deployment_conflicts(
        self,
        model_id: str,
        version: str,
        stage: DeploymentStage
    ) -> None:
        """Handle conflicts with existing deployments."""
        existing_deployments = await self._model_deployment_port.list_deployments(
            model_id=model_id,
            stage=stage
        )
        
        if existing_deployments:
            logger.info(f"Found {len(existing_deployments)} existing deployments for {model_id} in {stage.value}")
            # For now, we allow multiple deployments
            # In a real system, you might want to undeploy old versions
    
    async def _setup_deployment_monitoring(
        self,
        deployment_id: str,
        model: ModelInfo,
        config: DeploymentConfig
    ) -> None:
        """Set up monitoring for deployed model."""
        try:
            # This would integrate with monitoring systems
            logger.info(f"Setting up monitoring for deployment {deployment_id}")
            
        except Exception as e:
            logger.warning(f"Failed to set up deployment monitoring: {e}")
    
    async def _enhance_search_results(
        self,
        results: List[ModelInfo],
        query: Optional[str]
    ) -> List[ModelInfo]:
        """Enhance search results with business intelligence."""
        # Add relevance scoring, popularity metrics, etc.
        if query:
            # Simple relevance scoring
            query_lower = query.lower()
            
            def relevance_score(model: ModelInfo) -> float:
                score = 0.0
                
                if query_lower in model.model_id.lower():
                    score += 10.0
                
                if query_lower in model.metadata.algorithm.lower():
                    score += 5.0
                
                for tag in model.metadata.tags:
                    if query_lower in tag.lower():
                        score += 3.0
                
                # Performance bonus
                if model.metrics.accuracy:
                    score += model.metrics.accuracy * 2
                
                return score
            
            # Sort by relevance
            results.sort(key=relevance_score, reverse=True)
        
        return results
    
    async def _rank_model_recommendations(
        self,
        recommendations: List[ModelInfo],
        use_case: str,
        performance_requirements: Dict[str, float]
    ) -> List[ModelInfo]:
        """Rank model recommendations based on business criteria."""
        def recommendation_score(model: ModelInfo) -> float:
            score = 0.0
            
            # Performance score based on requirements
            for metric, required_value in performance_requirements.items():
                model_value = getattr(model.metrics, metric, None)
                if model_value and model_value >= required_value:
                    score += 10.0
                elif model_value:
                    # Partial score based on how close it is
                    score += (model_value / required_value) * 5.0
            
            # Status bonus (production models preferred)
            if model.status == ModelStatus.PRODUCTION:
                score += 5.0
            elif model.status == ModelStatus.STAGING:
                score += 3.0
            
            # Recency bonus
            days_old = (datetime.utcnow() - model.created_at).days
            if days_old < 30:
                score += 2.0
            
            return score
        
        # Sort by recommendation score
        ranked = sorted(recommendations, key=recommendation_score, reverse=True)
        return ranked
    
    async def _enhance_version_comparison(
        self,
        model1: ModelInfo,
        model2: ModelInfo,
        base_comparison: Dict[str, Any],
        include_performance: bool,
        include_parameters: bool
    ) -> Dict[str, Any]:
        """Enhance version comparison with business insights."""
        enhanced = base_comparison.copy()
        
        if include_performance and model1.metrics and model2.metrics:
            performance_comparison = {
                "accuracy_change": self._calculate_metric_change(
                    model1.metrics.accuracy, model2.metrics.accuracy
                ),
                "f1_score_change": self._calculate_metric_change(
                    model1.metrics.f1_score, model2.metrics.f1_score
                ),
                "precision_change": self._calculate_metric_change(
                    model1.metrics.precision, model2.metrics.precision
                ),
                "recall_change": self._calculate_metric_change(
                    model1.metrics.recall, model2.metrics.recall
                )
            }
            enhanced["performance_comparison"] = performance_comparison
        
        if include_parameters:
            param_comparison = {
                "parameter_changes": self._compare_parameters(
                    model1.metadata.hyperparameters,
                    model2.metadata.hyperparameters
                )
            }
            enhanced["parameter_comparison"] = param_comparison
        
        # Add business recommendation
        enhanced["recommendation"] = self._generate_version_recommendation(model1, model2)
        
        return enhanced
    
    def _calculate_metric_change(self, value1: Optional[float], value2: Optional[float]) -> Optional[Dict[str, Any]]:
        """Calculate change between two metric values."""
        if value1 is None or value2 is None:
            return None
        
        change = value2 - value1
        percent_change = (change / value1) * 100 if value1 != 0 else 0
        
        return {
            "from": value1,
            "to": value2,
            "absolute_change": change,
            "percent_change": percent_change,
            "improvement": change > 0
        }
    
    def _compare_parameters(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare hyperparameters between versions."""
        all_keys = set(params1.keys()) | set(params2.keys())
        
        changes = {
            "modified": {},
            "added": {},
            "removed": {}
        }
        
        for key in all_keys:
            if key in params1 and key in params2:
                if params1[key] != params2[key]:
                    changes["modified"][key] = {"from": params1[key], "to": params2[key]}
            elif key in params2:
                changes["added"][key] = params2[key]
            else:
                changes["removed"][key] = params1[key]
        
        return changes
    
    def _generate_version_recommendation(self, model1: ModelInfo, model2: ModelInfo) -> str:
        """Generate recommendation for version comparison."""
        # Simple heuristic based on performance
        if model1.metrics.accuracy and model2.metrics.accuracy:
            if model2.metrics.accuracy > model1.metrics.accuracy:
                return f"Version {model2.version} shows improved performance over {model1.version}"
            elif model1.metrics.accuracy > model2.metrics.accuracy:
                return f"Version {model1.version} performs better than {model2.version}"
        
        return "Both versions have similar performance characteristics"
    
    async def _check_registry_health(self, model: ModelInfo) -> Dict[str, Any]:
        """Check model registry health."""
        return {
            "status": "healthy",
            "metadata_complete": bool(model.metadata.description),
            "metrics_available": bool(model.metrics.accuracy),
            "tags_present": len(model.metadata.tags) > 0
        }
    
    async def _check_storage_health(self, model: ModelInfo) -> Dict[str, Any]:
        """Check model storage health."""
        try:
            # Verify model integrity
            integrity_check = await self._model_storage_port.verify_model_integrity(
                model.model_id, model.version
            )
            
            return {
                "status": "healthy" if integrity_check else "degraded",
                "integrity_verified": integrity_check,
                "storage_path": model.model_path
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_performance_health(self, model: ModelInfo) -> Dict[str, Any]:
        """Check model performance health."""
        health_status = "healthy"
        issues = []
        
        # Check if metrics are available
        if not model.metrics.accuracy:
            health_status = "degraded"
            issues.append("No accuracy metrics available")
        
        # Check performance thresholds
        if model.metrics.accuracy and model.metrics.accuracy < 0.7:
            health_status = "degraded"
            issues.append("Accuracy below recommended threshold")
        
        return {
            "status": health_status,
            "accuracy": model.metrics.accuracy,
            "f1_score": model.metrics.f1_score,
            "issues": issues
        }
    
    async def _check_deployment_health(self, model: ModelInfo) -> Dict[str, Any]:
        """Check deployment health for the model."""
        try:
            deployments = await self._model_deployment_port.list_deployments(
                model_id=model.model_id
            )
            
            active_deployments = [d for d in deployments if d.health_status == "healthy"]
            
            return {
                "total_deployments": len(deployments),
                "active_deployments": len(active_deployments),
                "deployment_stages": [d.stage.value for d in deployments],
                "overall_deployment_health": "healthy" if active_deployments else "no_deployments"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _compute_overall_health(self, health_status: Dict[str, Any]) -> str:
        """Compute overall health from individual health checks."""
        registry_health = health_status.get("registry_health", {}).get("status", "unknown")
        storage_health = health_status.get("storage_health", {}).get("status", "unknown")
        performance_health = health_status.get("performance_health", {}).get("status", "unknown")
        
        # Simple majority rule
        health_scores = [registry_health, storage_health, performance_health]
        
        if health_scores.count("healthy") >= 2:
            return "healthy"
        elif "unhealthy" in health_scores:
            return "unhealthy"
        else:
            return "degraded"