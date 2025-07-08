"""Configuration Template Generation Service.

This service generates reusable configuration templates from successful experiments,
enabling rapid deployment and standardization of proven anomaly detection workflows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml
from jinja2 import BaseLoader, Environment

from pynomaly.application.dto.configuration_dto import (
    ConfigurationLevel,
    ConfigurationSearchRequestDTO,
    ConfigurationSource,
    ConfigurationTemplateDTO,
    ExperimentConfigurationDTO,
    ExportFormat,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.infrastructure.config.feature_flags import require_feature
from pynomaly.infrastructure.persistence.configuration_repository import (
    ConfigurationRepository,
)

logger = logging.getLogger(__name__)


class ConfigurationTemplateService:
    """Service for generating and managing configuration templates."""

    def __init__(
        self,
        configuration_service: ConfigurationCaptureService,
        repository: ConfigurationRepository,
        template_storage_path: Path | None = None,
    ):
        """Initialize configuration template service.

        Args:
            configuration_service: Configuration capture service
            repository: Configuration repository
            template_storage_path: Path to store generated templates
        """
        self.configuration_service = configuration_service
        self.repository = repository
        self.template_storage_path = template_storage_path or Path("data/templates")
        self.template_storage_path.mkdir(parents=True, exist_ok=True)

        # Template generation statistics
        self.template_stats = {
            "total_templates_generated": 0,
            "templates_from_automl": 0,
            "templates_from_autonomous": 0,
            "templates_from_cli": 0,
            "parameterized_templates": 0,
            "static_templates": 0,
        }

        # Initialize Jinja2 environment for template generation
        self.jinja_env = Environment(loader=BaseLoader())

    @require_feature("advanced_automl")
    async def generate_template_from_configuration(
        self,
        config_id: UUID,
        template_name: str,
        description: str,
        parameterize: bool = True,
        difficulty_level: ConfigurationLevel = ConfigurationLevel.INTERMEDIATE,
        use_cases: list[str] | None = None,
        include_documentation: bool = True,
    ) -> ConfigurationTemplateDTO:
        """Generate template from existing configuration.

        Args:
            config_id: Configuration ID to base template on
            template_name: Name for the template
            description: Template description
            parameterize: Whether to parameterize the template
            difficulty_level: Template difficulty level
            use_cases: Recommended use cases
            include_documentation: Include documentation and examples

        Returns:
            Generated configuration template
        """
        # Load base configuration
        base_config = await self.repository.load_configuration(config_id)
        if not base_config:
            raise ValueError(f"Configuration {config_id} not found")

        logger.info(
            f"Generating template '{template_name}' from configuration {config_id}"
        )

        # Extract variable parameters if parameterizing
        if parameterize:
            (
                variable_parameters,
                parameter_constraints,
            ) = self._identify_variable_parameters(base_config)
        else:
            variable_parameters = {}
            parameter_constraints = {}

        # Generate template configuration
        template_config = self._create_template_configuration(
            base_config, variable_parameters
        )

        # Generate documentation
        documentation = ""
        examples = []
        if include_documentation:
            documentation = self._generate_template_documentation(
                base_config, template_name, description, variable_parameters
            )
            examples = self._generate_template_examples(
                template_config, variable_parameters
            )

        # Create template DTO
        template = ConfigurationTemplateDTO(
            id=uuid4(),
            name=template_name,
            description=description,
            template_config=template_config,
            variable_parameters=variable_parameters,
            parameter_constraints=parameter_constraints,
            use_cases=use_cases or self._infer_use_cases(base_config),
            difficulty_level=difficulty_level,
            estimated_runtime=self._estimate_template_runtime(base_config),
            usage_count=0,
            success_rate=None,
            rating=None,
            documentation=documentation,
            examples=examples,
        )

        # Save template
        await self.repository.save_template(template)

        # Update statistics
        self.template_stats["total_templates_generated"] += 1
        if parameterize:
            self.template_stats["parameterized_templates"] += 1
        else:
            self.template_stats["static_templates"] += 1

        # Update source-specific statistics
        source = base_config.metadata.source
        if source == ConfigurationSource.AUTOML:
            self.template_stats["templates_from_automl"] += 1
        elif source == ConfigurationSource.AUTONOMOUS:
            self.template_stats["templates_from_autonomous"] += 1
        elif source == ConfigurationSource.CLI:
            self.template_stats["templates_from_cli"] += 1

        logger.info(f"Template '{template_name}' generated successfully")
        return template

    async def generate_batch_templates(
        self,
        configuration_criteria: dict[str, Any],
        template_prefix: str = "auto_template",
        max_templates: int = 10,
        min_performance_threshold: float = 0.8,
    ) -> list[ConfigurationTemplateDTO]:
        """Generate multiple templates from high-performing configurations.

        Args:
            configuration_criteria: Criteria for selecting configurations
            template_prefix: Prefix for template names
            max_templates: Maximum number of templates to generate
            min_performance_threshold: Minimum performance threshold

        Returns:
            List of generated templates
        """
        logger.info(f"Generating batch templates with prefix '{template_prefix}'")

        # Search for high-performing configurations
        search_request = ConfigurationSearchRequestDTO(
            min_accuracy=min_performance_threshold,
            limit=max_templates * 2,  # Get more candidates
            sort_by="accuracy",
            sort_order="desc",
            **configuration_criteria,
        )

        configurations = await self.repository.search_configurations(search_request)

        if not configurations:
            logger.warning("No configurations found matching criteria")
            return []

        # Generate templates from top configurations
        templates = []
        for i, config in enumerate(configurations[:max_templates]):
            try:
                template_name = f"{template_prefix}_{i + 1}_{config.algorithm_config.algorithm_name}"
                description = f"Auto-generated template from high-performing {config.algorithm_config.algorithm_name} configuration"

                template = await self.generate_template_from_configuration(
                    config_id=config.id,
                    template_name=template_name,
                    description=description,
                    parameterize=True,
                    difficulty_level=self._determine_difficulty_level(config),
                    use_cases=self._infer_use_cases(config),
                    include_documentation=True,
                )

                templates.append(template)

            except Exception as e:
                logger.warning(
                    f"Failed to generate template from config {config.id}: {e}"
                )
                continue

        logger.info(f"Generated {len(templates)} batch templates")
        return templates

    async def generate_collection_template(
        self,
        collection_id: UUID,
        template_name: str,
        description: str,
        merge_strategy: str = "ensemble",
    ) -> ConfigurationTemplateDTO:
        """Generate template from configuration collection.

        Args:
            collection_id: Configuration collection ID
            template_name: Name for the template
            description: Template description
            merge_strategy: Strategy for merging configurations (ensemble, best, average)

        Returns:
            Generated template from collection
        """
        # Load collection
        collection = await self.repository.load_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        # Load all configurations in collection
        configurations = []
        for config_id in collection.configurations:
            config = await self.repository.load_configuration(config_id)
            if config:
                configurations.append(config)

        if not configurations:
            raise ValueError("No valid configurations found in collection")

        logger.info(
            f"Generating collection template '{template_name}' from {len(configurations)} configurations"
        )

        # Merge configurations based on strategy
        if merge_strategy == "ensemble":
            merged_config = self._create_ensemble_configuration(configurations)
        elif merge_strategy == "best":
            merged_config = self._select_best_configuration(configurations)
        elif merge_strategy == "average":
            merged_config = self._average_configurations(configurations)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Generate template from merged configuration
        (
            variable_parameters,
            parameter_constraints,
        ) = self._identify_variable_parameters_from_multiple(configurations)

        template = ConfigurationTemplateDTO(
            id=uuid4(),
            name=template_name,
            description=description,
            template_config=merged_config,
            variable_parameters=variable_parameters,
            parameter_constraints=parameter_constraints,
            use_cases=self._infer_use_cases_from_multiple(configurations),
            difficulty_level=ConfigurationLevel.ADVANCED,  # Collections are typically advanced
            estimated_runtime=self._estimate_collection_runtime(configurations),
            documentation=self._generate_collection_documentation(
                configurations, merge_strategy
            ),
            examples=self._generate_collection_examples(
                merged_config, variable_parameters
            ),
        )

        # Save template
        await self.repository.save_template(template)

        self.template_stats["total_templates_generated"] += 1
        self.template_stats["parameterized_templates"] += 1

        logger.info(f"Collection template '{template_name}' generated successfully")
        return template

    async def export_template(
        self,
        template_id: UUID,
        export_format: ExportFormat,
        output_path: Path | None = None,
        include_examples: bool = True,
        include_documentation: bool = True,
    ) -> str:
        """Export template in specified format.

        Args:
            template_id: Template ID to export
            export_format: Export format
            output_path: Output file path
            include_examples: Include usage examples
            include_documentation: Include documentation

        Returns:
            Exported template content as string
        """
        template = await self.repository.load_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")

        logger.info(f"Exporting template '{template.name}' as {export_format}")

        if export_format == ExportFormat.YAML:
            content = self._export_template_yaml(
                template, include_examples, include_documentation
            )
        elif export_format == ExportFormat.JSON:
            content = self._export_template_json(
                template, include_examples, include_documentation
            )
        elif export_format == ExportFormat.PYTHON:
            content = self._export_template_python(
                template, include_examples, include_documentation
            )
        elif export_format == ExportFormat.NOTEBOOK:
            content = self._export_template_notebook(
                template, include_examples, include_documentation
            )
        elif export_format == ExportFormat.DOCKER:
            content = self._export_template_docker(
                template, include_examples, include_documentation
            )
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        # Save to file if path specified
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Template exported to {output_path}")

        return content

    async def instantiate_template(
        self, template_id: UUID, parameters: dict[str, Any], configuration_name: str
    ) -> ExperimentConfigurationDTO:
        """Instantiate template with specific parameters.

        Args:
            template_id: Template ID to instantiate
            parameters: Parameter values for template variables
            configuration_name: Name for the new configuration

        Returns:
            Instantiated configuration
        """
        template = await self.repository.load_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")

        logger.info(
            f"Instantiating template '{template.name}' with configuration '{configuration_name}'"
        )

        # Validate parameters
        validation_errors = self._validate_template_parameters(template, parameters)
        if validation_errors:
            raise ValueError(f"Parameter validation failed: {validation_errors}")

        # Create instantiated configuration
        instantiated_config = self._instantiate_template_configuration(
            template.template_config, template.variable_parameters, parameters
        )

        # Update configuration details
        instantiated_config.name = configuration_name
        instantiated_config.id = uuid4()
        instantiated_config.metadata.source = ConfigurationSource.TEMPLATE
        instantiated_config.metadata.parent_id = template.id
        instantiated_config.metadata.derived_from = [template.template_config.id]
        instantiated_config.metadata.tags.extend(["template", "instantiated"])

        # Update template usage statistics
        template.usage_count += 1
        await self.repository.save_template(template)

        logger.info(f"Template instantiated as configuration '{configuration_name}'")
        return instantiated_config

    async def get_template_recommendations(
        self,
        dataset_characteristics: dict[str, Any],
        performance_requirements: dict[str, float],
        difficulty_preference: ConfigurationLevel | None = None,
    ) -> list[tuple[ConfigurationTemplateDTO, float]]:
        """Get template recommendations based on requirements.

        Args:
            dataset_characteristics: Target dataset characteristics
            performance_requirements: Performance requirements
            difficulty_preference: Preferred difficulty level

        Returns:
            List of (template, relevance_score) tuples sorted by relevance
        """
        logger.info("Generating template recommendations")

        # Get all templates (in production, would use more sophisticated querying)
        template_files = list(self.template_storage_path.glob("*.json"))
        templates = []

        for template_file in template_files:
            try:
                template = await self.repository.load_template(UUID(template_file.stem))
                if template:
                    templates.append(template)
            except Exception:
                continue

        # Score templates based on relevance
        scored_templates = []
        for template in templates:
            relevance_score = self._calculate_template_relevance(
                template,
                dataset_characteristics,
                performance_requirements,
                difficulty_preference,
            )
            scored_templates.append((template, relevance_score))

        # Sort by relevance score
        scored_templates.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Generated {len(scored_templates)} template recommendations")
        return scored_templates

    def get_template_statistics(self) -> dict[str, Any]:
        """Get template generation statistics.

        Returns:
            Template statistics dictionary
        """
        return {
            "template_stats": self.template_stats,
            "template_storage_path": str(self.template_storage_path),
            "total_template_files": len(
                list(self.template_storage_path.glob("*.json"))
            ),
        }

    # Private methods

    def _identify_variable_parameters(
        self, config: ExperimentConfigurationDTO
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Identify parameters that should be made variable in template."""
        variable_parameters = {}
        parameter_constraints = {}

        # Algorithm hyperparameters
        if config.algorithm_config.hyperparameters:
            for param, value in config.algorithm_config.hyperparameters.items():
                if self._is_tunable_parameter(param, value):
                    variable_parameters[f"algorithm.{param}"] = {
                        "default": value,
                        "type": self._get_parameter_type(value),
                        "description": f"Algorithm parameter: {param}",
                    }
                    parameter_constraints[
                        f"algorithm.{param}"
                    ] = self._get_parameter_constraints(param, value)

        # Common tunable parameters
        tunable_params = {
            "contamination": config.algorithm_config.contamination,
            "random_state": config.algorithm_config.random_state,
        }

        for param, value in tunable_params.items():
            if value is not None:
                variable_parameters[f"algorithm.{param}"] = {
                    "default": value,
                    "type": self._get_parameter_type(value),
                    "description": f"Algorithm parameter: {param}",
                }
                parameter_constraints[
                    f"algorithm.{param}"
                ] = self._get_parameter_constraints(param, value)

        # Evaluation parameters
        if config.evaluation_config:
            eval_params = {
                "cv_folds": config.evaluation_config.cv_folds,
                "test_size": config.evaluation_config.test_size,
            }

            for param, value in eval_params.items():
                if value is not None:
                    variable_parameters[f"evaluation.{param}"] = {
                        "default": value,
                        "type": self._get_parameter_type(value),
                        "description": f"Evaluation parameter: {param}",
                    }
                    parameter_constraints[
                        f"evaluation.{param}"
                    ] = self._get_parameter_constraints(param, value)

        return variable_parameters, parameter_constraints

    def _is_tunable_parameter(self, param_name: str, value: Any) -> bool:
        """Check if parameter should be made tunable."""
        # Skip parameters that are typically not tuned
        skip_params = {"algorithm", "class", "type", "name", "id"}
        if param_name.lower() in skip_params:
            return False

        # Include numeric parameters
        if isinstance(value, int | float):
            return True

        # Include boolean parameters
        if isinstance(value, bool):
            return True

        # Include string parameters that look like choices
        if isinstance(value, str) and len(value) < 50:
            return True

        return False

    def _get_parameter_type(self, value: Any) -> str:
        """Get parameter type for template."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "list"
        else:
            return "object"

    def _get_parameter_constraints(self, param_name: str, value: Any) -> dict[str, Any]:
        """Get parameter constraints for validation."""
        constraints = {}

        # Parameter-specific constraints
        if param_name == "contamination":
            constraints = {"min": 0.01, "max": 0.5, "type": "float"}
        elif param_name == "random_state":
            constraints = {"min": 0, "max": 2**31 - 1, "type": "integer"}
        elif param_name == "cv_folds":
            constraints = {"min": 2, "max": 20, "type": "integer"}
        elif param_name == "test_size":
            constraints = {"min": 0.1, "max": 0.5, "type": "float"}
        elif param_name in ["n_estimators", "n_neighbors"]:
            constraints = {"min": 10, "max": 1000, "type": "integer"}
        elif param_name in ["learning_rate", "alpha", "beta"]:
            constraints = {"min": 0.001, "max": 1.0, "type": "float"}
        else:
            # Generic constraints based on value type
            if isinstance(value, bool):
                constraints = {"type": "boolean", "choices": [True, False]}
            elif isinstance(value, int):
                constraints = {
                    "type": "integer",
                    "min": max(1, int(value * 0.1)),
                    "max": int(value * 10),
                }
            elif isinstance(value, float):
                constraints = {
                    "type": "float",
                    "min": max(0.001, value * 0.1),
                    "max": value * 10,
                }
            elif isinstance(value, str):
                constraints = {"type": "string", "max_length": 100}

        return constraints

    def _create_template_configuration(
        self,
        base_config: ExperimentConfigurationDTO,
        variable_parameters: dict[str, Any],
    ) -> ExperimentConfigurationDTO:
        """Create template configuration with variable placeholders."""
        # Deep copy base configuration
        template_config = base_config.model_copy(deep=True)

        # Replace variable parameters with placeholders
        for param_path, param_info in variable_parameters.items():
            self._set_template_placeholder(template_config, param_path, param_info)

        # Update metadata for template
        template_config.id = uuid4()
        template_config.metadata.source = ConfigurationSource.TEMPLATE
        template_config.metadata.tags.append("template")

        return template_config

    def _set_template_placeholder(
        self,
        config: ExperimentConfigurationDTO,
        param_path: str,
        param_info: dict[str, Any],
    ) -> None:
        """Set template placeholder for parameter."""
        parts = param_path.split(".")

        if parts[0] == "algorithm":
            if len(parts) == 2:
                param_name = parts[1]
                if param_name == "contamination":
                    config.algorithm_config.contamination = f"{{{{ {param_path} }}}}"
                elif param_name == "random_state":
                    config.algorithm_config.random_state = f"{{{{ {param_path} }}}}"
                elif param_name in config.algorithm_config.hyperparameters:
                    config.algorithm_config.hyperparameters[
                        param_name
                    ] = f"{{{{ {param_path} }}}}"

        elif parts[0] == "evaluation":
            if len(parts) == 2:
                param_name = parts[1]
                if param_name == "cv_folds":
                    config.evaluation_config.cv_folds = f"{{{{ {param_path} }}}}"
                elif param_name == "test_size":
                    config.evaluation_config.test_size = f"{{{{ {param_path} }}}}"

    def _generate_template_documentation(
        self,
        base_config: ExperimentConfigurationDTO,
        template_name: str,
        description: str,
        variable_parameters: dict[str, Any],
    ) -> str:
        """Generate comprehensive documentation for template."""
        doc_parts = [
            f"# {template_name}\n",
            f"{description}\n",
            "## Overview\n",
            f"This template is based on a successful {base_config.algorithm_config.algorithm_name} configuration ",
            (
                f"that achieved {base_config.performance_results.accuracy:.3f} accuracy.\n"
                if base_config.performance_results
                else ""
            ),
            "\n## Algorithm Details\n",
            f"- **Algorithm**: {base_config.algorithm_config.algorithm_name}\n",
            f"- **Source**: {base_config.metadata.source}\n",
            f"- **Created**: {base_config.metadata.created_at}\n",
        ]

        if variable_parameters:
            doc_parts.extend(
                [
                    "\n## Configurable Parameters\n",
                    "This template supports the following configurable parameters:\n",
                ]
            )

            for param_path, param_info in variable_parameters.items():
                doc_parts.append(
                    f"- **{param_path}**: {param_info.get('description', 'No description')} "
                    f"(default: {param_info.get('default', 'N/A')})\n"
                )

        if base_config.performance_results:
            doc_parts.extend(
                [
                    "\n## Expected Performance\n",
                    "Based on the original configuration, you can expect:\n",
                    (
                        f"- **Accuracy**: {base_config.performance_results.accuracy:.3f}\n"
                        if base_config.performance_results.accuracy
                        else ""
                    ),
                    (
                        f"- **Training Time**: {base_config.performance_results.training_time_seconds:.1f}s\n"
                        if base_config.performance_results.training_time_seconds
                        else ""
                    ),
                ]
            )

        doc_parts.extend(
            [
                "\n## Usage\n",
                "To use this template:\n",
                "1. Load the template\n",
                "2. Provide values for configurable parameters\n",
                "3. Instantiate the configuration\n",
                "4. Run anomaly detection\n",
            ]
        )

        return "".join(doc_parts)

    def _generate_template_examples(
        self,
        template_config: ExperimentConfigurationDTO,
        variable_parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate usage examples for template."""
        examples = []

        # Basic example
        basic_params = {}
        for param_path, param_info in variable_parameters.items():
            basic_params[param_path] = param_info.get("default")

        examples.append(
            {
                "name": "Basic Usage",
                "description": "Standard configuration with default parameters",
                "parameters": basic_params,
                "code": self._generate_example_code(template_config, basic_params),
            }
        )

        # High-performance example
        if len(variable_parameters) > 1:
            high_perf_params = basic_params.copy()
            # Adjust parameters for higher performance
            for param_path, param_info in variable_parameters.items():
                if "contamination" in param_path:
                    high_perf_params[param_path] = 0.05  # Lower contamination
                elif "cv_folds" in param_path:
                    high_perf_params[param_path] = 10  # More CV folds

            examples.append(
                {
                    "name": "High Performance",
                    "description": "Configuration optimized for higher accuracy",
                    "parameters": high_perf_params,
                    "code": self._generate_example_code(
                        template_config, high_perf_params
                    ),
                }
            )

        return examples

    def _generate_example_code(
        self, template_config: ExperimentConfigurationDTO, parameters: dict[str, Any]
    ) -> str:
        """Generate example code for using template."""
        code_lines = [
            "from pynomaly import ConfigurationTemplateService",
            "from pynomaly.infrastructure.data_loaders import CSVLoader",
            "",
            "# Load template service",
            "template_service = ConfigurationTemplateService()",
            "",
            "# Set parameters",
            f"parameters = {json.dumps(parameters, indent=2)}",
            "",
            "# Instantiate configuration",
            "config = template_service.instantiate_template(",
            f"    template_id='{template_config.id}',",
            "    parameters=parameters,",
            "    configuration_name='my_detection_config'",
            ")",
            "",
            "# Load your dataset",
            "loader = CSVLoader()",
            "dataset = loader.load('path/to/your/data.csv')",
            "",
            "# Run detection",
            "from pynomaly.application.services import DetectionService",
            "detection_service = DetectionService()",
            "result = detection_service.detect_anomalies(dataset, config)",
        ]

        return "\n".join(code_lines)

    def _infer_use_cases(self, config: ExperimentConfigurationDTO) -> list[str]:
        """Infer use cases from configuration characteristics."""
        use_cases = []

        algorithm = config.algorithm_config.algorithm_name.lower()

        if "isolation" in algorithm:
            use_cases.extend(
                ["high_dimensional_data", "large_datasets", "general_purpose"]
            )
        elif "lof" in algorithm or "outlier" in algorithm:
            use_cases.extend(
                [
                    "density_based_detection",
                    "local_anomalies",
                    "small_to_medium_datasets",
                ]
            )
        elif "svm" in algorithm:
            use_cases.extend(
                ["non_linear_boundaries", "robust_detection", "medium_datasets"]
            )
        elif "autoencoder" in algorithm or "neural" in algorithm:
            use_cases.extend(["complex_patterns", "large_datasets", "deep_learning"])

        # Add use cases based on performance
        if config.performance_results and config.performance_results.accuracy:
            if config.performance_results.accuracy > 0.9:
                use_cases.append("high_accuracy_required")
            if (
                config.performance_results.training_time_seconds
                and config.performance_results.training_time_seconds < 60
            ):
                use_cases.append("fast_training")

        # Add use cases based on source
        if config.metadata.source == ConfigurationSource.AUTOML:
            use_cases.append("automated_optimization")
        elif config.metadata.source == ConfigurationSource.AUTONOMOUS:
            use_cases.append("autonomous_detection")

        return list(set(use_cases))  # Remove duplicates

    def _determine_difficulty_level(
        self, config: ExperimentConfigurationDTO
    ) -> ConfigurationLevel:
        """Determine difficulty level based on configuration complexity."""
        complexity_score = 0

        # Algorithm complexity
        algorithm = config.algorithm_config.algorithm_name.lower()
        if any(term in algorithm for term in ["autoencoder", "neural", "deep"]):
            complexity_score += 3
        elif any(term in algorithm for term in ["svm", "ensemble"]):
            complexity_score += 2
        else:
            complexity_score += 1

        # Hyperparameter complexity
        if config.algorithm_config.hyperparameters:
            complexity_score += min(
                len(config.algorithm_config.hyperparameters) // 3, 2
            )

        # Preprocessing complexity
        if config.preprocessing_config:
            if config.preprocessing_config.apply_pca:
                complexity_score += 1
            if config.preprocessing_config.feature_selection_method:
                complexity_score += 1

        # Evaluation complexity
        if config.evaluation_config.cv_folds > 5:
            complexity_score += 1

        # Map score to difficulty level
        if complexity_score <= 2:
            return ConfigurationLevel.BASIC
        elif complexity_score <= 4:
            return ConfigurationLevel.INTERMEDIATE
        elif complexity_score <= 6:
            return ConfigurationLevel.ADVANCED
        else:
            return ConfigurationLevel.EXPERT

    def _estimate_template_runtime(
        self, config: ExperimentConfigurationDTO
    ) -> float | None:
        """Estimate template runtime based on configuration."""
        if (
            config.performance_results
            and config.performance_results.training_time_seconds
        ):
            return config.performance_results.training_time_seconds

        # Estimate based on algorithm type
        algorithm = config.algorithm_config.algorithm_name.lower()
        base_times = {
            "isolation": 30.0,
            "lof": 60.0,
            "svm": 120.0,
            "autoencoder": 300.0,
            "neural": 300.0,
        }

        for keyword, time in base_times.items():
            if keyword in algorithm:
                return time

        return 60.0  # Default estimate

    def _export_template_yaml(
        self,
        template: ConfigurationTemplateDTO,
        include_examples: bool,
        include_documentation: bool,
    ) -> str:
        """Export template as YAML."""
        template_data = {
            "template": {
                "name": template.name,
                "description": template.description,
                "difficulty_level": template.difficulty_level,
                "estimated_runtime": template.estimated_runtime,
                "use_cases": template.use_cases,
            },
            "configuration": template.template_config.model_dump(),
            "parameters": template.variable_parameters,
            "constraints": template.parameter_constraints,
        }

        if include_documentation and template.documentation:
            template_data["documentation"] = template.documentation

        if include_examples and template.examples:
            template_data["examples"] = template.examples

        return yaml.dump(template_data, default_flow_style=False, sort_keys=False)

    def _export_template_json(
        self,
        template: ConfigurationTemplateDTO,
        include_examples: bool,
        include_documentation: bool,
    ) -> str:
        """Export template as JSON."""
        template_data = template.model_dump()

        if not include_documentation:
            template_data.pop("documentation", None)

        if not include_examples:
            template_data.pop("examples", None)

        return json.dumps(template_data, indent=2, default=str)

    def _export_template_python(
        self,
        template: ConfigurationTemplateDTO,
        include_examples: bool,
        include_documentation: bool,
    ) -> str:
        """Export template as Python script."""
        script_lines = [
            "#!/usr/bin/env python3",
            f'"""Configuration template: {template.name}"""',
            "",
            "from pynomaly.application.services import ConfigurationTemplateService",
            "from pynomaly.application.services import DetectionService",
            "from pynomaly.infrastructure.data_loaders import CSVLoader",
            "",
            f"# Template: {template.name}",
            f"# {template.description}",
            "",
            "def create_configuration(**kwargs):",
            '    """Create configuration from template."""',
            "    # Default parameters",
            f"    default_params = {json.dumps(template.variable_parameters, indent=4)}",
            "",
            "    # Update with provided parameters",
            "    params = default_params.copy()",
            "    params.update(kwargs)",
            "",
            "    # Instantiate template",
            "    template_service = ConfigurationTemplateService()",
            "    config = template_service.instantiate_template(",
            f"        template_id='{template.id}',",
            "        parameters=params,",
            "        configuration_name='template_config'",
            "    )",
            "",
            "    return config",
            "",
            "def run_detection(dataset_path, **template_params):",
            '    """Run detection using template configuration."""',
            "    # Create configuration",
            "    config = create_configuration(**template_params)",
            "",
            "    # Load dataset",
            "    loader = CSVLoader()",
            "    dataset = loader.load(dataset_path)",
            "",
            "    # Run detection",
            "    detection_service = DetectionService()",
            "    result = detection_service.detect_anomalies(dataset, config)",
            "",
            "    return result",
        ]

        if include_examples and template.examples:
            script_lines.extend(
                [
                    "",
                    "# Example usage:",
                    "if __name__ == '__main__':",
                    "    # Basic usage",
                    "    result = run_detection('path/to/data.csv')",
                    "    print(f'Found {result.n_anomalies} anomalies')",
                ]
            )

        return "\n".join(script_lines)

    def _export_template_notebook(
        self,
        template: ConfigurationTemplateDTO,
        include_examples: bool,
        include_documentation: bool,
    ) -> str:
        """Export template as Jupyter notebook."""
        cells = []

        # Title cell
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Configuration Template: {template.name}\n",
                    f"\n{template.description}\n",
                    f"\n**Difficulty Level**: {template.difficulty_level}\n",
                    f"**Estimated Runtime**: {template.estimated_runtime}s\n",
                ],
            }
        )

        # Import cell
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "from pynomaly.application.services import ConfigurationTemplateService\n",
                    "from pynomaly.application.services import DetectionService\n",
                    "from pynomaly.infrastructure.data_loaders import CSVLoader\n",
                    "import pandas as pd\n",
                    "import numpy as np",
                ],
            }
        )

        # Configuration cell
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Template parameters\n",
                    f"template_id = '{template.id}'\n",
                    f"parameters = {json.dumps(template.variable_parameters, indent=2)}\n",
                    "\n",
                    "# Instantiate configuration\n",
                    "template_service = ConfigurationTemplateService()\n",
                    "config = template_service.instantiate_template(\n",
                    "    template_id=template_id,\n",
                    "    parameters=parameters,\n",
                    "    configuration_name='notebook_config'\n",
                    ")",
                ],
            }
        )

        # Usage cell
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load your dataset\n",
                    "# dataset_path = 'path/to/your/data.csv'\n",
                    "# loader = CSVLoader()\n",
                    "# dataset = loader.load(dataset_path)\n",
                    "\n",
                    "# Run detection\n",
                    "# detection_service = DetectionService()\n",
                    "# result = detection_service.detect_anomalies(dataset, config)\n",
                    "# print(f'Found {result.n_anomalies} anomalies out of {len(result.scores)} samples')",
                ],
            }
        )

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        return json.dumps(notebook, indent=2)

    def _export_template_docker(
        self,
        template: ConfigurationTemplateDTO,
        include_examples: bool,
        include_documentation: bool,
    ) -> str:
        """Export template as Docker Compose configuration."""
        docker_compose = {
            "version": "3.8",
            "services": {
                f"pynomaly-{template.name.lower().replace(' ', '-')}": {
                    "image": "pynomaly:latest",
                    "environment": {
                        "PYNOMALY_TEMPLATE_ID": str(template.id),
                        "PYNOMALY_TEMPLATE_NAME": template.name,
                        **{
                            f"PYNOMALY_{k.upper().replace('.', '_')}": str(
                                v.get("default", "")
                            )
                            for k, v in template.variable_parameters.items()
                        },
                    },
                    "volumes": [
                        "./data:/app/data:ro",
                        "./results:/app/results",
                        "./templates:/app/templates:ro",
                    ],
                    "command": [
                        "python",
                        "-m",
                        "pynomaly.cli.template",
                        "--template-id",
                        str(template.id),
                        "--data-path",
                        "/app/data",
                        "--output-path",
                        "/app/results",
                    ],
                }
            },
        }

        return yaml.dump(docker_compose, default_flow_style=False)

    def _calculate_template_relevance(
        self,
        template: ConfigurationTemplateDTO,
        dataset_characteristics: dict[str, Any],
        performance_requirements: dict[str, float],
        difficulty_preference: ConfigurationLevel | None,
    ) -> float:
        """Calculate template relevance score."""
        score = 0.0

        # Use case matching
        if "use_case" in dataset_characteristics:
            target_use_case = dataset_characteristics["use_case"]
            if target_use_case in template.use_cases:
                score += 0.3

        # Performance requirements
        if template.success_rate:
            required_accuracy = performance_requirements.get("min_accuracy", 0.7)
            if template.success_rate >= required_accuracy:
                score += 0.3

        # Difficulty preference
        if difficulty_preference:
            if template.difficulty_level == difficulty_preference:
                score += 0.2
            elif (
                abs(
                    list(ConfigurationLevel).index(template.difficulty_level)
                    - list(ConfigurationLevel).index(difficulty_preference)
                )
                == 1
            ):
                score += 0.1

        # Template usage and rating
        if template.usage_count > 0:
            score += min(0.1, template.usage_count / 100)  # Up to 0.1 for usage

        if template.rating:
            score += template.rating * 0.1  # Up to 0.1 for rating

        return min(1.0, score)

    # Additional helper methods for collection templates

    def _create_ensemble_configuration(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> ExperimentConfigurationDTO:
        """Create ensemble configuration from multiple configurations."""
        # Use the best performing configuration as base
        best_config = max(
            configurations,
            key=lambda c: (
                c.performance_results.accuracy
                if c.performance_results and c.performance_results.accuracy
                else 0
            ),
        )

        ensemble_config = best_config.model_copy(deep=True)
        ensemble_config.id = uuid4()
        ensemble_config.name = f"ensemble_{len(configurations)}_algorithms"
        ensemble_config.algorithm_config.is_ensemble = True
        ensemble_config.algorithm_config.ensemble_method = "voting"
        ensemble_config.algorithm_config.base_algorithms = [
            c.algorithm_config.algorithm_name for c in configurations
        ]

        return ensemble_config

    def _select_best_configuration(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> ExperimentConfigurationDTO:
        """Select best configuration from collection."""
        return max(
            configurations,
            key=lambda c: (
                c.performance_results.accuracy
                if c.performance_results and c.performance_results.accuracy
                else 0
            ),
        )

    def _average_configurations(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> ExperimentConfigurationDTO:
        """Create averaged configuration from collection."""
        # Use first configuration as base
        averaged_config = configurations[0].model_copy(deep=True)
        averaged_config.id = uuid4()
        averaged_config.name = f"averaged_{len(configurations)}_configs"

        # Average numeric hyperparameters
        numeric_params = {}
        for config in configurations:
            for param, value in config.algorithm_config.hyperparameters.items():
                if isinstance(value, int | float):
                    if param not in numeric_params:
                        numeric_params[param] = []
                    numeric_params[param].append(value)

        # Update with averaged values
        for param, values in numeric_params.items():
            if values:
                averaged_config.algorithm_config.hyperparameters[param] = sum(
                    values
                ) / len(values)

        return averaged_config

    def _identify_variable_parameters_from_multiple(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Identify variable parameters from multiple configurations."""
        # Find parameters that vary across configurations
        all_params = {}
        for config in configurations:
            single_params, _ = self._identify_variable_parameters(config)
            all_params.update(single_params)

        # Filter to parameters that actually vary
        variable_parameters = {}
        parameter_constraints = {}

        for param_path, param_info in all_params.items():
            values = []
            for config in configurations:
                # Extract actual values from configurations
                # This is simplified - would need more robust extraction
                values.append(param_info.get("default"))

            # Include if values vary
            if len({str(v) for v in values if v is not None}) > 1:
                variable_parameters[param_path] = param_info
                parameter_constraints[param_path] = self._get_parameter_constraints(
                    param_path.split(".")[-1], param_info.get("default")
                )

        return variable_parameters, parameter_constraints

    def _infer_use_cases_from_multiple(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> list[str]:
        """Infer use cases from multiple configurations."""
        all_use_cases = set()
        for config in configurations:
            use_cases = self._infer_use_cases(config)
            all_use_cases.update(use_cases)

        # Add collection-specific use cases
        all_use_cases.add("comparative_analysis")
        all_use_cases.add("multi_algorithm")

        return list(all_use_cases)

    def _estimate_collection_runtime(
        self, configurations: list[ExperimentConfigurationDTO]
    ) -> float | None:
        """Estimate runtime for collection-based template."""
        runtimes = []
        for config in configurations:
            runtime = self._estimate_template_runtime(config)
            if runtime:
                runtimes.append(runtime)

        if runtimes:
            return max(runtimes)  # Use maximum runtime

        return None

    def _generate_collection_documentation(
        self, configurations: list[ExperimentConfigurationDTO], merge_strategy: str
    ) -> str:
        """Generate documentation for collection-based template."""
        algorithms = [c.algorithm_config.algorithm_name for c in configurations]

        doc_parts = [
            f"# Collection Template ({merge_strategy} strategy)\n",
            f"This template is based on {len(configurations)} high-performing configurations.\n",
            "\n## Included Algorithms\n",
        ]

        for i, algorithm in enumerate(algorithms, 1):
            doc_parts.append(f"{i}. {algorithm}\n")

        doc_parts.extend(
            [
                "\n## Merge Strategy\n",
                f"Strategy used: **{merge_strategy}**\n",
                "\n## Expected Benefits\n",
                "- Combines strengths of multiple approaches\n",
                "- Robust performance across different data types\n",
                "- Proven effectiveness on real datasets\n",
            ]
        )

        return "".join(doc_parts)

    def _generate_collection_examples(
        self,
        template_config: ExperimentConfigurationDTO,
        variable_parameters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate examples for collection-based template."""
        return self._generate_template_examples(template_config, variable_parameters)

    def _validate_template_parameters(
        self, template: ConfigurationTemplateDTO, parameters: dict[str, Any]
    ) -> list[str]:
        """Validate parameters against template constraints."""
        errors = []

        for param_path, value in parameters.items():
            if param_path in template.parameter_constraints:
                constraints = template.parameter_constraints[param_path]

                # Type validation
                expected_type = constraints.get("type")
                if expected_type:
                    if expected_type == "integer" and not isinstance(value, int):
                        errors.append(
                            f"{param_path}: expected integer, got {type(value).__name__}"
                        )
                    elif expected_type == "float" and not isinstance(
                        value, int | float
                    ):
                        errors.append(
                            f"{param_path}: expected float, got {type(value).__name__}"
                        )
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(
                            f"{param_path}: expected boolean, got {type(value).__name__}"
                        )
                    elif expected_type == "string" and not isinstance(value, str):
                        errors.append(
                            f"{param_path}: expected string, got {type(value).__name__}"
                        )

                # Range validation
                if isinstance(value, int | float):
                    if "min" in constraints and value < constraints["min"]:
                        errors.append(
                            f"{param_path}: value {value} below minimum {constraints['min']}"
                        )
                    if "max" in constraints and value > constraints["max"]:
                        errors.append(
                            f"{param_path}: value {value} above maximum {constraints['max']}"
                        )

                # Choice validation
                if "choices" in constraints and value not in constraints["choices"]:
                    errors.append(
                        f"{param_path}: value {value} not in allowed choices {constraints['choices']}"
                    )

        return errors

    def _instantiate_template_configuration(
        self,
        template_config: ExperimentConfigurationDTO,
        variable_parameters: dict[str, Any],
        parameters: dict[str, Any],
    ) -> ExperimentConfigurationDTO:
        """Instantiate template with specific parameter values."""
        # Create copy of template configuration
        instantiated = template_config.model_copy(deep=True)

        # Replace placeholders with actual values
        for param_path, value in parameters.items():
            if param_path in variable_parameters:
                self._set_instantiated_value(instantiated, param_path, value)

        # Ensure all placeholders are replaced with defaults if not provided
        for param_path, param_info in variable_parameters.items():
            if param_path not in parameters:
                default_value = param_info.get("default")
                if default_value is not None:
                    self._set_instantiated_value(
                        instantiated, param_path, default_value
                    )

        return instantiated

    def _set_instantiated_value(
        self, config: ExperimentConfigurationDTO, param_path: str, value: Any
    ) -> None:
        """Set instantiated value for parameter path."""
        parts = param_path.split(".")

        if parts[0] == "algorithm":
            if len(parts) == 2:
                param_name = parts[1]
                if param_name == "contamination":
                    config.algorithm_config.contamination = value
                elif param_name == "random_state":
                    config.algorithm_config.random_state = value
                elif param_name in config.algorithm_config.hyperparameters:
                    config.algorithm_config.hyperparameters[param_name] = value

        elif parts[0] == "evaluation":
            if len(parts) == 2:
                param_name = parts[1]
                if param_name == "cv_folds":
                    config.evaluation_config.cv_folds = value
                elif param_name == "test_size":
                    config.evaluation_config.test_size = value
