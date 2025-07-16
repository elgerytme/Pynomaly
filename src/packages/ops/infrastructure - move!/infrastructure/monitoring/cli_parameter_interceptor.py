"""CLI Parameter Interceptor for transparent configuration capture.

This module provides transparent capture of CLI command parameters and results
for automatic configuration management and learning.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from monorepo.application.dto.configuration_dto import (
    ConfigurationCaptureRequestDTO,
    ConfigurationSource,
    PerformanceResultsDTO,
)
from monorepo.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from monorepo.infrastructure.config.feature_flags import feature_flags

logger = logging.getLogger(__name__)


class CLIParameterInterceptor:
    """Interceptor for capturing CLI command parameters and results."""

    def __init__(
        self,
        configuration_service: ConfigurationCaptureService,
        enable_automatic_capture: bool = True,
        capture_successful_only: bool = True,
        min_execution_time: float = 1.0,  # Only capture commands that take >1s
    ):
        """Initialize CLI parameter interceptor.

        Args:
            configuration_service: Configuration capture service
            enable_automatic_capture: Enable automatic parameter capture
            capture_successful_only: Only capture successful command executions
            min_execution_time: Minimum execution time to trigger capture
        """
        self.configuration_service = configuration_service
        self.enable_automatic_capture = enable_automatic_capture
        self.capture_successful_only = capture_successful_only
        self.min_execution_time = min_execution_time

        # Tracking
        self.active_commands: dict[str, dict[str, Any]] = {}
        self.capture_stats = {
            "total_commands": 0,
            "captured_commands": 0,
            "failed_captures": 0,
            "commands_by_type": {},
        }

    def capture_cli_command(
        self,
        command_type: str,
        capture_results: bool = True,
        tags: list[str] | None = None,
    ):
        """Decorator to capture CLI command parameters and results.

        Args:
            command_type: Type of command (detect, train, evaluate, etc.)
            capture_results: Whether to capture command results
            tags: Additional tags for the configuration

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enable_automatic_capture or not feature_flags.is_enabled(
                    "advanced_automl"
                ):
                    return func(*args, **kwargs)

                # Use async wrapper internally
                return asyncio.run(
                    self._async_wrapper(
                        func, args, kwargs, command_type, capture_results, tags
                    )
                )

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enable_automatic_capture or not feature_flags.is_enabled(
                    "advanced_automl"
                ):
                    return await func(*args, **kwargs)

                return await self._async_wrapper(
                    func, args, kwargs, command_type, capture_results, tags
                )

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    async def _async_wrapper(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        command_type: str,
        capture_results: bool,
        tags: list[str] | None,
    ):
        """Async wrapper for parameter capture."""
        command_id = f"{command_type}_{datetime.now().timestamp()}"
        start_time = datetime.now()

        self.capture_stats["total_commands"] += 1
        self.capture_stats["commands_by_type"][command_type] = (
            self.capture_stats["commands_by_type"].get(command_type, 0) + 1
        )

        # Extract parameters before execution
        parameters = self._extract_parameters(func, args, kwargs)

        # Store active command context
        self.active_commands[command_id] = {
            "command_type": command_type,
            "parameters": parameters,
            "start_time": start_time,
            "tags": tags or [],
        }

        try:
            # Execute the original function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Capture configuration if conditions are met
            if execution_time >= self.min_execution_time and (
                not self.capture_successful_only or result is not None
            ):
                await self._capture_command_configuration(
                    command_id, result, execution_time, success=True
                )

            return result

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Capture configuration for failed commands if not capturing successful only
            if not self.capture_successful_only:
                await self._capture_command_configuration(
                    command_id, None, execution_time, success=False, error=str(e)
                )

            raise e

        finally:
            # Clean up active command
            self.active_commands.pop(command_id, None)

    def _extract_parameters(
        self, func: Callable, args: tuple, kwargs: dict
    ) -> dict[str, Any]:
        """Extract parameters from function call."""
        try:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert to dict and handle special types
            parameters = {}
            for name, value in bound_args.arguments.items():
                parameters[name] = self._serialize_parameter_value(value)

            return parameters

        except Exception as e:
            logger.warning(f"Failed to extract parameters from {func.__name__}: {e}")
            return {"extraction_error": str(e)}

    def _serialize_parameter_value(self, value: Any) -> Any:
        """Serialize parameter value for storage."""
        if value is None:
            return None
        elif isinstance(value, str | int | float | bool):
            return value
        elif isinstance(value, list | tuple):
            return [self._serialize_parameter_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_parameter_value(v) for k, v in value.items()}
        elif isinstance(value, Path):
            return str(value)
        elif hasattr(value, "__dict__"):
            # For objects, capture key attributes
            return {
                "type": type(value).__name__,
                "attributes": {
                    k: self._serialize_parameter_value(v)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_")
                },
            }
        else:
            return str(value)

    async def _capture_command_configuration(
        self,
        command_id: str,
        result: Any,
        execution_time: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Capture configuration for executed command."""
        try:
            command_context = self.active_commands.get(command_id, {})

            if not command_context:
                logger.warning(f"No context found for command {command_id}")
                return

            # Extract performance results
            performance_results = self._extract_performance_results(
                result, execution_time, success, error
            )

            # Create capture request
            capture_request = ConfigurationCaptureRequestDTO(
                source=ConfigurationSource.CLI,
                raw_parameters=command_context["parameters"],
                execution_results=(
                    performance_results.model_dump() if performance_results else None
                ),
                source_context={
                    "command_type": command_context["command_type"],
                    "cli_intercepted": True,
                    "execution_successful": success,
                    "execution_time_seconds": execution_time,
                    "error_message": error,
                    "timestamp": datetime.now().isoformat(),
                },
                auto_save=True,
                generate_name=True,
                tags=["cli", command_context["command_type"], "intercepted"]
                + command_context["tags"],
            )

            # Capture configuration
            response = await self.configuration_service.capture_configuration(
                capture_request
            )

            if response.success:
                self.capture_stats["captured_commands"] += 1
                logger.info(
                    f"Captured CLI command configuration: {command_context['command_type']}"
                )
            else:
                self.capture_stats["failed_captures"] += 1
                logger.warning(
                    f"Failed to capture CLI command configuration: {response.message}"
                )

        except Exception as e:
            self.capture_stats["failed_captures"] += 1
            logger.error(f"Error capturing CLI command configuration: {e}")

    def _extract_performance_results(
        self, result: Any, execution_time: float, success: bool, error: str | None
    ) -> PerformanceResultsDTO | None:
        """Extract performance results from command result."""
        if not success:
            return PerformanceResultsDTO(
                training_time_seconds=execution_time,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
            )

        # Try to extract metrics from result
        accuracy = None
        precision = None
        recall = None
        f1_score = None
        roc_auc = None

        try:
            if hasattr(result, "accuracy"):
                accuracy = float(result.accuracy)
            elif hasattr(result, "score"):
                accuracy = float(result.score)
            elif isinstance(result, dict):
                accuracy = result.get("accuracy") or result.get("score")
                precision = result.get("precision")
                recall = result.get("recall")
                f1_score = result.get("f1_score")
                roc_auc = result.get("roc_auc")

        except (ValueError, TypeError, AttributeError):
            pass

        return PerformanceResultsDTO(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=roc_auc,
            training_time_seconds=execution_time,
        )

    def get_capture_statistics(self) -> dict[str, Any]:
        """Get capture statistics.

        Returns:
            Capture statistics dictionary
        """
        return {
            "capture_stats": self.capture_stats,
            "settings": {
                "enable_automatic_capture": self.enable_automatic_capture,
                "capture_successful_only": self.capture_successful_only,
                "min_execution_time": self.min_execution_time,
            },
            "active_commands": len(self.active_commands),
        }


# Global interceptor instance
_global_interceptor: CLIParameterInterceptor | None = None


def initialize_cli_interceptor(
    configuration_service: ConfigurationCaptureService,
) -> CLIParameterInterceptor:
    """Initialize global CLI parameter interceptor.

    Args:
        configuration_service: Configuration capture service

    Returns:
        CLI parameter interceptor instance
    """
    global _global_interceptor
    _global_interceptor = CLIParameterInterceptor(configuration_service)
    return _global_interceptor


def get_cli_interceptor() -> CLIParameterInterceptor | None:
    """Get global CLI parameter interceptor.

    Returns:
        CLI parameter interceptor instance or None if not initialized
    """
    return _global_interceptor


def capture_cli_command(
    command_type: str, capture_results: bool = True, tags: list[str] | None = None
):
    """Decorator to capture CLI command parameters and results.

    Args:
        command_type: Type of command (detect, train, evaluate, etc.)
        capture_results: Whether to capture command results
        tags: Additional tags for the configuration

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        if _global_interceptor is None:
            # Return original function if interceptor not initialized
            return func

        return _global_interceptor.capture_cli_command(
            command_type=command_type, capture_results=capture_results, tags=tags
        )(func)

    return decorator


# Convenience decorators for common CLI command types
def capture_detection_command(tags: list[str] | None = None):
    """Decorator for detection commands."""
    return capture_cli_command("detection", capture_results=True, tags=tags)


def capture_training_command(tags: list[str] | None = None):
    """Decorator for training commands."""
    return capture_cli_command("training", capture_results=True, tags=tags)


def capture_evaluation_command(tags: list[str] | None = None):
    """Decorator for evaluation commands."""
    return capture_cli_command("evaluation", capture_results=True, tags=tags)


def capture_preprocessing_command(tags: list[str] | None = None):
    """Decorator for preprocessing commands."""
    return capture_cli_command("preprocessing", capture_results=True, tags=tags)


def capture_automl_command(tags: list[str] | None = None):
    """Decorator for AutoML commands."""
    return capture_cli_command("automl", capture_results=True, tags=tags)


def capture_autonomous_command(tags: list[str] | None = None):
    """Decorator for autonomous commands."""
    return capture_cli_command("autonomous", capture_results=True, tags=tags)


class CLIConfigurationAnalytics:
    """Analytics for CLI command configuration patterns."""

    def __init__(self, configuration_service: ConfigurationCaptureService):
        """Initialize CLI configuration analytics.

        Args:
            configuration_service: Configuration capture service
        """
        self.configuration_service = configuration_service

    async def analyze_cli_usage_patterns(self, days_back: int = 30) -> dict[str, Any]:
        """Analyze CLI usage patterns from captured configurations.

        Args:
            days_back: Number of days to analyze

        Returns:
            Usage pattern analysis
        """
        from datetime import timedelta

        from monorepo.application.dto.configuration_dto import (
            ConfigurationSearchRequestDTO,
        )

        # Search for CLI configurations
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.CLI,
            created_after=datetime.now() - timedelta(days=days_back),
            limit=1000,
            sort_by="created_at",
            sort_order="desc",
        )

        configurations = await self.configuration_service.search_configurations(
            search_request
        )

        if not configurations:
            return {"message": "No CLI configurations found", "configurations": 0}

        # Analyze patterns
        command_types = {}
        parameter_patterns = {}
        performance_by_command = {}

        for config in configurations:
            if config.source_context:
                command_type = config.source_context.get("command_type", "unknown")
                command_types[command_type] = command_types.get(command_type, 0) + 1

                # Analyze common parameter patterns
                for param, value in config.raw_parameters.items():
                    if param not in parameter_patterns:
                        parameter_patterns[param] = {}

                    param_value = str(value)
                    parameter_patterns[param][param_value] = (
                        parameter_patterns[param].get(param_value, 0) + 1
                    )

                # Performance by command type
                if command_type not in performance_by_command:
                    performance_by_command[command_type] = []

                if (
                    config.performance_results
                    and config.performance_results.training_time_seconds
                ):
                    performance_by_command[command_type].append(
                        config.performance_results.training_time_seconds
                    )

        # Calculate average execution times
        avg_execution_times = {}
        for command_type, times in performance_by_command.items():
            if times:
                avg_execution_times[command_type] = sum(times) / len(times)

        return {
            "analysis_period_days": days_back,
            "total_cli_configurations": len(configurations),
            "command_type_usage": command_types,
            "most_used_command": (
                max(command_types.items(), key=lambda x: x[1])[0]
                if command_types
                else None
            ),
            "average_execution_times": avg_execution_times,
            "common_parameter_patterns": parameter_patterns,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    async def get_command_recommendations(
        self,
        command_type: str,
        dataset_characteristics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get parameter recommendations for a command type.

        Args:
            command_type: Type of CLI command
            dataset_characteristics: Characteristics of the target dataset

        Returns:
            Parameter recommendations
        """
        from monorepo.application.dto.configuration_dto import (
            ConfigurationSearchRequestDTO,
        )

        # Search for successful configurations of this command type
        search_request = ConfigurationSearchRequestDTO(
            source=ConfigurationSource.CLI,
            tags=[command_type],
            min_accuracy=0.7,
            limit=100,
            sort_by="accuracy",
            sort_order="desc",
        )

        configurations = await self.configuration_service.search_configurations(
            search_request
        )

        if not configurations:
            return {"message": f"No successful {command_type} configurations found"}

        # Analyze successful parameter combinations
        successful_params = {}
        for config in configurations:
            for param, value in config.raw_parameters.items():
                if param not in successful_params:
                    successful_params[param] = []
                successful_params[param].append(value)

        # Calculate recommendations
        recommendations = {}
        for param, values in successful_params.items():
            if values:
                # For numeric parameters, calculate mean
                try:
                    numeric_values = [float(v) for v in values if v is not None]
                    if numeric_values:
                        recommendations[param] = {
                            "recommended_value": sum(numeric_values)
                            / len(numeric_values),
                            "type": "numeric",
                            "range": [min(numeric_values), max(numeric_values)],
                        }
                    else:
                        raise ValueError("Not numeric")
                except (ValueError, TypeError):
                    # For categorical parameters, find most common
                    from collections import Counter

                    value_counts = Counter(str(v) for v in values if v is not None)
                    if value_counts:
                        recommendations[param] = {
                            "recommended_value": value_counts.most_common(1)[0][0],
                            "type": "categorical",
                            "options": dict(value_counts.most_common(5)),
                        }

        return {
            "command_type": command_type,
            "total_successful_configs": len(configurations),
            "parameter_recommendations": recommendations,
            "based_on_configurations": [
                str(config.id) for config in configurations[:10]
            ],
        }
