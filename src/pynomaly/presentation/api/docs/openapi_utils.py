"""OpenAPI generation utilities for Pynomaly API."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Type, get_type_hints

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from pydantic import BaseModel
from pydantic._internal._model_construction import complete_model_class


logger = logging.getLogger(__name__)

# Global cache for OpenAPI schemas
_openapi_schema_cache: Dict[str, Any] = {}


def clear_openapi_cache() -> None:
    """Clear the OpenAPI schema cache."""
    global _openapi_schema_cache
    _openapi_schema_cache.clear()


def get_all_pydantic_models() -> Dict[str, Type[BaseModel]]:
    """Get all Pydantic models from the application.
    
    Returns:
        Dictionary mapping model names to model classes
    """
    models = {}
    
    # Get all BaseModel subclasses
    for cls in BaseModel.__subclasses__():
        if hasattr(cls, '__name__'):
            models[cls.__name__] = cls
            
            # Also get nested subclasses
            for subcls in cls.__subclasses__():
                if hasattr(subcls, '__name__'):
                    models[subcls.__name__] = subcls
    
    return models


def rebuild_model_references(models: Dict[str, Type[BaseModel]]) -> None:
    """Rebuild all Pydantic models to resolve forward references.
    
    Args:
        models: Dictionary of model name to model class
    """
    for name, model in models.items():
        try:
            # Call model_rebuild() to resolve forward references
            model.model_rebuild()
            logger.debug(f"Rebuilt model: {name}")
        except Exception as e:
            logger.warning(f"Failed to rebuild model {name}: {e}")


def generate_model_schemas(models: Dict[str, Type[BaseModel]]) -> Dict[str, Any]:
    """Generate JSON schemas for all Pydantic models.
    
    Args:
        models: Dictionary of model name to model class
        
    Returns:
        Dictionary mapping model names to their JSON schemas
    """
    schemas = {}
    
    for name, model in models.items():
        try:
            # Generate JSON schema using model_json_schema()
            schema = model.model_json_schema()
            schemas[name] = schema
            logger.debug(f"Generated schema for model: {name}")
        except Exception as e:
            logger.warning(f"Failed to generate schema for model {name}: {e}")
            # Create a basic schema as fallback
            schemas[name] = {
                "type": "object",
                "properties": {},
                "title": name,
                "description": f"Schema for {name} (fallback)"
            }
    
    return schemas


def create_operation_objects(routes: List[APIRoute]) -> List[Dict[str, Any]]:
    """Manually build Operation objects for routes.
    
    Args:
        routes: List of FastAPI routes
        
    Returns:
        List of operation objects
    """
    operations = []
    
    for route in routes:
        if not isinstance(route, APIRoute):
            continue
            
        try:
            # Build operation object
            operation = {
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name,
                "tags": route.tags or [],
                "summary": route.summary,
                "description": route.description,
            }
            
            # Add response model info if available
            if hasattr(route, 'response_model') and route.response_model:
                operation["response_model"] = route.response_model.__name__
            
            # Add request model info if available
            if hasattr(route, 'endpoint') and route.endpoint:
                sig = inspect.signature(route.endpoint)
                for param_name, param in sig.parameters.items():
                    if hasattr(param.annotation, '__name__') and issubclass(param.annotation, BaseModel):
                        operation.setdefault("request_models", []).append(param.annotation.__name__)
            
            operations.append(operation)
            
        except Exception as e:
            logger.warning(f"Failed to create operation for route {route.path}: {e}")
    
    return operations


def custom_openapi_generator(app: FastAPI, config: Optional[Any] = None) -> Any:
    """Create a custom OpenAPI generator function.
    
    This function creates a custom OpenAPI generator that:
    1. Iterates through app.routes
    2. Builds Operation objects manually
    3. Calls model.model_json_schema() on every Pydantic model after invoking model_rebuild()
    4. Uses a cache so OpenAPI is generated once
    
    Args:
        app: FastAPI application instance
        config: Optional configuration object
        
    Returns:
        Custom OpenAPI generator function
    """
    def custom_openapi() -> Dict[str, Any]:
        # Use cached schema if available
        cache_key = f"{app.title}_{app.version}"
        if cache_key in _openapi_schema_cache:
            logger.debug(f"Using cached OpenAPI schema for {cache_key}")
            return _openapi_schema_cache[cache_key]
        
        logger.info(f"Generating OpenAPI schema for {app.title} v{app.version}")
        
        try:
            # 1. Get all routes from the app
            routes = [route for route in app.routes if isinstance(route, APIRoute)]
            logger.debug(f"Found {len(routes)} API routes")
            
            # 2. Create operation objects manually
            operations = create_operation_objects(routes)
            logger.debug(f"Created {len(operations)} operation objects")
            
            # 3. Get all Pydantic models
            models = get_all_pydantic_models()
            logger.debug(f"Found {len(models)} Pydantic models")
            
            # 4. Rebuild models to resolve forward references
            rebuild_model_references(models)
            
            # 5. Generate JSON schemas for all models
            schemas = generate_model_schemas(models)
            logger.debug(f"Generated {len(schemas)} model schemas")
            
            # 6. Generate base OpenAPI schema
            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=routes,
                servers=getattr(app, 'servers', None)
            )
            
            # 7. Add custom schemas to components
            if "components" not in openapi_schema:
                openapi_schema["components"] = {}
            
            openapi_schema["components"]["schemas"] = schemas
            
            # 8. Add custom metadata
            openapi_schema["x-operations"] = operations
            openapi_schema["x-model-count"] = len(models)
            openapi_schema["x-generated-at"] = {"timestamp": "now"}
            
            # 9. Cache the generated schema
            _openapi_schema_cache[cache_key] = openapi_schema
            
            logger.info(f"Successfully generated OpenAPI schema with {len(schemas)} schemas")
            return openapi_schema
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAPI schema: {e}")
            # Return minimal schema as fallback
            return {
                "openapi": "3.0.0",
                "info": {
                    "title": app.title,
                    "version": app.version,
                    "description": "API schema generation failed"
                },
                "paths": {},
                "components": {
                    "schemas": {}
                }
            }
    
    return custom_openapi


def apply_custom_openapi_to_app(app: FastAPI, config: Optional[Any] = None) -> None:
    """Apply custom OpenAPI generator to FastAPI app.
    
    Args:
        app: FastAPI application instance
        config: Optional configuration object
    """
    # Override the app's openapi method
    app.openapi = custom_openapi_generator(app, config)
    logger.info(f"Applied custom OpenAPI generator to {app.title}")

