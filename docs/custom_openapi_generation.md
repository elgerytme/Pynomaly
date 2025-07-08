# Custom OpenAPI Generation

The project includes a custom OpenAPI generator to address issues that arise from forward references in Pydantic models. This generator attempts multiple strategies to build the schema without errors.

## Features

1. **Route Analysis:**
   - Iterates through `app.routes` to identify and filter out problematic routes that may cause forward reference issues.

2. **Schema Generation:**
   - Builds operation objects manually.
   - Calls `model.model_json_schema()` after `model_rebuild()` to handle models with unresolved references.

3. **Fallback Strategy:**
   - Uses a cache to avoid regeneration.
   - Generates a manual OpenAPI schema with basic structures if normal generation fails.

## Usage

The custom OpenAPI generator is integrated directly within the application's configuration. It overrides the default `app.openapi` method with a custom function that handles the schema generation.

```
from pynomaly.presentation.api.docs.openapi_utils import apply_custom_openapi_to_app
apply_custom_openapi_to_app(app)
```

## Testing

Automated tests validate the OpenAPI generation process, ensuring that the schema includes all necessary paths and is free of circular references.

- Run the unit tests using:
```
pytest tests/unit/test_openapi_utils.py
```

- The testing includes mock verifications and checks for expected paths and schemas.
