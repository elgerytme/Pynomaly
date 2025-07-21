#!/usr/bin/env python3
"""
Simple API Documentation Generator for anomaly_detection
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DOCS_DIR = PROJECT_DIR / "docs" / "api"
OUTPUT_DIR = DOCS_DIR / "generated"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(message):
    """Log a message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def generate_openapi_spec():
    """Generate OpenAPI 3.0 specification."""
    log("Generating OpenAPI 3.0 specification...")

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "anomaly detection API",
            "description": "State-of-the-art anomaly detection API",
            "version": "1.0.0",
        },
        "servers": [
            {"url": "https://api.anomaly_detection.com", "description": "Production"},
            {"url": "http://localhost:8000", "description": "Development"},
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "tags": ["Health"],
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "timestamp": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/v1/detectors": {
                "get": {
                    "summary": "List detectors",
                    "tags": ["Detectors"],
                    "responses": {"200": {"description": "List of detectors"}},
                },
                "post": {
                    "summary": "Create detector",
                    "tags": ["Detectors"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "algorithm": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"201": {"description": "Detector created"}},
                },
            },
        },
    }

    # Save JSON format
    json_path = OUTPUT_DIR / "openapi.json"
    with open(json_path, "w") as f:
        json.dump(spec, f, indent=2)

    log(f"OpenAPI specification saved to {json_path}")
    return spec


def generate_swagger_ui():
    """Generate Swagger UI."""
    log("Generating Swagger UI...")

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>anomaly detection API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css" />
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            SwaggerUIBundle({
                url: './openapi.json',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "StandaloneLayout"
            });
        };
    </script>
</body>
</html>"""

    html_path = OUTPUT_DIR / "index.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    log(f"Swagger UI generated at {html_path}")


def generate_postman_collection():
    """Generate Postman collection."""
    log("Generating Postman collection...")

    collection = {
        "info": {
            "name": "anomaly detection API",
            "description": "API collection for anomaly_detection",
            "version": "1.0.0",
        },
        "item": [
            {
                "name": "Health Check",
                "request": {
                    "method": "GET",
                    "header": [{"key": "X-API-Key", "value": "{{api_key}}"}],
                    "url": {
                        "raw": "{{base_url}}/health",
                        "host": ["{{base_url}}"],
                        "path": ["health"],
                    },
                },
            },
            {
                "name": "List Detectors",
                "request": {
                    "method": "GET",
                    "header": [{"key": "X-API-Key", "value": "{{api_key}}"}],
                    "url": {
                        "raw": "{{base_url}}/api/v1/detectors",
                        "host": ["{{base_url}}"],
                        "path": ["api", "v1", "detectors"],
                    },
                },
            },
        ],
        "variable": [
            {"key": "base_url", "value": "https://api.anomaly_detection.com"},
            {"key": "api_key", "value": "your-api-key-here"},
        ],
    }

    postman_path = OUTPUT_DIR / "anomaly_detection_api.postman_collection.json"
    with open(postman_path, "w") as f:
        json.dump(collection, f, indent=2)

    log(f"Postman collection generated at {postman_path}")


def generate_code_examples():
    """Generate code examples."""
    log("Generating code examples...")

    examples_dir = OUTPUT_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)

    # Python example
    python_dir = examples_dir / "python"
    python_dir.mkdir(exist_ok=True)

    python_example = """import requests

API_KEY = "your-api-key-here"
BASE_URL = "https://api.anomaly_detection.com"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Health check
response = requests.get(f"{BASE_URL}/health", headers=headers)
print(response.json())

# List detectors
response = requests.get(f"{BASE_URL}/api/v1/detectors", headers=headers)
print(response.json())
"""

    with open(python_dir / "basic_usage.py", "w") as f:
        f.write(python_example)

    # cURL example
    curl_dir = examples_dir / "curl"
    curl_dir.mkdir(exist_ok=True)

    curl_example = """# Health check
curl -X GET "https://api.anomaly_detection.com/health" \\
  -H "X-API-Key: your-api-key-here"

# List detectors
curl -X GET "https://api.anomaly_detection.com/api/v1/detectors" \\
  -H "X-API-Key: your-api-key-here"
"""

    with open(curl_dir / "examples.sh", "w") as f:
        f.write(curl_example)

    log(f"Code examples generated in {examples_dir}")


def generate_readme():
    """Generate README."""
    log("Generating README...")

    readme_content = f"""# anomaly detection API Documentation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Generated

- `openapi.json` - OpenAPI 3.0 specification
- `index.html` - Interactive Swagger UI documentation
- `anomaly_detection_api.postman_collection.json` - Postman collection
- `examples/` - Code examples in different languages

## Usage

1. Open `index.html` in your browser for interactive documentation
2. Import the Postman collection for API testing
3. Use the code examples to integrate with your applications

## API Endpoints

### Health Check
- `GET /health` - Check API health status

### Detectors
- `GET /api/v1/detectors` - List all detectors
- `POST /api/v1/detectors` - Create a new detector

## Authentication

All API endpoints require authentication via API key in the X-API-Key header.

## Support

For more information, visit the comprehensive API documentation.
"""

    readme_path = DOCS_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    log(f"README generated at {readme_path}")


def main():
    """Main function."""
    log("Starting API documentation generation...")

    try:
        generate_openapi_spec()
        generate_swagger_ui()
        generate_postman_collection()
        generate_code_examples()
        generate_readme()

        log("API documentation generation completed successfully!")

        print("\n" + "=" * 50)
        print("📚 API Documentation Generated!")
        print("=" * 50)
        print(f"📁 Output Directory: {OUTPUT_DIR}")
        print(f"🌐 Interactive Docs: {OUTPUT_DIR / 'index.html'}")
        print(f"📄 OpenAPI Spec: {OUTPUT_DIR / 'openapi.json'}")
        print(
            f"🔧 Postman Collection: {OUTPUT_DIR / 'anomaly_detection_api.postman_collection.json'}"
        )
        print(f"💻 Code Examples: {OUTPUT_DIR / 'examples'}")
        print("=" * 50)

        print("\n🚀 Next Steps:")
        print("1. Open the interactive documentation in your browser")
        print("2. Import the Postman collection for API testing")
        print("3. Use the code examples to integrate with your applications")
        print("\n✅ Documentation is ready!")

    except Exception as e:
        print(f"ERROR: Documentation generation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
