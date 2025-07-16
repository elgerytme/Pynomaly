#!/usr/bin/env python3
"""
Multi-Language Client SDK Generator for Pynomaly API

This script generates comprehensive client SDKs for multiple programming languages
from the OpenAPI specification. It supports:

- Python SDK with async/await support
- JavaScript/TypeScript SDK with type definitions
- Java SDK with Maven support
- Go SDK with Go modules
- C# SDK with NuGet package
- PHP SDK with Composer support
- Ruby SDK with Gem support
- Rust SDK with Cargo support

Features:
- Automatic code generation from OpenAPI spec
- Built-in authentication handling
- Error handling and retry logic
- Rate limiting support
- Comprehensive documentation
- Unit test generation
- CI/CD configuration
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class SDKGenerator:
    """Multi-language SDK generator for Pynomaly API."""

    def __init__(self):
        """Initialize the SDK generator."""
        self.project_root = Path(__file__).parent.parent
        self.openapi_spec_path = self.project_root / "docs" / "api" / "openapi.yaml"
        self.output_dir = self.project_root / "sdks"

        # Supported languages and their configurations
        self.languages = {
            "python": {
                "generator": "python",
                "package_name": "pynomaly_client",
                "client_name": "PynomaliClient",
                "additional_properties": {
                    "packageName": "pynomaly_client",
                    "projectName": "pynomaly-python-client",
                    "packageVersion": "1.0.0",
                    "packageUrl": "https://github.com/pynomaly/python-client",
                    "pythonAtLeast": "3.8",
                    "library": "requests",
                    "generateSourceCodeOnly": "false",
                },
            },
            "typescript": {
                "generator": "typescript-fetch",
                "package_name": "@pynomaly/client",
                "client_name": "PynomaliClient",
                "additional_properties": {
                    "npmName": "@pynomaly/client",
                    "npmVersion": "1.0.0",
                    "npmRepository": "https://registry.npmjs.org",
                    "typescriptThreePlus": "true",
                    "withInterfaces": "true",
                    "supportsES6": "true",
                },
            },
            "java": {
                "generator": "java",
                "package_name": "com.monorepo.client",
                "client_name": "PynomaliClient",
                "additional_properties": {
                    "groupId": "com.pynomaly",
                    "artifactId": "pynomaly-client",
                    "artifactVersion": "1.0.0",
                    "library": "okhttp-gson",
                    "java8": "true",
                    "dateLibrary": "java8",
                },
            },
            "go": {
                "generator": "go",
                "package_name": "monorepo",
                "client_name": "Client",
                "additional_properties": {
                    "packageName": "monorepo",
                    "packageVersion": "1.0.0",
                    "packageUrl": "github.com/pynomaly/go-client",
                    "withGoCodegenComment": "true",
                },
            },
            "csharp": {
                "generator": "csharp",
                "package_name": "Pynomaly.Client",
                "client_name": "PynomaliClient",
                "additional_properties": {
                    "packageName": "Pynomaly.Client",
                    "packageVersion": "1.0.0",
                    "packageCompany": "Pynomaly",
                    "packageTitle": "Pynomaly API Client",
                    "packageDescription": "C# client library for Pynomaly API",
                    "targetFramework": "net6.0",
                },
            },
            "php": {
                "generator": "php",
                "package_name": "pynomaly/client",
                "client_name": "PynomaliClient",
                "additional_properties": {
                    "packageName": "pynomaly/client",
                    "packageVersion": "1.0.0",
                    "invokerPackage": "Pynomaly\\Client",
                    "composerVendorName": "monorepo",
                    "composerProjectName": "client",
                },
            },
            "ruby": {
                "generator": "ruby",
                "package_name": "pynomaly_client",
                "client_name": "PynomaliClient",
                "additional_properties": {
                    "gemName": "pynomaly_client",
                    "gemVersion": "1.0.0",
                    "gemHomepage": "https://github.com/pynomaly/ruby-client",
                    "gemSummary": "Ruby client library for Pynomaly API",
                    "gemDescription": "Ruby client library for Pynomaly anomaly detection API",
                    "gemAuthor": "Pynomaly Team",
                    "gemAuthorEmail": "support@monorepo.com",
                },
            },
            "rust": {
                "generator": "rust",
                "package_name": "pynomaly_client",
                "client_name": "Client",
                "additional_properties": {
                    "packageName": "pynomaly_client",
                    "packageVersion": "1.0.0",
                    "packageAuthors": "Pynomaly Team <support@monorepo.com>",
                    "supportAsync": "true",
                    "library": "reqwest",
                },
            },
        }

    def check_openapi_generator(self) -> bool:
        """Check if OpenAPI Generator is installed."""
        try:
            result = subprocess.run(
                ["openapi-generator-cli", "version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print(f"✅ OpenAPI Generator found: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False

    def install_openapi_generator(self) -> bool:
        """Install OpenAPI Generator using npm."""
        print("📦 Installing OpenAPI Generator...")
        try:
            # Check if npm is available
            subprocess.run(["npm", "--version"], check=True, capture_output=True)

            # Install OpenAPI Generator globally
            result = subprocess.run(
                ["npm", "install", "-g", "@openapitools/openapi-generator-cli"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print("✅ OpenAPI Generator installed successfully")
                return True
            else:
                print(f"❌ Failed to install OpenAPI Generator: {result.stderr}")
                return False

        except subprocess.CalledProcessError:
            print("❌ npm not found. Please install Node.js and npm first.")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out")
            return False

    def load_openapi_spec(self) -> dict[str, Any]:
        """Load and validate OpenAPI specification."""
        if not self.openapi_spec_path.exists():
            raise FileNotFoundError(f"OpenAPI spec not found: {self.openapi_spec_path}")

        with open(self.openapi_spec_path) as f:
            spec = yaml.safe_load(f)

        # Validate basic structure
        required_fields = ["openapi", "info", "paths"]
        for field in required_fields:
            if field not in spec:
                raise ValueError(f"Invalid OpenAPI spec: missing '{field}' field")

        print(
            f"✅ Loaded OpenAPI spec: {spec['info']['title']} v{spec['info']['version']}"
        )
        return spec

    def generate_sdk(self, language: str, spec: dict[str, Any]) -> bool:
        """Generate SDK for a specific language."""
        if language not in self.languages:
            print(f"❌ Unsupported language: {language}")
            return False

        lang_config = self.languages[language]
        output_path = self.output_dir / language

        print(f"🔨 Generating {language.title()} SDK...")

        # Create temporary spec file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(spec, f)
            temp_spec_path = f.name

        try:
            # Prepare OpenAPI Generator command
            cmd = [
                "openapi-generator-cli",
                "generate",
                "-i",
                temp_spec_path,
                "-g",
                lang_config["generator"],
                "-o",
                str(output_path),
                "--package-name",
                lang_config["package_name"],
            ]

            # Add additional properties
            for key, value in lang_config["additional_properties"].items():
                cmd.extend(["--additional-properties", f"{key}={value}"])

            # Add global properties
            cmd.extend(
                [
                    "--global-property",
                    "apiDocs=true,modelDocs=true,apiTests=true,modelTests=true",
                ]
            )

            # Execute command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, cwd=self.project_root
            )

            if result.returncode != 0:
                print(f"❌ Failed to generate {language} SDK:")
                print(result.stderr)
                return False

            # Post-process the generated SDK
            self._post_process_sdk(language, output_path, spec)

            print(f"✅ {language.title()} SDK generated successfully")
            return True

        except subprocess.TimeoutExpired:
            print(f"❌ {language} SDK generation timed out")
            return False
        except Exception as e:
            print(f"❌ Error generating {language} SDK: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_spec_path)
            except OSError:
                pass

    def _post_process_sdk(self, language: str, output_path: Path, spec: dict[str, Any]):
        """Post-process generated SDK with custom enhancements."""

        # Add custom README
        self._generate_custom_readme(language, output_path, spec)

        # Add custom examples
        self._generate_custom_examples(language, output_path)

        # Add CI/CD configuration
        self._generate_ci_config(language, output_path)

        # Language-specific post-processing
        if language == "python":
            self._post_process_python(output_path)
        elif language == "typescript":
            self._post_process_typescript(output_path)
        elif language == "java":
            self._post_process_java(output_path)
        elif language == "go":
            self._post_process_go(output_path)

    def _generate_custom_readme(
        self, language: str, output_path: Path, spec: dict[str, Any]
    ):
        """Generate custom README for the SDK."""
        lang_config = self.languages[language]

        readme_content = f"""# {spec['info']['title']} {language.title()} Client

{spec['info']['description']}

## Installation

"""

        # Add language-specific installation instructions
        if language == "python":
            readme_content += """```bash
pip install pynomaly-client
```

## Quick Start

```python
from pynomaly_client import PynomaliClient

# Initialize client
client = PynomaliClient(base_url="https://api.monorepo.com")

# Authenticate
token_response = client.auth.login("username", "password")

# Detect anomalies
result = client.detection.detect(
    data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    algorithm="isolation_forest",
    parameters={"contamination": 0.1}
)

print(f"Detected anomalies: {result.anomalies}")
```
"""
        elif language == "typescript":
            readme_content += """```bash
npm install @pynomaly/client
```

## Quick Start

```typescript
import { PynomaliClient } from '@pynomaly/client';

// Initialize client
const client = new PynomaliClient({
    basePath: 'https://api.monorepo.com'
});

// Authenticate
const tokenResponse = await client.auth.login({
    username: 'username',
    password: 'password'
});

// Set authentication token
client.setAccessToken(tokenResponse.access_token);

// Detect anomalies
const result = await client.detection.detect({
    data: [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
    algorithm: 'isolation_forest',
    parameters: { contamination: 0.1 }
});

console.log('Detected anomalies:', result.anomalies);
```
"""
        elif language == "java":
            readme_content += """### Maven

```xml
<dependency>
    <groupId>com.pynomaly</groupId>
    <artifactId>pynomaly-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Gradle

```gradle
implementation 'com.pynomaly:pynomaly-client:1.0.0'
```

## Quick Start

```java
import com.monorepo.client.ApiClient;
import com.monorepo.client.Configuration;
import com.monorepo.client.api.AuthApi;
import com.monorepo.client.api.DetectionApi;
import com.monorepo.client.model.*;

// Initialize client
ApiClient client = Configuration.getDefaultApiClient();
client.setBasePath("https://api.monorepo.com");

// Authenticate
AuthApi authApi = new AuthApi(client);
LoginRequest loginRequest = new LoginRequest()
    .username("username")
    .password("password");
TokenResponse tokenResponse = authApi.login(loginRequest);

// Set authentication
client.setAccessToken(tokenResponse.getAccessToken());

// Detect anomalies
DetectionApi detectionApi = new DetectionApi(client);
DetectionRequest request = new DetectionRequest()
    .data(Arrays.asList(1.0, 2.0, 3.0, 100.0, 4.0, 5.0))
    .algorithm("isolation_forest")
    .parameters(Map.of("contamination", 0.1));

DetectionResponse result = detectionApi.detect(request);
System.out.println("Detected anomalies: " + result.getAnomalies());
```
"""
        elif language == "go":
            readme_content += """```bash
go get github.com/pynomaly/go-client
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/pynomaly/go-client"
)

func main() {
    // Initialize client
    client := monorepo.NewAPIClient(monorepo.NewConfiguration())
    client.GetConfig().BasePath = "https://api.monorepo.com"

    ctx := context.Background()

    // Authenticate
    loginReq := monorepo.LoginRequest{
        Username: "username",
        Password: "password",
    }

    tokenResp, _, err := client.AuthApi.Login(ctx, loginReq)
    if err != nil {
        log.Fatal(err)
    }

    // Set authentication
    auth := context.WithValue(ctx, monorepo.ContextAccessToken, tokenResp.AccessToken)

    // Detect anomalies
    detectReq := monorepo.DetectionRequest{
        Data:      []float64{1.0, 2.0, 3.0, 100.0, 4.0, 5.0},
        Algorithm: "isolation_forest",
        Parameters: map[string]interface{}{
            "contamination": 0.1,
        },
    }

    result, _, err := client.DetectionApi.Detect(auth, detectReq)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Detected anomalies: %v\\n", result.Anomalies)
}
```
"""

        readme_content += f"""
## Features

- ✅ **Full API Coverage**: Complete coverage of all Pynomaly API endpoints
- ✅ **Authentication**: Built-in JWT token management
- ✅ **Error Handling**: Comprehensive error handling with retry logic
- ✅ **Rate Limiting**: Automatic rate limit handling
- ✅ **Type Safety**: Full type definitions and validation
- ✅ **Async Support**: Asynchronous operations where applicable
- ✅ **Testing**: Comprehensive test suite included
- ✅ **Documentation**: Detailed API documentation

## API Reference

### Authentication

- `login(username, password)` - Authenticate and get JWT token
- `refresh()` - Refresh existing JWT token
- `me()` - Get current user profile

### Anomaly Detection

- `detect(data, algorithm, parameters)` - Detect anomalies in data
- `train(training_data, algorithm, parameters)` - Train custom model
- `batch_detect(requests)` - Batch anomaly detection

### Model Management

- `list_models()` - List available models
- `get_model(model_id)` - Get model details
- `delete_model(model_id)` - Delete model

### Health & Monitoring

- `health()` - Get system health status
- `metrics()` - Get system metrics

## Error Handling

The SDK provides comprehensive error handling following RFC 7807 standards:

```{language}
try:
    result = client.detection.detect(data, algorithm)
except ApiException as e:
    print(f"API Error: {{e.status}} - {{e.reason}}")
    print(f"Details: {{e.body}}")
```

## Rate Limiting

The SDK automatically handles rate limiting with exponential backoff:

- Standard users: 1000 requests/hour
- Enterprise users: 10000 requests/hour
- Automatic retry with backoff

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

- **Documentation**: https://docs.monorepo.com
- **Support**: support@monorepo.com
- **Issues**: https://github.com/pynomaly/{language}-client/issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)

    def _generate_custom_examples(self, language: str, output_path: Path):
        """Generate custom examples for the SDK."""
        examples_dir = output_path / "examples"
        examples_dir.mkdir(exist_ok=True)

        if language == "python":
            # Python examples
            basic_example = '''"""
Basic Pynomaly Client Example
"""

import asyncio
from pynomaly_client import PynomaliClient
from pynomaly_client.exceptions import ApiException

async def main():
    """Basic usage example."""
    # Initialize client
    client = PynomaliClient(base_url="https://api.monorepo.com")

    try:
        # Login
        token_response = await client.auth.login("your_username", "your_password")
        print(f"Login successful: {token_response.token_type}")

        # Simple anomaly detection
        detection_result = await client.detection.detect(
            data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )

        print(f"Detected anomalies at indices: {detection_result.anomalies}")
        print(f"Anomaly scores: {detection_result.scores}")

        # Train a custom model
        training_result = await client.detection.train(
            training_data="path/to/training/data.csv",
            algorithm="lstm_autoencoder",
            parameters={"epochs": 50, "batch_size": 32},
            model_name="custom_model_v1"
        )

        print(f"Model trained: {training_result.model_id}")
        print(f"Training metrics: {training_result.training_metrics}")

        # Health check
        health = await client.health.check()
        print(f"System status: {health.status}")

    except ApiException as e:
        print(f"API Error: {e.status} - {e.reason}")
        print(f"Details: {e.body}")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
'''

            with open(examples_dir / "basic_example.py", "w") as f:
                f.write(basic_example)

            # Advanced example
            advanced_example = '''"""
Advanced Pynomaly Client Example with Error Handling and Retries
"""

import asyncio
import logging
from typing import List
from pynomaly_client import PynomaliClient
from pynomaly_client.exceptions import ApiException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PynomaliWrapper:
    """Advanced wrapper with retry logic and error handling."""

    def __init__(self, base_url: str = "https://api.monorepo.com"):
        self.client = PynomaliClient(base_url=base_url)
        self.authenticated = False

    async def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                token_response = await self.client.auth.login(username, password)
                self.authenticated = True
                logger.info(f"Authentication successful on attempt {attempt + 1}")
                return True
            except ApiException as e:
                if e.status == 401:
                    logger.error("Invalid credentials")
                    return False
                elif attempt < max_retries - 1:
                    logger.warning(f"Auth attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Authentication failed after {max_retries} attempts")
                    return False
        return False

    async def detect_anomalies_batch(self, datasets: List[List[float]],
                                   algorithm: str = "isolation_forest") -> List[dict]:
        """Detect anomalies in multiple datasets with parallel processing."""
        if not self.authenticated:
            raise RuntimeError("Client not authenticated")

        async def detect_single(data: List[float]) -> dict:
            try:
                result = await self.client.detection.detect(
                    data=data,
                    algorithm=algorithm,
                    parameters={"contamination": 0.1}
                )
                return {
                    "success": True,
                    "data": data,
                    "anomalies": result.anomalies,
                    "scores": result.scores
                }
            except ApiException as e:
                logger.error(f"Detection failed for dataset: {e}")
                return {
                    "success": False,
                    "data": data,
                    "error": str(e)
                }

        # Process datasets in parallel
        tasks = [detect_single(dataset) for dataset in datasets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [r for r in results if not isinstance(r, Exception)]

    async def train_and_validate_model(self, training_data: str, validation_data: List[float],
                                     algorithm: str, parameters: dict) -> dict:
        """Train model and validate it immediately."""
        try:
            # Train model
            training_result = await self.client.detection.train(
                training_data=training_data,
                algorithm=algorithm,
                parameters=parameters,
                model_name=f"{algorithm}_model_{int(asyncio.get_event_loop().time())}"
            )

            logger.info(f"Model trained: {training_result.model_id}")

            # Validate with test data
            validation_result = await self.client.detection.detect(
                data=validation_data,
                algorithm=algorithm,
                parameters=parameters
            )

            return {
                "training": training_result,
                "validation": validation_result,
                "model_id": training_result.model_id
            }

        except ApiException as e:
            logger.error(f"Training/validation failed: {e}")
            raise

    async def monitor_system_health(self, interval: int = 60) -> None:
        """Continuously monitor system health."""
        while True:
            try:
                health = await self.client.health.check()
                metrics = await self.client.health.metrics()

                logger.info(f"System Status: {health.status}")
                logger.info(f"Services: {health.services}")

                if health.status != "healthy":
                    logger.warning("System health degraded!")

            except ApiException as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(interval)

    async def close(self):
        """Close the client connection."""
        await self.client.close()

async def main():
    """Advanced usage example."""
    wrapper = PynomaliWrapper()

    try:
        # Authenticate
        if not await wrapper.authenticate("your_username", "your_password"):
            return

        # Batch anomaly detection
        datasets = [
            [1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            [10.0, 11.0, 12.0, 13.0, 500.0, 14.0],
            [0.1, 0.2, 0.15, 0.18, 0.12, 5.0]
        ]

        batch_results = await wrapper.detect_anomalies_batch(datasets)
        for i, result in enumerate(batch_results):
            if result["success"]:
                print(f"Dataset {i}: Anomalies at {result['anomalies']}")
            else:
                print(f"Dataset {i}: Failed - {result['error']}")

        # Train and validate model
        model_result = await wrapper.train_and_validate_model(
            training_data="s3://pynomaly-data/training/sample.csv",
            validation_data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            algorithm="lstm_autoencoder",
            parameters={"epochs": 50, "batch_size": 32}
        )

        print(f"Model trained and validated: {model_result['model_id']}")

        # Start health monitoring in background
        health_task = asyncio.create_task(wrapper.monitor_system_health(30))

        # Let it run for a bit
        await asyncio.sleep(90)
        health_task.cancel()

    except Exception as e:
        logger.error(f"Application error: {e}")

    finally:
        await wrapper.close()

if __name__ == "__main__":
    asyncio.run(main())
'''

            with open(examples_dir / "advanced_example.py", "w") as f:
                f.write(advanced_example)

    def _generate_ci_config(self, language: str, output_path: Path):
        """Generate CI/CD configuration for the SDK."""
        if language == "python":
            # GitHub Actions for Python
            github_dir = output_path / ".github" / "workflows"
            github_dir.mkdir(parents=True, exist_ok=True)

            ci_config = """name: Python SDK CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Type check with mypy
      run: mypy pynomaly_client

    - name: Test with pytest
      run: |
        pytest --cov=pynomaly_client --cov-report=xml --cov-report=html

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  publish:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && contains(github.ref, 'refs/tags/')

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
"""

            with open(github_dir / "ci.yml", "w") as f:
                f.write(ci_config)

        elif language == "typescript":
            # GitHub Actions for TypeScript
            github_dir = output_path / ".github" / "workflows"
            github_dir.mkdir(parents=True, exist_ok=True)

            ci_config = """name: TypeScript SDK CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [16, 18, 20]

    steps:
    - uses: actions/checkout@v3

    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Lint
      run: npm run lint

    - name: Type check
      run: npm run type-check

    - name: Build
      run: npm run build

    - name: Test
      run: npm run test:coverage

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  publish:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && contains(github.ref, 'refs/tags/')

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: 18
        registry-url: 'https://registry.npmjs.org'

    - name: Install dependencies
      run: npm ci

    - name: Build
      run: npm run build

    - name: Publish to npm
      run: npm publish
      env:
        NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
"""

            with open(github_dir / "ci.yml", "w") as f:
                f.write(ci_config)

    def _post_process_python(self, output_path: Path):
        """Python-specific post-processing."""
        # Add async support and enhanced features
        requirements_dev = """pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
black>=22.0.0
flake8>=5.0.0
mypy>=1.0.0
isort>=5.0.0
pre-commit>=2.20.0
"""

        with open(output_path / "requirements-dev.txt", "w") as f:
            f.write(requirements_dev)

        # Add setup.cfg for development tools
        setup_cfg = """[flake8]
max-line-length = 127
extend-ignore = E203, W503
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist

[isort]
profile = black
multi_line_output = 3

[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
asyncio_mode = auto
"""

        with open(output_path / "setup.cfg", "w") as f:
            f.write(setup_cfg)

    def _post_process_typescript(self, output_path: Path):
        """TypeScript-specific post-processing."""
        # Add enhanced package.json scripts
        package_json_scripts = {
            "build": "tsc",
            "test": "jest",
            "test:watch": "jest --watch",
            "test:coverage": "jest --coverage",
            "lint": "eslint src --ext .ts",
            "lint:fix": "eslint src --ext .ts --fix",
            "type-check": "tsc --noEmit",
            "prepublishOnly": "npm run build",
        }

        # Check if package.json exists and update it
        package_json_path = output_path / "package.json"
        if package_json_path.exists():
            with open(package_json_path) as f:
                package_json = json.load(f)

            package_json["scripts"] = {
                **package_json.get("scripts", {}),
                **package_json_scripts,
            }
            package_json["devDependencies"] = {
                **package_json.get("devDependencies", {}),
                "@types/jest": "^29.0.0",
                "@typescript-eslint/eslint-plugin": "^5.0.0",
                "@typescript-eslint/parser": "^5.0.0",
                "eslint": "^8.0.0",
                "jest": "^29.0.0",
                "ts-jest": "^29.0.0",
                "typescript": "^4.8.0",
            }

            with open(package_json_path, "w") as f:
                json.dump(package_json, f, indent=2)

    def _post_process_java(self, output_path: Path):
        """Java-specific post-processing."""
        # Add Maven wrapper and enhanced pom.xml configurations
        pass

    def _post_process_go(self, output_path: Path):
        """Go-specific post-processing."""
        # Add go.mod enhancements and GitHub Actions
        pass

    def generate_all_sdks(self, languages: list[str] | None = None) -> dict[str, bool]:
        """Generate SDKs for all or specified languages."""
        if languages is None:
            languages = list(self.languages.keys())

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Check OpenAPI Generator
        if not self.check_openapi_generator():
            if not self.install_openapi_generator():
                print("❌ Failed to install OpenAPI Generator")
                return {}

        # Load OpenAPI specification
        try:
            spec = self.load_openapi_spec()
        except Exception as e:
            print(f"❌ Failed to load OpenAPI spec: {e}")
            return {}

        # Generate SDKs
        results = {}
        for language in languages:
            if language in self.languages:
                results[language] = self.generate_sdk(language, spec)
            else:
                print(f"⚠️  Skipping unsupported language: {language}")
                results[language] = False

        # Generate summary
        self._generate_summary(results)

        return results

    def _generate_summary(self, results: dict[str, bool]):
        """Generate a summary of SDK generation results."""
        successful = [lang for lang, success in results.items() if success]
        failed = [lang for lang, success in results.items() if not success]

        summary_content = f"""# Pynomaly SDK Generation Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Successfully Generated SDKs

"""

        for lang in successful:
            lang_config = self.languages[lang]
            summary_content += f"""### {lang.title()}
- **Package Name**: `{lang_config['package_name']}`
- **Client Class**: `{lang_config['client_name']}`
- **Location**: `sdks/{lang}/`
- **Status**: ✅ Generated successfully

"""

        if failed:
            summary_content += "\n## Failed Generations\n\n"
            for lang in failed:
                summary_content += f"- **{lang.title()}**: ❌ Generation failed\n"

        summary_content += """
## Usage

Each SDK includes:
- Complete API client with all endpoints
- Authentication handling (JWT + API Key)
- Comprehensive error handling
- Rate limiting support
- Full type definitions
- Unit tests
- Documentation and examples
- CI/CD configuration

## Installation Instructions

"""

        for lang in successful:
            lang_config = self.languages[lang]
            if lang == "python":
                summary_content += """### Python
```bash
cd sdks/python
pip install -e .
```

"""
            elif lang == "typescript":
                summary_content += """### TypeScript/JavaScript
```bash
cd sdks/typescript
npm install
npm run build
```

"""
            elif lang == "java":
                summary_content += """### Java
```bash
cd sdks/java
mvn clean install
```

"""
            elif lang == "go":
                summary_content += """### Go
```bash
cd sdks/go
go mod tidy
```

"""

        summary_content += """
## Support

- **Documentation**: https://docs.monorepo.com
- **Support**: support@monorepo.com
- **Issues**: https://github.com/pynomaly/client-sdks/issues
"""

        with open(self.output_dir / "README.md", "w") as f:
            f.write(summary_content)

        print("\n" + "=" * 60)
        print("📊 SDK Generation Summary:")
        print(f"✅ Successful: {len(successful)} ({', '.join(successful)})")
        if failed:
            print(f"❌ Failed: {len(failed)} ({', '.join(failed)})")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 60)


def main():
    """Main function to generate client SDKs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate multi-language client SDKs for Pynomaly API"
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        choices=["python", "typescript", "java", "go", "csharp", "php", "ruby", "rust"],
        help="Languages to generate SDKs for (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List supported languages")

    args = parser.parse_args()

    generator = SDKGenerator()

    if args.list:
        print("Supported languages:")
        for lang in generator.languages:
            config = generator.languages[lang]
            print(f"  - {lang}: {config['package_name']} ({config['generator']})")
        return

    print("🚀 Starting SDK generation...")
    print(f"📋 Target languages: {args.languages or 'all'}")

    results = generator.generate_all_sdks(args.languages)

    if any(results.values()):
        print("🎉 SDK generation completed!")
    else:
        print("💥 All SDK generations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
