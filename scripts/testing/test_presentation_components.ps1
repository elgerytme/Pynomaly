# Pynomaly Presentation Components Test Suite - PowerShell Version
# Tests CLI, API, and Web UI components in fresh environments

param(
    [switch]$Verbose = $false,
    [switch]$SkipOptional = $false
)

# Test results tracking
$script:TotalTests = 0
$script:PassedTests = 0
$script:FailedTests = 0

# Helper functions
function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
    $script:PassedTests++
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
    $script:FailedTests++
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Invoke-PynomaryTest {
    param(
        [string]$TestName,
        [scriptblock]$TestScript
    )
    
    Write-Host ""
    Write-Host "Testing: $TestName" -ForegroundColor Blue
    Write-Host "----------------------------------------"
    
    $script:TotalTests++
    
    try {
        $result = & $TestScript
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
            Write-Success "$TestName passed"
            return $true
        } else {
            Write-Error-Custom "$TestName failed (exit code: $LASTEXITCODE)"
            return $false
        }
    }
    catch {
        Write-Error-Custom "$TestName failed with exception: $($_.Exception.Message)"
        return $false
    }
}

function Test-Environment {
    Write-Info "Setting up PowerShell test environment..."
    
    # Check Python availability
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    
    if (-not $pythonCmd) {
        Write-Error-Custom "Python not found in PATH"
        return $false
    }
    
    # Set environment variables
    $env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
    $env:PYNOMALY_TEST_ENV = "PowerShell"
    
    Write-Success "PowerShell environment setup complete"
    return $true
}

function Test-Dependencies {
    $testScript = {
        $pythonCode = @"
import sys
sys.path.insert(0, "src")

dependencies = {
    "Core": ["pyod", "numpy", "pandas", "polars", "pydantic", "structlog"],
    "API": ["fastapi", "uvicorn", "httpx", "requests", "jinja2", "aiofiles"],
    "CLI": ["typer", "rich"],
    "Optional": ["shap", "lime"]
}

all_available = True

for category, deps in dependencies.items():
    print(f"{category} dependencies:")
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            if category == "Optional":
                print(f"  ⚠️  {dep} (optional - not installed)")
            else:
                print(f"  ❌ {dep} (required but missing)")
                all_available = False
    print()

if all_available:
    print("✅ All required dependencies available")
else:
    print("❌ Some required dependencies missing")
    sys.exit(1)
"@
        
        python -c $pythonCode
    }
    
    return Invoke-PynomaryTest "Dependency Availability" $testScript
}

function Test-CLIComponent {
    $testScript = {
        $pythonCode = @"
import sys
sys.path.insert(0, "src")

try:
    # Test CLI imports
    from pynomaly.presentation.cli.app import app
    print("✅ CLI app imported successfully")
    
    # Test dependencies
    import typer
    import rich
    from rich.console import Console
    
    print("✅ CLI dependencies available")
    
    # Test CLI functionality
    console = Console()
    console.print("✅ Rich console working", style="green")
    
    print("✅ CLI component test completed successfully")
    
except Exception as e:
    print(f"❌ CLI test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
        
        python -c $pythonCode
    }
    
    return Invoke-PynomaryTest "CLI Component" $testScript
}

function Test-APIComponent {
    $testScript = {
        $pythonCode = @"
import sys
sys.path.insert(0, "src")

try:
    # Test API imports
    from pynomaly.presentation.api.app import create_app
    print("✅ API create_app imported successfully")
    
    # Test dependencies
    import fastapi
    import uvicorn
    from fastapi.testclient import TestClient
    
    print("✅ API dependencies available")
    
    # Test app creation
    app = create_app()
    print("✅ API application created successfully")
    
    # Test with client
    client = TestClient(app)
    response = client.get("/api/health")
    
    if response.status_code == 200:
        print(f"✅ Health endpoint working (status: {response.status_code})")
    else:
        print(f"⚠️  Health endpoint returned: {response.status_code}")
    
    print("✅ API component test completed successfully")
    
except Exception as e:
    print(f"❌ API test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
        
        python -c $pythonCode
    }
    
    return Invoke-PynomaryTest "API Component" $testScript
}

function Test-WebUIComponent {
    $testScript = {
        $pythonCode = @"
import sys
sys.path.insert(0, "src")

try:
    # Test Web UI imports
    from pynomaly.presentation.web.app import create_web_app, mount_web_ui
    print("✅ Web UI functions imported successfully")
    
    # Test dependencies
    from jinja2 import Template
    from fastapi.staticfiles import StaticFiles
    from fastapi.testclient import TestClient
    
    print("✅ Web UI dependencies available")
    
    # Test app creation
    app = create_web_app()
    print("✅ Complete web application created successfully")
    
    # Count routes
    total_routes = len([r for r in app.routes if hasattr(r, "path")])
    web_routes = len([r for r in app.routes if hasattr(r, "path") and r.path.startswith("/web")])
    api_routes = len([r for r in app.routes if hasattr(r, "path") and r.path.startswith("/api")])
    
    print(f"✅ Routes configured - Total: {total_routes}, Web: {web_routes}, API: {api_routes}")
    
    # Test with client
    client = TestClient(app)
    
    # Test API health
    response = client.get("/api/health")
    if response.status_code == 200:
        print(f"✅ API health endpoint working (status: {response.status_code})")
    
    # Test web UI root
    response = client.get("/web/")
    if response.status_code in [200, 302]:  # 200 for content, 302 for redirect
        print(f"✅ Web UI root endpoint working (status: {response.status_code})")
    
    print("✅ Web UI component test completed successfully")
    
except Exception as e:
    print(f"❌ Web UI test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
        
        python -c $pythonCode
    }
    
    return Invoke-PynomaryTest "Web UI Component" $testScript
}

function Test-PowerShellEnvironment {
    $testScript = {
        $pythonCode = @"
import sys
import os
import platform

print("PowerShell Environment Test:")
print(f"  System: {platform.system()}")
print(f"  Python: {platform.python_version()}")
print(f"  Test Env: {os.getenv('PYNOMALY_TEST_ENV', 'Unknown')}")
print()

try:
    sys.path.insert(0, "src")
    
    # Test imports
    from pynomaly.presentation.cli.app import app as cli_app
    from pynomaly.presentation.api.app import create_app
    from pynomaly.presentation.web.app import create_web_app
    
    print("✅ All presentation components importable")
    
    # Test functionality
    api_app = create_app()
    web_app = create_web_app()
    
    print("✅ All components instantiated successfully")
    print("✅ PowerShell environment test completed")
    
except Exception as e:
    print(f"❌ PowerShell environment test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@
        
        python -c $pythonCode
    }
    
    return Invoke-PynomaryTest "PowerShell Environment" $testScript
}

# Main execution
function Main {
    Write-Host "🧪 Pynomaly Presentation Components Test Suite (PowerShell)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Test environment: Windows PowerShell"
    Write-Host "PowerShell version: $($PSVersionTable.PSVersion)"
    
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "Python version: $pythonVersion"
    }
    catch {
        Write-Host "Python version: Not available"
    }
    
    Write-Host ""
    
    # Setup
    if (-not (Test-Environment)) {
        Write-Host ""
        Write-Host "❌ Environment setup failed. Exiting." -ForegroundColor Red
        exit 1
    }
    
    # Run tests
    Write-Host ""
    Write-Host "🧪 Running Test Suite" -ForegroundColor Blue
    Write-Host "====================="
    
    $testResults = @()
    
    $testResults += Test-Dependencies
    $testResults += Test-CLIComponent
    $testResults += Test-APIComponent
    $testResults += Test-WebUIComponent
    $testResults += Test-PowerShellEnvironment
    
    # Test summary
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "📊 Test Results Summary" -ForegroundColor Blue
    Write-Host "=============================================="
    Write-Host "Total Tests: $script:TotalTests"
    Write-Host "Passed: $script:PassedTests" -ForegroundColor Green
    Write-Host "Failed: $script:FailedTests" -ForegroundColor Red
    
    if ($script:FailedTests -eq 0) {
        Write-Host ""
        Write-Host "🎉 All tests passed! Presentation components are working correctly." -ForegroundColor Green
        Write-Host ""
        Write-Host "✅ CLI Component: Ready for use"
        Write-Host "✅ API Component: Ready for use"
        Write-Host "✅ Web UI Component: Ready for use"
        Write-Host ""
        Write-Host "You can now run:"
        Write-Host "  - CLI: pynomaly --help"
        Write-Host "  - API: uvicorn pynomaly.presentation.api:app"
        Write-Host "  - Web UI: uvicorn pynomaly.presentation.web.app:create_web_app"
        
        exit 0
    }
    else {
        Write-Host ""
        Write-Host "❌ Some tests failed. Please check the output above for details." -ForegroundColor Red
        exit 1
    }
}

# Run main function
Main