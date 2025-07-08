#!/usr/bin/env python3
"""
Development environment setup script for Pynomaly circular import debugging.

This script sets up a reproducible development environment that guarantees
everyone sees the same circular import tracebacks.
"""

import sys
import os
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the development environment."""
    project_root = Path(__file__).parent
    
    print("🔧 Setting up Pynomaly Development Environment")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print()
    
    # Create necessary directories
    directories = [
        "debug_reports",
        "logs", 
        "temp"
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"  ✓ {directory}")
    
    # Set environment variables
    print("\n🌍 Setting environment variables...")
    env_vars = {
        "PYTHONPATH": str(project_root / "src"),
        "PYNOMALY_ENVIRONMENT": "development", 
        "PYNOMALY_DEBUG": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✓ {key}={value}")
    
    # Check Python path
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print(f"\n🐍 Python path configured:")
    print(f"  ✓ {src_path}")
    
    return True

def run_tests():
    """Run the circular import reproduction tests."""
    print("\n🧪 Running circular import tests...")
    print("=" * 50)
    
    # Run the reproduction script
    try:
        result = subprocess.run([
            sys.executable, "reproduce_circular_imports.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_detailed_analysis():
    """Run detailed circular import analysis."""
    print("\n🔍 Running detailed circular import analysis...")
    print("=" * 50)
    
    try:
        # Run the debug script
        result = subprocess.run([
            sys.executable, "debug_imports.py"
        ], capture_output=True, text=True, timeout=30)
        
        # Save the output
        debug_reports_dir = Path("debug_reports")
        output_file = debug_reports_dir / "import_trace_latest.log"
        
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)
        
        print(f"📄 Detailed trace saved to: {output_file}")
        
        # Show summary
        if "CIRCULAR IMPORT" in result.stdout:
            print("🔄 Circular imports detected!")
            circular_count = result.stdout.count("POTENTIAL CIRCULAR IMPORT")
            print(f"   Found {circular_count} circular import patterns")
        else:
            print("✅ No circular imports detected in basic trace")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Analysis timed out (imports may be hanging)")
        return False
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

def start_web_application():
    """Start the web application for testing."""
    print("\n🌐 Starting web application...")
    print("=" * 50)
    
    try:
        # Test if we can import and create the web app
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        from pynomaly.presentation.web.app import create_web_app
        app = create_web_app()
        
        print("✅ Web application created successfully")
        print("🚀 Application ready for testing")
        print()
        print("Available endpoints:")
        print("  - Web UI: http://localhost:8000/web/")
        print("  - API Docs: http://localhost:8000/api/docs") 
        print("  - Health Check: http://localhost:8000/api/health")
        print()
        print("To start the server, run:")
        print("  python scripts/development/run_web_app.py --port 8000")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating web application: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup function."""
    try:
        print("🎯 Pynomaly Development Environment Setup")
        print("=" * 50)
        print("This script will:")
        print("1. Set up the development environment")
        print("2. Run circular import reproduction tests")
        print("3. Generate detailed analysis reports")
        print("4. Test web application creation")
        print()
        
        # Step 1: Setup environment
        if not setup_environment():
            print("❌ Environment setup failed")
            return False
        
        # Step 2: Run tests
        if not run_tests():
            print("⚠️ Some tests failed (this may be expected)")
        
        # Step 3: Run detailed analysis
        run_detailed_analysis()
        
        # Step 4: Test web application
        start_web_application()
        
        print("\n🎉 Development environment setup complete!")
        print("=" * 50)
        print()
        print("Next steps:")
        print("1. Review the generated reports in debug_reports/")
        print("2. Run: python scripts/development/run_web_app.py --port 8000")
        print("3. Check for circular imports with debug scripts")
        print()
        print("Debug commands:")
        print("  python debug_imports.py          # Simple import trace")
        print("  python debug_circular_imports.py # Comprehensive analysis")
        print("  python reproduce_circular_imports.py # Reproduction test")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
