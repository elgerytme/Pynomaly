#!/usr/bin/env python3
"""
Simple Integration Test Runner

A basic test runner for integration tests that works with the current setup.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add paths to support imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleIntegrationTestRunner:
    """Simple integration test runner."""
    
    def __init__(self):
        """Initialize test runner."""
        self.results = {}
        self.start_time = None
        
    def run_tests(self):
        """Run integration tests."""
        logger.info("Starting integration tests...")
        self.start_time = time.time()
        
        # Import subprocess to run pytest
        import subprocess
        
        # Run pytest with the virtual environment
        venv_python = "./environments/.venv/bin/python"
        cmd = [
            venv_python, "-m", "pytest",
            "tests/integration/test_simple_integration.py",
            "-v",
            "--tb=short"
        ]
        
        try:
            # Create results directory
            results_dir = Path("test-results/integration")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Run tests
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            # Parse results
            self.results = {
                "start_time": datetime.now().isoformat(),
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": time.time() - self.start_time,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }
            
            # Log results
            logger.info(f"Test execution completed with return code: {result.returncode}")
            logger.info(f"Execution time: {self.results['execution_time']:.2f}s")
            
            # Save detailed results
            self._save_results()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            self.results = {
                "start_time": datetime.now().isoformat(),
                "error": str(e),
                "status": "ERROR"
            }
            return self.results
    
    def _save_results(self):
        """Save test results."""
        results_dir = Path("test-results/integration")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = results_dir / f"simple_integration_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save latest results
        latest_file = results_dir / "latest_simple_results.json"
        with open(latest_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        if not self.results:
            logger.error("No results available")
            return
        
        print("\n" + "="*50)
        print("INTEGRATION TEST SUMMARY")
        print("="*50)
        print(f"Status: {self.results['status']}")
        print(f"Execution Time: {self.results.get('execution_time', 0):.2f}s")
        print(f"Return Code: {self.results.get('returncode', 'N/A')}")
        print("="*50)
        
        if self.results.get('stdout'):
            print("\nSTDOUT:")
            print(self.results['stdout'])
        
        if self.results.get('stderr'):
            print("\nSTDERR:")
            print(self.results['stderr'])


def main():
    """Main entry point."""
    runner = SimpleIntegrationTestRunner()
    
    try:
        results = runner.run_tests()
        runner.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if results["status"] == "PASSED" else 1)
        
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()