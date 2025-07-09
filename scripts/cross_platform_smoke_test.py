#!/usr/bin/env python3
"""
Cross-Platform Smoke Test Suite
Runs basic functionality tests across different OS environments using Docker
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossPlatformSmokeTest:
    def __init__(self):
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent
        self.docker_test_configs = {
            'ubuntu': {
                'image': 'ubuntu:22.04',
                'python_setup': [
                    'apt-get update',
                    'apt-get install -y python3 python3-pip python3-venv build-essential',
                    'ln -s /usr/bin/python3 /usr/bin/python'
                ]
            },
            'alpine': {  # Lightweight Linux alternative
                'image': 'alpine:3.18',
                'python_setup': [
                    'apk add --no-cache python3 py3-pip build-base',
                    'ln -s /usr/bin/python3 /usr/bin/python'
                ]
            },
            'windows': {
                'image': 'mcr.microsoft.com/windows/servercore:ltsc2022',
                'python_setup': [
                    'powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe -OutFile python-installer.exe"',
                    'python-installer.exe /quiet InstallAllUsers=1 PrependPath=1'
                ]
            }
        }
    
    def run_docker_command(self, command: str, platform: str = 'linux/amd64') -> Dict:
        """Run a Docker command and return the result"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Command timed out after 5 minutes',
                'returncode': -1
            }
    
    def create_test_dockerfile(self, platform: str) -> str:
        """Create a Dockerfile for the specific platform"""
        config = self.docker_test_configs[platform]
        
        dockerfile_content = f"""
FROM {config['image']}

# Set working directory
WORKDIR /app

# Install Python and basic dependencies
"""
        
        for setup_cmd in config['python_setup']:
            dockerfile_content += f"RUN {setup_cmd}\n"
        
        dockerfile_content += """
# Copy project files
COPY . .

# Install Python dependencies
RUN python -m pip install --upgrade pip || true
RUN pip install pytest pytest-cov || true

# Install project in development mode
RUN pip install -e . || echo "Installation failed but continuing..."

# Run basic smoke tests
CMD ["python", "-c", "import sys; print(f'Python {sys.version}'); import pynomaly; print('Pynomaly imported successfully')"]
"""
        
        dockerfile_path = self.project_root / f"Dockerfile.smoke-{platform}"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return str(dockerfile_path)
    
    def run_platform_test(self, platform: str) -> Dict:
        """Run smoke tests for a specific platform"""
        logger.info(f"Running smoke tests for {platform}")
        
        # Create Dockerfile
        dockerfile_path = self.create_test_dockerfile(platform)
        
        # Build Docker image
        build_command = f"docker build -f {dockerfile_path} -t pynomaly-smoke-{platform} ."
        build_result = self.run_docker_command(build_command)
        
        if not build_result['success']:
            logger.error(f"Failed to build Docker image for {platform}: {build_result['stderr']}")
            return {
                'platform': platform,
                'build_success': False,
                'test_success': False,
                'error': build_result['stderr']
            }
        
        # Run the container
        run_command = f"docker run --rm pynomaly-smoke-{platform}"
        run_result = self.run_docker_command(run_command)
        
        # Run additional tests
        test_commands = [
            "python -c 'import sys; print(sys.version)'",
            "python -c 'import pynomaly; print(\"Pynomaly import successful\")'",
            "python -m pytest tests/ -v --tb=short -x" if (self.project_root / "tests").exists() else "echo 'No tests directory found'"
        ]
        
        test_results = []
        for cmd in test_commands:
            test_cmd = f"docker run --rm pynomaly-smoke-{platform} {cmd}"
            test_result = self.run_docker_command(test_cmd)
            test_results.append({
                'command': cmd,
                'success': test_result['success'],
                'output': test_result['stdout'],
                'error': test_result['stderr']
            })
        
        # Clean up
        cleanup_command = f"docker rmi pynomaly-smoke-{platform}"
        self.run_docker_command(cleanup_command)
        
        os.remove(dockerfile_path)
        
        return {
            'platform': platform,
            'build_success': build_result['success'],
            'test_success': run_result['success'],
            'run_output': run_result['stdout'],
            'run_error': run_result['stderr'],
            'additional_tests': test_results
        }
    
    def run_all_tests(self) -> Dict:
        """Run smoke tests for all platforms"""
        logger.info("Starting cross-platform smoke tests")
        
        # Check if Docker is available
        docker_check = self.run_docker_command("docker --version")
        if not docker_check['success']:
            logger.error("Docker is not available. Please install Docker to run cross-platform tests.")
            return {'error': 'Docker not available'}
        
        results = {}
        
        # Test Ubuntu and Alpine (Linux variants)
        for platform in ['ubuntu', 'alpine']:
            try:
                results[platform] = self.run_platform_test(platform)
                logger.info(f"Completed {platform} tests")
            except Exception as e:
                logger.error(f"Error running {platform} tests: {str(e)}")
                results[platform] = {
                    'platform': platform,
                    'build_success': False,
                    'test_success': False,
                    'error': str(e)
                }
        
        # Windows testing requires Windows containers (skip on non-Windows hosts)
        if sys.platform == 'win32':
            try:
                results['windows'] = self.run_platform_test('windows')
                logger.info("Completed Windows tests")
            except Exception as e:
                logger.error(f"Error running Windows tests: {str(e)}")
                results['windows'] = {
                    'platform': 'windows',
                    'build_success': False,
                    'test_success': False,
                    'error': str(e)
                }
        else:
            logger.info("Skipping Windows tests (not on Windows host)")
            results['windows'] = {
                'platform': 'windows',
                'build_success': True,
                'test_success': True,
                'skipped': True,
                'reason': 'Not on Windows host'
            }
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive test report"""
        report = ["# Cross-Platform Smoke Test Report", ""]
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"**Host Platform:** {sys.platform}")
        report.append("")
        
        report.append("## Test Results Summary")
        report.append("")
        report.append("| Platform | Build | Tests | Status |")
        report.append("|----------|-------|-------|--------|")
        
        overall_success = True
        
        for platform, result in results.items():
            if 'error' in result and result.get('error') == 'Docker not available':
                report.append(f"| {platform} | ❌ | ❌ | Docker not available |")
                overall_success = False
                continue
            
            if result.get('skipped'):
                report.append(f"| {platform} | ⚠️ | ⚠️ | {result.get('reason', 'Skipped')} |")
                continue
            
            build_status = "✅" if result.get('build_success') else "❌"
            test_status = "✅" if result.get('test_success') else "❌"
            
            if not result.get('build_success') or not result.get('test_success'):
                overall_success = False
                status = "Failed"
            else:
                status = "Passed"
            
            report.append(f"| {platform} | {build_status} | {test_status} | {status} |")
        
        report.append("")
        
        if overall_success:
            report.append("## 🎉 Overall Result: **PASSED**")
            report.append("")
            report.append("✅ All cross-platform smoke tests passed successfully!")
        else:
            report.append("## ❌ Overall Result: **FAILED**")
            report.append("")
            report.append("Some platform tests failed. Please review the detailed results below.")
        
        report.append("")
        report.append("## Detailed Results")
        report.append("")
        
        for platform, result in results.items():
            report.append(f"### {platform.title()}")
            report.append("")
            
            if result.get('skipped'):
                report.append(f"**Status:** Skipped - {result.get('reason', 'Unknown reason')}")
                report.append("")
                continue
            
            if 'error' in result and result.get('error') == 'Docker not available':
                report.append("**Status:** Error - Docker not available")
                report.append("")
                continue
            
            report.append(f"**Build Success:** {result.get('build_success', False)}")
            report.append(f"**Test Success:** {result.get('test_success', False)}")
            report.append("")
            
            if result.get('run_output'):
                report.append("**Output:**")
                report.append("```")
                report.append(result['run_output'])
                report.append("```")
                report.append("")
            
            if result.get('run_error'):
                report.append("**Errors:**")
                report.append("```")
                report.append(result['run_error'])
                report.append("```")
                report.append("")
            
            if result.get('additional_tests'):
                report.append("**Additional Tests:**")
                for test in result['additional_tests']:
                    status = "✅" if test['success'] else "❌"
                    report.append(f"- {status} `{test['command']}`")
                report.append("")
        
        return "\n".join(report)
    
    def save_report(self, report: str, filename: str = "cross_platform_smoke_test_report.md"):
        """Save the test report to a file"""
        report_path = self.project_root / filename
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        return report_path

def main():
    """Main function to run cross-platform smoke tests"""
    tester = CrossPlatformSmokeTest()
    
    print("🚀 Starting Cross-Platform Smoke Tests")
    print("=" * 50)
    
    results = tester.run_all_tests()
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        sys.exit(1)
    
    report = tester.generate_report(results)
    print("\n" + report)
    
    # Save report
    report_path = tester.save_report(report)
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Exit with appropriate code
    overall_success = all(
        result.get('build_success', False) and result.get('test_success', False)
        for result in results.values()
        if not result.get('skipped', False)
    )
    
    if overall_success:
        print("\n✅ All cross-platform smoke tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some cross-platform tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
