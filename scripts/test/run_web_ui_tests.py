#!/usr/bin/env python3
"""
Comprehensive Web UI Test Runner
Runs all web UI tests including integration, frontend, and end-to-end tests
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import signal
from contextlib import contextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class WebUITestRunner:
    """Comprehensive test runner for web UI components"""
    
    def __init__(self, args):
        self.args = args
        self.project_root = project_root
        self.test_results = {}
        self.server_process = None
        self.start_time = time.time()
        
    def run_all_tests(self):
        """Run all web UI tests"""
        print("🚀 Starting Comprehensive Web UI Test Suite")
        print("=" * 60)
        
        try:
            # 1. Setup test environment
            self.setup_test_environment()
            
            # 2. Start test server if needed
            if self.args.integration or self.args.all:
                self.start_test_server()
            
            # 3. Run Python integration tests
            if self.args.integration or self.args.all:
                self.run_python_integration_tests()
            
            # 4. Run frontend JavaScript tests
            if self.args.frontend or self.args.all:
                self.run_frontend_tests()
            
            # 5. Run end-to-end tests
            if self.args.e2e or self.args.all:
                self.run_e2e_tests()
            
            # 6. Run performance tests
            if self.args.performance or self.args.all:
                self.run_performance_tests()
            
            # 7. Run security tests
            if self.args.security or self.args.all:
                self.run_security_tests()
            
            # 8. Generate test report
            self.generate_test_report()
            
        except KeyboardInterrupt:
            print("\n❌ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Test runner error: {e}")
            sys.exit(1)
        finally:
            self.cleanup()
    
    def setup_test_environment(self):
        """Setup test environment"""
        print("📋 Setting up test environment...")
        
        # Check Python dependencies
        self.check_python_dependencies()
        
        # Check test files exist
        self.check_test_files()
        
        # Set environment variables
        os.environ['PYNOMALY_ENVIRONMENT'] = 'test'
        os.environ['PYNOMALY_DEBUG'] = 'false'
        os.environ['PYNOMALY_USE_DATABASE_REPOSITORIES'] = 'false'
        os.environ['PYNOMALY_CACHE_BACKEND'] = 'memory'
        
        print("✅ Test environment ready")
    
    def check_python_dependencies(self):
        """Check required Python dependencies"""
        required_packages = [
            'pytest',
            'fastapi',
            'uvicorn',
            'httpx',
            'starlette'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing Python packages: {', '.join(missing)}")
            print("Install with: pip install pytest fastapi uvicorn httpx starlette")
            sys.exit(1)
    
    def check_test_files(self):
        """Check that test files exist"""
        test_files = [
            'tests/ui/test_web_ui_integration.py',
            'tests/ui/test_frontend_javascript.py'
        ]
        
        for test_file in test_files:
            file_path = self.project_root / test_file
            if not file_path.exists():
                print(f"❌ Test file not found: {test_file}")
                sys.exit(1)
    
    def start_test_server(self):
        """Start test server for integration tests"""
        print("🚀 Starting test server...")
        
        server_script = self.project_root / "scripts" / "run" / "run_web_app.py"
        if not server_script.exists():
            print(f"❌ Server script not found: {server_script}")
            return
        
        # Start server in background
        try:
            env = os.environ.copy()
            env['PYNOMALY_ENVIRONMENT'] = 'test'
            
            self.server_process = subprocess.Popen([
                sys.executable, str(server_script),
                '--port', '8888',
                '--log-level', 'error'
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(5)
            
            # Check if server is running
            if self.server_process.poll() is None:
                print("✅ Test server started on port 8888")
            else:
                print("❌ Failed to start test server")
                
        except Exception as e:
            print(f"❌ Error starting test server: {e}")
    
    def run_python_integration_tests(self):
        """Run Python integration tests"""
        print("\n🧪 Running Python Integration Tests...")
        
        test_file = self.project_root / "tests" / "ui" / "test_web_ui_integration.py"
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_file),
            '-v',
            '--tb=short'
        ]
        
        if self.args.coverage:
            cmd.extend(['--cov=src/pynomaly/presentation', '--cov-report=term'])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.test_results['python_integration'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Python integration tests passed")
        else:
            print("❌ Python integration tests failed")
            if self.args.verbose:
                print(result.stdout)
                print(result.stderr)
    
    def run_frontend_tests(self):
        """Run frontend JavaScript tests"""
        print("\n🌐 Running Frontend JavaScript Tests...")
        
        test_file = self.project_root / "tests" / "ui" / "test_frontend_javascript.py"
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_file),
            '-v',
            '--tb=short'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.test_results['frontend_javascript'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Frontend JavaScript tests passed")
        else:
            print("❌ Frontend JavaScript tests failed")
            if self.args.verbose:
                print(result.stdout)
                print(result.stderr)
    
    def run_e2e_tests(self):
        """Run end-to-end tests"""
        print("\n🎭 Running End-to-End Tests...")
        
        # Check if Playwright is available
        try:
            result = subprocess.run(['npx', 'playwright', '--version'], 
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️  Playwright not found, skipping E2E tests")
                return
        except FileNotFoundError:
            print("⚠️  Node.js/npm not found, skipping E2E tests")
            return
        
        # Run Playwright tests
        cmd = ['npx', 'playwright', 'test', '--reporter=line']
        
        if self.args.headed:
            cmd.append('--headed')
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        self.test_results['e2e'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ End-to-end tests passed")
        else:
            print("❌ End-to-end tests failed")
            if self.args.verbose:
                print(result.stdout)
                print(result.stderr)
    
    def run_performance_tests(self):
        """Run performance tests"""
        print("\n⚡ Running Performance Tests...")
        
        # Check if Lighthouse is available
        try:
            result = subprocess.run(['npx', 'lighthouse', '--version'], 
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️  Lighthouse not found, skipping performance tests")
                return
        except FileNotFoundError:
            print("⚠️  Node.js/npm not found, skipping performance tests")
            return
        
        # Run Lighthouse audit
        cmd = [
            'npx', 'lighthouse',
            'http://localhost:8888/web/',
            '--output=json',
            '--output-path=test_reports/lighthouse.json',
            '--chrome-flags=--headless'
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        self.test_results['performance'] = {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Performance tests completed")
            self.analyze_performance_results()
        else:
            print("❌ Performance tests failed")
            if self.args.verbose:
                print(result.stdout)
                print(result.stderr)
    
    def analyze_performance_results(self):
        """Analyze Lighthouse performance results"""
        lighthouse_file = self.project_root / "test_reports" / "lighthouse.json"
        
        if not lighthouse_file.exists():
            return
        
        try:
            with open(lighthouse_file, 'r') as f:
                data = json.load(f)
            
            categories = data.get('categories', {})
            
            print("\n📊 Performance Metrics:")
            for category_name, category_data in categories.items():
                score = category_data.get('score', 0) * 100
                print(f"  {category_name}: {score:.1f}/100")
            
            # Check Core Web Vitals
            audits = data.get('audits', {})
            vitals = {
                'largest-contentful-paint': 'LCP',
                'first-input-delay': 'FID', 
                'cumulative-layout-shift': 'CLS'
            }
            
            print("\n🎯 Core Web Vitals:")
            for audit_id, vital_name in vitals.items():
                if audit_id in audits:
                    audit = audits[audit_id]
                    value = audit.get('numericValue', 0)
                    score = audit.get('score', 0)
                    print(f"  {vital_name}: {value:.1f} (score: {score:.2f})")
                    
        except Exception as e:
            print(f"❌ Error analyzing performance results: {e}")
    
    def run_security_tests(self):
        """Run security tests"""
        print("\n🔒 Running Security Tests...")
        
        # Basic security checks
        security_checks = [
            self.check_security_headers,
            self.check_csrf_protection,
            self.check_content_security_policy,
            self.check_https_redirect
        ]
        
        passed = 0
        total = len(security_checks)
        
        for check in security_checks:
            try:
                if check():
                    passed += 1
            except Exception as e:
                print(f"❌ Security check failed: {e}")
        
        self.test_results['security'] = {
            'passed': passed == total,
            'total_checks': total,
            'passed_checks': passed
        }
        
        if passed == total:
            print(f"✅ All {total} security checks passed")
        else:
            print(f"❌ {passed}/{total} security checks passed")
    
    def check_security_headers(self):
        """Check security headers"""
        import requests
        
        try:
            response = requests.get('http://localhost:8888/web/', timeout=5)
            headers = response.headers
            
            required_headers = [
                'X-Frame-Options',
                'X-Content-Type-Options',
                'Content-Security-Policy',
                'Referrer-Policy'
            ]
            
            for header in required_headers:
                if header not in headers:
                    print(f"❌ Missing security header: {header}")
                    return False
            
            print("✅ Security headers present")
            return True
            
        except Exception as e:
            print(f"❌ Error checking security headers: {e}")
            return False
    
    def check_csrf_protection(self):
        """Check CSRF protection"""
        import requests
        
        try:
            response = requests.get('http://localhost:8888/web/', timeout=5)
            content = response.text
            
            if 'csrf-token' not in content:
                print("❌ CSRF token not found in HTML")
                return False
            
            print("✅ CSRF protection enabled")
            return True
            
        except Exception as e:
            print(f"❌ Error checking CSRF protection: {e}")
            return False
    
    def check_content_security_policy(self):
        """Check Content Security Policy"""
        import requests
        
        try:
            response = requests.get('http://localhost:8888/web/', timeout=5)
            csp = response.headers.get('Content-Security-Policy', '')
            
            if 'default-src' not in csp:
                print("❌ CSP default-src directive missing")
                return False
            
            print("✅ Content Security Policy configured")
            return True
            
        except Exception as e:
            print(f"❌ Error checking CSP: {e}")
            return False
    
    def check_https_redirect(self):
        """Check HTTPS redirect (in production)"""
        # This would be more relevant in production
        print("✅ HTTPS redirect check skipped (development mode)")
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📄 Generating Test Report...")
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': f"{duration:.2f}s",
            'results': self.test_results,
            'summary': self.get_test_summary()
        }
        
        # Save report to file
        report_file = self.project_root / "test_reports" / "web_ui_test_report.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_test_summary()
        
        print(f"\n📄 Full report saved to: {report_file}")
    
    def get_test_summary(self):
        """Get test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('passed', False))
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
        }
    
    def print_test_summary(self):
        """Print test summary"""
        summary = self.get_test_summary()
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']}")
        
        if summary['failed_tests'] > 0:
            print("\n❌ Some tests failed. Check logs for details.")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
    
    def cleanup(self):
        """Cleanup test environment"""
        if self.server_process:
            print("🧹 Stopping test server...")
            self.server_process.terminate()
            self.server_process.wait()
    
    @contextmanager
    def signal_handler(self):
        """Handle interruption signals"""
        def handler(signum, frame):
            print("\n⚠️  Received interrupt signal, cleaning up...")
            self.cleanup()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Comprehensive Web UI Test Runner')
    
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--frontend', action='store_true', help='Run frontend tests')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--security', action='store_true', help='Run security tests')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--headed', action='store_true', help='Run E2E tests in headed mode')
    
    args = parser.parse_args()
    
    # Default to all tests if no specific tests specified
    if not any([args.integration, args.frontend, args.e2e, args.performance, args.security]):
        args.all = True
    
    runner = WebUITestRunner(args)
    
    with runner.signal_handler():
        runner.run_all_tests()


if __name__ == "__main__":
    main()