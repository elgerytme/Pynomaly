#!/usr/bin/env python3
"""
Deployment Verification Script for Pynomaly Multi-Environment Deployment

This script performs comprehensive verification of Pynomaly deployment across
different environments (development, staging, production).
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import click
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class DeploymentVerifier:
    """Comprehensive deployment verification system."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.verification_results = {
            "environment": environment,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_status": "pass",
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0
            }
        }
        
        # Environment-specific configuration
        self.config = self._load_environment_config()
    
    def _load_environment_config(self) -> Dict:
        """Load environment-specific configuration."""
        config_map = {
            "development": {
                "api_url": "http://localhost:8000",
                "db_host": "localhost",
                "redis_host": "localhost",
                "expected_services": ["api", "db", "redis"],
                "health_threshold": 0.7,
                "response_timeout": 30,
            },
            "staging": {
                "api_url": "https://staging-api.monorepo.com",
                "db_host": "staging-db.monorepo.com",
                "redis_host": "staging-redis.monorepo.com",
                "expected_services": ["api", "db", "redis", "monitoring"],
                "health_threshold": 0.8,
                "response_timeout": 15,
            },
            "production": {
                "api_url": "https://api.monorepo.com",
                "db_host": "prod-db.monorepo.com",
                "redis_host": "prod-redis.monorepo.com",
                "expected_services": ["api", "db", "redis", "monitoring", "backup"],
                "health_threshold": 0.9,
                "response_timeout": 10,
            },
        }
        
        return config_map.get(self.environment, config_map["development"])
    
    async def verify_api_health(self) -> Dict:
        """Verify API service health."""
        check_name = "api_health"
        console.print(f"[blue]Checking API health at {self.config['api_url']}...[/blue]")
        
        try:
            # Test basic health endpoint
            response = requests.get(
                f"{self.config['api_url']}/health",
                timeout=self.config['response_timeout']
            )
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check health score
                health_score = health_data.get("health_score", 0)
                if health_score >= self.config['health_threshold']:
                    result = {
                        "status": "pass",
                        "message": f"API health check passed (score: {health_score:.2f})",
                        "details": {
                            "health_score": health_score,
                            "response_time": response.elapsed.total_seconds(),
                            "timestamp": health_data.get("timestamp"),
                            "version": health_data.get("version")
                        }
                    }
                else:
                    result = {
                        "status": "fail",
                        "message": f"API health score below threshold ({health_score:.2f} < {self.config['health_threshold']})",
                        "details": {"health_score": health_score}
                    }
            else:
                result = {
                    "status": "fail",
                    "message": f"API health check failed (status: {response.status_code})",
                    "details": {"status_code": response.status_code}
                }
        
        except requests.exceptions.RequestException as e:
            result = {
                "status": "fail",
                "message": f"API health check failed: {str(e)}",
                "details": {"error": str(e)}
            }
        
        self.verification_results["checks"][check_name] = result
        return result
    
    async def verify_database_connectivity(self) -> Dict:
        """Verify database connectivity."""
        check_name = "database_connectivity"
        console.print(f"[blue]Checking database connectivity to {self.config['db_host']}...[/blue]")
        
        try:
            # Test database connectivity via API
            response = requests.get(
                f"{self.config['api_url']}/health/database",
                timeout=self.config['response_timeout']
            )
            
            if response.status_code == 200:
                db_health = response.json()
                
                if db_health.get("connected", False):
                    result = {
                        "status": "pass",
                        "message": "Database connectivity verified",
                        "details": {
                            "connection_pool_size": db_health.get("pool_size", 0),
                            "active_connections": db_health.get("active_connections", 0),
                            "response_time": response.elapsed.total_seconds()
                        }
                    }
                else:
                    result = {
                        "status": "fail",
                        "message": "Database connection failed",
                        "details": db_health
                    }
            else:
                result = {
                    "status": "fail",
                    "message": f"Database health check failed (status: {response.status_code})",
                    "details": {"status_code": response.status_code}
                }
        
        except requests.exceptions.RequestException as e:
            result = {
                "status": "fail",
                "message": f"Database connectivity check failed: {str(e)}",
                "details": {"error": str(e)}
            }
        
        self.verification_results["checks"][check_name] = result
        return result
    
    async def verify_redis_connectivity(self) -> Dict:
        """Verify Redis connectivity."""
        check_name = "redis_connectivity"
        console.print(f"[blue]Checking Redis connectivity to {self.config['redis_host']}...[/blue]")
        
        try:
            # Test Redis connectivity via API
            response = requests.get(
                f"{self.config['api_url']}/health/redis",
                timeout=self.config['response_timeout']
            )
            
            if response.status_code == 200:
                redis_health = response.json()
                
                if redis_health.get("connected", False):
                    result = {
                        "status": "pass",
                        "message": "Redis connectivity verified",
                        "details": {
                            "ping_response": redis_health.get("ping_response"),
                            "memory_usage": redis_health.get("memory_usage"),
                            "response_time": response.elapsed.total_seconds()
                        }
                    }
                else:
                    result = {
                        "status": "fail",
                        "message": "Redis connection failed",
                        "details": redis_health
                    }
            else:
                result = {
                    "status": "fail",
                    "message": f"Redis health check failed (status: {response.status_code})",
                    "details": {"status_code": response.status_code}
                }
        
        except requests.exceptions.RequestException as e:
            result = {
                "status": "fail",
                "message": f"Redis connectivity check failed: {str(e)}",
                "details": {"error": str(e)}
            }
        
        self.verification_results["checks"][check_name] = result
        return result
    
    async def verify_anomaly_detection_functionality(self) -> Dict:
        """Verify core anomaly detection functionality."""
        check_name = "anomaly_detection"
        console.print("[blue]Testing anomaly detection functionality...[/blue]")
        
        try:
            # Test anomaly detection endpoint with sample data
            test_data = {
                "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                "algorithm": "IsolationForest",
                "contamination": 0.1
            }
            
            response = requests.post(
                f"{self.config['api_url']}/api/v1/detect",
                json=test_data,
                timeout=self.config['response_timeout']
            )
            
            if response.status_code == 200:
                detection_result = response.json()
                
                if "anomalies" in detection_result:
                    result = {
                        "status": "pass",
                        "message": "Anomaly detection functionality verified",
                        "details": {
                            "anomalies_detected": len(detection_result["anomalies"]),
                            "processing_time": detection_result.get("processing_time", 0),
                            "algorithm_used": detection_result.get("algorithm"),
                            "response_time": response.elapsed.total_seconds()
                        }
                    }
                else:
                    result = {
                        "status": "fail",
                        "message": "Anomaly detection response missing expected fields",
                        "details": detection_result
                    }
            else:
                result = {
                    "status": "fail",
                    "message": f"Anomaly detection test failed (status: {response.status_code})",
                    "details": {"status_code": response.status_code}
                }
        
        except requests.exceptions.RequestException as e:
            result = {
                "status": "fail",
                "message": f"Anomaly detection test failed: {str(e)}",
                "details": {"error": str(e)}
            }
        
        self.verification_results["checks"][check_name] = result
        return result
    
    async def verify_performance_metrics(self) -> Dict:
        """Verify performance metrics."""
        check_name = "performance_metrics"
        console.print("[blue]Checking performance metrics...[/blue]")
        
        try:
            # Test performance metrics endpoint
            response = requests.get(
                f"{self.config['api_url']}/metrics",
                timeout=self.config['response_timeout']
            )
            
            if response.status_code == 200:
                # Check if metrics are in expected format
                metrics_text = response.text
                
                # Look for key metrics
                expected_metrics = [
                    "pynomaly_requests_total",
                    "pynomaly_request_duration_seconds",
                    "pynomaly_anomalies_detected_total",
                    "pynomaly_database_connections_active"
                ]
                
                found_metrics = [m for m in expected_metrics if m in metrics_text]
                
                if len(found_metrics) >= len(expected_metrics) * 0.8:  # 80% threshold
                    result = {
                        "status": "pass",
                        "message": "Performance metrics verified",
                        "details": {
                            "metrics_found": len(found_metrics),
                            "expected_metrics": len(expected_metrics),
                            "response_time": response.elapsed.total_seconds(),
                            "metrics_size": len(metrics_text)
                        }
                    }
                else:
                    result = {
                        "status": "warning",
                        "message": f"Some performance metrics missing ({len(found_metrics)}/{len(expected_metrics)})",
                        "details": {
                            "missing_metrics": [m for m in expected_metrics if m not in metrics_text]
                        }
                    }
            else:
                result = {
                    "status": "fail",
                    "message": f"Performance metrics check failed (status: {response.status_code})",
                    "details": {"status_code": response.status_code}
                }
        
        except requests.exceptions.RequestException as e:
            result = {
                "status": "fail",
                "message": f"Performance metrics check failed: {str(e)}",
                "details": {"error": str(e)}
            }
        
        self.verification_results["checks"][check_name] = result
        return result
    
    async def verify_security_configurations(self) -> Dict:
        """Verify security configurations."""
        check_name = "security_configurations"
        console.print("[blue]Checking security configurations...[/blue]")
        
        security_checks = []
        
        try:
            # Check HTTPS configuration
            if self.config['api_url'].startswith('https://'):
                security_checks.append(("HTTPS", "enabled"))
            else:
                security_checks.append(("HTTPS", "disabled"))
            
            # Check security headers
            response = requests.get(
                f"{self.config['api_url']}/health",
                timeout=self.config['response_timeout']
            )
            
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Content-Security-Policy"
            ]
            
            present_headers = [h for h in security_headers if h in response.headers]
            security_checks.append(("Security Headers", f"{len(present_headers)}/{len(security_headers)}"))
            
            # Check authentication
            auth_response = requests.get(
                f"{self.config['api_url']}/api/v1/auth/verify",
                timeout=self.config['response_timeout']
            )
            
            if auth_response.status_code in [401, 403]:  # Expecting authentication required
                security_checks.append(("Authentication", "required"))
            else:
                security_checks.append(("Authentication", "optional"))
            
            # Overall security assessment
            security_score = 0
            if any("HTTPS" in check and "enabled" in check for check in security_checks):
                security_score += 0.4
            if any("Security Headers" in check and "4/4" in check for check in security_checks):
                security_score += 0.3
            elif any("Security Headers" in check and "3/4" in check for check in security_checks):
                security_score += 0.2
            if any("Authentication" in check and "required" in check for check in security_checks):
                security_score += 0.3
            
            if security_score >= 0.8:
                result = {
                    "status": "pass",
                    "message": "Security configurations verified",
                    "details": {
                        "security_score": security_score,
                        "checks": security_checks
                    }
                }
            elif security_score >= 0.6:
                result = {
                    "status": "warning",
                    "message": f"Security configurations partially configured (score: {security_score:.2f})",
                    "details": {
                        "security_score": security_score,
                        "checks": security_checks
                    }
                }
            else:
                result = {
                    "status": "fail",
                    "message": f"Security configurations insufficient (score: {security_score:.2f})",
                    "details": {
                        "security_score": security_score,
                        "checks": security_checks
                    }
                }
        
        except requests.exceptions.RequestException as e:
            result = {
                "status": "fail",
                "message": f"Security configurations check failed: {str(e)}",
                "details": {"error": str(e)}
            }
        
        self.verification_results["checks"][check_name] = result
        return result
    
    async def run_all_verifications(self) -> Dict:
        """Run all verification checks."""
        console.print(f"[bold blue]Starting deployment verification for {self.environment} environment[/bold blue]")
        console.print("=" * 80)
        
        # List of verification functions
        verification_functions = [
            self.verify_api_health,
            self.verify_database_connectivity,
            self.verify_redis_connectivity,
            self.verify_anomaly_detection_functionality,
            self.verify_performance_metrics,
            self.verify_security_configurations,
        ]
        
        # Run all verifications
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            for verify_func in verification_functions:
                task = progress.add_task(f"Running {verify_func.__name__}...", total=None)
                
                try:
                    await verify_func()
                    progress.update(task, description=f"Completed {verify_func.__name__}")
                except Exception as e:
                    logger.error(f"Error in {verify_func.__name__}: {e}")
                    progress.update(task, description=f"Failed {verify_func.__name__}")
        
        # Calculate summary
        self._calculate_summary()
        
        return self.verification_results
    
    def _calculate_summary(self):
        """Calculate verification summary."""
        total_checks = len(self.verification_results["checks"])
        passed_checks = sum(1 for check in self.verification_results["checks"].values() if check["status"] == "pass")
        failed_checks = sum(1 for check in self.verification_results["checks"].values() if check["status"] == "fail")
        warnings = sum(1 for check in self.verification_results["checks"].values() if check["status"] == "warning")
        
        self.verification_results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "warnings": warnings,
            "success_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }
        
        # Overall status
        if failed_checks == 0:
            self.verification_results["overall_status"] = "pass"
        elif failed_checks <= total_checks * 0.2:  # 20% tolerance
            self.verification_results["overall_status"] = "warning"
        else:
            self.verification_results["overall_status"] = "fail"
    
    def display_results(self):
        """Display verification results."""
        console.print("\n" + "=" * 80)
        console.print(f"[bold blue]Deployment Verification Results - {self.environment.upper()}[/bold blue]")
        console.print("=" * 80)
        
        # Summary
        summary = self.verification_results["summary"]
        status_color = {
            "pass": "green",
            "warning": "yellow",
            "fail": "red"
        }.get(self.verification_results["overall_status"], "white")
        
        console.print(f"\n[bold {status_color}]Overall Status: {self.verification_results['overall_status'].upper()}[/bold {status_color}]")
        console.print(f"[bold]Success Rate: {summary['success_rate']:.1f}%[/bold]")
        console.print(f"Total Checks: {summary['total_checks']}")
        console.print(f"Passed: [green]{summary['passed_checks']}[/green]")
        console.print(f"Failed: [red]{summary['failed_checks']}[/red]")
        console.print(f"Warnings: [yellow]{summary['warnings']}[/yellow]")
        
        # Detailed results table
        table = Table(title="Detailed Verification Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Message", style="white")
        
        for check_name, check_result in self.verification_results["checks"].items():
            status_color = {
                "pass": "green",
                "warning": "yellow",
                "fail": "red"
            }.get(check_result["status"], "white")
            
            status_symbol = {
                "pass": "✓",
                "warning": "⚠",
                "fail": "✗"
            }.get(check_result["status"], "?")
            
            table.add_row(
                check_name.replace("_", " ").title(),
                f"[{status_color}]{status_symbol} {check_result['status'].upper()}[/{status_color}]",
                check_result["message"]
            )
        
        console.print("\n")
        console.print(table)
        
        # Show recommendations if any issues
        if self.verification_results["overall_status"] != "pass":
            console.print("\n[yellow]🔧 Recommendations:[/yellow]")
            
            for check_name, check_result in self.verification_results["checks"].items():
                if check_result["status"] in ["fail", "warning"]:
                    console.print(f"  • {check_name}: {check_result['message']}")
        
        console.print(f"\n[dim]Verification completed at {self.verification_results['timestamp']}[/dim]")


@click.command()
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["development", "staging", "production"]),
    default="production",
    help="Environment to verify"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for results (JSON format)"
)
@click.option(
    "--timeout",
    "-t",
    type=int,
    default=30,
    help="Request timeout in seconds"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
async def main(environment: str, output: Optional[str], timeout: int, verbose: bool):
    """
    Verify Pynomaly deployment across environments.
    
    This script performs comprehensive verification of the Pynomaly deployment
    including API health, database connectivity, functionality tests, and
    security configurations.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize verifier
    verifier = DeploymentVerifier(environment=environment)
    
    # Override timeout if provided
    verifier.config['response_timeout'] = timeout
    
    try:
        # Run all verifications
        results = await verifier.run_all_verifications()
        
        # Display results
        verifier.display_results()
        
        # Save results to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Results saved to {output}[/green]")
        
        # Exit with appropriate code
        if results["overall_status"] == "pass":
            console.print("\n[green]✅ All verification checks passed![/green]")
            sys.exit(0)
        elif results["overall_status"] == "warning":
            console.print("\n[yellow]⚠️  Verification completed with warnings[/yellow]")
            sys.exit(1)
        else:
            console.print("\n[red]❌ Verification failed[/red]")
            sys.exit(2)
    
    except Exception as e:
        console.print(f"\n[red]💥 Verification error: {e}[/red]")
        logger.exception("Verification failed with exception")
        sys.exit(3)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())