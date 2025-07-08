#!/usr/bin/env python3
"""Test script to simulate security scan with proper exit codes"""

import sys
import subprocess
import json
import os

def run_security_scan():
    """Run security scan tools and check for high severity issues"""
    
    print("🔒 Running comprehensive security scan...")
    
    # Create artifacts directory
    os.makedirs('artifacts/security', exist_ok=True)
    
    # Run bandit security scan
    print("1️⃣ Running bandit security scan...")
    bandit_result = subprocess.run([
        'bandit', '-r', 'src/', 
        '-f', 'json', 
        '-o', 'artifacts/security/bandit_results.json',
        '-ll'
    ], capture_output=True, text=True)
    
    # Check bandit results for high severity issues
    if os.path.exists('artifacts/security/bandit_results.json'):
        try:
            with open('artifacts/security/bandit_results.json', 'r') as f:
                bandit_data = json.load(f)
            
            high_severity_count = bandit_data.get('metrics', {}).get('_totals', {}).get('SEVERITY.HIGH', 0)
            print(f"   - Found {high_severity_count} high severity issues")
            
            if high_severity_count > 0:
                print("❌ High severity security issues found!")
                return 1
                
        except Exception as e:
            print(f"   - Error reading bandit results: {e}")
            return 1
    
    # Run safety scan
    print("2️⃣ Running safety vulnerability scan...")
    safety_result = subprocess.run([
        'safety', 'check', 
        '--json', 
        '--output', 'artifacts/security/safety_results.json',
        '--continue-on-error'
    ], capture_output=True, text=True)
    
    # Run pip-audit scan
    print("3️⃣ Running pip-audit scan...")
    pip_audit_result = subprocess.run([
        'pip-audit', 
        '--format=json', 
        '--output=artifacts/security/pip_audit_results.json'
    ], capture_output=True, text=True)
    
    print("✅ Security scan completed!")
    print(f"📄 Results saved to artifacts/security/")
    
    # Check if we found any high severity issues
    if bandit_result.returncode != 0:
        print("🚨 Security scan failed due to high severity issues")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = run_security_scan()
    sys.exit(exit_code)
