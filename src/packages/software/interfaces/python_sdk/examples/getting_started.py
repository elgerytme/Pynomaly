"""
Getting Started with Software SDK

A comprehensive guide to getting started with the generic software SDK.
"""

import os
import json
from typing import Dict, Any, Optional

class SoftwareSDK:
    """Main SDK class for software interactions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.authenticated = False
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "api_url": os.getenv("SOFTWARE_API_URL", "https://api.software.example.com"),
            "timeout": int(os.getenv("SOFTWARE_TIMEOUT", "30")),
            "retries": int(os.getenv("SOFTWARE_RETRIES", "3")),
            "debug": os.getenv("SOFTWARE_DEBUG", "false").lower() == "true"
        }
    
    def authenticate(self, api_key: str) -> bool:
        """Authenticate with the service"""
        if not api_key:
            print("❌ API key is required")
            return False
        
        print("🔐 Authenticating...")
        # Placeholder authentication logic
        self.authenticated = True
        print("✅ Authentication successful")
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        if not self.authenticated:
            return {"error": "Not authenticated"}
        
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0.0"
        }
    
    def submit_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a processing job"""
        if not self.authenticated:
            return {"error": "Not authenticated"}
        
        print(f"📤 Submitting job: {job_data.get('name', 'unnamed')}")
        
        # Placeholder job submission logic
        return {
            "job_id": "job_12345",
            "status": "submitted",
            "estimated_time": "5 minutes"
        }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if not self.authenticated:
            return {"error": "Not authenticated"}
        
        # Placeholder job status logic
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "result": "success"
        }

def step_by_step_example():
    """Step-by-step usage example"""
    print("Getting Started with Software SDK")
    print("=" * 50)
    
    # Step 1: Initialize SDK
    print("\n🚀 Step 1: Initialize SDK")
    sdk = SoftwareSDK()
    print(f"Configuration: {json.dumps(sdk.config, indent=2)}")
    
    # Step 2: Authenticate
    print("\n🔐 Step 2: Authenticate")
    api_key = "your-api-key-here"  # Replace with actual API key
    if not sdk.authenticate(api_key):
        print("❌ Failed to authenticate")
        return
    
    # Step 3: Health check
    print("\n🏥 Step 3: Health check")
    health = sdk.health_check()
    print(f"Health status: {json.dumps(health, indent=2)}")
    
    # Step 4: Submit a job
    print("\n📤 Step 4: Submit job")
    job_data = {
        "name": "sample_job",
        "type": "processing",
        "parameters": {
            "input_format": "json",
            "output_format": "json",
            "validate": True
        }
    }
    
    job_response = sdk.submit_job(job_data)
    print(f"Job response: {json.dumps(job_response, indent=2)}")
    
    # Step 5: Check job status
    print("\n📊 Step 5: Check job status")
    if "job_id" in job_response:
        status = sdk.get_job_status(job_response["job_id"])
        print(f"Job status: {json.dumps(status, indent=2)}")
    
    print("\n✅ Getting started example completed!")

def configuration_example():
    """Example of different configuration options"""
    print("\n" + "=" * 50)
    print("Configuration Examples")
    print("=" * 50)
    
    # Custom configuration
    custom_config = {
        "api_url": "https://custom.api.com",
        "timeout": 60,
        "retries": 5,
        "debug": True
    }
    
    print("\n🔧 Custom Configuration:")
    sdk_custom = SoftwareSDK(custom_config)
    print(f"Custom config: {json.dumps(sdk_custom.config, indent=2)}")
    
    # Environment-based configuration
    print("\n🌍 Environment-based Configuration:")
    print("Set these environment variables:")
    print("  SOFTWARE_API_URL=https://api.example.com")
    print("  SOFTWARE_TIMEOUT=45")
    print("  SOFTWARE_RETRIES=3")
    print("  SOFTWARE_DEBUG=true")
    
    sdk_env = SoftwareSDK()
    print(f"Environment config: {json.dumps(sdk_env.config, indent=2)}")

def error_handling_example():
    """Example of error handling"""
    print("\n" + "=" * 50)
    print("Error Handling Examples")
    print("=" * 50)
    
    sdk = SoftwareSDK()
    
    # Example 1: Authentication error
    print("\n❌ Example 1: Authentication error")
    try:
        if not sdk.authenticate(""):
            print("Handle authentication failure")
    except Exception as e:
        print(f"Authentication exception: {e}")
    
    # Example 2: Unauthenticated request
    print("\n❌ Example 2: Unauthenticated request")
    health = sdk.health_check()
    if "error" in health:
        print(f"Error: {health['error']}")
    
    # Example 3: Invalid job data
    print("\n❌ Example 3: Invalid job data")
    invalid_job = {}
    result = sdk.submit_job(invalid_job)
    if "error" in result:
        print(f"Job submission error: {result['error']}")

def main():
    """Main function running all examples"""
    # Run step-by-step example
    step_by_step_example()
    
    # Run configuration example
    configuration_example()
    
    # Run error handling example
    error_handling_example()
    
    print("\n🎉 All examples completed successfully!")
    print("\nNext steps:")
    print("1. Get your API key from the service dashboard")
    print("2. Set up environment variables")
    print("3. Start building your application")
    print("4. Check the documentation for advanced features")

if __name__ == "__main__":
    main()