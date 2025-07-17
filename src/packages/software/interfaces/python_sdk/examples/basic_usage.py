"""
Basic Software SDK Usage Example

Demonstrates basic usage patterns for the generic software SDK.
"""

from typing import Dict, Any
import json

class SoftwareClient:
    """Generic software client for API interactions"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    def connect(self) -> bool:
        """Connect to the software service"""
        print(f"Connecting to {self.base_url}")
        # Placeholder implementation
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "status": "operational",
            "version": "1.0.0",
            "uptime": "24h"
        }
    
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the service"""
        print(f"Processing data: {data}")
        return {
            "processed": True,
            "result": "success",
            "data": data
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get service configuration"""
        return {
            "environment": "production",
            "debug": False,
            "timeout": 30
        }

def main():
    """Main example function"""
    print("Software SDK Basic Usage Example")
    print("=" * 40)
    
    # Initialize client
    client = SoftwareClient(
        base_url="https://api.software.example.com",
        api_key="your-api-key-here"
    )
    
    # Connect to service
    if client.connect():
        print("✅ Connected successfully")
    else:
        print("❌ Connection failed")
        return
    
    # Get status
    status = client.get_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Process some data
    sample_data = {
        "input": "sample data",
        "parameters": {
            "format": "json",
            "validate": True
        }
    }
    
    result = client.process_data(sample_data)
    print(f"Processing result: {json.dumps(result, indent=2)}")
    
    # Get configuration
    config = client.get_configuration()
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    print("\n✅ Example completed successfully")

if __name__ == "__main__":
    main()