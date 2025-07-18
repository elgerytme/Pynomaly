"""
AWS Cloud Integration for Pynomaly Detection
===========================================

This module provides comprehensive AWS cloud integration including:
- EKS cluster management
- S3 data storage and model persistence
- CloudWatch monitoring and logging
- Parameter Store configuration management
- IAM role and policy management
- Auto-scaling and load balancing
"""

from .eks_manager import EKSManager
from .s3_storage import S3StorageManager
from .cloudwatch_monitor import CloudWatchMonitor
from .parameter_store import ParameterStoreManager
from .iam_manager import IAMManager
from .deployment_manager import AWSDeploymentManager

__all__ = [
    'EKSManager',
    'S3StorageManager', 
    'CloudWatchMonitor',
    'ParameterStoreManager',
    'IAMManager',
    'AWSDeploymentManager'
]

# AWS Integration Version
__version__ = "1.0.0"

def get_aws_integration_info():
    """Get AWS integration information."""
    return {
        "provider": "AWS",
        "version": __version__,
        "services": {
            "eks": "Elastic Kubernetes Service",
            "s3": "Simple Storage Service",
            "cloudwatch": "CloudWatch Monitoring",
            "parameter_store": "Systems Manager Parameter Store",
            "iam": "Identity and Access Management"
        },
        "regions": [
            "us-east-1", "us-west-2", "eu-west-1", 
            "ap-southeast-1", "ap-northeast-1"
        ]
    }