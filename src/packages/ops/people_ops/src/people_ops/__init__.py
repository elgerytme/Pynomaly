"""
People Operations Package - Human Resources and Team Management

This package provides people operations capabilities including:
- Employee management
- Team organization
- Performance tracking
- Onboarding processes
- Skills assessment
- Training management
"""

__version__ = "0.1.0"
__author__ = "Pynomaly Team"
__email__ = "support@pynomaly.com"

# Core imports
from .core import (
    Employee,
    Team,
    Role,
    Skill,
    Performance,
)

# Service imports
from .services import (
    EmployeeService,
    TeamService,
    PerformanceService,
    OnboardingService,
    TrainingService,
)

__all__ = [
    # Core
    "Employee",
    "Team",
    "Role",
    "Skill",
    "Performance",
    
    # Services
    "EmployeeService",
    "TeamService",
    "PerformanceService",
    "OnboardingService",
    "TrainingService",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]