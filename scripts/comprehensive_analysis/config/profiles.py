"""Analysis profiles for different use cases."""

from typing import Dict, Any

# Strict profile - Maximum analysis depth and requirements
STRICT_PROFILE: Dict[str, Any] = {
    "type_checking": {
        "strict_mode": True,
        "require_type_annotations": True,
        "check_untyped_defs": True,
        "disallow_any_generics": True,
        "disallow_incomplete_defs": True,
        "warn_unused_ignores": True,
        "warn_redundant_casts": True,
        "warn_unreachable": True,
    },
    "security": {
        "level": "high",
        "confidence_threshold": 90,
        "fail_on_vulnerability": True,
    },
    "performance": {
        "check_complexity": True,
        "max_complexity": 8,
        "check_memory_usage": True,
        "detect_antipatterns": True,
    },
    "documentation": {
        "require_docstrings": True,
        "min_coverage": 90,
        "check_examples": True,
    },
    "tools": {
        "mypy": {"strict": True},
        "ruff": {"select": ["ALL"], "ignore": []},
        "bandit": {"confidence_level": "high"},
    }
}

# Balanced profile - Practical analysis for most projects
BALANCED_PROFILE: Dict[str, Any] = {
    "type_checking": {
        "strict_mode": False,
        "require_type_annotations": False,
        "check_untyped_defs": True,
        "warn_unused_ignores": False,
    },
    "security": {
        "level": "medium",
        "confidence_threshold": 70,
        "fail_on_vulnerability": False,
    },
    "performance": {
        "check_complexity": True,
        "max_complexity": 12,
        "check_memory_usage": False,
        "detect_antipatterns": True,
    },
    "documentation": {
        "require_docstrings": False,
        "min_coverage": 60,
        "check_examples": False,
    },
    "tools": {
        "mypy": {"strict": False},
        "ruff": {"select": ["E", "F", "W", "C", "N"], "ignore": ["E501"]},
        "bandit": {"confidence_level": "medium"},
    }
}

# Permissive profile - Minimal analysis for legacy code
PERMISSIVE_PROFILE: Dict[str, Any] = {
    "type_checking": {
        "strict_mode": False,
        "require_type_annotations": False,
        "check_untyped_defs": False,
        "ignore_errors": True,
    },
    "security": {
        "level": "low",
        "confidence_threshold": 50,
        "fail_on_vulnerability": False,
    },
    "performance": {
        "check_complexity": False,
        "max_complexity": 20,
        "check_memory_usage": False,
        "detect_antipatterns": False,
    },
    "documentation": {
        "require_docstrings": False,
        "min_coverage": 30,
        "check_examples": False,
    },
    "tools": {
        "mypy": {"strict": False, "ignore_errors": True},
        "ruff": {"select": ["E", "F"], "ignore": ["E501", "W503"]},
        "bandit": {"confidence_level": "low"},
    }
}

ANALYSIS_PROFILES = {
    "strict": STRICT_PROFILE,
    "balanced": BALANCED_PROFILE,
    "permissive": PERMISSIVE_PROFILE,
}