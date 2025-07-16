"""Backward compatibility wrapper for detection service.

This module provides a compatibility layer for tests that import the old DetectionService.
"""

from pynomaly_detection.domain.services.advanced_detection_service import AdvancedDetectionService

# Backward compatibility alias
DetectionService = AdvancedDetectionService

__all__ = ["DetectionService"]
