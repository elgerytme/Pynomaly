"""Configuration management for advanced threat detection systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field


class ThreatDetectionConfig(BaseModel):
    """Configuration for threat detection systems."""
    
    # Global settings
    enabled: bool = Field(default=True, description="Enable threat detection")
    strict_mode: bool = Field(default=False, description="Enable strict threat detection")
    
    # Behavioral analysis settings
    behavior_analysis_enabled: bool = Field(default=True, description="Enable behavioral analysis")
    learning_period_days: int = Field(default=14, ge=1, le=90, description="Learning period in days")
    anomaly_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Anomaly detection threshold")
    min_samples_for_profile: int = Field(default=50, ge=10, le=1000, description="Minimum samples to build profile")
    
    # Threat intelligence settings
    threat_intelligence_enabled: bool = Field(default=True, description="Enable threat intelligence")
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0, description="Threat intelligence confidence threshold")
    update_interval_seconds: int = Field(default=3600, ge=300, le=86400, description="Feed update interval")
    
    # Data exfiltration detection
    data_exfiltration_enabled: bool = Field(default=True, description="Enable data exfiltration detection")
    size_threshold_mb: float = Field(default=100.0, ge=1.0, le=10000.0, description="Data size threshold in MB")
    time_window_seconds: int = Field(default=300, ge=60, le=3600, description="Detection time window")
    request_count_threshold: int = Field(default=50, ge=5, le=1000, description="Request count threshold")
    
    # Alert settings
    auto_mitigation_enabled: bool = Field(default=True, description="Enable automatic mitigation")
    alert_cooldown_seconds: int = Field(default=300, ge=60, le=3600, description="Alert cooldown period")
    max_alerts_per_user: int = Field(default=10, ge=1, le=100, description="Maximum alerts per user")
    
    # IP and network settings
    trusted_networks: List[str] = Field(default_factory=list, description="Trusted network CIDRs")
    blocked_ips: Set[str] = Field(default_factory=set, description="Manually blocked IP addresses")
    whitelist_ips: Set[str] = Field(default_factory=set, description="Whitelisted IP addresses")
    
    # User agent settings
    suspicious_user_agents: List[str] = Field(
        default_factory=lambda: [
            "sqlmap", "nikto", "nmap", "masscan", "w3af", "burp", "dirbuster",
            "gobuster", "ffuf", "dirb", "wpscan", "nuclei", "zap"
        ],
        description="Known suspicious user agents"
    )
    
    # Performance settings
    max_history_size: int = Field(default=10000, ge=1000, le=100000, description="Maximum alert history size")
    cleanup_interval_seconds: int = Field(default=3600, ge=300, le=86400, description="Cleanup interval")
    max_memory_usage_mb: int = Field(default=500, ge=100, le=2000, description="Maximum memory usage in MB")


@dataclass
class ThreatDetectorSettings:
    """Settings for individual threat detectors."""
    
    name: str
    enabled: bool = True
    config: Dict[str, any] = field(default_factory=dict)
    priority: int = 50  # 1-100, higher = more important
    auto_mitigation: bool = True
    alert_threshold: float = 0.7


class ThreatDetectionManager:
    """Manager for threat detection configuration and lifecycle."""
    
    def __init__(self, config: Optional[ThreatDetectionConfig] = None):
        """Initialize threat detection manager.
        
        Args:
            config: Threat detection configuration
        """
        self.config = config or ThreatDetectionConfig()
        self.detector_settings: Dict[str, ThreatDetectorSettings] = {}
        self._initialize_default_settings()
    
    def _initialize_default_settings(self) -> None:
        """Initialize default detector settings."""
        # Advanced behavior analyzer
        self.detector_settings["advanced_behavior"] = ThreatDetectorSettings(
            name="advanced_behavior",
            enabled=self.config.behavior_analysis_enabled,
            config={
                "learning_period_days": self.config.learning_period_days,
                "anomaly_threshold": self.config.anomaly_threshold,
                "min_samples_for_profile": self.config.min_samples_for_profile
            },
            priority=80,
            auto_mitigation=False,  # Behavioral anomalies should be reviewed
            alert_threshold=self.config.anomaly_threshold
        )
        
        # Threat intelligence detector
        self.detector_settings["threat_intelligence"] = ThreatDetectorSettings(
            name="threat_intelligence",
            enabled=self.config.threat_intelligence_enabled,
            config={
                "confidence_threshold": self.config.confidence_threshold,
                "update_interval": self.config.update_interval_seconds
            },
            priority=95,
            auto_mitigation=True,  # Known threats should be auto-blocked
            alert_threshold=self.config.confidence_threshold
        )
        
        # Data exfiltration detector
        self.detector_settings["data_exfiltration"] = ThreatDetectorSettings(
            name="data_exfiltration",
            enabled=self.config.data_exfiltration_enabled,
            config={
                "size_threshold_mb": self.config.size_threshold_mb,
                "time_window_seconds": self.config.time_window_seconds,
                "request_count_threshold": self.config.request_count_threshold
            },
            priority=90,
            auto_mitigation=False,  # Data access might be legitimate
            alert_threshold=0.8
        )
        
        # Legacy detectors with updated settings
        self.detector_settings["brute_force"] = ThreatDetectorSettings(
            name="brute_force",
            enabled=True,
            config={
                "max_attempts": 5,
                "time_window": 300,
                "block_duration": 3600
            },
            priority=85,
            auto_mitigation=True,
            alert_threshold=1.0  # Always alert on brute force
        )
        
        self.detector_settings["anomalous_access"] = ThreatDetectorSettings(
            name="anomalous_access",
            enabled=True,
            config={
                "learning_period_days": self.config.learning_period_days
            },
            priority=70,
            auto_mitigation=False,
            alert_threshold=0.6
        )
        
        self.detector_settings["injection_attack"] = ThreatDetectorSettings(
            name="injection_attack",
            enabled=True,
            config={},
            priority=100,  # Highest priority
            auto_mitigation=True,
            alert_threshold=1.0  # Always alert on injection
        )
    
    def get_detector_config(self, detector_name: str) -> Optional[ThreatDetectorSettings]:
        """Get configuration for a specific detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Detector settings or None if not found
        """
        return self.detector_settings.get(detector_name)
    
    def update_detector_config(self, detector_name: str, settings: ThreatDetectorSettings) -> None:
        """Update configuration for a detector.
        
        Args:
            detector_name: Name of the detector
            settings: New detector settings
        """
        self.detector_settings[detector_name] = settings
    
    def enable_detector(self, detector_name: str) -> bool:
        """Enable a threat detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            True if detector was enabled
        """
        if detector_name in self.detector_settings:
            self.detector_settings[detector_name].enabled = True
            return True
        return False
    
    def disable_detector(self, detector_name: str) -> bool:
        """Disable a threat detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            True if detector was disabled
        """
        if detector_name in self.detector_settings:
            self.detector_settings[detector_name].enabled = False
            return True
        return False
    
    def get_enabled_detectors(self) -> List[str]:
        """Get list of enabled detector names.
        
        Returns:
            List of enabled detector names
        """
        return [
            name for name, settings in self.detector_settings.items()
            if settings.enabled
        ]
    
    def is_ip_whitelisted(self, ip_address: str) -> bool:
        """Check if IP address is whitelisted.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if IP is whitelisted
        """
        return ip_address in self.config.whitelist_ips
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if IP is blocked
        """
        return ip_address in self.config.blocked_ips
    
    def add_blocked_ip(self, ip_address: str) -> None:
        """Add IP address to blocked list.
        
        Args:
            ip_address: IP address to block
        """
        self.config.blocked_ips.add(ip_address)
    
    def remove_blocked_ip(self, ip_address: str) -> None:
        """Remove IP address from blocked list.
        
        Args:
            ip_address: IP address to unblock
        """
        self.config.blocked_ips.discard(ip_address)
    
    def add_whitelist_ip(self, ip_address: str) -> None:
        """Add IP address to whitelist.
        
        Args:
            ip_address: IP address to whitelist
        """
        self.config.whitelist_ips.add(ip_address)
    
    def remove_whitelist_ip(self, ip_address: str) -> None:
        """Remove IP address from whitelist.
        
        Args:
            ip_address: IP address to remove from whitelist
        """
        self.config.whitelist_ips.discard(ip_address)
    
    def export_config(self) -> Dict[str, any]:
        """Export configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return {
            "global_config": self.config.dict(),
            "detector_settings": {
                name: {
                    "enabled": settings.enabled,
                    "config": settings.config,
                    "priority": settings.priority,
                    "auto_mitigation": settings.auto_mitigation,
                    "alert_threshold": settings.alert_threshold
                }
                for name, settings in self.detector_settings.items()
            }
        }
    
    def import_config(self, config_dict: Dict[str, any]) -> None:
        """Import configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        if "global_config" in config_dict:
            self.config = ThreatDetectionConfig(**config_dict["global_config"])
        
        if "detector_settings" in config_dict:
            for name, settings_dict in config_dict["detector_settings"].items():
                self.detector_settings[name] = ThreatDetectorSettings(
                    name=name,
                    enabled=settings_dict.get("enabled", True),
                    config=settings_dict.get("config", {}),
                    priority=settings_dict.get("priority", 50),
                    auto_mitigation=settings_dict.get("auto_mitigation", True),
                    alert_threshold=settings_dict.get("alert_threshold", 0.7)
                )


# Global configuration manager instance
_threat_detection_manager: Optional[ThreatDetectionManager] = None


def get_threat_detection_manager() -> ThreatDetectionManager:
    """Get global threat detection manager instance.
    
    Returns:
        Threat detection manager
    """
    global _threat_detection_manager
    if _threat_detection_manager is None:
        _threat_detection_manager = ThreatDetectionManager()
    return _threat_detection_manager


def init_threat_detection_manager(config: Optional[ThreatDetectionConfig] = None) -> ThreatDetectionManager:
    """Initialize global threat detection manager.
    
    Args:
        config: Threat detection configuration
        
    Returns:
        Threat detection manager
    """
    global _threat_detection_manager
    _threat_detection_manager = ThreatDetectionManager(config)
    return _threat_detection_manager