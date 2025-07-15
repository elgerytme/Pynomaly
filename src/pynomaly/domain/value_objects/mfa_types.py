"""MFA types and value objects for domain layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MFAMethodType(Enum):
    """MFA method types."""

    TOTP = "totp"
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODES = "backup_codes"
    HARDWARE_TOKEN = "hardware_token"


class MFAMethodStatus(Enum):
    """MFA method status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    DISABLED = "disabled"


@dataclass(frozen=True)
class TOTPSetupResponse:
    """TOTP setup response value object."""

    secret: str
    qr_code_url: str
    manual_entry_key: str
    backup_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MFAMethodDTO:
    """MFA method data transfer object."""

    id: str
    method_type: MFAMethodType
    status: MFAMethodStatus
    display_name: str
    created_at: datetime
    last_used: datetime | None = None
    is_primary: bool = False
    backup_codes_remaining: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MFADeviceDTO:
    """MFA device data transfer object."""

    device_id: str
    device_name: str
    device_type: str
    last_used: datetime
    created_at: datetime
    is_trusted: bool = False
    user_agent: str | None = None
    ip_address: str | None = None


@dataclass(frozen=True)
class MFAStatisticsDTO:
    """MFA statistics data transfer object."""

    total_users: int
    mfa_enabled_users: int
    mfa_adoption_rate: float
    method_usage: dict[str, int] = field(default_factory=dict)
    recent_authentications: int = 0
    failed_attempts_24h: int = 0
