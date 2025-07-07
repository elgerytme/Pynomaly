"""Service for managing rate limits."""

from typing import Dict, List, Optional
from pydantic import BaseModel


class RateLimitRule(BaseModel):
    """Rate limit rule model."""
    id: str
    path_regex: str
    limit: int
    period_seconds: int
    description: Optional[str] = None
    is_active: bool = True


class RateLimitService:
    """Service for managing rate limits."""

    def __init__(self):
        self._rules = {}

    def create_rule(
        self,
        path_regex: str,
        limit: int,
        period_seconds: int,
        description: Optional[str] = None
    ) -> RateLimitRule:
        """Create a new rate limit rule."""
        rule_id = f"rl_{len(self._rules) + 1}"
        rule = RateLimitRule(
            id=rule_id,
            path_regex=path_regex,
            limit=limit,
            period_seconds=period_seconds,
            description=description,
            is_active=True,
        )
        self._rules[rule_id] = rule
        return rule

    def get_rule(self, rule_id: str) -> Optional[RateLimitRule]:
        """Get a rate limit rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(self) -> List[RateLimitRule]:
        """List all rate limit rules."""
        return list(self._rules.values())

    def update_rule(
        self,
        rule_id: str,
        path_regex: Optional[str] = None,
        limit: Optional[int] = None,
        period_seconds: Optional[int] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[RateLimitRule]:
        """Update an existing rate limit rule."""
        rule = self._rules.get(rule_id)
        if not rule:
            return None

        if path_regex is not None:
            rule.path_regex = path_regex
        if limit is not None:
            rule.limit = limit
        if period_seconds is not None:
            rule.period_seconds = period_seconds
        if description is not None:
            rule.description = description
        if is_active is not None:
            rule.is_active = is_active
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rate limit rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

