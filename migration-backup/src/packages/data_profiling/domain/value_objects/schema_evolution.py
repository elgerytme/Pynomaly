"""Schema evolution value objects for tracking data structure changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set
from uuid import UUID


class ChangeType(str, Enum):
    """Types of schema changes."""
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    COLUMN_TYPE_CHANGED = "column_type_changed"
    COLUMN_NULLABLE_CHANGED = "column_nullable_changed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    INDEX_ADDED = "index_added"
    INDEX_REMOVED = "index_removed"
    PRIMARY_KEY_CHANGED = "primary_key_changed"
    FOREIGN_KEY_ADDED = "foreign_key_added"
    FOREIGN_KEY_REMOVED = "foreign_key_removed"
    TABLE_ADDED = "table_added"
    TABLE_REMOVED = "table_removed"
    TABLE_RENAMED = "table_renamed"
    CARDINALITY_CHANGED = "cardinality_changed"
    DISTRIBUTION_CHANGED = "distribution_changed"
    PATTERN_CHANGED = "pattern_changed"


class ImpactLevel(str, Enum):
    """Impact level of schema changes."""
    BREAKING = "breaking"
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    COSMETIC = "cosmetic"


class CompatibilityStatus(str, Enum):
    """Compatibility status between schema versions."""
    FULLY_COMPATIBLE = "fully_compatible"
    BACKWARD_COMPATIBLE = "backward_compatible"
    FORWARD_COMPATIBLE = "forward_compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class SchemaChange:
    """Represents a single schema change."""
    change_id: str
    change_type: ChangeType
    element_name: str  # column, table, index, etc.
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    impact_level: ImpactLevel = ImpactLevel.MINOR
    description: str = ""
    affected_queries: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    automated_fix_available: bool = False
    detection_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_breaking_change(self) -> bool:
        """Check if this is a breaking change."""
        return self.impact_level == ImpactLevel.BREAKING
    
    @property
    def requires_manual_intervention(self) -> bool:
        """Check if change requires manual intervention."""
        return self.is_breaking_change and not self.automated_fix_available
    
    @classmethod
    def create_column_added(
        cls,
        change_id: str,
        column_name: str,
        column_type: str,
        nullable: bool = True,
        default_value: Any = None
    ) -> SchemaChange:
        """Create a column addition change."""
        impact = ImpactLevel.MINOR if nullable or default_value is not None else ImpactLevel.MAJOR
        
        return cls(
            change_id=change_id,
            change_type=ChangeType.COLUMN_ADDED,
            element_name=column_name,
            new_value={"type": column_type, "nullable": nullable, "default": default_value},
            impact_level=impact,
            description=f"Added column '{column_name}' of type {column_type}",
            automated_fix_available=True,
            metadata={"column_type": column_type, "nullable": nullable, "default_value": default_value}
        )
    
    @classmethod
    def create_column_removed(
        cls,
        change_id: str,
        column_name: str,
        column_type: str
    ) -> SchemaChange:
        """Create a column removal change."""
        return cls(
            change_id=change_id,
            change_type=ChangeType.COLUMN_REMOVED,
            element_name=column_name,
            old_value={"type": column_type},
            impact_level=ImpactLevel.BREAKING,
            description=f"Removed column '{column_name}' of type {column_type}",
            automated_fix_available=False,
            mitigation_strategies=[
                "Update queries to exclude removed column",
                "Consider data migration for dependent applications"
            ]
        )
    
    @classmethod
    def create_type_change(
        cls,
        change_id: str,
        column_name: str,
        old_type: str,
        new_type: str
    ) -> SchemaChange:
        """Create a column type change."""
        # Determine impact based on type compatibility
        compatible_changes = {
            ("int", "bigint"),
            ("float", "double"),
            ("varchar", "text"),
            ("char", "varchar")
        }
        
        is_compatible = (old_type, new_type) in compatible_changes
        impact = ImpactLevel.MINOR if is_compatible else ImpactLevel.MAJOR
        
        return cls(
            change_id=change_id,
            change_type=ChangeType.COLUMN_TYPE_CHANGED,
            element_name=column_name,
            old_value=old_type,
            new_value=new_type,
            impact_level=impact,
            description=f"Changed column '{column_name}' type from {old_type} to {new_type}",
            automated_fix_available=is_compatible,
            metadata={"old_type": old_type, "new_type": new_type, "compatible": is_compatible}
        )


@dataclass(frozen=True)
class SchemaVersion:
    """Represents a specific version of a schema."""
    version_id: str
    version_number: str
    schema_hash: str
    timestamp: datetime
    author: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Schema structure
    tables: List[str] = field(default_factory=list)
    columns: Dict[str, List[str]] = field(default_factory=dict)  # table -> column names
    data_types: Dict[str, str] = field(default_factory=dict)  # column -> type
    constraints: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    
    # Statistics
    total_tables: int = 0
    total_columns: int = 0
    total_rows: int = 0
    estimated_size_bytes: Optional[int] = None
    
    # Quality metrics
    schema_quality_score: float = 0.0
    data_quality_score: float = 0.0
    completeness_percentage: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_baseline(self) -> bool:
        """Check if this is a baseline version."""
        return self.version_number == "1.0.0" or "baseline" in self.tags
    
    @property
    def column_count_by_table(self) -> Dict[str, int]:
        """Get column count for each table."""
        return {table: len(columns) for table, columns in self.columns.items()}
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """Get columns for specific table."""
        return self.columns.get(table_name, [])
    
    def has_table(self, table_name: str) -> bool:
        """Check if schema contains specific table."""
        return table_name in self.tables
    
    def has_column(self, table_name: str, column_name: str) -> bool:
        """Check if schema contains specific column."""
        return column_name in self.get_table_columns(table_name)


@dataclass(frozen=True)
class SchemaComparison:
    """Result of comparing two schema versions."""
    comparison_id: str
    from_version: SchemaVersion
    to_version: SchemaVersion
    changes: List[SchemaChange] = field(default_factory=list)
    compatibility_status: CompatibilityStatus = CompatibilityStatus.FULLY_COMPATIBLE
    overall_impact: ImpactLevel = ImpactLevel.PATCH
    comparison_timestamp: datetime = field(default_factory=datetime.now)
    
    # Change statistics
    breaking_changes_count: int = 0
    major_changes_count: int = 0
    minor_changes_count: int = 0
    
    # Compatibility analysis
    backward_compatibility_issues: List[str] = field(default_factory=list)
    forward_compatibility_issues: List[str] = field(default_factory=list)
    migration_required: bool = False
    
    # Impact analysis
    affected_tables: Set[str] = field(default_factory=set)
    affected_columns: Set[str] = field(default_factory=set)
    performance_impact_score: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Calculate change counts
        breaking_count = sum(1 for c in self.changes if c.impact_level == ImpactLevel.BREAKING)
        major_count = sum(1 for c in self.changes if c.impact_level == ImpactLevel.MAJOR)
        minor_count = sum(1 for c in self.changes if c.impact_level == ImpactLevel.MINOR)
        
        object.__setattr__(self, 'breaking_changes_count', breaking_count)
        object.__setattr__(self, 'major_changes_count', major_count)
        object.__setattr__(self, 'minor_changes_count', minor_count)
        
        # Determine overall impact
        if breaking_count > 0:
            object.__setattr__(self, 'overall_impact', ImpactLevel.BREAKING)
        elif major_count > 0:
            object.__setattr__(self, 'overall_impact', ImpactLevel.MAJOR)
        elif minor_count > 0:
            object.__setattr__(self, 'overall_impact', ImpactLevel.MINOR)
        
        # Determine compatibility
        if breaking_count > 0:
            object.__setattr__(self, 'compatibility_status', CompatibilityStatus.INCOMPATIBLE)
        elif major_count > 0:
            object.__setattr__(self, 'compatibility_status', CompatibilityStatus.PARTIALLY_COMPATIBLE)
        
        # Calculate migration requirement
        object.__setattr__(self, 'migration_required', breaking_count > 0 or major_count > 0)
    
    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return len(self.changes)
    
    @property
    def is_safe_upgrade(self) -> bool:
        """Check if upgrade is safe (no breaking changes)."""
        return self.breaking_changes_count == 0
    
    @property
    def requires_attention(self) -> bool:
        """Check if comparison requires manual attention."""
        return (self.breaking_changes_count > 0 or 
                self.major_changes_count > 0 or
                not self.is_safe_upgrade)
    
    def get_changes_by_type(self, change_type: ChangeType) -> List[SchemaChange]:
        """Get changes of specific type."""
        return [c for c in self.changes if c.change_type == change_type]
    
    def get_changes_by_impact(self, impact_level: ImpactLevel) -> List[SchemaChange]:
        """Get changes of specific impact level."""
        return [c for c in self.changes if c.impact_level == impact_level]
    
    def get_breaking_changes(self) -> List[SchemaChange]:
        """Get all breaking changes."""
        return self.get_changes_by_impact(ImpactLevel.BREAKING)


@dataclass(frozen=True)
class SchemaEvolutionHistory:
    """Complete history of schema evolution."""
    dataset_id: str
    versions: List[SchemaVersion] = field(default_factory=list)
    comparisons: List[SchemaComparison] = field(default_factory=list)
    baseline_version: Optional[SchemaVersion] = None
    current_version: Optional[SchemaVersion] = None
    
    # Evolution metrics
    total_changes: int = 0
    evolution_rate: float = 0.0  # changes per day
    stability_score: float = 1.0  # 1.0 = very stable, 0.0 = very unstable
    
    # Tracking metadata
    first_recorded: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    monitoring_enabled: bool = True
    
    def __post_init__(self):
        if self.versions:
            # Set baseline and current versions
            sorted_versions = sorted(self.versions, key=lambda v: v.timestamp)
            object.__setattr__(self, 'baseline_version', sorted_versions[0])
            object.__setattr__(self, 'current_version', sorted_versions[-1])
            object.__setattr__(self, 'first_recorded', sorted_versions[0].timestamp)
            
            # Calculate total changes
            total = sum(len(comp.changes) for comp in self.comparisons)
            object.__setattr__(self, 'total_changes', total)
            
            # Calculate evolution rate
            if len(sorted_versions) > 1:
                time_span = (sorted_versions[-1].timestamp - sorted_versions[0].timestamp).days
                if time_span > 0:
                    rate = total / time_span
                    object.__setattr__(self, 'evolution_rate', rate)
            
            # Calculate stability score
            if self.comparisons:
                breaking_changes = sum(comp.breaking_changes_count for comp in self.comparisons)
                stability = max(0.0, 1.0 - (breaking_changes / max(1, total)))
                object.__setattr__(self, 'stability_score', stability)
    
    @property
    def version_count(self) -> int:
        """Get total number of versions."""
        return len(self.versions)
    
    @property
    def has_breaking_changes(self) -> bool:
        """Check if any version introduced breaking changes."""
        return any(comp.breaking_changes_count > 0 for comp in self.comparisons)
    
    @property
    def latest_version_number(self) -> Optional[str]:
        """Get latest version number."""
        return self.current_version.version_number if self.current_version else None
    
    def get_version_by_number(self, version_number: str) -> Optional[SchemaVersion]:
        """Get version by version number."""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None
    
    def get_changes_since_version(self, version_number: str) -> List[SchemaChange]:
        """Get all changes since specific version."""
        changes = []
        target_version = self.get_version_by_number(version_number)
        if not target_version:
            return changes
        
        for comparison in self.comparisons:
            if comparison.from_version.timestamp >= target_version.timestamp:
                changes.extend(comparison.changes)
        
        return changes
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of schema evolution."""
        return {
            "dataset_id": self.dataset_id,
            "version_count": self.version_count,
            "total_changes": self.total_changes,
            "evolution_rate_per_day": self.evolution_rate,
            "stability_score": self.stability_score,
            "has_breaking_changes": self.has_breaking_changes,
            "current_version": self.latest_version_number,
            "monitoring_period_days": (
                (self.last_updated - self.first_recorded).days 
                if self.first_recorded else 0
            ),
            "breaking_changes_count": sum(
                comp.breaking_changes_count for comp in self.comparisons
            ),
            "major_changes_count": sum(
                comp.major_changes_count for comp in self.comparisons
            )
        }
    
    def add_version(self, version: SchemaVersion) -> SchemaEvolutionHistory:
        """Add new version to history."""
        new_versions = list(self.versions) + [version]
        return dataclass.replace(self, versions=new_versions, last_updated=datetime.now())
    
    def add_comparison(self, comparison: SchemaComparison) -> SchemaEvolutionHistory:
        """Add new comparison to history."""
        new_comparisons = list(self.comparisons) + [comparison]
        return dataclass.replace(self, comparisons=new_comparisons, last_updated=datetime.now())
    
    @classmethod
    def create_initial(cls, dataset_id: str, baseline_version: SchemaVersion) -> SchemaEvolutionHistory:
        """Create initial evolution history with baseline version."""
        return cls(
            dataset_id=dataset_id,
            versions=[baseline_version],
            baseline_version=baseline_version,
            current_version=baseline_version,
            first_recorded=baseline_version.timestamp
        )