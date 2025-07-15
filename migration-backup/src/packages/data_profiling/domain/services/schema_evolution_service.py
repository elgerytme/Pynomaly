"""Domain service for schema evolution analysis and tracking."""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import hashlib
import json

from ..entities.data_profile import SchemaProfile, ColumnProfile
from ..value_objects.schema_evolution import (
    SchemaChange, SchemaVersion, SchemaComparison, SchemaEvolutionHistory,
    ChangeType, ImpactLevel, CompatibilityStatus
)


class SchemaEvolutionService:
    """Domain service for schema evolution analysis and tracking."""

    def create_schema_version(
        self,
        schema_profile: SchemaProfile,
        version_number: str,
        author: str = "system",
        description: str = ""
    ) -> SchemaVersion:
        """Create a schema version from a schema profile.
        
        Args:
            schema_profile: Schema profile to create version from
            version_number: Version number string
            author: Author of the schema version
            description: Description of changes
            
        Returns:
            SchemaVersion instance
        """
        # Extract schema structure
        tables = ["main_table"]  # Default table name
        columns = {"main_table": [col.column_name for col in schema_profile.columns]}
        data_types = {col.column_name: col.data_type.value for col in schema_profile.columns}
        
        # Generate schema hash
        schema_hash = self._generate_schema_hash(schema_profile)
        
        return SchemaVersion(
            version_id=f"schema_{version_number}_{int(datetime.now().timestamp())}",
            version_number=version_number,
            schema_hash=schema_hash,
            timestamp=datetime.now(),
            author=author,
            description=description,
            tables=tables,
            columns=columns,
            data_types=data_types,
            constraints=schema_profile.constraints,
            indexes=[idx.index_name for idx in schema_profile.indexes],
            total_tables=schema_profile.total_tables,
            total_columns=schema_profile.total_columns,
            total_rows=schema_profile.total_rows,
            estimated_size_bytes=schema_profile.estimated_size_bytes,
            schema_quality_score=self._calculate_schema_quality(schema_profile),
            data_quality_score=0.0,  # To be filled from quality assessment
            completeness_percentage=self._calculate_completeness(schema_profile)
        )

    def compare_schemas(
        self,
        from_version: SchemaVersion,
        to_version: SchemaVersion
    ) -> SchemaComparison:
        """Compare two schema versions and identify changes.
        
        Args:
            from_version: Source schema version
            to_version: Target schema version
            
        Returns:
            SchemaComparison with detected changes
        """
        comparison_id = f"comp_{from_version.version_id}_{to_version.version_id}"
        changes = []
        
        # Compare columns
        changes.extend(self._compare_columns(from_version, to_version))
        
        # Compare data types
        changes.extend(self._compare_data_types(from_version, to_version))
        
        # Compare constraints
        changes.extend(self._compare_constraints(from_version, to_version))
        
        # Compare indexes
        changes.extend(self._compare_indexes(from_version, to_version))
        
        # Calculate compatibility
        compatibility_status = self._determine_compatibility(changes)
        
        # Calculate affected elements
        affected_tables = {change.element_name for change in changes if "table" in change.change_type.value}
        affected_columns = {change.element_name for change in changes if "column" in change.change_type.value}
        
        return SchemaComparison(
            comparison_id=comparison_id,
            from_version=from_version,
            to_version=to_version,
            changes=changes,
            compatibility_status=compatibility_status,
            affected_tables=affected_tables,
            affected_columns=affected_columns,
            performance_impact_score=self._calculate_performance_impact(changes)
        )

    def track_evolution(
        self,
        dataset_id: str,
        existing_history: Optional[SchemaEvolutionHistory],
        new_version: SchemaVersion
    ) -> SchemaEvolutionHistory:
        """Track schema evolution by adding new version to history.
        
        Args:
            dataset_id: Dataset identifier
            existing_history: Existing evolution history (if any)
            new_version: New schema version to add
            
        Returns:
            Updated SchemaEvolutionHistory
        """
        if existing_history is None:
            # Create initial history
            return SchemaEvolutionHistory.create_initial(dataset_id, new_version)
        
        # Add new version
        updated_history = existing_history.add_version(new_version)
        
        # If there's a previous version, create comparison
        if len(updated_history.versions) > 1:
            previous_version = sorted(updated_history.versions, key=lambda v: v.timestamp)[-2]
            comparison = self.compare_schemas(previous_version, new_version)
            updated_history = updated_history.add_comparison(comparison)
        
        return updated_history

    def detect_breaking_changes(self, comparison: SchemaComparison) -> List[SchemaChange]:
        """Detect breaking changes in schema comparison.
        
        Args:
            comparison: Schema comparison to analyze
            
        Returns:
            List of breaking changes
        """
        return comparison.get_breaking_changes()

    def suggest_migration_strategies(self, changes: List[SchemaChange]) -> List[str]:
        """Suggest migration strategies for schema changes.
        
        Args:
            changes: List of schema changes
            
        Returns:
            List of migration strategy suggestions
        """
        strategies = []
        
        for change in changes:
            if change.change_type == ChangeType.COLUMN_REMOVED:
                strategies.extend([
                    f"Create migration script to handle removal of column '{change.element_name}'",
                    f"Update all queries and applications using column '{change.element_name}'",
                    "Consider data archival before column removal"
                ])
            
            elif change.change_type == ChangeType.COLUMN_TYPE_CHANGED:
                strategies.extend([
                    f"Implement type conversion for column '{change.element_name}'",
                    "Validate data compatibility before type change",
                    "Create rollback plan in case of conversion issues"
                ])
            
            elif change.change_type == ChangeType.COLUMN_NULLABLE_CHANGED:
                old_nullable = change.old_value
                new_nullable = change.new_value
                if old_nullable and not new_nullable:
                    strategies.append(f"Ensure no NULL values exist in column '{change.element_name}' before making it NOT NULL")
            
            elif change.change_type == ChangeType.PRIMARY_KEY_CHANGED:
                strategies.extend([
                    "Update all foreign key references",
                    "Rebuild dependent indexes",
                    "Update application code using primary key"
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        return unique_strategies

    def calculate_evolution_metrics(self, history: SchemaEvolutionHistory) -> Dict[str, Any]:
        """Calculate metrics for schema evolution.
        
        Args:
            history: Schema evolution history
            
        Returns:
            Dictionary containing evolution metrics
        """
        if not history.versions:
            return {}
        
        metrics = {
            "total_versions": len(history.versions),
            "total_changes": history.total_changes,
            "evolution_rate_per_day": history.evolution_rate,
            "stability_score": history.stability_score,
            "average_changes_per_version": history.total_changes / len(history.versions) if history.versions else 0,
        }
        
        # Calculate change type distribution
        change_type_counts = {}
        for comparison in history.comparisons:
            for change in comparison.changes:
                change_type = change.change_type.value
                change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        metrics["change_type_distribution"] = change_type_counts
        
        # Calculate impact distribution
        impact_counts = {}
        for comparison in history.comparisons:
            for change in comparison.changes:
                impact = change.impact_level.value
                impact_counts[impact] = impact_counts.get(impact, 0) + 1
        
        metrics["impact_distribution"] = impact_counts
        
        # Calculate compatibility trends
        compatibility_trend = []
        for comparison in history.comparisons:
            compatibility_trend.append({
                "timestamp": comparison.comparison_timestamp,
                "status": comparison.compatibility_status.value,
                "breaking_changes": comparison.breaking_changes_count
            })
        
        metrics["compatibility_trend"] = compatibility_trend
        
        return metrics

    def _generate_schema_hash(self, schema_profile: SchemaProfile) -> str:
        """Generate hash for schema structure."""
        schema_data = {
            "columns": sorted([
                {
                    "name": col.column_name,
                    "type": col.data_type.value,
                    "nullable": col.nullable
                } for col in schema_profile.columns
            ], key=lambda x: x["name"]),
            "constraints": sorted(schema_profile.constraints),
            "indexes": sorted([idx.index_name for idx in schema_profile.indexes])
        }
        
        schema_json = json.dumps(schema_data, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()

    def _calculate_schema_quality(self, schema_profile: SchemaProfile) -> float:
        """Calculate schema quality score."""
        if not schema_profile.columns:
            return 0.0
        
        quality_factors = []
        
        # Primary key presence
        has_primary_key = len(schema_profile.primary_keys) > 0
        quality_factors.append(1.0 if has_primary_key else 0.5)
        
        # Column naming consistency
        column_names = [col.column_name for col in schema_profile.columns]
        naming_score = self._calculate_naming_consistency(column_names)
        quality_factors.append(naming_score)
        
        # Type consistency
        type_score = self._calculate_type_consistency(schema_profile.columns)
        quality_factors.append(type_score)
        
        # Index coverage
        indexed_columns = set()
        for index in schema_profile.indexes:
            indexed_columns.update(index.column_names)
        
        index_coverage = len(indexed_columns) / len(schema_profile.columns)
        quality_factors.append(min(1.0, index_coverage * 2))  # Cap at 1.0
        
        return sum(quality_factors) / len(quality_factors)

    def _calculate_completeness(self, schema_profile: SchemaProfile) -> float:
        """Calculate schema completeness percentage."""
        if not schema_profile.columns:
            return 0.0
        
        completeness_factors = []
        
        for column in schema_profile.columns:
            column_completeness = column.completeness_ratio
            completeness_factors.append(column_completeness)
        
        return sum(completeness_factors) / len(completeness_factors) * 100

    def _compare_columns(self, from_version: SchemaVersion, to_version: SchemaVersion) -> List[SchemaChange]:
        """Compare columns between versions."""
        changes = []
        from_columns = set()
        to_columns = set()
        
        # Get all columns from both versions
        for table_columns in from_version.columns.values():
            from_columns.update(table_columns)
        
        for table_columns in to_version.columns.values():
            to_columns.update(table_columns)
        
        # Detect added columns
        added_columns = to_columns - from_columns
        for column in added_columns:
            change_id = f"add_col_{column}_{int(datetime.now().timestamp())}"
            column_type = to_version.data_types.get(column, "unknown")
            changes.append(SchemaChange.create_column_added(change_id, column, column_type))
        
        # Detect removed columns
        removed_columns = from_columns - to_columns
        for column in removed_columns:
            change_id = f"rem_col_{column}_{int(datetime.now().timestamp())}"
            column_type = from_version.data_types.get(column, "unknown")
            changes.append(SchemaChange.create_column_removed(change_id, column, column_type))
        
        return changes

    def _compare_data_types(self, from_version: SchemaVersion, to_version: SchemaVersion) -> List[SchemaChange]:
        """Compare data types between versions."""
        changes = []
        
        # Get common columns
        from_columns = set()
        to_columns = set()
        
        for table_columns in from_version.columns.values():
            from_columns.update(table_columns)
        
        for table_columns in to_version.columns.values():
            to_columns.update(table_columns)
        
        common_columns = from_columns & to_columns
        
        # Compare types for common columns
        for column in common_columns:
            from_type = from_version.data_types.get(column)
            to_type = to_version.data_types.get(column)
            
            if from_type and to_type and from_type != to_type:
                change_id = f"type_change_{column}_{int(datetime.now().timestamp())}"
                changes.append(SchemaChange.create_type_change(change_id, column, from_type, to_type))
        
        return changes

    def _compare_constraints(self, from_version: SchemaVersion, to_version: SchemaVersion) -> List[SchemaChange]:
        """Compare constraints between versions."""
        changes = []
        
        from_constraints = set(from_version.constraints)
        to_constraints = set(to_version.constraints)
        
        # Detect added constraints
        added_constraints = to_constraints - from_constraints
        for constraint in added_constraints:
            changes.append(SchemaChange(
                change_id=f"add_constraint_{hash(constraint)}_{int(datetime.now().timestamp())}",
                change_type=ChangeType.CONSTRAINT_ADDED,
                element_name=constraint,
                new_value=constraint,
                impact_level=ImpactLevel.MINOR,
                description=f"Added constraint: {constraint}"
            ))
        
        # Detect removed constraints
        removed_constraints = from_constraints - to_constraints
        for constraint in removed_constraints:
            changes.append(SchemaChange(
                change_id=f"rem_constraint_{hash(constraint)}_{int(datetime.now().timestamp())}",
                change_type=ChangeType.CONSTRAINT_REMOVED,
                element_name=constraint,
                old_value=constraint,
                impact_level=ImpactLevel.MAJOR,
                description=f"Removed constraint: {constraint}"
            ))
        
        return changes

    def _compare_indexes(self, from_version: SchemaVersion, to_version: SchemaVersion) -> List[SchemaChange]:
        """Compare indexes between versions."""
        changes = []
        
        from_indexes = set(from_version.indexes)
        to_indexes = set(to_version.indexes)
        
        # Detect added indexes
        added_indexes = to_indexes - from_indexes
        for index in added_indexes:
            changes.append(SchemaChange(
                change_id=f"add_index_{hash(index)}_{int(datetime.now().timestamp())}",
                change_type=ChangeType.INDEX_ADDED,
                element_name=index,
                new_value=index,
                impact_level=ImpactLevel.MINOR,
                description=f"Added index: {index}"
            ))
        
        # Detect removed indexes
        removed_indexes = from_indexes - to_indexes
        for index in removed_indexes:
            changes.append(SchemaChange(
                change_id=f"rem_index_{hash(index)}_{int(datetime.now().timestamp())}",
                change_type=ChangeType.INDEX_REMOVED,
                element_name=index,
                old_value=index,
                impact_level=ImpactLevel.MINOR,
                description=f"Removed index: {index}"
            ))
        
        return changes

    def _determine_compatibility(self, changes: List[SchemaChange]) -> CompatibilityStatus:
        """Determine compatibility status based on changes."""
        if not changes:
            return CompatibilityStatus.FULLY_COMPATIBLE
        
        breaking_changes = [c for c in changes if c.impact_level == ImpactLevel.BREAKING]
        major_changes = [c for c in changes if c.impact_level == ImpactLevel.MAJOR]
        
        if breaking_changes:
            return CompatibilityStatus.INCOMPATIBLE
        elif major_changes:
            return CompatibilityStatus.PARTIALLY_COMPATIBLE
        else:
            return CompatibilityStatus.BACKWARD_COMPATIBLE

    def _calculate_performance_impact(self, changes: List[SchemaChange]) -> float:
        """Calculate performance impact score for changes."""
        if not changes:
            return 0.0
        
        impact_scores = []
        
        for change in changes:
            if change.change_type in [ChangeType.INDEX_ADDED, ChangeType.INDEX_REMOVED]:
                impact_scores.append(0.8)  # High impact for index changes
            elif change.change_type in [ChangeType.COLUMN_TYPE_CHANGED]:
                impact_scores.append(0.6)  # Medium-high impact for type changes
            elif change.change_type in [ChangeType.CONSTRAINT_ADDED, ChangeType.CONSTRAINT_REMOVED]:
                impact_scores.append(0.4)  # Medium impact for constraint changes
            else:
                impact_scores.append(0.2)  # Low impact for other changes
        
        return sum(impact_scores) / len(impact_scores)

    def _calculate_naming_consistency(self, column_names: List[str]) -> float:
        """Calculate naming consistency score."""
        if not column_names:
            return 1.0
        
        # Simple heuristics for naming consistency
        consistent_factors = []
        
        # Check for consistent case
        lower_count = sum(1 for name in column_names if name.islower())
        upper_count = sum(1 for name in column_names if name.isupper())
        mixed_count = len(column_names) - lower_count - upper_count
        
        case_consistency = max(lower_count, upper_count, mixed_count) / len(column_names)
        consistent_factors.append(case_consistency)
        
        # Check for consistent separators
        underscore_count = sum(1 for name in column_names if "_" in name)
        separator_consistency = max(underscore_count, len(column_names) - underscore_count) / len(column_names)
        consistent_factors.append(separator_consistency)
        
        return sum(consistent_factors) / len(consistent_factors)

    def _calculate_type_consistency(self, columns: List[ColumnProfile]) -> float:
        """Calculate type consistency score."""
        if not columns:
            return 1.0
        
        # Check for appropriate type usage
        appropriate_types = 0
        
        for column in columns:
            # Simple heuristics for type appropriateness
            if column.data_type.value in ["integer", "float"] and column.unique_count > 1:
                appropriate_types += 1
            elif column.data_type.value == "string" and column.unique_count > 0:
                appropriate_types += 1
            elif column.data_type.value == "boolean" and column.unique_count <= 2:
                appropriate_types += 1
            else:
                # Default to appropriate for other cases
                appropriate_types += 1
        
        return appropriate_types / len(columns)