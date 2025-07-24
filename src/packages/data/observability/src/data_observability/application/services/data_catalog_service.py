"""
Data Catalog Service

Provides application-level services for managing data catalog operations,
including asset registration, discovery, search, and metadata management.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID

from ...domain.entities.data_catalog import (
    DataCatalogEntry,
    DataSchema,
    DataUsage,
    DataAssetType,
    DataFormat,
    AccessLevel,
    DataClassification,
    DataQuality,
    ColumnMetadata
)


from ...domain.repositories.data_catalog_repository import DataCatalogRepository
from ...infrastructure.errors.exceptions import AssetNotFoundError

class DataCatalogService:
    """Service for managing data catalog operations."""
    
    def __init__(self, repository: DataCatalogRepository):
        self._repository = repository
        self._usage_history: Dict[UUID, List[DataUsage]] = {} # This will need to be persisted later
        self._schemas: Dict[UUID, DataSchema] = {} # This will need to be persisted later
        
        # Business glossary - will need to be persisted later
        self._business_terms: Dict[str, Dict[str, Any]] = {}
        
        # Auto-discovery patterns - will need to be persisted later
        self._discovery_patterns: List[Dict[str, Any]] = []
    
    async def register_asset(
        self,
        name: str,
        asset_type: DataAssetType,
        location: str,
        data_format: DataFormat,
        description: str = None,
        owner: str = None,
        domain: str = None,
        schema: DataSchema = None,
        pipeline_id: Optional[UUID] = None,
        **kwargs
    ) -> DataCatalogEntry:
        """Register a new data asset in the catalog."""
        
        entry = DataCatalogEntry(
            name=name,
            type=asset_type,
            location=location,
            format=data_format,
            description=description,
            owner=owner,
            domain=domain,
            schema_=schema,
            **kwargs
        )
        
        # Auto-classify based on patterns (will be handled by domain service or repository)
        # self._auto_classify_asset(entry)
        
        # Auto-tag based on discovery patterns (will be handled by domain service or repository)
        # self._auto_tag_asset(entry)
        
        # Store in catalog
        await self._repository.save(entry)
        self._usage_history[entry.id] = []
        
        if schema:
            self._schemas[schema.id] = schema
        
        return entry
    
    async def get_asset(self, asset_id: UUID) -> Optional[DataCatalogEntry]:
        """Get an asset by ID."""
        return await self._repository.get_by_id(asset_id)
    
    async def get_asset_by_name(self, name: str, domain: str = None) -> Optional[DataCatalogEntry]:
        """Get an asset by name and optional domain."""
        # The repository's get_by_name might return a list, so we need to filter by domain if provided
        assets = await self._repository.get_by_name(name)
        for asset in assets:
            if (domain is None or asset.domain == domain):
                return asset
        return None
    
    async def search_assets(
        self,
        query: str = None,
        asset_type: DataAssetType = None,
        owner: str = None,
        domain: str = None,
        tags: List[str] = None,
        classification: DataClassification = None,
        quality_min: float = None,
        limit: int = 50
    ) -> List[DataCatalogEntry]:
        """Search for assets with various filters."""
        
        # Fetch all assets from the repository (temporary, will be optimized with proper search in repo)
        all_assets = await self._repository.get_all()
        candidates = []
        
        for asset in all_assets:
            # Apply filters
            if asset_type and asset.type != asset_type:
                continue
            if owner and asset.owner != owner:
                continue
            if domain and asset.domain != domain:
                continue
            if tags and not all(tag.lower() in [t.lower() for t in asset.tags] for tag in tags):
                continue
            if classification and asset.classification != classification:
                continue
            if quality_min is not None and (asset.quality_score is None or asset.quality_score < quality_min):
                continue
            
            # Calculate relevance score for text query
            relevance_score = 1.0
            if query:
                query_terms = self._parse_query(query)
                relevance_score = asset.get_relevance_score(query_terms)
                
                # Skip assets with very low relevance
                if relevance_score < 0.1:
                    continue
            
            candidates.append((asset, relevance_score))
        
        # Sort by relevance, popularity, and quality
        candidates.sort(key=lambda x: (
            x[1],  # Relevance score
            x[0].get_popularity_score(),  # Popularity
            x[0].quality_score or 0,  # Quality
            x[0].get_freshness_score()  # Freshness
        ), reverse=True)
        
        return [asset for asset, _ in candidates[:limit]]
    
    async def discover_similar_assets(self, asset_id: UUID, limit: int = 10) -> List[Tuple[DataCatalogEntry, float]]:
        """Discover assets similar to the given asset."""
        source_asset = await self._repository.get_by_id(asset_id)
        if not source_asset:
            return []
        
        similarities = []
        
        all_assets = await self._repository.get_all()
        for other_asset in all_assets:
            if other_asset.id == asset_id:
                continue
            
            similarity = self._calculate_similarity(source_asset, other_asset)
            if similarity > 0.1:  # Minimum similarity threshold
                similarities.append((other_asset, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:limit]
    
    def record_usage(
        self,
        asset_id: UUID,
        user_id: str = None,
        user_name: str = None,
        usage_type: str = "read",
        query: str = None,
        rows_accessed: int = None,
        columns_accessed: List[str] = None,
        duration_ms: int = None,
        application: str = None,
        purpose: str = None
    ) -> DataUsage:
        """Record usage of a data asset."""
        
        asset = await self._repository.get_by_id(asset_id)
        if not asset:
            raise AssetNotFoundError(f"Asset {asset_id} not found")
        
        usage = DataUsage(
            asset_id=asset_id,
            user_id=user_id,
            user_name=user_name,
            usage_type=usage_type,
            query=query,
            rows_accessed=rows_accessed,
            columns_accessed=columns_accessed or [],
            duration_ms=duration_ms,
            application=application,
            purpose=purpose
        )
        
        # Store usage (this will need to be persisted later)
        if asset_id not in self._usage_history:
            self._usage_history[asset_id] = []
        self._usage_history[asset_id].append(usage)
        
        # Update asset access tracking and save the asset
        asset.record_access(user_name or user_id, usage_type)
        await self._repository.update(asset)
        
        # Trim usage history to last 10000 entries (this will be handled by persistence later)
        if len(self._usage_history[asset_id]) > 10000:
            self._usage_history[asset_id] = self._usage_history[asset_id][-10000:]
        
        return usage
    
    def get_usage_analytics(self, asset_id: UUID, days: int = 30) -> Dict[str, Any]:
        """Get usage analytics for an asset."""
        if asset_id not in self._usage_history:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_usage = [
            usage for usage in self._usage_history[asset_id]
            if usage.timestamp >= cutoff_time
        ]
        
        if not recent_usage:
            return {
                "total_accesses": 0,
                "unique_users": 0,
                "usage_by_type": {},
                "usage_by_application": {},
                "daily_usage": [],
                "peak_usage_hour": None,
                "avg_duration_ms": None
            }
        
        # Calculate analytics
        total_accesses = len(recent_usage)
        unique_users = len(set(u.user_id for u in recent_usage if u.user_id))
        
        # Usage by type
        usage_by_type = {}
        for usage in recent_usage:
            usage_by_type[usage.usage_type] = usage_by_type.get(usage.usage_type, 0) + 1
        
        # Usage by application
        usage_by_application = {}
        for usage in recent_usage:
            if usage.application:
                app = usage.application
                usage_by_application[app] = usage_by_application.get(app, 0) + 1
        
        # Daily usage
        daily_usage = {}
        for usage in recent_usage:
            day = usage.timestamp.date().isoformat()
            daily_usage[day] = daily_usage.get(day, 0) + 1
        
        # Peak usage hour
        hourly_usage = {}
        for usage in recent_usage:
            hour = usage.timestamp.hour
            hourly_usage[hour] = hourly_usage.get(hour, 0) + 1
        
        peak_hour = max(hourly_usage.items(), key=lambda x: x[1])[0] if hourly_usage else None
        
        # Average duration
        durations = [u.duration_ms for u in recent_usage if u.duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else None
        
        return {
            "total_accesses": total_accesses,
            "unique_users": unique_users,
            "usage_by_type": usage_by_type,
            "usage_by_application": usage_by_application,
            "daily_usage": [{"date": k, "count": v} for k, v in daily_usage.items()],
            "peak_usage_hour": peak_hour,
            "avg_duration_ms": avg_duration
        }
    
    def add_business_term(self, term: str, definition: str, category: str = None, related_terms: List[str] = None) -> None:
        """Add a business term to the glossary."""
        # This will need to be persisted later
        self._business_terms[term.lower()] = {
            "term": term,
            "definition": definition,
            "category": category,
            "related_terms": related_terms or [],
            "created_at": datetime.utcnow().isoformat()
        }
    
    def get_business_term(self, term: str) -> Optional[Dict[str, Any]]:
        """Get a business term definition."""
        return self._business_terms.get(term.lower())
    
    def suggest_business_terms(self, asset: DataCatalogEntry) -> List[str]:
        """Suggest business terms for an asset based on its metadata."""
        suggestions = []
        
        # Check name and description for business terms
        text = f"{asset.name} {asset.description or ''}".lower()
        
        for term in self._business_terms:
            if term in text:
                suggestions.append(self._business_terms[term]["term"])
        
        return suggestions
    
    def add_discovery_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add an auto-discovery pattern."""
        self._discovery_patterns.append({
            "id": len(self._discovery_patterns),
            "created_at": datetime.utcnow(),
            **pattern
        })
    
    def auto_discover_assets(self, scan_locations: List[str]) -> List[DataCatalogEntry]:
        """Auto-discover assets in specified locations."""
        discovered = []
        
        # This is a simplified implementation - in practice, you'd scan
        # actual file systems, databases, APIs, etc.
        
        for location in scan_locations:
            # Apply discovery patterns
            for pattern in self._discovery_patterns:
                if self._location_matches_pattern(location, pattern):
                    asset = self._create_asset_from_pattern(location, pattern)
                    if asset:
                        discovered.append(asset)
        
        return discovered
    
    async def get_catalog_statistics(self) -> Dict[str, Any]:
        """Get overall catalog statistics."""
        all_assets = await self._repository.get_all()
        total_assets = len(all_assets)
        
        # Assets by type
        by_type = {}
        for asset in all_assets:
            by_type[asset.type.value] = by_type.get(asset.type.value, 0) + 1
        
        # Assets by classification
        by_classification = {}
        for asset in all_assets:
            cls = asset.classification.value
            by_classification[cls] = by_classification.get(cls, 0) + 1
        
        # Assets by quality
        by_quality = {}
        for asset in all_assets:
            qual = asset.quality.value
            by_quality[qual] = by_quality.get(qual, 0) + 1
        
        # Top domains
        domain_counts = {}
        for asset in all_assets:
            if asset.domain:
                domain_counts[asset.domain] = domain_counts.get(asset.domain, 0) + 1
        
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top owners
        owner_counts = {}
        for asset in all_assets:
            if asset.owner:
                owner_counts[asset.owner] = owner_counts.get(asset.owner, 0) + 1
        
        top_owners = sorted(owner_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Quality statistics
        quality_scores = [a.quality_score for a in all_assets if a.quality_score]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "total_assets": total_assets,
            "assets_by_type": by_type,
            "assets_by_classification": by_classification,
            "assets_by_quality": by_quality,
            "top_domains": top_domains,
            "top_owners": top_owners,
            "avg_quality_score": avg_quality,
            "total_business_terms": len(self._business_terms), # Still in-memory
            "total_schemas": len(self._schemas) # Still in-memory
        }
    
    async def get_data_lineage_graph(self, asset_id: UUID, depth: int = 2) -> Dict[str, Any]:
        """Get lineage graph for an asset."""
        asset = await self._repository.get_by_id(asset_id)
        if not asset:
            return {}
        
        # This method needs to interact with the DataLineageService
        # For now, we'll return a simplified structure.
        return {
            "nodes": [],
            "edges": [],
            "center_node": str(asset_id),
            "message": "Lineage graph integration is pending with DataLineageService."
        }
    
    def _parse_query(self, query: str) -> List[str]:
        """Parse search query into terms."""
        # Simple parsing - split by whitespace and remove special characters
        terms = re.findall(r'\b\w+\b', query.lower())
        return terms
    
    def _calculate_similarity(self, asset1: DataCatalogEntry, asset2: DataCatalogEntry) -> float:
        """Calculate similarity between two assets."""
        similarity = 0.0
        
        # Type similarity
        if asset1.type == asset2.type:
            similarity += 0.3
        
        # Tag similarity
        tag1 = asset1.tags
        tag2 = asset2.tags
        if tag1 and tag2:
            intersection = len(tag1.intersection(tag2))
            union = len(tag1.union(tag2))
            tag_similarity = intersection / union if union > 0 else 0
            similarity += tag_similarity * 0.2
        
        # Domain similarity
        if asset1.domain and asset2.domain and asset1.domain == asset2.domain:
            similarity += 0.2
        
        # Owner similarity
        if asset1.owner and asset2.owner and asset1.owner == asset2.owner:
            similarity += 0.1
        
        # Name similarity (simple string matching)
        name1_words = set(asset1.name.lower().split())
        name2_words = set(asset2.name.lower().split())
        if name1_words and name2_words:
            name_intersection = len(name1_words.intersection(name2_words))
            name_union = len(name1_words.union(name2_words))
            name_similarity = name_intersection / name_union if name_union > 0 else 0
            similarity += name_similarity * 0.2
        
        return min(1.0, similarity)
    
    def _auto_classify_asset(self, asset: DataCatalogEntry) -> None:
        """Auto-classify asset based on patterns."""
        name_lower = asset.name.lower()
        location_lower = asset.location.lower()
        
        # PII detection patterns
        pii_patterns = [
            r'\b(ssn|social.security|tax.id)\b',
            r'\b(email|phone|address)\b',
            r'\b(credit.card|cc|payment)\b',
            r'\b(password|pwd|secret)\b'
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, name_lower) or re.search(pattern, location_lower):
                asset.classification = DataClassification.PII
                asset.access_level = AccessLevel.RESTRICTED
                break
        
        # Financial data patterns
        financial_patterns = [
            r'\b(financial|finance|revenue|profit)\b',
            r'\b(salary|wage|compensation)\b',
            r'\b(budget|cost|expense)\b'
        ]
        
        for pattern in financial_patterns:
            if re.search(pattern, name_lower):
                asset.classification = DataClassification.FINANCIAL
                asset.access_level = AccessLevel.CONFIDENTIAL
                break
    
    def _auto_tag_asset(self, asset: DataCatalogEntry) -> None:
        """Auto-tag asset based on discovery patterns."""
        name_lower = asset.name.lower()
        
        # Common tag patterns
        tag_patterns = {
            "customer": [r'\b(customer|client|user)\b'],
            "product": [r'\b(product|item|catalog)\b'],
            "transaction": [r'\b(transaction|order|purchase|sale)\b'],
            "analytics": [r'\b(analytics|metric|kpi|report)\b'],
            "log": [r'\b(log|audit|event)\b'],
            "reference": [r'\b(reference|lookup|dimension)\b']
        }
        
        for tag, patterns in tag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    asset.add_tag(tag)
                    break
    
    def _location_matches_pattern(self, location: str, pattern: Dict[str, Any]) -> bool:
        """Check if location matches discovery pattern."""
        pattern_regex = pattern.get("location_pattern")
        if pattern_regex:
            return bool(re.search(pattern_regex, location))
        return False
    
    def _create_asset_from_pattern(self, location: str, pattern: Dict[str, Any]) -> Optional[DataCatalogEntry]:
        """Create asset from discovery pattern."""
        try:
            # Extract name from location
            name = pattern.get("name_extractor", location.split("/")[-1])
            
            asset = DataCatalogEntry(
                name=name,
                type=DataAssetType(pattern.get("asset_type", "file")),
                location=location,
                format=DataFormat(pattern.get("format", "unknown")),
                description=pattern.get("description", f"Auto-discovered asset at {location}"),
                domain=pattern.get("domain")
            )
            
            # Apply pattern tags
            for tag in pattern.get("tags", []):
                asset.add_tag(tag)
            
            # Store in catalog
            self._catalog[asset.id] = asset
            self._usage_history[asset.id] = []
            self._update_indexes(asset)
            
            return asset
            
        except Exception:
            return None
    
    