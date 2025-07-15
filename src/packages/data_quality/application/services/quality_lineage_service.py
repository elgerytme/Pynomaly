"""Quality Lineage and Impact Analysis Service.

Service for building quality lineage graphs, analyzing impact propagation,
and providing comprehensive quality dependency tracking.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from functools import lru_cache, wraps
import sqlite3
import pickle
import gzip
import os
from pathlib import Path

from ...domain.entities.quality_lineage import (
    QualityLineageGraph, LineageNode, LineageEdge, QualityImpactAnalysis,
    LineageNodeId, LineageEdgeId, LineageNodeType, ImpactType, ImpactSeverity
)
from ...domain.entities.quality_anomaly import QualityAnomaly, AnomalySeverity
from ...domain.entities.quality_profile import DataQualityProfile, DatasetId
from ...domain.entities.quality_scores import QualityScores

logger = logging.getLogger(__name__)


class LineageAnalysisType(Enum):
    """Types of lineage analysis."""
    UPSTREAM_IMPACT = "upstream_impact"
    DOWNSTREAM_IMPACT = "downstream_impact"
    BIDIRECTIONAL_IMPACT = "bidirectional_impact"
    CROSS_DATASET_IMPACT = "cross_dataset_impact"
    TEMPORAL_IMPACT = "temporal_impact"


@dataclass(frozen=True)
class LineageServiceConfig:
    """Configuration for quality lineage service."""
    # Impact analysis
    max_propagation_depth: int = 10
    min_impact_threshold: float = 0.1
    enable_impact_scoring: bool = True
    
    # Performance optimizations
    max_nodes_per_analysis: int = 1000000  # Support up to 1 million nodes
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    
    # Advanced caching
    enable_persistent_cache: bool = True
    cache_compression: bool = True
    cache_partitioning: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    lazy_loading: bool = True
    node_attribute_compression: bool = True
    
    # Lineage building
    auto_discover_relationships: bool = True
    confidence_threshold: float = 0.7
    enable_weak_relationships: bool = False
    
    # Impact propagation
    propagation_decay_factor: float = 0.8
    amplification_factor: float = 1.2
    mitigation_factor: float = 0.5
    
    # Reporting
    include_recommendations: bool = True
    max_recommendations: int = 10
    generate_visualizations: bool = False


@dataclass
class ImpactPropagationRule:
    """Rule for how quality impacts propagate through lineage."""
    source_node_type: LineageNodeType
    target_node_type: LineageNodeType
    relationship_type: str
    impact_factor: float
    impact_type: ImpactType
    conditions: Dict[str, Any] = field(default_factory=dict)


class QualityLineageService:
    """Service for quality lineage and impact analysis."""
    
    def __init__(self, config: LineageServiceConfig = None):
        """Initialize quality lineage service.
        
        Args:
            config: Service configuration
        """
        self.config = config or LineageServiceConfig()
        self._lineage_graphs: Dict[str, QualityLineageGraph] = {}
        self._impact_cache: Dict[str, QualityImpactAnalysis] = {}
        
        # Performance optimizations
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers) if self.config.enable_parallel_processing else None
        self._cache_lock = threading.RLock()
        self._persistent_cache_db = None
        
        # Initialize persistent cache if enabled
        if self.config.enable_persistent_cache:
            self._initialize_persistent_cache()
        
        # Define default propagation rules
        self._propagation_rules = self._initialize_propagation_rules()
        
        # Advanced caching structures
        self._node_cache: Dict[str, Any] = {}
        self._edge_cache: Dict[str, Any] = {}
        self._path_cache: Dict[str, List[List[LineageNodeId]]] = {}
        
        logger.info("Quality Lineage Service initialized with performance optimizations")
    
    def _initialize_persistent_cache(self):
        """Initialize persistent cache using SQLite."""
        try:
            cache_dir = Path.home() / ".pynomaly" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = cache_dir / "lineage_cache.db"
            self._persistent_cache_db = sqlite3.connect(str(cache_file), check_same_thread=False)
            
            # Create tables
            self._persistent_cache_db.execute('''
                CREATE TABLE IF NOT EXISTS impact_cache (
                    cache_key TEXT PRIMARY KEY,
                    analysis_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
            
            self._persistent_cache_db.execute('''
                CREATE TABLE IF NOT EXISTS path_cache (
                    cache_key TEXT PRIMARY KEY,
                    path_data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self._persistent_cache_db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to initialize persistent cache: {e}")
            self._persistent_cache_db = None
    
    def _get_from_persistent_cache(self, cache_key: str, table: str = "impact_cache") -> Optional[Any]:
        """Get item from persistent cache."""
        if not self._persistent_cache_db:
            return None
        
        try:
            cursor = self._persistent_cache_db.cursor()
            if table == "impact_cache":
                cursor.execute(
                    "SELECT analysis_data FROM impact_cache WHERE cache_key = ? AND expires_at > datetime('now')",
                    (cache_key,)
                )
            else:
                cursor.execute(
                    "SELECT path_data FROM path_cache WHERE cache_key = ?",
                    (cache_key,)
                )
            
            row = cursor.fetchone()
            if row:
                data = pickle.loads(gzip.decompress(row[0]) if self.config.cache_compression else row[0])
                return data
                
        except Exception as e:
            logger.warning(f"Failed to retrieve from persistent cache: {e}")
        
        return None
    
    def _save_to_persistent_cache(self, cache_key: str, data: Any, table: str = "impact_cache"):
        """Save item to persistent cache."""
        if not self._persistent_cache_db:
            return
        
        try:
            serialized_data = pickle.dumps(data)
            if self.config.cache_compression:
                serialized_data = gzip.compress(serialized_data)
            
            cursor = self._persistent_cache_db.cursor()
            if table == "impact_cache":
                expires_at = datetime.now() + timedelta(hours=self.config.cache_ttl_hours)
                cursor.execute(
                    "INSERT OR REPLACE INTO impact_cache (cache_key, analysis_data, expires_at) VALUES (?, ?, ?)",
                    (cache_key, serialized_data, expires_at)
                )
            else:
                cursor.execute(
                    "INSERT OR REPLACE INTO path_cache (cache_key, path_data) VALUES (?, ?)",
                    (cache_key, serialized_data)
                )
            
            self._persistent_cache_db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to save to persistent cache: {e}")
    
    def _batch_process_nodes(self, nodes: List[LineageNode], processor_func, batch_size: int = None) -> List[Any]:
        """Process nodes in batches for performance."""
        batch_size = batch_size or self.config.batch_size
        results = []
        
        if self._thread_pool and self.config.enable_parallel_processing:
            # Parallel processing
            futures = []
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                future = self._thread_pool.submit(processor_func, batch)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
        else:
            # Sequential processing
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                try:
                    batch_results = processor_func(batch)
                    results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
        
        return results
    
    def _initialize_propagation_rules(self) -> List[ImpactPropagationRule]:
        """Initialize default impact propagation rules."""
        return [
            # Data source to dataset
            ImpactPropagationRule(
                source_node_type=LineageNodeType.DATA_SOURCE,
                target_node_type=LineageNodeType.DATASET,
                relationship_type="feeds",
                impact_factor=0.9,
                impact_type=ImpactType.DIRECT
            ),
            # Dataset to column
            ImpactPropagationRule(
                source_node_type=LineageNodeType.DATASET,
                target_node_type=LineageNodeType.COLUMN,
                relationship_type="contains",
                impact_factor=0.8,
                impact_type=ImpactType.DIRECT
            ),
            # Column to transformation
            ImpactPropagationRule(
                source_node_type=LineageNodeType.COLUMN,
                target_node_type=LineageNodeType.TRANSFORMATION,
                relationship_type="input_to",
                impact_factor=0.7,
                impact_type=ImpactType.CASCADING
            ),
            # Transformation to dataset
            ImpactPropagationRule(
                source_node_type=LineageNodeType.TRANSFORMATION,
                target_node_type=LineageNodeType.DATASET,
                relationship_type="produces",
                impact_factor=0.8,
                impact_type=ImpactType.CASCADING
            ),
            # Dataset to downstream system
            ImpactPropagationRule(
                source_node_type=LineageNodeType.DATASET,
                target_node_type=LineageNodeType.DOWNSTREAM_SYSTEM,
                relationship_type="consumed_by",
                impact_factor=0.6,
                impact_type=ImpactType.INDIRECT
            ),
            # Dataset to ML model
            ImpactPropagationRule(
                source_node_type=LineageNodeType.DATASET,
                target_node_type=LineageNodeType.ML_MODEL,
                relationship_type="trains",
                impact_factor=0.9,
                impact_type=ImpactType.AMPLIFIED
            ),
            # Dataset to report/dashboard
            ImpactPropagationRule(
                source_node_type=LineageNodeType.DATASET,
                target_node_type=LineageNodeType.REPORT,
                relationship_type="feeds",
                impact_factor=0.7,
                impact_type=ImpactType.DIRECT
            ),
            ImpactPropagationRule(
                source_node_type=LineageNodeType.DATASET,
                target_node_type=LineageNodeType.DASHBOARD,
                relationship_type="feeds",
                impact_factor=0.7,
                impact_type=ImpactType.DIRECT
            )
        ]
    
    def create_lineage_graph(self, 
                           graph_name: str = None,
                           description: str = None) -> QualityLineageGraph:
        """Create a new quality lineage graph.
        
        Args:
            graph_name: Name for the graph
            description: Description of the graph
            
        Returns:
            New quality lineage graph
        """
        graph = QualityLineageGraph(
            name=graph_name or "Quality Lineage Graph",
            description=description or "Graph representing quality relationships and dependencies"
        )
        
        self._lineage_graphs[graph.graph_id] = graph
        logger.info(f"Created lineage graph: {graph.graph_id}")
        
        return graph
    
    def add_dataset_node(self,
                        graph: QualityLineageGraph,
                        dataset_id: str,
                        dataset_name: str,
                        quality_profile: Optional[DataQualityProfile] = None,
                        **kwargs) -> LineageNode:
        """Add a dataset node to the lineage graph.
        
        Args:
            graph: Target lineage graph
            dataset_id: Dataset identifier
            dataset_name: Dataset name
            quality_profile: Quality profile for the dataset
            **kwargs: Additional node attributes
            
        Returns:
            Created lineage node
        """
        node = LineageNode(
            node_id=LineageNodeId(),
            node_type=LineageNodeType.DATASET,
            name=dataset_name,
            description=f"Dataset: {dataset_name}",
            attributes={
                'dataset_id': dataset_id,
                **kwargs
            }
        )
        
        if quality_profile:
            node.quality_score = quality_profile.overall_quality_score
            node.update_attribute('row_count', quality_profile.row_count)
            node.update_attribute('column_count', quality_profile.column_count)
        
        graph.add_node(node)
        logger.debug(f"Added dataset node: {dataset_name}")
        
        return node
    
    def add_column_node(self,
                       graph: QualityLineageGraph,
                       dataset_node: LineageNode,
                       column_name: str,
                       column_type: str,
                       quality_score: Optional[float] = None,
                       **kwargs) -> LineageNode:
        """Add a column node to the lineage graph.
        
        Args:
            graph: Target lineage graph
            dataset_node: Parent dataset node
            column_name: Column name
            column_type: Column data type
            quality_score: Quality score for the column
            **kwargs: Additional node attributes
            
        Returns:
            Created lineage node
        """
        node = LineageNode(
            node_id=LineageNodeId(),
            node_type=LineageNodeType.COLUMN,
            name=column_name,
            description=f"Column: {column_name}",
            quality_score=quality_score,
            attributes={
                'column_name': column_name,
                'column_type': column_type,
                'parent_dataset': str(dataset_node.node_id),
                **kwargs
            }
        )
        
        graph.add_node(node)
        
        # Create relationship edge
        edge = LineageEdge(
            edge_id=LineageEdgeId(),
            source_node_id=dataset_node.node_id,
            target_node_id=node.node_id,
            relationship_type="contains",
            impact_type=ImpactType.DIRECT,
            quality_impact_factor=0.8
        )
        
        graph.add_edge(edge)
        logger.debug(f"Added column node: {column_name}")
        
        return node
    
    def add_transformation_node(self,
                              graph: QualityLineageGraph,
                              transformation_name: str,
                              transformation_type: str,
                              input_nodes: List[LineageNode],
                              output_nodes: List[LineageNode],
                              **kwargs) -> LineageNode:
        """Add a transformation node to the lineage graph.
        
        Args:
            graph: Target lineage graph
            transformation_name: Transformation name
            transformation_type: Type of transformation
            input_nodes: Input nodes to the transformation
            output_nodes: Output nodes from the transformation
            **kwargs: Additional node attributes
            
        Returns:
            Created lineage node
        """
        node = LineageNode(
            node_id=LineageNodeId(),
            node_type=LineageNodeType.TRANSFORMATION,
            name=transformation_name,
            description=f"Transformation: {transformation_name}",
            attributes={
                'transformation_type': transformation_type,
                'input_count': len(input_nodes),
                'output_count': len(output_nodes),
                **kwargs
            }
        )
        
        graph.add_node(node)
        
        # Create input edges
        for input_node in input_nodes:
            edge = LineageEdge(
                edge_id=LineageEdgeId(),
                source_node_id=input_node.node_id,
                target_node_id=node.node_id,
                relationship_type="input_to",
                impact_type=ImpactType.CASCADING,
                quality_impact_factor=0.7
            )
            graph.add_edge(edge)
        
        # Create output edges
        for output_node in output_nodes:
            edge = LineageEdge(
                edge_id=LineageEdgeId(),
                source_node_id=node.node_id,
                target_node_id=output_node.node_id,
                relationship_type="produces",
                impact_type=ImpactType.CASCADING,
                quality_impact_factor=0.8
            )
            graph.add_edge(edge)
        
        logger.debug(f"Added transformation node: {transformation_name}")
        return node
    
    def add_downstream_system_node(self,
                                 graph: QualityLineageGraph,
                                 system_name: str,
                                 system_type: str,
                                 source_datasets: List[LineageNode],
                                 **kwargs) -> LineageNode:
        """Add a downstream system node to the lineage graph.
        
        Args:
            graph: Target lineage graph
            system_name: System name
            system_type: Type of system (API, ML model, dashboard, etc.)
            source_datasets: Dataset nodes that feed this system
            **kwargs: Additional node attributes
            
        Returns:
            Created lineage node
        """
        # Determine node type based on system type
        node_type_mapping = {
            'ml_model': LineageNodeType.ML_MODEL,
            'api': LineageNodeType.API_ENDPOINT,
            'dashboard': LineageNodeType.DASHBOARD,
            'report': LineageNodeType.REPORT,
            'system': LineageNodeType.DOWNSTREAM_SYSTEM
        }
        
        node_type = node_type_mapping.get(system_type.lower(), LineageNodeType.DOWNSTREAM_SYSTEM)
        
        node = LineageNode(
            node_id=LineageNodeId(),
            node_type=node_type,
            name=system_name,
            description=f"{system_type}: {system_name}",
            attributes={
                'system_type': system_type,
                'source_count': len(source_datasets),
                **kwargs
            }
        )
        
        graph.add_node(node)
        
        # Create dependency edges
        for source_dataset in source_datasets:
            # Determine impact factor based on system type
            impact_factor = 0.9 if system_type.lower() == 'ml_model' else 0.7
            impact_type = ImpactType.AMPLIFIED if system_type.lower() == 'ml_model' else ImpactType.INDIRECT
            
            edge = LineageEdge(
                edge_id=LineageEdgeId(),
                source_node_id=source_dataset.node_id,
                target_node_id=node.node_id,
                relationship_type="consumed_by",
                impact_type=impact_type,
                quality_impact_factor=impact_factor
            )
            graph.add_edge(edge)
        
        logger.debug(f"Added downstream system node: {system_name}")
        return node
    
    def analyze_quality_impact(self,
                             graph: QualityLineageGraph,
                             root_node_id: LineageNodeId,
                             impact_scope: str = "downstream",
                             quality_issue: Optional[QualityAnomaly] = None) -> QualityImpactAnalysis:
        """Analyze quality impact propagation from a root node.
        
        Args:
            graph: Lineage graph to analyze
            root_node_id: Starting node for impact analysis
            impact_scope: Scope of analysis (upstream, downstream, both)
            quality_issue: Optional quality issue to analyze
            
        Returns:
            Quality impact analysis results
        """
        start_time = time.time()
        
        # Check cache if enabled (both in-memory and persistent)
        cache_key = f"{graph.graph_id}:{root_node_id}:{impact_scope}"
        
        # Check in-memory cache first
        if self.config.enable_caching:
            with self._cache_lock:
                if cache_key in self._impact_cache:
                    cached_analysis = self._impact_cache[cache_key]
                    cache_age = datetime.now() - cached_analysis.analysis_date
                    if cache_age.total_seconds() < (self.config.cache_ttl_hours * 3600):
                        logger.debug(f"Returning cached impact analysis: {cache_key}")
                        return cached_analysis
        
        # Check persistent cache
        if self.config.enable_persistent_cache:
            cached_analysis = self._get_from_persistent_cache(cache_key)
            if cached_analysis:
                logger.debug(f"Returning persistent cached impact analysis: {cache_key}")
                with self._cache_lock:
                    self._impact_cache[cache_key] = cached_analysis
                return cached_analysis
        
        # Validate graph size for performance
        if len(graph.nodes) > self.config.max_nodes_per_analysis:
            logger.warning(f"Graph has {len(graph.nodes)} nodes, exceeding max of {self.config.max_nodes_per_analysis}")
            return self._analyze_quality_impact_large_graph(graph, root_node_id, impact_scope, quality_issue)
        
        analysis = QualityImpactAnalysis(
            root_node_id=root_node_id,
            impact_scope=impact_scope
        )
        
        root_node = graph.get_node(root_node_id)
        if not root_node:
            raise ValueError(f"Root node not found: {root_node_id}")
        
        # Track visited nodes to avoid cycles
        visited_nodes = set()
        nodes_to_analyze = 0
        
        if impact_scope in ["downstream", "both"]:
            nodes_to_analyze += self._analyze_downstream_impact(
                graph, root_node, analysis, visited_nodes, quality_issue
            )
        
        if impact_scope in ["upstream", "both"]:
            nodes_to_analyze += self._analyze_upstream_impact(
                graph, root_node, analysis, visited_nodes, quality_issue
            )
        
        # Calculate analysis summary
        analysis.total_nodes_analyzed = nodes_to_analyze
        analysis.analysis_duration_seconds = time.time() - start_time
        analysis.calculate_summary()
        
        # Cache results if enabled
        if self.config.enable_caching:
            with self._cache_lock:
                self._impact_cache[cache_key] = analysis
        
        if self.config.enable_persistent_cache:
            self._save_to_persistent_cache(cache_key, analysis)
        
        logger.info(f"Completed impact analysis for {root_node.name}: "
                   f"{len(analysis.impacted_nodes)} impacted nodes in {analysis.analysis_duration_seconds:.2f}s")
        
        return analysis
    
    def _analyze_quality_impact_large_graph(self,
                                           graph: QualityLineageGraph,
                                           root_node_id: LineageNodeId,
                                           impact_scope: str,
                                           quality_issue: Optional[QualityAnomaly] = None) -> QualityImpactAnalysis:
        """Analyze quality impact for large graphs using optimized algorithms."""
        start_time = time.time()
        
        analysis = QualityImpactAnalysis(
            root_node_id=root_node_id,
            impact_scope=impact_scope
        )
        
        root_node = graph.get_node(root_node_id)
        if not root_node:
            raise ValueError(f"Root node not found: {root_node_id}")
        
        # Use breadth-first search with limited depth for large graphs
        visited_nodes = set()
        nodes_to_process = deque([(root_node, 0, 1.0)])  # (node, depth, impact_score)
        
        while nodes_to_process and len(analysis.impacted_nodes) < self.config.max_nodes_per_analysis:
            current_node, depth, current_impact = nodes_to_process.popleft()
            
            if depth >= self.config.max_propagation_depth:
                continue
            
            current_node_str = str(current_node.node_id)
            if current_node_str in visited_nodes:
                continue
            
            visited_nodes.add(current_node_str)
            
            # Process current node
            if current_node != root_node and current_impact >= self.config.min_impact_threshold:
                impact_severity = self._calculate_impact_severity(current_impact, current_node)
                analysis.add_impacted_node(
                    node_id=current_node.node_id,
                    node_name=current_node.name,
                    impact_severity=impact_severity,
                    impact_type=ImpactType.CASCADING,
                    impact_score=current_impact,
                    propagation_path=[root_node_id, current_node.node_id]
                )
            
            # Add neighbors to queue
            if impact_scope in ["downstream", "both"]:
                for downstream_node in graph.get_downstream_nodes(current_node.node_id):
                    if str(downstream_node.node_id) not in visited_nodes:
                        propagated_impact = current_impact * 0.8  # Simplified propagation
                        nodes_to_process.append((downstream_node, depth + 1, propagated_impact))
            
            if impact_scope in ["upstream", "both"]:
                for upstream_node in graph.get_upstream_nodes(current_node.node_id):
                    if str(upstream_node.node_id) not in visited_nodes:
                        propagated_impact = current_impact * 0.8  # Simplified propagation
                        nodes_to_process.append((upstream_node, depth + 1, propagated_impact))
        
        # Calculate analysis summary
        analysis.total_nodes_analyzed = len(visited_nodes)
        analysis.analysis_duration_seconds = time.time() - start_time
        analysis.calculate_summary()
        
        logger.info(f"Completed large graph impact analysis for {root_node.name}: "
                   f"{len(analysis.impacted_nodes)} impacted nodes in {analysis.analysis_duration_seconds:.2f}s")
        
        return analysis
    
    def _analyze_downstream_impact(self,
                                 graph: QualityLineageGraph,
                                 root_node: LineageNode,
                                 analysis: QualityImpactAnalysis,
                                 visited_nodes: Set[str],
                                 quality_issue: Optional[QualityAnomaly],
                                 current_depth: int = 0,
                                 current_impact_score: float = 1.0) -> int:
        """Analyze downstream impact propagation."""
        if current_depth >= self.config.max_propagation_depth:
            return 0
        
        root_node_str = str(root_node.node_id)
        if root_node_str in visited_nodes:
            return 0
        
        visited_nodes.add(root_node_str)
        nodes_analyzed = 1
        
        # Get downstream nodes
        downstream_nodes = graph.get_downstream_nodes(root_node.node_id)
        outgoing_edges = graph.get_outgoing_edges(root_node.node_id)
        
        for i, downstream_node in enumerate(downstream_nodes):
            if i < len(outgoing_edges):
                edge = outgoing_edges[i]
                
                # Calculate impact score for this node
                propagated_score = self._calculate_propagated_impact(
                    current_impact_score, edge, root_node, downstream_node
                )
                
                if propagated_score >= self.config.min_impact_threshold:
                    # Determine impact severity
                    impact_severity = self._calculate_impact_severity(propagated_score, downstream_node)
                    
                    # Create propagation path
                    propagation_path = analysis.propagation_paths[-1] if analysis.propagation_paths else [root_node.node_id]
                    propagation_path = propagation_path + [downstream_node.node_id]
                    
                    # Add to analysis
                    analysis.add_impacted_node(
                        node_id=downstream_node.node_id,
                        node_name=downstream_node.name,
                        impact_severity=impact_severity,
                        impact_type=edge.impact_type,
                        impact_score=propagated_score,
                        propagation_path=propagation_path
                    )
                    
                    # Store propagation path
                    if propagation_path not in analysis.propagation_paths:
                        analysis.propagation_paths.append(propagation_path)
                    
                    # Recursively analyze downstream
                    nodes_analyzed += self._analyze_downstream_impact(
                        graph, downstream_node, analysis, visited_nodes,
                        quality_issue, current_depth + 1, propagated_score
                    )
        
        return nodes_analyzed
    
    def _analyze_upstream_impact(self,
                               graph: QualityLineageGraph,
                               root_node: LineageNode,
                               analysis: QualityImpactAnalysis,
                               visited_nodes: Set[str],
                               quality_issue: Optional[QualityAnomaly],
                               current_depth: int = 0,
                               current_impact_score: float = 1.0) -> int:
        """Analyze upstream impact propagation."""
        if current_depth >= self.config.max_propagation_depth:
            return 0
        
        root_node_str = str(root_node.node_id)
        if root_node_str in visited_nodes:
            return 0
        
        visited_nodes.add(root_node_str)
        nodes_analyzed = 1
        
        # Get upstream nodes
        upstream_nodes = graph.get_upstream_nodes(root_node.node_id)
        incoming_edges = graph.get_incoming_edges(root_node.node_id)
        
        for i, upstream_node in enumerate(upstream_nodes):
            if i < len(incoming_edges):
                edge = incoming_edges[i]
                
                # Calculate impact score for this node
                propagated_score = self._calculate_propagated_impact(
                    current_impact_score, edge, upstream_node, root_node
                )
                
                if propagated_score >= self.config.min_impact_threshold:
                    # Determine impact severity
                    impact_severity = self._calculate_impact_severity(propagated_score, upstream_node)
                    
                    # Create propagation path (reverse for upstream)
                    propagation_path = [upstream_node.node_id, root_node.node_id]
                    
                    # Add to analysis
                    analysis.add_impacted_node(
                        node_id=upstream_node.node_id,
                        node_name=upstream_node.name,
                        impact_severity=impact_severity,
                        impact_type=edge.impact_type,
                        impact_score=propagated_score,
                        propagation_path=propagation_path
                    )
                    
                    # Store propagation path
                    if propagation_path not in analysis.propagation_paths:
                        analysis.propagation_paths.append(propagation_path)
                    
                    # Recursively analyze upstream
                    nodes_analyzed += self._analyze_upstream_impact(
                        graph, upstream_node, analysis, visited_nodes,
                        quality_issue, current_depth + 1, propagated_score
                    )
        
        return nodes_analyzed
    
    def _calculate_propagated_impact(self,
                                   current_score: float,
                                   edge: LineageEdge,
                                   source_node: LineageNode,
                                   target_node: LineageNode) -> float:
        """Calculate how impact propagates through an edge."""
        base_impact = current_score * edge.quality_impact_factor
        
        # Apply impact type modifiers
        if edge.impact_type == ImpactType.AMPLIFIED:
            base_impact *= self.config.amplification_factor
        elif edge.impact_type == ImpactType.MITIGATED:
            base_impact *= self.config.mitigation_factor
        elif edge.impact_type == ImpactType.CASCADING:
            base_impact *= self.config.propagation_decay_factor
        
        # Consider edge strength and confidence
        base_impact *= edge.strength * edge.confidence
        
        # Consider target node quality score
        if target_node.quality_score is not None:
            # Lower quality nodes amplify impact
            quality_modifier = 2.0 - target_node.quality_score
            base_impact *= quality_modifier
        
        return min(base_impact, 1.0)  # Cap at 1.0
    
    def _calculate_impact_severity(self,
                                 impact_score: float,
                                 node: LineageNode) -> ImpactSeverity:
        """Calculate impact severity based on score and node characteristics."""
        # Base severity from impact score
        if impact_score >= 0.8:
            base_severity = ImpactSeverity.CRITICAL
        elif impact_score >= 0.6:
            base_severity = ImpactSeverity.HIGH
        elif impact_score >= 0.4:
            base_severity = ImpactSeverity.MEDIUM
        elif impact_score >= 0.2:
            base_severity = ImpactSeverity.LOW
        else:
            base_severity = ImpactSeverity.MINIMAL
        
        # Escalate severity for critical node types
        critical_node_types = {
            LineageNodeType.ML_MODEL,
            LineageNodeType.API_ENDPOINT,
            LineageNodeType.DOWNSTREAM_SYSTEM
        }
        
        if node.node_type in critical_node_types and base_severity != ImpactSeverity.MINIMAL:
            severity_escalation = {
                ImpactSeverity.LOW: ImpactSeverity.MEDIUM,
                ImpactSeverity.MEDIUM: ImpactSeverity.HIGH,
                ImpactSeverity.HIGH: ImpactSeverity.CRITICAL
            }
            base_severity = severity_escalation.get(base_severity, base_severity)
        
        return base_severity
    
    def find_quality_dependencies(self,
                                graph: QualityLineageGraph,
                                node_id: LineageNodeId,
                                dependency_type: str = "all") -> Dict[str, List[LineageNode]]:
        """Find quality dependencies for a node.
        
        Args:
            graph: Lineage graph to search
            node_id: Node to find dependencies for
            dependency_type: Type of dependencies (upstream, downstream, all)
            
        Returns:
            Dictionary of dependency relationships
        """
        dependencies = {
            'upstream': [],
            'downstream': [],
            'critical_dependencies': [],
            'weak_dependencies': []
        }
        
        node = graph.get_node(node_id)
        if not node:
            return dependencies
        
        if dependency_type in ["upstream", "all"]:
            upstream_nodes = graph.get_upstream_nodes(node_id)
            dependencies['upstream'] = upstream_nodes
            
            # Identify critical upstream dependencies
            for upstream_node in upstream_nodes:
                incoming_edges = graph.get_incoming_edges(node_id)
                for edge in incoming_edges:
                    if str(edge.source_node_id) == str(upstream_node.node_id):
                        if edge.quality_impact_factor >= 0.8:
                            dependencies['critical_dependencies'].append(upstream_node)
                        elif edge.quality_impact_factor <= 0.3:
                            dependencies['weak_dependencies'].append(upstream_node)
        
        if dependency_type in ["downstream", "all"]:
            downstream_nodes = graph.get_downstream_nodes(node_id)
            dependencies['downstream'] = downstream_nodes
        
        return dependencies
    
    def get_quality_lineage_summary(self, graph: QualityLineageGraph) -> Dict[str, Any]:
        """Get comprehensive summary of quality lineage graph.
        
        Args:
            graph: Lineage graph to summarize
            
        Returns:
            Comprehensive lineage summary
        """
        stats = graph.get_graph_statistics()
        
        # Calculate quality metrics
        total_quality_score = 0
        quality_scored_nodes = 0
        quality_issues = 0
        
        for node in graph.nodes.values():
            if node.quality_score is not None:
                total_quality_score += node.quality_score
                quality_scored_nodes += 1
            
            quality_issues += len(node.quality_issues)
        
        avg_quality = total_quality_score / quality_scored_nodes if quality_scored_nodes > 0 else 0
        
        # Find critical paths
        critical_paths = self._find_critical_paths(graph)
        
        # Find quality bottlenecks
        bottlenecks = self._find_quality_bottlenecks(graph)
        
        return {
            'graph_statistics': stats,
            'quality_metrics': {
                'average_quality_score': round(avg_quality, 3),
                'total_quality_issues': quality_issues,
                'quality_coverage': round(quality_scored_nodes / len(graph.nodes), 3) if graph.nodes else 0
            },
            'critical_paths': critical_paths,
            'quality_bottlenecks': bottlenecks,
            'risk_assessment': self._assess_lineage_risk(graph),
            'recommendations': self._generate_lineage_recommendations(graph)
        }
    
    def _find_critical_paths(self, graph: QualityLineageGraph) -> List[Dict[str, Any]]:
        """Find critical quality propagation paths in the graph."""
        critical_paths = []
        
        # Find data sources and ML models as critical endpoints
        sources = graph.get_nodes_by_type(LineageNodeType.DATA_SOURCE)
        ml_models = graph.get_nodes_by_type(LineageNodeType.ML_MODEL)
        
        for source in sources:
            for ml_model in ml_models:
                paths = graph.find_all_paths(source.node_id, ml_model.node_id, max_depth=8)
                
                for path in paths:
                    # Calculate path quality impact
                    path_impact = self._calculate_path_impact(graph, path)
                    
                    if path_impact >= 0.7:  # High impact paths
                        critical_paths.append({
                            'source': source.name,
                            'target': ml_model.name,
                            'path_length': len(path),
                            'impact_score': path_impact,
                            'path_nodes': [graph.get_node(node_id).name for node_id in path]
                        })
        
        # Sort by impact score
        critical_paths.sort(key=lambda x: x['impact_score'], reverse=True)
        return critical_paths[:10]  # Top 10 critical paths
    
    def _find_quality_bottlenecks(self, graph: QualityLineageGraph) -> List[Dict[str, Any]]:
        """Find quality bottlenecks in the lineage graph."""
        bottlenecks = []
        
        for node in graph.nodes.values():
            downstream_count = len(graph.get_downstream_nodes(node.node_id))
            
            # Nodes with high fan-out and low quality are bottlenecks
            if downstream_count >= 3 and node.quality_score is not None and node.quality_score < 0.7:
                bottlenecks.append({
                    'node_name': node.name,
                    'node_type': node.node_type.value,
                    'quality_score': node.quality_score,
                    'downstream_impact': downstream_count,
                    'risk_level': 'high' if node.quality_score < 0.5 else 'medium'
                })
        
        # Sort by risk level and downstream impact
        bottlenecks.sort(key=lambda x: (x['risk_level'] == 'high', x['downstream_impact']), reverse=True)
        return bottlenecks
    
    def _calculate_path_impact(self, graph: QualityLineageGraph, path: List[LineageNodeId]) -> float:
        """Calculate cumulative impact score for a path."""
        if len(path) < 2:
            return 0.0
        
        cumulative_impact = 1.0
        
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # Find edge between nodes
            source_node = graph.get_node(source_id)
            outgoing_edges = graph.get_outgoing_edges(source_id)
            
            edge_impact = 0.5  # Default if no specific edge found
            for edge in outgoing_edges:
                if edge.target_node_id == target_id:
                    edge_impact = edge.quality_impact_factor
                    break
            
            cumulative_impact *= edge_impact
            
            # Apply decay factor
            cumulative_impact *= self.config.propagation_decay_factor
        
        return cumulative_impact
    
    def _assess_lineage_risk(self, graph: QualityLineageGraph) -> Dict[str, Any]:
        """Assess overall risk in the lineage graph."""
        total_nodes = len(graph.nodes)
        if total_nodes == 0:
            return {'overall_risk': 'unknown', 'risk_score': 0}
        
        # Calculate risk factors
        low_quality_nodes = sum(1 for node in graph.nodes.values() 
                               if node.quality_score is not None and node.quality_score < 0.7)
        
        nodes_with_issues = sum(1 for node in graph.nodes.values() if node.quality_issues)
        
        critical_nodes = len(graph.get_nodes_by_type(LineageNodeType.ML_MODEL)) + \
                        len(graph.get_nodes_by_type(LineageNodeType.API_ENDPOINT))
        
        # Calculate risk score
        quality_risk = (low_quality_nodes / total_nodes) * 0.4
        issue_risk = (nodes_with_issues / total_nodes) * 0.3
        critical_exposure = min(critical_nodes / max(total_nodes, 1), 1.0) * 0.3
        
        risk_score = quality_risk + issue_risk + critical_exposure
        
        if risk_score >= 0.7:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'overall_risk': risk_level,
            'risk_score': round(risk_score, 3),
            'risk_factors': {
                'low_quality_nodes': low_quality_nodes,
                'nodes_with_issues': nodes_with_issues,
                'critical_node_exposure': critical_nodes
            }
        }
    
    def _generate_lineage_recommendations(self, graph: QualityLineageGraph) -> List[str]:
        """Generate recommendations for improving quality lineage."""
        recommendations = []
        
        # Check for bottlenecks
        bottlenecks = self._find_quality_bottlenecks(graph)
        if bottlenecks:
            recommendations.append(f"Address {len(bottlenecks)} quality bottlenecks to improve downstream impact")
        
        # Check quality coverage
        quality_scored = sum(1 for node in graph.nodes.values() if node.quality_score is not None)
        coverage = quality_scored / len(graph.nodes) if graph.nodes else 0
        if coverage < 0.8:
            recommendations.append("Increase quality monitoring coverage - many nodes lack quality scores")
        
        # Check for single points of failure
        critical_singles = 0
        for node in graph.nodes.values():
            upstream_count = len(graph.get_upstream_nodes(node.node_id))
            downstream_count = len(graph.get_downstream_nodes(node.node_id))
            
            if upstream_count == 1 and downstream_count >= 3:
                critical_singles += 1
        
        if critical_singles > 0:
            recommendations.append(f"Consider redundancy for {critical_singles} single points of failure")
        
        # Check for orphaned nodes
        orphaned = sum(1 for node in graph.nodes.values() 
                      if len(graph.get_upstream_nodes(node.node_id)) == 0 and 
                         len(graph.get_downstream_nodes(node.node_id)) == 0)
        
        if orphaned > 0:
            recommendations.append(f"Review {orphaned} orphaned nodes that may indicate missing relationships")
        
        return recommendations[:self.config.max_recommendations]
    
    def perform_root_cause_analysis(self,
                                   graph: QualityLineageGraph,
                                   quality_issue_node_id: LineageNodeId,
                                   analysis_depth: int = 5) -> Dict[str, Any]:
        """Perform advanced root cause analysis for quality issues.
        
        Args:
            graph: Lineage graph to analyze
            quality_issue_node_id: Node where quality issue was detected
            analysis_depth: Maximum depth to search for root causes
            
        Returns:
            Root cause analysis results
        """
        start_time = time.time()
        
        issue_node = graph.get_node(quality_issue_node_id)
        if not issue_node:
            raise ValueError(f"Quality issue node not found: {quality_issue_node_id}")
        
        # Initialize analysis results
        root_causes = []
        contributing_factors = []
        confidence_scores = {}
        
        # Perform backward traversal to find potential root causes
        visited_nodes = set()
        candidates = deque([(issue_node, 0, 1.0)])  # (node, depth, confidence)
        
        while candidates:
            current_node, depth, confidence = candidates.popleft()
            
            if depth >= analysis_depth:
                continue
            
            current_node_str = str(current_node.node_id)
            if current_node_str in visited_nodes:
                continue
            
            visited_nodes.add(current_node_str)
            
            # Analyze node for root cause potential
            root_cause_score = self._calculate_root_cause_score(current_node, issue_node, depth)
            
            if root_cause_score > 0.7:  # High confidence root cause
                root_causes.append({
                    'node_id': str(current_node.node_id),
                    'node_name': current_node.name,
                    'node_type': current_node.node_type.value,
                    'confidence_score': root_cause_score,
                    'distance_from_issue': depth,
                    'quality_score': current_node.quality_score,
                    'contributing_factors': self._identify_contributing_factors(current_node)
                })
            elif root_cause_score > 0.4:  # Contributing factor
                contributing_factors.append({
                    'node_id': str(current_node.node_id),
                    'node_name': current_node.name,
                    'node_type': current_node.node_type.value,
                    'contribution_score': root_cause_score,
                    'distance_from_issue': depth
                })
            
            # Add upstream nodes for further analysis
            for upstream_node in graph.get_upstream_nodes(current_node.node_id):
                if str(upstream_node.node_id) not in visited_nodes:
                    # Calculate propagated confidence
                    propagated_confidence = confidence * 0.9  # Slight decay
                    candidates.append((upstream_node, depth + 1, propagated_confidence))
        
        # Rank root causes by confidence
        root_causes.sort(key=lambda x: x['confidence_score'], reverse=True)
        contributing_factors.sort(key=lambda x: x['contribution_score'], reverse=True)
        
        # Generate recommendations
        recommendations = self._generate_root_cause_recommendations(root_causes, contributing_factors)
        
        analysis_duration = time.time() - start_time
        
        return {
            'issue_node': {
                'node_id': str(quality_issue_node_id),
                'node_name': issue_node.name,
                'node_type': issue_node.node_type.value,
                'quality_score': issue_node.quality_score,
                'quality_issues': issue_node.quality_issues
            },
            'root_causes': root_causes,
            'contributing_factors': contributing_factors,
            'analysis_summary': {
                'total_nodes_analyzed': len(visited_nodes),
                'root_causes_found': len(root_causes),
                'contributing_factors_found': len(contributing_factors),
                'analysis_duration_seconds': analysis_duration,
                'confidence_threshold': 0.7,
                'analysis_depth': analysis_depth
            },
            'recommendations': recommendations,
            'next_steps': self._generate_next_steps(root_causes, contributing_factors)
        }
    
    def _calculate_root_cause_score(self, candidate_node: LineageNode, issue_node: LineageNode, depth: int) -> float:
        """Calculate root cause confidence score for a candidate node."""
        score = 0.0
        
        # Distance penalty (closer nodes are more likely to be root causes)
        distance_score = max(0, 1.0 - (depth * 0.15))
        score += distance_score * 0.3
        
        # Quality score factor (lower quality = higher root cause probability)
        if candidate_node.quality_score is not None:
            quality_factor = 1.0 - candidate_node.quality_score
            score += quality_factor * 0.4
        
        # Node type factor (data sources and transformations are more likely root causes)
        type_weights = {
            LineageNodeType.DATA_SOURCE: 0.9,
            LineageNodeType.TRANSFORMATION: 0.8,
            LineageNodeType.DATASET: 0.6,
            LineageNodeType.COLUMN: 0.5,
            LineageNodeType.QUALITY_RULE: 0.7,
            LineageNodeType.API_ENDPOINT: 0.3,
            LineageNodeType.REPORT: 0.2,
            LineageNodeType.DASHBOARD: 0.1
        }
        
        type_weight = type_weights.get(candidate_node.node_type, 0.5)
        score += type_weight * 0.2
        
        # Quality issues factor
        if candidate_node.quality_issues:
            issues_factor = min(1.0, len(candidate_node.quality_issues) * 0.2)
            score += issues_factor * 0.1
        
        return min(score, 1.0)
    
    def _identify_contributing_factors(self, node: LineageNode) -> List[str]:
        """Identify contributing factors for a potential root cause."""
        factors = []
        
        if node.quality_score is not None and node.quality_score < 0.7:
            factors.append("Low quality score")
        
        if node.quality_issues:
            factors.append(f"Has {len(node.quality_issues)} quality issues")
        
        if node.node_type == LineageNodeType.DATA_SOURCE:
            factors.append("External data source - potential upstream issues")
        
        if node.node_type == LineageNodeType.TRANSFORMATION:
            factors.append("Data transformation - potential processing issues")
        
        # Check for recent updates
        if node.updated_at and (datetime.now() - node.updated_at).days <= 7:
            factors.append("Recent changes detected")
        
        return factors
    
    def _generate_root_cause_recommendations(self, root_causes: List[Dict], contributing_factors: List[Dict]) -> List[str]:
        """Generate recommendations based on root cause analysis."""
        recommendations = []
        
        if not root_causes:
            recommendations.append("No high-confidence root causes found - consider expanding analysis depth")
            return recommendations
        
        # Address top root causes
        for i, root_cause in enumerate(root_causes[:3]):  # Top 3 root causes
            node_name = root_cause['node_name']
            node_type = root_cause['node_type']
            
            if node_type == 'data_source':
                recommendations.append(f"Investigate data source '{node_name}' - validate data quality at origin")
            elif node_type == 'transformation':
                recommendations.append(f"Review transformation logic in '{node_name}' - check for processing errors")
            elif node_type == 'dataset':
                recommendations.append(f"Validate dataset '{node_name}' - run comprehensive quality checks")
            else:
                recommendations.append(f"Investigate '{node_name}' ({node_type}) - address quality issues")
        
        # Address contributing factors
        if contributing_factors:
            recommendations.append("Address contributing factors to prevent future issues")
        
        return recommendations
    
    def _generate_next_steps(self, root_causes: List[Dict], contributing_factors: List[Dict]) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        if root_causes:
            next_steps.append("Prioritize investigation of identified root causes")
            next_steps.append("Implement monitoring for root cause nodes")
            next_steps.append("Develop remediation plan for top root causes")
        
        if contributing_factors:
            next_steps.append("Monitor contributing factors for early warning")
        
        next_steps.extend([
            "Set up alerts for quality degradation in critical paths",
            "Consider implementing data quality gates",
            "Review and update quality rules based on findings"
        ])
        
        return next_steps
    
    def export_lineage_graph(self, 
                           graph: QualityLineageGraph,
                           format: str = "json",
                           include_analysis: bool = True) -> Dict[str, Any]:
        """Export lineage graph with optional analysis.
        
        Args:
            graph: Lineage graph to export
            format: Export format (json, graphml, etc.)
            include_analysis: Include analysis summary
            
        Returns:
            Exported graph data
        """
        export_data = graph.export_graph(format)
        
        if include_analysis:
            export_data['lineage_summary'] = self.get_quality_lineage_summary(graph)
        
        return export_data
    
    def get_lineage_graph(self, graph_id: str) -> Optional[QualityLineageGraph]:
        """Get lineage graph by ID.
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Lineage graph if found
        """
        return self._lineage_graphs.get(graph_id)
    
    def list_lineage_graphs(self) -> List[Dict[str, Any]]:
        """List all lineage graphs.
        
        Returns:
            List of graph summaries
        """
        return [
            {
                'graph_id': graph.graph_id,
                'name': graph.name,
                'description': graph.description,
                'node_count': len(graph.nodes),
                'edge_count': len(graph.edges),
                'created_at': graph.created_at.isoformat(),
                'updated_at': graph.updated_at.isoformat()
            }
            for graph in self._lineage_graphs.values()
        ]