# Knowledge Graph Package

A comprehensive knowledge graph and semantic data modeling platform for representing and querying complex relationships in data.

## Overview

The Knowledge Graph package provides advanced capabilities for building, managing, and querying knowledge graphs. It enables semantic data modeling, relationship discovery, and intelligent data connections across the platform.

## Features

- **Graph Database Integration**: Support for multiple graph databases (Neo4j, Amazon Neptune, ArangoDB)
- **Semantic Modeling**: RDF/OWL-based semantic data modeling and ontology management
- **Entity Resolution**: Automatic entity deduplication and relationship discovery
- **Graph Analytics**: Advanced graph algorithms for pattern detection and insights
- **SPARQL Queries**: Powerful semantic query capabilities
- **Knowledge Extraction**: Automatic knowledge extraction from unstructured data
- **Graph Visualization**: Interactive graph visualization and exploration tools
- **Schema Management**: Flexible schema definition and evolution

## Architecture

```
knowledge_graph/
├── domain/                 # Core knowledge graph business logic
│   ├── entities/          # Graph, Node, Edge, Ontology entities
│   ├── services/          # Graph management and query services
│   └── value_objects/     # Graph-specific value objects
├── application/           # Use cases and orchestration  
│   ├── services/          # Application services
│   ├── use_cases/         # Knowledge graph workflows
│   └── dto/               # Data transfer objects
├── infrastructure/        # External integrations
│   ├── repositories/      # Graph database implementations
│   ├── adapters/          # Database-specific adapters
│   └── parsers/           # Data format parsers (RDF, JSON-LD)
└── presentation/          # Interfaces
    ├── api/               # REST and GraphQL APIs
    ├── query/             # SPARQL endpoint
    └── visualization/     # Graph visualization components
```

## Quick Start

```python
from src.packages.data.knowledge_graph.application.services import KnowledgeGraphService
from src.packages.data.knowledge_graph.domain.entities import Graph, Node, Edge

# Initialize knowledge graph service
kg_service = KnowledgeGraphService()

# Create a new knowledge graph
graph = kg_service.create_graph(
    name="Customer Knowledge Graph",
    description="Comprehensive customer and product relationships"
)

# Add entities and relationships
customer = kg_service.add_entity(
    graph_id=graph.id,
    entity_type="Customer",
    properties={
        "name": "John Doe",
        "email": "john.doe@example.com",
        "segment": "premium"
    }
)

product = kg_service.add_entity(
    graph_id=graph.id,
    entity_type="Product",
    properties={
        "name": "Premium Analytics Suite",
        "category": "software",
        "price": 299.99
    }
)

# Create relationship
relationship = kg_service.add_relationship(
    graph_id=graph.id,
    source_id=customer.id,
    target_id=product.id,
    relationship_type="PURCHASED",
    properties={
        "date": "2024-01-15",
        "amount": 299.99,
        "channel": "online"
    }
)

# Query the graph
results = kg_service.query(
    graph_id=graph.id,
    query="""
    MATCH (c:Customer)-[p:PURCHASED]->(prod:Product)
    WHERE c.segment = 'premium'
    RETURN c.name, prod.name, p.amount
    """
)

# Discover patterns
patterns = kg_service.discover_patterns(
    graph_id=graph.id,
    pattern_type="customer_product_affinity"
)
```

## Core Capabilities

### Semantic Modeling
```python
# Define ontology
ontology = kg_service.create_ontology(
    name="ecommerce_ontology",
    namespaces={
        "ecom": "http://example.com/ecommerce/",
        "foaf": "http://xmlns.com/foaf/0.1/"
    }
)

# Define entity classes
kg_service.define_entity_class(
    ontology_id=ontology.id,
    class_name="Customer",
    parent_class="foaf:Person",
    properties={
        "hasEmail": {"type": "string", "required": True},
        "hasSegment": {"type": "enum", "values": ["basic", "premium", "enterprise"]}
    }
)
```

### Entity Resolution
```python
# Configure entity resolution
resolver = kg_service.get_entity_resolver()
resolver.configure_matching_rules(
    entity_type="Customer",
    rules=[
        {"field": "email", "weight": 0.8, "threshold": 0.9},
        {"field": "name", "weight": 0.6, "threshold": 0.7, "fuzzy": True}
    ]
)

# Resolve duplicate entities
duplicates = resolver.find_duplicates(graph_id=graph.id)
resolved_entities = resolver.merge_entities(duplicates)
```

### Graph Analytics
```python
# Run graph algorithms
centrality = kg_service.calculate_centrality(
    graph_id=graph.id,
    algorithm="pagerank",
    entity_types=["Customer", "Product"]
)

communities = kg_service.detect_communities(
    graph_id=graph.id,
    algorithm="louvain"
)

# Path finding
paths = kg_service.find_paths(
    graph_id=graph.id,
    start_entity=customer.id,
    end_entity=product.id,
    max_depth=3
)
```

## Use Cases

- **Customer 360**: Complete customer view with all relationships and interactions
- **Product Recommendations**: Graph-based recommendation systems
- **Fraud Detection**: Pattern analysis for fraudulent behavior detection
- **Supply Chain Analysis**: Complex supply chain relationship modeling
- **Compliance Mapping**: Regulatory compliance and audit trail management
- **Research & Discovery**: Scientific and research data relationship modeling

## Integration

Works seamlessly with other data domain packages:

```python
# With data quality for graph validation
from src.packages.data.data_quality.application.services import DataQualityService

quality_service = DataQualityService()
graph_quality = quality_service.assess_graph_quality(graph_id)

# With data observability for lineage tracking
from src.packages.data.observability.application.services import LineageService

lineage_service = LineageService()
lineage_service.track_graph_transformation(source_graph, target_graph)
```

## Installation

```bash
# Install from package directory
cd src/packages/data/knowledge_graph
pip install -e .

# Install with graph database support
pip install -e ".[neo4j,neptune,arango]"
```

## Configuration

```yaml
# knowledge_graph_config.yaml
knowledge_graph:
  database:
    provider: "neo4j"  # neo4j, neptune, arango
    connection:
      host: "localhost"
      port: 7687
      username: "neo4j"
      password: "${NEO4J_PASSWORD}"
  
  ontology:
    auto_inference: true
    validation: "strict"
    namespaces:
      default: "http://example.com/kg/"
  
  entity_resolution:
    auto_merge_threshold: 0.95
    manual_review_threshold: 0.80
    batch_processing: true
  
  analytics:
    algorithms: ["pagerank", "betweenness", "louvain"]
    cache_results: true
    parallel_processing: true
```

## Performance

Optimized for large-scale graph operations:

- **Indexed Queries**: Automatic index optimization for common query patterns
- **Parallel Processing**: Multi-threaded graph algorithm execution
- **Caching**: Intelligent caching of frequently accessed subgraphs
- **Partitioning**: Graph partitioning for distributed processing
- **Streaming**: Streaming ingestion for large datasets

## License

MIT License
