"""Knowledge graph entity for semantic reasoning."""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
import uuid
import warnings
from pathlib import Path

try:
    from rdflib import Graph, URIRef, Literal, BNode, Namespace
    from rdflib.namespace import RDF, RDFS, OWL, XSD
    from rdflib.plugins.parsers.notation3 import BadSyntax
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    warnings.warn("RDFLib not available. Advanced knowledge graph features will be limited.")

from ...infrastructure.adapters.symbolic_adapter import RDFReasoner, LogicalRule, LogicType


@dataclass
class Triple:
    """Represents an RDF triple (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Triple):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object
        )
    
    def __hash__(self) -> int:
        return hash((self.subject, self.predicate, self.object))


@dataclass
class KnowledgeGraph:
    """
    Entity representing a knowledge graph for semantic reasoning.
    Contains semantic facts and rules for logical inference.
    """
    
    id: str
    name: str
    triples: List[Triple] = field(default_factory=list)
    rules: List[LogicalRule] = field(default_factory=list)
    namespaces: Dict[str, str] = field(default_factory=dict)
    _rdf_reasoner: Optional[RDFReasoner] = field(default=None, init=False)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Initialize RDF reasoner if available
        if HAS_RDFLIB:
            try:
                self._rdf_reasoner = RDFReasoner()
                self._initialize_default_namespaces()
                self._populate_rdf_graph()
            except Exception as e:
                warnings.warn(f"Failed to initialize RDF reasoner: {e}")
                self._rdf_reasoner = None
    
    def _initialize_default_namespaces(self):
        """Initialize default RDF namespaces."""
        if self._rdf_reasoner:
            default_namespaces = {
                'rdf': str(RDF),
                'rdfs': str(RDFS),
                'owl': str(OWL),
                'xsd': str(XSD),
                'ex': 'http://example.org/'
            }
            
            for prefix, uri in default_namespaces.items():
                if prefix not in self.namespaces:
                    self.namespaces[prefix] = uri
                self._rdf_reasoner.add_namespace(prefix, uri)
    
    def _populate_rdf_graph(self):
        """Populate the RDF graph with current triples."""
        if self._rdf_reasoner:
            for triple in self.triples:
                try:
                    self._rdf_reasoner.add_triple(
                        triple.subject,
                        triple.predicate,
                        triple.object
                    )
                except Exception as e:
                    warnings.warn(f"Failed to add triple to RDF graph: {e}")
    
    @classmethod
    def create(cls, name: str) -> "KnowledgeGraph":
        """Create a new knowledge graph."""
        return cls(
            id=str(uuid.uuid4()),
            name=name
        )
    
    @classmethod
    def from_file(cls, file_path: str) -> "KnowledgeGraph":
        """Load knowledge graph from OWL/RDF file."""
        kg = cls.create(f"KG from {Path(file_path).name}")
        
        if not HAS_RDFLIB:
            warnings.warn("RDFLib not available. Cannot load from file.")
            return kg
        
        try:
            # Load RDF file using rdflib
            temp_graph = Graph()
            temp_graph.parse(file_path)
            
            # Convert RDF graph to our Triple format
            for s, p, o in temp_graph:
                kg.add_triple(str(s), str(p), str(o))
            
            # Extract namespaces
            for prefix, namespace in temp_graph.namespaces():
                if prefix:
                    kg.add_namespace(str(prefix), str(namespace))
            
        except Exception as e:
            warnings.warn(f"Failed to load knowledge graph from {file_path}: {e}")
        
        return kg
    
    @classmethod
    def from_triples(cls, name: str, triples: List[Triple]) -> "KnowledgeGraph":
        """Create knowledge graph from a list of triples."""
        kg = cls.create(name)
        for triple in triples:
            kg.add_triple(triple.subject, triple.predicate, triple.object, triple.confidence)
        return kg
    
    def add_namespace(self, prefix: str, uri: str) -> None:
        """Add a namespace to the knowledge graph."""
        self.namespaces[prefix] = uri
        if self._rdf_reasoner:
            self._rdf_reasoner.add_namespace(prefix, uri)
    
    def add_triple(
        self, 
        subject: str, 
        predicate: str, 
        obj: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new triple to the knowledge graph."""
        triple = Triple(subject, predicate, obj, confidence, metadata)
        if triple not in self.triples:
            self.triples.append(triple)
            
            # Add to RDF reasoner if available
            if self._rdf_reasoner:
                try:
                    self._rdf_reasoner.add_triple(subject, predicate, obj)
                except Exception as e:
                    warnings.warn(f"Failed to add triple to RDF reasoner: {e}")
    
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a logical rule for inference."""
        self.rules.append(rule)
    
    def create_rule(
        self,
        rule_id: str,
        antecedent: str,
        consequent: str,
        confidence: float = 1.0,
        logic_type: LogicType = LogicType.FIRST_ORDER
    ) -> LogicalRule:
        """Create and add a logical rule."""
        rule = LogicalRule(
            id=rule_id,
            antecedent=antecedent,
            consequent=consequent,
            logic_type=logic_type,
            confidence=confidence
        )
        self.add_rule(rule)
        return rule
    
    def query(self, sparql_query: str) -> List[Dict[str, str]]:
        """Execute SPARQL query on the knowledge graph."""
        if not self._rdf_reasoner:
            warnings.warn("RDF reasoner not available. Cannot execute SPARQL query.")
            return []
        
        return self._rdf_reasoner.query(sparql_query)
    
    def get_entities(self) -> Set[str]:
        """Get all entities in the knowledge graph."""
        entities = set()
        for triple in self.triples:
            entities.add(triple.subject)
            entities.add(triple.object)
        return entities
    
    def get_predicates(self) -> Set[str]:
        """Get all predicates/relations in the knowledge graph."""
        return {triple.predicate for triple in self.triples}
    
    def get_triples_by_subject(self, subject: str) -> List[Triple]:
        """Get all triples with a specific subject."""
        return [t for t in self.triples if t.subject == subject]
    
    def get_triples_by_predicate(self, predicate: str) -> List[Triple]:
        """Get all triples with a specific predicate."""
        return [t for t in self.triples if t.predicate == predicate]
    
    def get_triples_by_object(self, obj: str) -> List[Triple]:
        """Get all triples with a specific object."""
        return [t for t in self.triples if t.object == obj]
    
    def find_paths(self, start: str, end: str, max_depth: int = 3) -> List[List[str]]:
        """Find paths between two entities."""
        paths = []
        
        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current == target and len(path) > 1:
                paths.append(path.copy())
                return
            
            # Find connected entities
            for triple in self.triples:
                next_entity = None
                if triple.subject == current:
                    next_entity = triple.object
                elif triple.object == current:
                    next_entity = triple.subject
                
                if next_entity and next_entity not in path:
                    path.append(next_entity)
                    dfs(next_entity, target, path, depth + 1)
                    path.pop()
        
        dfs(start, end, [start], 0)
        return paths
    
    def get_neighbors(self, entity: str) -> Dict[str, List[str]]:
        """Get all neighboring entities and their relationships."""
        neighbors = {'outgoing': [], 'incoming': []}
        
        for triple in self.triples:
            if triple.subject == entity:
                neighbors['outgoing'].append({
                    'predicate': triple.predicate,
                    'object': triple.object,
                    'confidence': triple.confidence
                })
            elif triple.object == entity:
                neighbors['incoming'].append({
                    'predicate': triple.predicate,
                    'subject': triple.subject,
                    'confidence': triple.confidence
                })
        
        return neighbors
    
    def reason(self) -> List[Triple]:
        """Apply inference rules to derive new facts."""
        new_triples = []
        
        # Apply RDFS inference if RDF reasoner is available
        if self._rdf_reasoner:
            try:
                self._rdf_reasoner.infer_rdfs()
                
                # Query for new inferred triples
                inferred_query = """
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                }
                """
                results = self.query(inferred_query)
                
                for result in results:
                    triple = Triple(
                        subject=result.get('s', ''),
                        predicate=result.get('p', ''),
                        object=result.get('o', ''),
                        confidence=0.8  # Lower confidence for inferred triples
                    )
                    if triple not in self.triples:
                        new_triples.append(triple)
                        
            except Exception as e:
                warnings.warn(f"RDF reasoning failed: {e}")
        
        # Apply custom logical rules
        for rule in self.rules:
            # Simplified rule application - would need proper unification
            for triple in self.triples:
                if rule.antecedent in f"{triple.subject} {triple.predicate} {triple.object}":
                    # Create new triple based on rule consequent
                    # This is highly simplified - real implementation would need parsing
                    parts = rule.consequent.split()
                    if len(parts) >= 3:
                        new_triple = Triple(
                            subject=parts[0],
                            predicate=parts[1], 
                            object=parts[2],
                            confidence=min(triple.confidence, rule.confidence)
                        )
                        if new_triple not in self.triples and new_triple not in new_triples:
                            new_triples.append(new_triple)
        
        return new_triples
    
    def merge_with(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Merge this knowledge graph with another."""
        merged = KnowledgeGraph.create(f"{self.name}_merged_with_{other.name}")
        
        # Merge triples
        all_triples = self.triples + other.triples
        unique_triples = list(set(all_triples))
        merged.triples = unique_triples
        
        # Merge rules
        merged.rules = self.rules + other.rules
        
        # Merge namespaces
        merged.namespaces.update(self.namespaces)
        merged.namespaces.update(other.namespaces)
        
        return merged
    
    def export_to_rdf(self, file_path: str, format: str = "turtle") -> bool:
        """Export knowledge graph to RDF file."""
        if not self._rdf_reasoner:
            warnings.warn("RDF reasoner not available. Cannot export to RDF.")
            return False
        
        try:
            self._rdf_reasoner.graph.serialize(destination=file_path, format=format)
            return True
        except Exception as e:
            warnings.warn(f"Failed to export to RDF: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            'num_triples': len(self.triples),
            'num_entities': len(self.get_entities()),
            'num_predicates': len(self.get_predicates()),
            'num_rules': len(self.rules),
            'num_namespaces': len(self.namespaces),
            'avg_confidence': sum(t.confidence for t in self.triples) / len(self.triples) if self.triples else 0.0
        }