"""Symbolic reasoning adapters for neuro-symbolic models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import warnings
from dataclasses import dataclass
from enum import Enum

try:
    import sympy as sp
    from sympy.logic.boolalg import BooleanFunction
    from sympy import symbols, sympify, solve, simplify
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    warnings.warn("SymPy not available. Symbolic reasoning features will be limited.")

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    warnings.warn("Z3 solver not available. SMT solving features will be disabled.")

try:
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL
    import rdflib.plugins.sparql as sparql
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    warnings.warn("RDFLib not available. RDF/SPARQL features will be disabled.")


class LogicType(Enum):
    """Types of logical reasoning systems."""
    PROPOSITIONAL = "propositional"
    FIRST_ORDER = "first_order"
    TEMPORAL = "temporal"
    FUZZY = "fuzzy"
    MODAL = "modal"


@dataclass
class LogicalRule:
    """Represents a logical rule for symbolic reasoning."""
    id: str
    antecedent: str  # Premise/condition
    consequent: str  # Conclusion
    logic_type: LogicType
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self) -> str:
        return f"{self.antecedent} → {self.consequent} (confidence: {self.confidence})"


@dataclass
class InferenceResult:
    """Result of symbolic inference."""
    conclusion: str
    premises: List[str]
    reasoning_chain: List[str]
    confidence: float
    logic_type: LogicType
    metadata: Optional[Dict[str, Any]] = None


class SymbolicReasoner(ABC):
    """Abstract base class for symbolic reasoning engines."""
    
    @abstractmethod
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a logical rule to the reasoning system."""
        pass
    
    @abstractmethod
    def add_fact(self, fact: str, confidence: float = 1.0) -> None:
        """Add a fact to the knowledge base."""
        pass
    
    @abstractmethod
    def infer(self, query: str) -> List[InferenceResult]:
        """Perform inference on a query."""
        pass
    
    @abstractmethod
    def explain(self, conclusion: str) -> List[str]:
        """Explain how a conclusion was reached."""
        pass


class PropositionalReasoner(SymbolicReasoner):
    """Propositional logic reasoner using SymPy."""
    
    def __init__(self):
        if not HAS_SYMPY:
            raise ImportError("SymPy required for PropositionalReasoner")
        
        self.rules: List[LogicalRule] = []
        self.facts: Dict[str, float] = {}  # fact -> confidence
        self.symbol_table: Dict[str, sp.Symbol] = {}
    
    def _get_symbol(self, name: str) -> sp.Symbol:
        """Get or create a SymPy symbol for a proposition."""
        if name not in self.symbol_table:
            self.symbol_table[name] = symbols(name)
        return self.symbol_table[name]
    
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a logical rule."""
        if rule.logic_type != LogicType.PROPOSITIONAL:
            raise ValueError("PropositionalReasoner only supports propositional logic")
        self.rules.append(rule)
    
    def add_fact(self, fact: str, confidence: float = 1.0) -> None:
        """Add a fact to the knowledge base."""
        self.facts[fact] = confidence
    
    def infer(self, query: str) -> List[InferenceResult]:
        """Perform propositional inference."""
        results = []
        
        # Forward chaining inference
        new_facts = self.facts.copy()
        reasoning_chain = []
        
        # Iteratively apply rules
        max_iterations = 100
        for iteration in range(max_iterations):
            added_new_fact = False
            
            for rule in self.rules:
                # Check if antecedent is satisfied
                try:
                    antecedent_expr = sympify(rule.antecedent, self.symbol_table)
                    
                    # Create substitution dict from known facts
                    substitutions = {}
                    for fact, conf in new_facts.items():
                        symbol = self._get_symbol(fact)
                        substitutions[symbol] = True if conf > 0.5 else False
                    
                    # Evaluate antecedent
                    if substitutions and antecedent_expr.subs(substitutions) == True:
                        # Derive consequent
                        consequent = rule.consequent
                        if consequent not in new_facts:
                            confidence = min(rule.confidence, min(
                                new_facts.get(str(s), 1.0) 
                                for s in antecedent_expr.free_symbols
                                if str(s) in new_facts
                            ) if antecedent_expr.free_symbols else rule.confidence)
                            
                            new_facts[consequent] = confidence
                            reasoning_chain.append(
                                f"Applied rule '{rule.id}': {rule.antecedent} → {rule.consequent}"
                            )
                            added_new_fact = True
                
                except Exception as e:
                    # Skip malformed rules
                    continue
            
            if not added_new_fact:
                break
        
        # Check if query is satisfied
        if query in new_facts:
            # Find premises that led to the query
            premises = [f for f in self.facts.keys() if f != query]
            
            results.append(InferenceResult(
                conclusion=query,
                premises=premises,
                reasoning_chain=reasoning_chain,
                confidence=new_facts[query],
                logic_type=LogicType.PROPOSITIONAL
            ))
        
        return results
    
    def explain(self, conclusion: str) -> List[str]:
        """Explain how a conclusion was reached."""
        if conclusion not in self.facts:
            # Try to infer it first
            results = self.infer(conclusion)
            if not results:
                return [f"Cannot derive conclusion: {conclusion}"]
            return results[0].reasoning_chain
        else:
            return [f"Direct fact: {conclusion}"]


class FirstOrderReasoner(SymbolicReasoner):
    """First-order logic reasoner with basic theorem proving."""
    
    def __init__(self):
        self.rules: List[LogicalRule] = []
        self.facts: List[str] = []
        self.predicates: Set[str] = set()
        self.constants: Set[str] = set()
        self.variables: Set[str] = set()
    
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a first-order logic rule."""
        if rule.logic_type != LogicType.FIRST_ORDER:
            raise ValueError("FirstOrderReasoner only supports first-order logic")
        self.rules.append(rule)
        self._parse_rule_elements(rule)
    
    def add_fact(self, fact: str, confidence: float = 1.0) -> None:
        """Add a fact to the knowledge base."""
        self.facts.append(fact)
        self._parse_fact_elements(fact)
    
    def _parse_rule_elements(self, rule: LogicalRule) -> None:
        """Parse predicates, constants, and variables from a rule."""
        # Simple parsing - in practice would need proper FOL parser
        for text in [rule.antecedent, rule.consequent]:
            # Extract predicates (functions followed by parentheses)
            import re
            predicates = re.findall(r'(\w+)\(', text)
            self.predicates.update(predicates)
            
            # Extract constants and variables
            terms = re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', text)
            for term in terms:
                if term.isupper():
                    self.variables.add(term)
                else:
                    self.constants.add(term)
    
    def _parse_fact_elements(self, fact: str) -> None:
        """Parse elements from a fact."""
        import re
        predicates = re.findall(r'(\w+)\(', fact)
        self.predicates.update(predicates)
        
        terms = re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', fact)
        self.constants.update(terms)
    
    def infer(self, query: str) -> List[InferenceResult]:
        """Perform first-order inference (simplified)."""
        results = []
        
        # Simple pattern matching for basic inference
        # In practice, would implement unification and resolution
        for rule in self.rules:
            if self._matches_pattern(rule.consequent, query):
                # Check if antecedent can be satisfied
                if self._can_satisfy_antecedent(rule.antecedent):
                    premises = [f for f in self.facts if f in rule.antecedent]
                    
                    results.append(InferenceResult(
                        conclusion=query,
                        premises=premises,
                        reasoning_chain=[f"Applied rule: {rule}"],
                        confidence=rule.confidence,
                        logic_type=LogicType.FIRST_ORDER
                    ))
        
        return results
    
    def _matches_pattern(self, pattern: str, query: str) -> bool:
        """Check if query matches a rule pattern (simplified)."""
        # Very basic pattern matching - would need proper unification
        return pattern.split('(')[0] == query.split('(')[0]
    
    def _can_satisfy_antecedent(self, antecedent: str) -> bool:
        """Check if antecedent can be satisfied by known facts."""
        # Simplified check
        for fact in self.facts:
            if fact in antecedent:
                return True
        return False
    
    def explain(self, conclusion: str) -> List[str]:
        """Explain first-order reasoning."""
        results = self.infer(conclusion)
        if results:
            return results[0].reasoning_chain
        return [f"Cannot prove: {conclusion}"]


class SMTSolver(SymbolicReasoner):
    """SMT (Satisfiability Modulo Theories) solver using Z3."""
    
    def __init__(self):
        if not HAS_Z3:
            raise ImportError("Z3 solver required for SMTSolver")
        
        self.solver = z3.Solver()
        self.variables: Dict[str, z3.ExprRef] = {}
        self.rules: List[LogicalRule] = []
    
    def _get_variable(self, name: str, sort_type: str = "Bool") -> z3.ExprRef:
        """Get or create a Z3 variable."""
        if name not in self.variables:
            if sort_type == "Bool":
                self.variables[name] = z3.Bool(name)
            elif sort_type == "Int":
                self.variables[name] = z3.Int(name)
            elif sort_type == "Real":
                self.variables[name] = z3.Real(name)
            else:
                raise ValueError(f"Unsupported sort type: {sort_type}")
        return self.variables[name]
    
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a rule as a Z3 constraint."""
        self.rules.append(rule)
        
        try:
            # Parse rule into Z3 formula (simplified)
            antecedent_var = self._get_variable(rule.antecedent.replace(" ", "_"))
            consequent_var = self._get_variable(rule.consequent.replace(" ", "_"))
            
            # Add implication: antecedent → consequent
            self.solver.add(z3.Implies(antecedent_var, consequent_var))
            
        except Exception as e:
            warnings.warn(f"Failed to add rule {rule.id}: {e}")
    
    def add_fact(self, fact: str, confidence: float = 1.0) -> None:
        """Add a fact as a Z3 assertion."""
        try:
            fact_var = self._get_variable(fact.replace(" ", "_"))
            if confidence > 0.5:
                self.solver.add(fact_var)
            else:
                self.solver.add(z3.Not(fact_var))
        except Exception as e:
            warnings.warn(f"Failed to add fact {fact}: {e}")
    
    def infer(self, query: str) -> List[InferenceResult]:
        """Check satisfiability and perform inference."""
        results = []
        
        try:
            query_var = self._get_variable(query.replace(" ", "_"))
            
            # Check if query is satisfiable
            self.solver.push()
            self.solver.add(query_var)
            
            if self.solver.check() == z3.sat:
                model = self.solver.model()
                premises = []
                
                # Extract relevant facts from model
                for var_name, var in self.variables.items():
                    if model[var] is not None:
                        premises.append(f"{var_name}: {model[var]}")
                
                results.append(InferenceResult(
                    conclusion=query,
                    premises=premises,
                    reasoning_chain=[f"SMT solving confirmed: {query}"],
                    confidence=1.0,  # SMT solving gives definitive results
                    logic_type=LogicType.FIRST_ORDER
                ))
            
            self.solver.pop()
            
        except Exception as e:
            warnings.warn(f"SMT solving failed for query {query}: {e}")
        
        return results
    
    def explain(self, conclusion: str) -> List[str]:
        """Explain SMT reasoning."""
        results = self.infer(conclusion)
        if results:
            return [f"SMT model satisfies: {conclusion}"]
        return [f"SMT model does not satisfy: {conclusion}"]


class RDFReasoner:
    """RDF-based reasoning using SPARQL queries."""
    
    def __init__(self):
        if not HAS_RDFLIB:
            raise ImportError("RDFLib required for RDF reasoning")
        
        self.graph = Graph()
        self.namespaces: Dict[str, Namespace] = {
            'rdf': RDF,
            'rdfs': RDFS,
            'owl': OWL
        }
    
    def add_namespace(self, prefix: str, uri: str) -> None:
        """Add a namespace to the RDF graph."""
        namespace = Namespace(uri)
        self.namespaces[prefix] = namespace
        self.graph.bind(prefix, namespace)
    
    def add_triple(self, subject: str, predicate: str, obj: str) -> None:
        """Add a triple to the RDF graph."""
        subj_ref = URIRef(subject) if subject.startswith('http') else BNode(subject)
        pred_ref = URIRef(predicate) if predicate.startswith('http') else URIRef(f"ex:{predicate}")
        obj_ref = URIRef(obj) if obj.startswith('http') else Literal(obj)
        
        self.graph.add((subj_ref, pred_ref, obj_ref))
    
    def query(self, sparql_query: str) -> List[Dict[str, str]]:
        """Execute SPARQL query."""
        try:
            results = self.graph.query(sparql_query)
            return [
                {str(var): str(row[var]) for var in results.vars}
                for row in results
            ]
        except Exception as e:
            warnings.warn(f"SPARQL query failed: {e}")
            return []
    
    def infer_rdfs(self) -> None:
        """Apply RDFS inference rules."""
        # Basic RDFS inference - in practice would use more sophisticated reasoner
        rdfs_rules = [
            # Subclass transitivity
            """
            INSERT {
                ?x rdfs:subClassOf ?z .
            }
            WHERE {
                ?x rdfs:subClassOf ?y .
                ?y rdfs:subClassOf ?z .
                FILTER(?x != ?z)
            }
            """,
            # Domain/range inference
            """
            INSERT {
                ?x a ?class .
            }
            WHERE {
                ?property rdfs:domain ?class .
                ?x ?property ?y .
            }
            """
        ]
        
        for rule in rdfs_rules:
            try:
                self.graph.update(rule)
            except Exception as e:
                warnings.warn(f"RDFS inference rule failed: {e}")


class SymbolicAdapter:
    """Adapter for managing symbolic reasoning engines."""
    
    REASONER_REGISTRY = {
        LogicType.PROPOSITIONAL: PropositionalReasoner,
        LogicType.FIRST_ORDER: FirstOrderReasoner,
        # SMTSolver can handle multiple logic types
        "smt": SMTSolver
    }
    
    def __init__(self):
        self.reasoners: Dict[str, SymbolicReasoner] = {}
        self.rdf_reasoner: Optional[RDFReasoner] = None
    
    def create_reasoner(
        self,
        reasoner_id: str,
        logic_type: Union[LogicType, str],
        **kwargs
    ) -> SymbolicReasoner:
        """Create and register a symbolic reasoner."""
        if logic_type in self.REASONER_REGISTRY:
            reasoner_class = self.REASONER_REGISTRY[logic_type]
        elif isinstance(logic_type, str) and logic_type in self.REASONER_REGISTRY:
            reasoner_class = self.REASONER_REGISTRY[logic_type]
        else:
            raise ValueError(f"Unknown logic type: {logic_type}")
        
        reasoner = reasoner_class(**kwargs)
        self.reasoners[reasoner_id] = reasoner
        return reasoner
    
    def get_reasoner(self, reasoner_id: str) -> SymbolicReasoner:
        """Get a registered reasoner."""
        if reasoner_id not in self.reasoners:
            raise ValueError(f"Reasoner {reasoner_id} not found")
        return self.reasoners[reasoner_id]
    
    def create_rdf_reasoner(self) -> RDFReasoner:
        """Create RDF reasoner."""
        self.rdf_reasoner = RDFReasoner()
        return self.rdf_reasoner
    
    def add_rule_to_all(self, rule: LogicalRule) -> None:
        """Add a rule to all compatible reasoners."""
        for reasoner_id, reasoner in self.reasoners.items():
            try:
                reasoner.add_rule(rule)
            except ValueError:
                # Reasoner doesn't support this logic type
                continue
    
    def multi_reasoner_inference(self, query: str) -> Dict[str, List[InferenceResult]]:
        """Perform inference using multiple reasoners."""
        results = {}
        
        for reasoner_id, reasoner in self.reasoners.items():
            try:
                reasoner_results = reasoner.infer(query)
                if reasoner_results:
                    results[reasoner_id] = reasoner_results
            except Exception as e:
                warnings.warn(f"Inference failed for reasoner {reasoner_id}: {e}")
        
        return results
    
    def consensus_inference(self, query: str, threshold: float = 0.5) -> Optional[InferenceResult]:
        """Perform consensus inference across multiple reasoners."""
        all_results = self.multi_reasoner_inference(query)
        
        if not all_results:
            return None
        
        # Simple consensus: average confidence scores
        total_confidence = 0.0
        total_reasoners = 0
        all_premises = set()
        all_reasoning_chains = []
        
        for reasoner_id, reasoner_results in all_results.items():
            for result in reasoner_results:
                if result.conclusion == query:
                    total_confidence += result.confidence
                    total_reasoners += 1
                    all_premises.update(result.premises)
                    all_reasoning_chains.extend([
                        f"[{reasoner_id}] {chain}" for chain in result.reasoning_chain
                    ])
        
        if total_reasoners == 0:
            return None
        
        avg_confidence = total_confidence / total_reasoners
        
        if avg_confidence >= threshold:
            return InferenceResult(
                conclusion=query,
                premises=list(all_premises),
                reasoning_chain=all_reasoning_chains,
                confidence=avg_confidence,
                logic_type=LogicType.FIRST_ORDER,  # Mixed logic types
                metadata={
                    'consensus_reasoners': total_reasoners,
                    'participating_reasoners': list(all_results.keys())
                }
            )
        
        return None