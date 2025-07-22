"""Unit tests for symbolic reasoning adapter classes."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from neuro_symbolic.infrastructure.symbolic_adapters import (
    PropositionalReasoner,
    FirstOrderReasoner,
    SMTSolver
)


class TestPropositionalReasoner:
    """Test cases for PropositionalReasoner adapter."""
    
    def test_create_propositional_reasoner(self):
        """Test creating a valid propositional reasoner."""
        reasoner = PropositionalReasoner()
        
        assert reasoner.knowledge_base == []
        assert reasoner.inference_rules == []
        assert not reasoner.is_consistent()  # Empty KB is inconsistent
    
    def test_add_proposition(self):
        """Test adding propositions to knowledge base."""
        reasoner = PropositionalReasoner()
        
        reasoner.add_proposition("P")
        reasoner.add_proposition("Q")
        reasoner.add_proposition("P -> Q")
        
        assert len(reasoner.knowledge_base) == 3
        assert "P" in reasoner.knowledge_base
        assert "Q" in reasoner.knowledge_base
        assert "P -> Q" in reasoner.knowledge_base
    
    def test_add_inference_rule(self):
        """Test adding inference rules."""
        reasoner = PropositionalReasoner()
        
        reasoner.add_inference_rule("modus_ponens", ["P", "P -> Q"], "Q")
        reasoner.add_inference_rule("modus_tollens", ["~Q", "P -> Q"], "~P")
        
        assert len(reasoner.inference_rules) == 2
        assert reasoner.inference_rules[0]["name"] == "modus_ponens"
        assert reasoner.inference_rules[1]["name"] == "modus_tollens"
    
    def test_propositional_inference(self):
        """Test propositional inference."""
        reasoner = PropositionalReasoner()
        
        # Set up knowledge base
        reasoner.add_proposition("A")
        reasoner.add_proposition("A -> B")
        reasoner.add_proposition("B -> C")
        
        # Add modus ponens rule
        reasoner.add_inference_rule("modus_ponens", ["X", "X -> Y"], "Y")
        
        # Test inference
        result = reasoner.infer("C")
        
        assert result["conclusion"] == "C"
        assert result["derivable"] is True
        assert "A" in result["proof_chain"]
        assert "B" in result["proof_chain"]
        assert "C" in result["proof_chain"]
    
    def test_check_consistency(self):
        """Test consistency checking."""
        reasoner = PropositionalReasoner()
        
        # Consistent KB
        reasoner.add_proposition("P")
        reasoner.add_proposition("Q")
        assert reasoner.is_consistent()
        
        # Add contradiction
        reasoner.add_proposition("~P")
        assert not reasoner.is_consistent()
    
    def test_propositional_satisfiability(self):
        """Test satisfiability checking."""
        reasoner = PropositionalReasoner()
        
        # Satisfiable formula
        formula = "(P | Q) & (~P | R)"
        result = reasoner.is_satisfiable(formula)
        
        assert result["satisfiable"] is True
        assert "model" in result
        assert isinstance(result["model"], dict)
    
    def test_truth_table_generation(self):
        """Test truth table generation."""
        reasoner = PropositionalReasoner()
        
        formula = "P -> Q"
        truth_table = reasoner.generate_truth_table(formula)
        
        assert len(truth_table) == 4  # 2^2 rows for P and Q
        assert all("P" in row and "Q" in row and "result" in row for row in truth_table)
    
    def test_invalid_proposition_format(self):
        """Test that invalid propositions raise errors."""
        reasoner = PropositionalReasoner()
        
        with pytest.raises(ValueError, match="Invalid proposition format"):
            reasoner.add_proposition("")
        
        with pytest.raises(ValueError, match="Invalid proposition format"):
            reasoner.add_proposition("123")  # Must start with letter
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        reasoner = PropositionalReasoner()
        
        reasoner.add_proposition("A -> B")
        reasoner.add_proposition("B -> C")
        reasoner.add_proposition("C -> A")
        
        cycles = reasoner.detect_circular_dependencies()
        assert len(cycles) > 0
        assert any("A" in cycle and "B" in cycle and "C" in cycle for cycle in cycles)


class TestFirstOrderReasoner:
    """Test cases for FirstOrderReasoner adapter."""
    
    def test_create_first_order_reasoner(self):
        """Test creating a valid first-order reasoner."""
        reasoner = FirstOrderReasoner()
        
        assert reasoner.predicates == {}
        assert reasoner.facts == []
        assert reasoner.rules == []
        assert reasoner.constants == set()
    
    def test_define_predicate(self):
        """Test defining predicates."""
        reasoner = FirstOrderReasoner()
        
        reasoner.define_predicate("Human", 1, "x is a human")
        reasoner.define_predicate("Mortal", 1, "x is mortal")
        reasoner.define_predicate("Parent", 2, "x is parent of y")
        
        assert len(reasoner.predicates) == 3
        assert reasoner.predicates["Human"]["arity"] == 1
        assert reasoner.predicates["Parent"]["arity"] == 2
    
    def test_add_fact(self):
        """Test adding facts to knowledge base."""
        reasoner = FirstOrderReasoner()
        
        reasoner.define_predicate("Human", 1)
        reasoner.define_predicate("Age", 2)
        
        reasoner.add_fact("Human(socrates)")
        reasoner.add_fact("Age(socrates, 70)")
        
        assert len(reasoner.facts) == 2
        assert "socrates" in reasoner.constants
        assert "70" in reasoner.constants
    
    def test_add_rule(self):
        """Test adding inference rules."""
        reasoner = FirstOrderReasoner()
        
        reasoner.define_predicate("Human", 1)
        reasoner.define_predicate("Mortal", 1)
        
        rule = "∀x (Human(x) → Mortal(x))"
        reasoner.add_rule(rule, "All humans are mortal")
        
        assert len(reasoner.rules) == 1
        assert reasoner.rules[0]["rule"] == rule
        assert reasoner.rules[0]["description"] == "All humans are mortal"
    
    def test_first_order_inference(self):
        """Test first-order logic inference."""
        reasoner = FirstOrderReasoner()
        
        # Set up knowledge base
        reasoner.define_predicate("Human", 1)
        reasoner.define_predicate("Mortal", 1)
        
        reasoner.add_fact("Human(socrates)")
        reasoner.add_rule("∀x (Human(x) → Mortal(x))")
        
        # Test inference
        result = reasoner.infer("Mortal(socrates)")
        
        assert result["conclusion"] == "Mortal(socrates)"
        assert result["provable"] is True
        assert len(result["proof_steps"]) > 0
    
    def test_unification(self):
        """Test unification algorithm."""
        reasoner = FirstOrderReasoner()
        
        term1 = "Parent(x, john)"
        term2 = "Parent(mary, y)"
        
        unifier = reasoner.unify(term1, term2)
        
        assert unifier is not None
        assert unifier["x"] == "mary"
        assert unifier["y"] == "john"
    
    def test_variable_substitution(self):
        """Test variable substitution in formulas."""
        reasoner = FirstOrderReasoner()
        
        formula = "∀x (Human(x) → Mortal(x))"
        substitution = {"x": "socrates"}
        
        result = reasoner.substitute_variables(formula, substitution)
        
        assert "socrates" in result
        assert "Human(socrates)" in result or "Mortal(socrates)" in result
    
    def test_existential_instantiation(self):
        """Test existential instantiation."""
        reasoner = FirstOrderReasoner()
        
        reasoner.define_predicate("Loves", 2)
        formula = "∃x Loves(x, mary)"
        
        instantiated = reasoner.existential_instantiation(formula)
        
        assert "Loves(" in instantiated
        assert "mary)" in instantiated
        # Should contain a Skolem constant/function
    
    def test_resolution_inference(self):
        """Test resolution-based inference."""
        reasoner = FirstOrderReasoner()
        
        # Set up clauses for resolution
        reasoner.add_clause("~Human(x) | Mortal(x)")  # Human(x) → Mortal(x)
        reasoner.add_clause("Human(socrates)")
        
        result = reasoner.resolve("Mortal(socrates)")
        
        assert result["provable"] is True
        assert len(result["resolution_steps"]) > 0
    
    def test_invalid_predicate_definition(self):
        """Test that invalid predicate definitions raise errors."""
        reasoner = FirstOrderReasoner()
        
        with pytest.raises(ValueError, match="Predicate name must be non-empty"):
            reasoner.define_predicate("", 1)
        
        with pytest.raises(ValueError, match="Arity must be non-negative"):
            reasoner.define_predicate("Test", -1)
    
    def test_model_finding(self):
        """Test model finding for satisfiable formulas."""
        reasoner = FirstOrderReasoner()
        
        reasoner.define_predicate("Human", 1)
        reasoner.add_fact("Human(alice)")
        reasoner.add_fact("Human(bob)")
        
        model = reasoner.find_model()
        
        assert model is not None
        assert "Human" in model["predicates"]
        assert len(model["domain"]) >= 2


class TestSMTSolver:
    """Test cases for SMT (Satisfiability Modulo Theories) solver adapter."""
    
    def test_create_smt_solver(self):
        """Test creating a valid SMT solver."""
        solver = SMTSolver(theory="QF_LIA")  # Quantifier-free linear integer arithmetic
        
        assert solver.theory == "QF_LIA"
        assert solver.assertions == []
        assert solver.variables == {}
    
    def test_declare_variables(self):
        """Test declaring variables in SMT solver."""
        solver = SMTSolver(theory="QF_LIA")
        
        solver.declare_int("x")
        solver.declare_int("y")
        solver.declare_bool("p")
        solver.declare_real("z")
        
        assert len(solver.variables) == 4
        assert solver.variables["x"]["type"] == "Int"
        assert solver.variables["y"]["type"] == "Int"
        assert solver.variables["p"]["type"] == "Bool"
        assert solver.variables["z"]["type"] == "Real"
    
    def test_add_constraints(self):
        """Test adding constraints to SMT solver."""
        solver = SMTSolver(theory="QF_LIA")
        
        solver.declare_int("x")
        solver.declare_int("y")
        
        solver.add_constraint("x > 0")
        solver.add_constraint("y < 10")
        solver.add_constraint("x + y = 5")
        
        assert len(solver.assertions) == 3
    
    def test_smt_satisfiability_check(self):
        """Test SMT satisfiability checking."""
        solver = SMTSolver(theory="QF_LIA")
        
        solver.declare_int("x")
        solver.declare_int("y")
        
        # Satisfiable constraints
        solver.add_constraint("x > 0")
        solver.add_constraint("y > 0")
        solver.add_constraint("x + y < 10")
        
        result = solver.check_sat()
        
        assert result["status"] == "sat"
        assert "model" in result
    
    def test_smt_unsatisfiability(self):
        """Test SMT unsatisfiability detection."""
        solver = SMTSolver(theory="QF_LIA")
        
        solver.declare_int("x")
        
        # Contradictory constraints
        solver.add_constraint("x > 10")
        solver.add_constraint("x < 5")
        
        result = solver.check_sat()
        
        assert result["status"] == "unsat"
        assert "unsat_core" in result
    
    def test_smt_model_extraction(self):
        """Test model extraction from SMT solver."""
        solver = SMTSolver(theory="QF_LIA")
        
        solver.declare_int("a")
        solver.declare_int("b")
        solver.add_constraint("a + b = 10")
        solver.add_constraint("a > b")
        
        result = solver.check_sat()
        
        if result["status"] == "sat":
            model = solver.get_model()
            
            assert "a" in model
            assert "b" in model
            assert int(model["a"]) + int(model["b"]) == 10
            assert int(model["a"]) > int(model["b"])
    
    def test_incremental_solving(self):
        """Test incremental SMT solving with push/pop."""
        solver = SMTSolver(theory="QF_LIA")
        
        solver.declare_int("x")
        solver.add_constraint("x > 0")
        
        # Push context
        solver.push()
        solver.add_constraint("x < 5")
        
        result1 = solver.check_sat()
        assert result1["status"] == "sat"
        
        # Pop context and add different constraint
        solver.pop()
        solver.add_constraint("x > 100")
        
        result2 = solver.check_sat()
        assert result2["status"] == "sat"
    
    def test_optimization_objectives(self):
        """Test optimization objectives in SMT solver."""
        solver = SMTSolver(theory="QF_LIA", enable_optimization=True)
        
        solver.declare_int("profit")
        solver.declare_int("cost")
        
        solver.add_constraint("profit - cost > 0")
        solver.add_constraint("profit < 1000")
        solver.add_constraint("cost > 100")
        
        # Maximize profit - cost
        solver.maximize("profit - cost")
        
        result = solver.optimize()
        
        assert result["status"] == "optimal"
        assert "optimal_value" in result
        assert "model" in result
    
    def test_theory_specific_functions(self):
        """Test theory-specific functions and predicates."""
        solver = SMTSolver(theory="QF_ABV")  # Quantifier-free arrays and bit-vectors
        
        solver.declare_array("arr", "Int", "Int")
        solver.declare_bitvector("bv", 8)
        
        solver.add_constraint("(select arr 0) = 42")
        solver.add_constraint("bv = #b00001010")  # 10 in binary
        
        result = solver.check_sat()
        
        if result["status"] == "sat":
            model = solver.get_model()
            assert "arr" in model or "bv" in model
    
    def test_unsat_core_generation(self):
        """Test unsat core generation for debugging."""
        solver = SMTSolver(theory="QF_LIA", generate_unsat_core=True)
        
        solver.declare_int("x")
        
        # Named assertions for unsat core
        solver.add_named_constraint("c1", "x > 10")
        solver.add_named_constraint("c2", "x < 5")
        solver.add_named_constraint("c3", "x = x")  # Trivially true
        
        result = solver.check_sat()
        
        assert result["status"] == "unsat"
        unsat_core = result["unsat_core"]
        assert "c1" in unsat_core or "c2" in unsat_core
        assert "c3" not in unsat_core  # Should not be in minimal unsat core
    
    def test_invalid_smt_theory(self):
        """Test that invalid SMT theories raise errors."""
        with pytest.raises(ValueError, match="Unsupported SMT theory"):
            SMTSolver(theory="INVALID_THEORY")
    
    def test_timeout_handling(self):
        """Test timeout handling in SMT solving."""
        solver = SMTSolver(theory="QF_LIA", timeout=1000)  # 1 second timeout
        
        solver.declare_int("x")
        # Complex constraint that might timeout
        for i in range(100):
            solver.add_constraint(f"x != {i}")
        
        result = solver.check_sat()
        
        # Result should be either "sat", "unsat", or "timeout"
        assert result["status"] in ["sat", "unsat", "timeout"]


class TestSymbolicReasoningIntegration:
    """Test integration between different symbolic reasoning components."""
    
    def test_propositional_to_first_order_translation(self):
        """Test translating propositional logic to first-order logic."""
        prop_reasoner = PropositionalReasoner()
        fol_reasoner = FirstOrderReasoner()
        
        # Propositional formula
        prop_reasoner.add_proposition("P -> Q")
        
        # Translate to FOL
        fol_formula = fol_reasoner.translate_from_propositional("P -> Q")
        
        assert "→" in fol_formula or "->" in fol_formula
    
    def test_first_order_to_smt_encoding(self):
        """Test encoding first-order formulas into SMT."""
        fol_reasoner = FirstOrderReasoner()
        smt_solver = SMTSolver(theory="QF_LIA")
        
        # FOL constraint
        fol_reasoner.define_predicate("Age", 2)
        fol_reasoner.add_fact("Age(alice, 25)")
        
        # Encode to SMT
        encoding = smt_solver.encode_first_order_formula("Age(alice, x) ∧ x > 18")
        
        assert encoding is not None
    
    def test_reasoning_result_consistency(self):
        """Test consistency of reasoning results across different engines."""
        # Simple tautology that should be provable in all systems
        prop_reasoner = PropositionalReasoner()
        fol_reasoner = FirstOrderReasoner()
        
        # Propositional: P ∨ ¬P (law of excluded middle)
        prop_result = prop_reasoner.is_tautology("P | ~P")
        
        # First-order: ∀x (P(x) ∨ ¬P(x))
        fol_reasoner.define_predicate("P", 1)
        fol_result = fol_reasoner.is_tautology("∀x (P(x) ∨ ¬P(x))")
        
        assert prop_result is True
        assert fol_result is True
    
    def test_hybrid_reasoning_workflow(self):
        """Test a complete hybrid reasoning workflow."""
        # Start with propositional reasoning
        prop_reasoner = PropositionalReasoner()
        prop_reasoner.add_proposition("A -> B")
        prop_reasoner.add_proposition("A")
        
        prop_conclusion = prop_reasoner.infer("B")
        assert prop_conclusion["derivable"] is True
        
        # Lift to first-order logic
        fol_reasoner = FirstOrderReasoner()
        fol_reasoner.define_predicate("A", 0)
        fol_reasoner.define_predicate("B", 0)
        fol_reasoner.add_rule("A() → B()")
        fol_reasoner.add_fact("A()")
        
        fol_conclusion = fol_reasoner.infer("B()")
        assert fol_conclusion["provable"] is True
        
        # Verify with SMT solver
        smt_solver = SMTSolver(theory="QF_LIA")
        smt_solver.declare_bool("a")
        smt_solver.declare_bool("b")
        smt_solver.add_constraint("a")
        smt_solver.add_constraint("(not a) or b")  # a -> b
        smt_solver.add_constraint("not b")  # Negation of conclusion
        
        smt_result = smt_solver.check_sat()
        assert smt_result["status"] == "unsat"  # Contradiction, so B is entailed