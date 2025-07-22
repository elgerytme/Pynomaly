"""Unit tests for CausalExplanation value objects."""

import pytest
from datetime import datetime
from neuro_symbolic.domain.value_objects.causal_explanation import (
    CausalExplanation, CausalFactor, CausalLink, CausalChain,
    CausalAnalysisResult, CausalRelationType, TemporalRelation
)


class TestCausalFactor:
    """Test cases for CausalFactor value object."""
    
    def test_create_valid_causal_factor(self):
        """Test creating a valid causal factor."""
        factor = CausalFactor(
            id="factor_1",
            name="Temperature",
            value=85.5,
            confidence=0.9,
            evidence=["Sensor reading", "Historical data"]
        )
        
        assert factor.id == "factor_1"
        assert factor.name == "Temperature"
        assert factor.value == 85.5
        assert factor.confidence == 0.9
        assert len(factor.evidence) == 2
        assert isinstance(factor.metadata, dict)
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CausalFactor(
                id="factor_1",
                name="Temperature",
                value=85.5,
                confidence=1.5
            )
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            CausalFactor(
                id="factor_1", 
                name="Temperature",
                value=85.5,
                confidence=-0.1
            )
    
    def test_causal_factor_string_representation(self):
        """Test string representation of causal factor."""
        factor = CausalFactor(
            id="factor_1",
            name="Temperature",
            value=85.5,
            confidence=0.9
        )
        
        expected = "Temperature=85.5 (confidence: 0.900)"
        assert str(factor) == expected


class TestCausalLink:
    """Test cases for CausalLink value object."""
    
    def test_create_valid_causal_link(self):
        """Test creating a valid causal link."""
        cause = CausalFactor(
            id="cause_1", name="High Temperature", value=90, confidence=0.8
        )
        effect = CausalFactor(
            id="effect_1", name="System Failure", value=True, confidence=0.9
        )
        
        link = CausalLink(
            cause=cause,
            effect=effect,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.85,
            temporal_relation=TemporalRelation.BEFORE,
            evidence=["Statistical correlation", "Domain knowledge"]
        )
        
        assert link.cause == cause
        assert link.effect == effect
        assert link.relation_type == CausalRelationType.DIRECT_CAUSE
        assert link.strength == 0.85
        assert link.temporal_relation == TemporalRelation.BEFORE
        assert len(link.evidence) == 2
    
    def test_invalid_strength_raises_error(self):
        """Test that invalid strength values raise ValueError."""
        cause = CausalFactor(id="c1", name="Cause", value=1, confidence=0.8)
        effect = CausalFactor(id="e1", name="Effect", value=1, confidence=0.8)
        
        with pytest.raises(ValueError, match="Causal strength must be between 0 and 1"):
            CausalLink(
                cause=cause,
                effect=effect,
                relation_type=CausalRelationType.DIRECT_CAUSE,
                strength=1.5,
                temporal_relation=TemporalRelation.BEFORE
            )
    
    def test_causal_link_string_representation(self):
        """Test string representation of causal link."""
        cause = CausalFactor(id="c1", name="Temperature", value=90, confidence=0.8)
        effect = CausalFactor(id="e1", name="Failure", value=True, confidence=0.9)
        
        link = CausalLink(
            cause=cause,
            effect=effect,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.85,
            temporal_relation=TemporalRelation.BEFORE
        )
        
        assert str(link) == "Temperature → Failure (strength: 0.850)"


class TestCausalChain:
    """Test cases for CausalChain value object."""
    
    def test_create_valid_causal_chain(self):
        """Test creating a valid causal chain."""
        # Create factors
        factor_a = CausalFactor(id="a", name="A", value=1, confidence=0.8)
        factor_b = CausalFactor(id="b", name="B", value=2, confidence=0.9)
        factor_c = CausalFactor(id="c", name="C", value=3, confidence=0.7)
        
        # Create links
        link1 = CausalLink(
            cause=factor_a, effect=factor_b,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        link2 = CausalLink(
            cause=factor_b, effect=factor_c,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.9, temporal_relation=TemporalRelation.BEFORE
        )
        
        chain = CausalChain(
            links=[link1, link2],
            total_strength=0.85,
            confidence=0.8
        )
        
        assert len(chain.links) == 2
        assert chain.total_strength == 0.85
        assert chain.confidence == 0.8
        assert chain.length == 2
        assert chain.root_cause == factor_a
        assert chain.final_effect == factor_c
    
    def test_empty_chain_raises_error(self):
        """Test that empty chain raises ValueError."""
        with pytest.raises(ValueError, match="Causal chain must contain at least one link"):
            CausalChain(links=[], total_strength=0.5, confidence=0.5)
    
    def test_get_intermediate_factors(self):
        """Test getting intermediate factors from chain."""
        factor_a = CausalFactor(id="a", name="A", value=1, confidence=0.8)
        factor_b = CausalFactor(id="b", name="B", value=2, confidence=0.9)
        factor_c = CausalFactor(id="c", name="C", value=3, confidence=0.7)
        
        link1 = CausalLink(
            cause=factor_a, effect=factor_b,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        link2 = CausalLink(
            cause=factor_b, effect=factor_c,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.9, temporal_relation=TemporalRelation.BEFORE
        )
        
        chain = CausalChain(
            links=[link1, link2],
            total_strength=0.85,
            confidence=0.8
        )
        
        intermediates = chain.get_intermediate_factors()
        assert len(intermediates) == 1
        assert intermediates[0] == factor_b


class TestCausalExplanation:
    """Test cases for CausalExplanation value object."""
    
    def test_create_valid_causal_explanation(self):
        """Test creating a valid causal explanation."""
        # Create target outcome
        target = CausalFactor(
            id="target", name="System Outcome", value=True, confidence=0.9
        )
        
        # Create primary causes
        cause1 = CausalFactor(
            id="cause1", name="High CPU", value=95, confidence=0.8
        )
        cause2 = CausalFactor(
            id="cause2", name="Memory Leak", value=True, confidence=0.7
        )
        
        # Create causal chain
        link1 = CausalLink(
            cause=cause1, effect=target,
            relation_type=CausalRelationType.CONTRIBUTING_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        link2 = CausalLink(
            cause=cause2, effect=target,
            relation_type=CausalRelationType.CONTRIBUTING_CAUSE,
            strength=0.7, temporal_relation=TemporalRelation.BEFORE
        )
        
        chain = CausalChain(
            links=[link1, link2],
            total_strength=0.75,
            confidence=0.8
        )
        
        explanation = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause1, cause2],
            causal_chains=[chain],
            confidence=0.8,
            methodology="test_methodology",
            assumptions=["Test assumption"],
            limitations=["Test limitation"]
        )
        
        assert explanation.target_outcome == target
        assert len(explanation.primary_causes) == 2
        assert len(explanation.causal_chains) == 1
        assert explanation.confidence == 0.8
        assert len(explanation.assumptions) == 1
        assert len(explanation.limitations) == 1
        assert isinstance(explanation.timestamp, datetime)
    
    def test_get_all_factors(self):
        """Test getting all factors from explanation."""
        target = CausalFactor(id="target", name="Target", value=True, confidence=0.9)
        cause1 = CausalFactor(id="cause1", name="Cause1", value=1, confidence=0.8)
        cause2 = CausalFactor(id="cause2", name="Cause2", value=2, confidence=0.7)
        
        link = CausalLink(
            cause=cause1, effect=target,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        chain = CausalChain(links=[link], total_strength=0.8, confidence=0.8)
        
        explanation = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause1, cause2],
            causal_chains=[chain]
        )
        
        all_factors = explanation.get_all_factors()
        assert len(all_factors) == 3
        assert target in all_factors
        assert cause1 in all_factors
        assert cause2 in all_factors
    
    def test_get_strongest_chain(self):
        """Test getting strongest causal chain."""
        target = CausalFactor(id="target", name="Target", value=True, confidence=0.9)
        cause = CausalFactor(id="cause", name="Cause", value=1, confidence=0.8)
        
        # Create two chains with different strengths
        link1 = CausalLink(
            cause=cause, effect=target,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        chain1 = CausalChain(links=[link1], total_strength=0.6, confidence=0.8)
        
        link2 = CausalLink(
            cause=cause, effect=target,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.9, temporal_relation=TemporalRelation.BEFORE
        )
        chain2 = CausalChain(links=[link2], total_strength=0.9, confidence=0.9)
        
        explanation = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause],
            causal_chains=[chain1, chain2]
        )
        
        strongest = explanation.get_strongest_chain()
        assert strongest == chain2
        assert strongest.total_strength == 0.9
    
    def test_validate_consistency(self):
        """Test consistency validation."""
        target = CausalFactor(id="target", name="Target", value=True, confidence=0.9)
        cause1 = CausalFactor(id="cause1", name="Cause1", value=1, confidence=0.8)
        cause2 = CausalFactor(id="cause2", name="Cause2", value=2, confidence=0.7)
        
        # Create chain that doesn't include all primary causes
        link = CausalLink(
            cause=cause1, effect=target,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        chain = CausalChain(links=[link], total_strength=0.8, confidence=0.8)
        
        explanation = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause1, cause2],  # cause2 not in chain
            causal_chains=[chain]
        )
        
        issues = explanation.validate_consistency()
        assert len(issues) > 0
        assert any("Primary causes not found in chains" in issue for issue in issues)
    
    def test_compare_with_other_explanation(self):
        """Test comparison with another explanation."""
        # Create first explanation
        target1 = CausalFactor(id="target1", name="Target", value=True, confidence=0.9)
        cause1 = CausalFactor(id="cause1", name="CommonCause", value=1, confidence=0.8)
        
        link1 = CausalLink(
            cause=cause1, effect=target1,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        chain1 = CausalChain(links=[link1], total_strength=0.8, confidence=0.8)
        
        explanation1 = CausalExplanation.create(
            target_outcome=target1,
            primary_causes=[cause1],
            causal_chains=[chain1]
        )
        
        # Create second explanation with shared factor
        target2 = CausalFactor(id="target2", name="Target", value=True, confidence=0.8)
        cause2 = CausalFactor(id="cause1", name="CommonCause", value=1, confidence=0.8)  # Same name
        
        link2 = CausalLink(
            cause=cause2, effect=target2,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.7, temporal_relation=TemporalRelation.BEFORE
        )
        chain2 = CausalChain(links=[link2], total_strength=0.7, confidence=0.7)
        
        explanation2 = CausalExplanation.create(
            target_outcome=target2,
            primary_causes=[cause2],
            causal_chains=[chain2]
        )
        
        comparison = explanation1.compare_with(explanation2)
        
        assert 'agreement_score' in comparison
        assert 'confidence_difference' in comparison
        assert comparison['confidence_difference'] == 0.1
        assert comparison['methodology_match'] is True


class TestCausalAnalysisResult:
    """Test cases for CausalAnalysisResult."""
    
    def test_create_analysis_result(self):
        """Test creating causal analysis result."""
        target = CausalFactor(id="target", name="Target", value=True, confidence=0.9)
        cause = CausalFactor(id="cause", name="Cause", value=1, confidence=0.8)
        
        link = CausalLink(
            cause=cause, effect=target,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, temporal_relation=TemporalRelation.BEFORE
        )
        chain = CausalChain(links=[link], total_strength=0.8, confidence=0.8)
        
        explanation1 = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause],
            causal_chains=[chain],
            confidence=0.9
        )
        
        explanation2 = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause],
            causal_chains=[chain],
            confidence=0.7
        )
        
        result = CausalAnalysisResult.create(
            explanations=[explanation1, explanation2],
            analysis_metadata={"test": "metadata"}
        )
        
        assert len(result.explanations) == 2
        assert result.best_explanation == explanation1  # Higher confidence
        assert result.confidence_score == 0.8  # Average of 0.9 and 0.7
        assert result.analysis_metadata["test"] == "metadata"
    
    def test_empty_explanations_raises_error(self):
        """Test that empty explanations list raises ValueError."""
        with pytest.raises(ValueError, match="Must have at least one explanation"):
            CausalAnalysisResult.create(explanations=[])
    
    def test_best_explanation_selection(self):
        """Test that best explanation is correctly selected."""
        target = CausalFactor(id="target", name="Target", value=True, confidence=0.9)
        cause = CausalFactor(id="cause", name="Cause", value=1, confidence=0.8)
        
        # Create explanation with higher confidence and chain strength
        link_strong = CausalLink(
            cause=cause, effect=target,
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.9, temporal_relation=TemporalRelation.BEFORE
        )
        chain_strong = CausalChain(links=[link_strong], total_strength=0.9, confidence=0.9)
        
        explanation_strong = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause],
            causal_chains=[chain_strong],
            confidence=0.8
        )
        
        # Create explanation with lower strength
        link_weak = CausalLink(
            cause=cause, effect=target,
            relation_type=CausalRelationType.INDIRECT_CAUSE,
            strength=0.5, temporal_relation=TemporalRelation.BEFORE
        )
        chain_weak = CausalChain(links=[link_weak], total_strength=0.5, confidence=0.6)
        
        explanation_weak = CausalExplanation.create(
            target_outcome=target,
            primary_causes=[cause],
            causal_chains=[chain_weak],
            confidence=0.6
        )
        
        result = CausalAnalysisResult.create(
            explanations=[explanation_weak, explanation_strong]
        )
        
        # Best explanation should be the one with higher confidence * chain strength
        assert result.best_explanation == explanation_strong