"""Test for domain explainable AI service without shap/lime."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestExplainableAIServiceFallback:
    """Test explainable AI service graceful fallback when dependencies are missing."""

    @pytest.mark.asyncio
    async def test_domain_service_import_without_shap_lime(self):
        """Test importing domain service without shap or lime."""
        # Store original modules
        original_shap = sys.modules.get('shap')
        original_lime = sys.modules.get('lime')
        
        try:
            # Mock ImportError for shap and lime
            with patch.dict('sys.modules', {'shap': None, 'lime': None}):
                # Re-import the module to trigger the import checks
                import importlib
                import pynomaly.domain.services.explainable_ai_service
                importlib.reload(pynomaly.domain.services.explainable_ai_service)
                
                from pynomaly.domain.services.explainable_ai_service import (
                    ExplainableAIService, 
                    SHAP_AVAILABLE, 
                    LIME_AVAILABLE
                )
                
                # Check that flags are correctly set to False
                assert not SHAP_AVAILABLE, "SHAP_AVAILABLE should be False"
                assert not LIME_AVAILABLE, "LIME_AVAILABLE should be False"
                
                # Service should still be creatable
                service = ExplainableAIService()
                
                # Should be able to call methods (even if they fall back to basic implementations)
                assert service is not None
                
        finally:
            # Restore original modules
            if original_shap is not None:
                sys.modules['shap'] = original_shap
            if original_lime is not None:
                sys.modules['lime'] = original_lime

    def test_flags_always_defined(self):
        """Test that SHAP_AVAILABLE and LIME_AVAILABLE are always defined."""
        from pynomaly.domain.services.explainable_ai_service import (
            SHAP_AVAILABLE, 
            LIME_AVAILABLE
        )
        
        # These should always be defined as boolean values
        assert isinstance(SHAP_AVAILABLE, bool)
        assert isinstance(LIME_AVAILABLE, bool)
        
        # Whether they're True or False depends on whether libraries are installed
        # But they should never be None or undefined
        assert SHAP_AVAILABLE is not None
        assert LIME_AVAILABLE is not None
