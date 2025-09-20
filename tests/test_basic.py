"""Basic tests for Vertixia SDK."""

import pytest
from vertixia_sdk import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_sdk_imports():
    """Test that main SDK components can be imported."""
    try:
        from vertixia_sdk.base import (
            AIServiceComponent,
            AIAgentComponent, 
            AIToolComponent,
            AIWorkflowTemplate
        )
        from vertixia_sdk.config import ComponentConfig
        from vertixia_sdk.utils import discover_components
        
        # Basic instantiation test
        assert AIServiceComponent is not None
        assert AIAgentComponent is not None
        assert AIToolComponent is not None
        assert AIWorkflowTemplate is not None
        assert ComponentConfig is not None
        assert discover_components is not None
        
    except ImportError as e:
        pytest.fail(f"Failed to import SDK components: {e}")


def test_component_config():
    """Test component configuration."""
    from vertixia_sdk.config import ComponentConfig
    
    config = ComponentConfig(
        name="test-component",
        description="Test component",
        version="1.0.0"
    )
    
    assert config.name == "test-component"
    assert config.description == "Test component"
    assert config.version == "1.0.0"