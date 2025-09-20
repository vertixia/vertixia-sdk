"""
AI-OS SDK - Python SDK for extending AI Operating System

This SDK provides base classes, utilities, and tools for creating
AI-OS components, agents, workflows, and services.
"""

__version__ = "0.1.0"
__author__ = "AI-OS Development Team"

# Base component classes
from .base.component import AIServiceComponent
from .base.agent import AIAgentComponent  
from .base.tool import AIToolComponent
from .base.workflow import AIWorkflowTemplate

# Configuration system
from .config.models import (
    ComponentConfig, AgentConfig, WorkflowConfig, ReasoningConfig,
    ComponentType, ExecutionMode, ParameterType
)
from .config.validation import ConfigValidator, validate_config_file

# Discovery and registry
from .utils.discovery import ComponentRegistry, auto_discover_components, get_registry

# Marketplace integration
from .marketplace.store_extension import AIStoreService
from .marketplace.registry_integration import MarketplaceRegistry

# Component templates
from .templates.itrs_reasoning import ITRSReasoningComponent

# CLI (optional import)
try:
    from .cli.main import main as cli_main
except ImportError:
    cli_main = None

__all__ = [
    # Base classes
    "AIServiceComponent",
    "AIAgentComponent", 
    "AIToolComponent",
    "AIWorkflowTemplate",
    
    # Configuration
    "ComponentConfig",
    "AgentConfig", 
    "WorkflowConfig",
    "ReasoningConfig",
    "ComponentType",
    "ExecutionMode",
    "ParameterType",
    "ConfigValidator",
    "validate_config_file",
    
    # Discovery
    "ComponentRegistry",
    "auto_discover_components",
    "get_registry",
    
    # Marketplace
    "AIStoreService",
    "MarketplaceRegistry",
    
    # Templates
    "ITRSReasoningComponent",
    
    # CLI
    "cli_main",
]


def create_component(component_name: str, config=None, **kwargs):
    """
    Convenience function to create a component instance from the registry
    
    Args:
        component_name: Name of the component to create
        config: Optional configuration override
        **kwargs: Additional component arguments
        
    Returns:
        Component instance
    """
    registry = get_registry()
    return registry.create_component(component_name, config, **kwargs)


def discover_components(search_paths=None):
    """
    Convenience function to discover components
    
    Args:
        search_paths: Optional paths to search
        
    Returns:
        Number of components discovered
    """
    return auto_discover_components(search_paths)


# Add convenience functions to __all__
__all__.extend(["create_component", "discover_components"])