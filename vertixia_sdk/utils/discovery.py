"""
Component auto-discovery and registry system for AI-OS
"""

import os
import sys
import importlib
import importlib.util
import inspect
from typing import Dict, List, Type, Optional, Any, Set
from pathlib import Path
import logging

from ..base.component import AIServiceComponent
from ..base.agent import AIAgentComponent
from ..base.tool import AIToolComponent  
from ..base.workflow import AIWorkflowTemplate
from ..config.models import ComponentType, ComponentConfig
from ..config.validation import validate_config_file


logger = logging.getLogger(__name__)


class ComponentRegistry:
    """
    Central registry for AI-OS components with auto-discovery capabilities
    """
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Any]] = {}
        self._component_classes: Dict[str, Type[AIServiceComponent]] = {}
        self._search_paths: Set[str] = set()
        
        # Register built-in component types
        self._base_classes = {
            ComponentType.SERVICE: AIServiceComponent,
            ComponentType.AGENT: AIAgentComponent,
            ComponentType.TOOL: AIToolComponent,
            ComponentType.WORKFLOW: AIWorkflowTemplate,
            ComponentType.REASONING: AIServiceComponent,  # Uses base for now
            ComponentType.AUTOMATION: AIServiceComponent,  # Uses base for now
            ComponentType.INTELLIGENCE: AIServiceComponent,  # Uses base for now
            ComponentType.INTEGRATION: AIServiceComponent  # Uses base for now
        }
        
    def register_component(
        self,
        component_class: Type[AIServiceComponent],
        config_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a component class with the registry
        
        Args:
            component_class: The component class to register
            config_path: Optional path to component configuration file
            metadata: Optional component metadata
            
        Returns:
            True if registration successful
        """
        try:
            # Extract component information
            component_name = getattr(component_class, '__name__', str(component_class))
            module_name = getattr(component_class, '__module__', 'unknown')
            
            # Load configuration if provided
            config = None
            if config_path and os.path.exists(config_path):
                is_valid, config, errors, warnings = validate_config_file(config_path)
                if not is_valid:
                    logger.error(f"Invalid configuration for {component_name}: {errors}")
                    return False
                if warnings:
                    logger.warning(f"Configuration warnings for {component_name}: {warnings}")
            
            # Build component info
            component_info = {
                "class": component_class,
                "name": component_name,
                "module": module_name,
                "config_path": config_path,
                "config": config,
                "metadata": metadata or {},
                "type": self._infer_component_type(component_class),
                "description": inspect.getdoc(component_class) or "",
                "file_path": inspect.getfile(component_class) if hasattr(component_class, '__file__') else None
            }
            
            # Store in registry
            self._components[component_name] = component_info
            self._component_classes[component_name] = component_class
            
            logger.info(f"Registered component: {component_name} ({component_info['type']})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register component {component_class}: {e}")
            return False
    
    def _infer_component_type(self, component_class: Type[AIServiceComponent]) -> ComponentType:
        """Infer component type from class hierarchy"""
        if issubclass(component_class, AIAgentComponent):
            return ComponentType.AGENT
        elif issubclass(component_class, AIToolComponent):
            return ComponentType.TOOL
        elif issubclass(component_class, AIWorkflowTemplate):
            return ComponentType.WORKFLOW
        else:
            return ComponentType.SERVICE
    
    def unregister_component(self, component_name: str) -> bool:
        """Unregister a component from the registry"""
        if component_name in self._components:
            del self._components[component_name]
            del self._component_classes[component_name]
            logger.info(f"Unregistered component: {component_name}")
            return True
        return False
    
    def get_component(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get component information by name"""
        return self._components.get(component_name)
    
    def get_component_class(self, component_name: str) -> Optional[Type[AIServiceComponent]]:
        """Get component class by name"""
        return self._component_classes.get(component_name)
    
    def list_components(
        self,
        component_type: Optional[ComponentType] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List all registered components
        
        Args:
            component_type: Filter by component type
            include_metadata: Include full metadata in results
            
        Returns:
            List of component information
        """
        components = []
        
        for name, info in self._components.items():
            if component_type and info["type"] != component_type:
                continue
                
            component_data = {
                "name": name,
                "type": info["type"].value if hasattr(info["type"], 'value') else str(info["type"]),
                "module": info["module"],
                "description": info["description"]
            }
            
            if include_metadata:
                component_data.update({
                    "config_path": info["config_path"],
                    "metadata": info["metadata"],
                    "file_path": info["file_path"]
                })
            
            components.append(component_data)
        
        return sorted(components, key=lambda x: x["name"])
    
    def create_component(
        self,
        component_name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[AIServiceComponent]:
        """
        Create an instance of a registered component
        
        Args:
            component_name: Name of component to create
            config: Configuration override
            **kwargs: Additional arguments for component initialization
            
        Returns:
            Component instance or None if creation failed
        """
        component_class = self.get_component_class(component_name)
        if not component_class:
            logger.error(f"Component not found: {component_name}")
            return None
        
        try:
            # Use provided config or registry config
            if config:
                instance = component_class(config, **kwargs)
            else:
                component_info = self.get_component(component_name)
                if component_info and component_info["config"]:
                    instance = component_class(component_info["config"], **kwargs)
                else:
                    # Try to create with minimal config
                    minimal_config = {
                        "metadata": {"name": component_name},
                        "type": component_info["type"].value if component_info else "service"
                    }
                    instance = component_class(minimal_config, **kwargs)
            
            logger.info(f"Created component instance: {component_name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create component {component_name}: {e}")
            return None
    
    def add_search_path(self, path: str):
        """Add a directory to the component search path"""
        if os.path.isdir(path):
            self._search_paths.add(os.path.abspath(path))
            logger.debug(f"Added search path: {path}")
        else:
            logger.warning(f"Invalid search path: {path}")
    
    def discover_components(self, search_paths: Optional[List[str]] = None) -> int:
        """
        Auto-discover components in search paths
        
        Args:
            search_paths: Optional list of paths to search
            
        Returns:
            Number of components discovered
        """
        paths_to_search = set(self._search_paths)
        
        if search_paths:
            paths_to_search.update(search_paths)
        
        # Add current working directory if no paths specified
        if not paths_to_search:
            paths_to_search.add(os.getcwd())
        
        discovered_count = 0
        
        for search_path in paths_to_search:
            discovered_count += self._discover_in_path(search_path)
        
        logger.info(f"Discovery complete. Found {discovered_count} components.")
        return discovered_count
    
    def _discover_in_path(self, search_path: str) -> int:
        """Discover components in a specific path"""
        discovered_count = 0
        
        try:
            for root, dirs, files in os.walk(search_path):
                # Skip hidden directories and common non-component directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'tests', 'test']]
                
                for file in files:
                    if file.endswith('.py') and not file.startswith('_'):
                        file_path = os.path.join(root, file)
                        components_found = self._discover_in_file(file_path)
                        discovered_count += components_found
                        
                    elif file.endswith(('.yaml', '.yml', '.json')):
                        # Look for standalone configuration files
                        config_path = os.path.join(root, file)
                        if self._is_component_config(config_path):
                            discovered_count += self._register_from_config(config_path)
        
        except Exception as e:
            logger.error(f"Error discovering components in {search_path}: {e}")
        
        return discovered_count
    
    def _discover_in_file(self, file_path: str) -> int:
        """Discover components in a Python file"""
        discovered_count = 0
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location("discovery_module", file_path)
            if not spec or not spec.loader:
                return 0
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find component classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj != AIServiceComponent and 
                    issubclass(obj, AIServiceComponent) and 
                    obj.__module__ == module.__name__):
                    
                    # Look for accompanying config file
                    config_path = self._find_config_for_component(file_path, name)
                    
                    if self.register_component(obj, config_path):
                        discovered_count += 1
        
        except Exception as e:
            logger.debug(f"Could not discover components in {file_path}: {e}")
        
        return discovered_count
    
    def _find_config_for_component(self, file_path: str, component_name: str) -> Optional[str]:
        """Find configuration file for a component"""
        file_dir = os.path.dirname(file_path)
        file_stem = Path(file_path).stem
        
        # Common config file patterns
        config_patterns = [
            f"{component_name.lower()}.yaml",
            f"{component_name.lower()}.yml", 
            f"{component_name.lower()}.json",
            f"{file_stem}.yaml",
            f"{file_stem}.yml",
            f"{file_stem}.json",
            "config.yaml",
            "config.yml",
            "config.json"
        ]
        
        for pattern in config_patterns:
            config_path = os.path.join(file_dir, pattern)
            if os.path.exists(config_path) and self._is_component_config(config_path):
                return config_path
        
        return None
    
    def _is_component_config(self, config_path: str) -> bool:
        """Check if a file is a valid component configuration"""
        try:
            is_valid, config, errors, warnings = validate_config_file(config_path)
            return is_valid and config is not None
        except:
            return False
    
    def _register_from_config(self, config_path: str) -> int:
        """Register component from standalone config file"""
        try:
            is_valid, config, errors, warnings = validate_config_file(config_path)
            if not is_valid:
                return 0
            
            # Try to find corresponding Python file
            config_dir = os.path.dirname(config_path)
            config_stem = Path(config_path).stem
            
            python_patterns = [
                f"{config.metadata.name}.py",
                f"{config_stem}.py",
                "component.py",
                "main.py"
            ]
            
            for pattern in python_patterns:
                py_path = os.path.join(config_dir, pattern)
                if os.path.exists(py_path):
                    return self._discover_in_file(py_path)
            
            return 0
            
        except Exception as e:
            logger.debug(f"Could not register from config {config_path}: {e}")
            return 0
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry information for marketplace"""
        return {
            "components": self.list_components(include_metadata=True),
            "search_paths": list(self._search_paths),
            "component_count": len(self._components),
            "supported_types": [t.value for t in ComponentType]
        }
    
    def clear_registry(self):
        """Clear all registered components"""
        self._components.clear()
        self._component_classes.clear()
        logger.info("Component registry cleared")


# Global registry instance
_global_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry"""
    return _global_registry


def auto_discover_components(
    search_paths: Optional[List[str]] = None,
    registry: Optional[ComponentRegistry] = None
) -> int:
    """
    Auto-discover AI-OS components in specified paths
    
    Args:
        search_paths: Paths to search for components
        registry: Component registry to use (uses global if None)
        
    Returns:
        Number of components discovered
    """
    if registry is None:
        registry = get_registry()
    
    return registry.discover_components(search_paths)


def register_component(
    component_class: Type[AIServiceComponent],
    config_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    registry: Optional[ComponentRegistry] = None
) -> bool:
    """
    Register a component with the global registry
    
    Args:
        component_class: Component class to register
        config_path: Optional configuration file path
        metadata: Optional component metadata
        registry: Component registry to use (uses global if None)
        
    Returns:
        True if registration successful
    """
    if registry is None:
        registry = get_registry()
    
    return registry.register_component(component_class, config_path, metadata)


def create_component(
    component_name: str,
    config: Optional[Dict[str, Any]] = None,
    registry: Optional[ComponentRegistry] = None,
    **kwargs
) -> Optional[AIServiceComponent]:
    """
    Create a component instance from the registry
    
    Args:
        component_name: Name of component to create
        config: Configuration override
        registry: Component registry to use (uses global if None)
        **kwargs: Additional component arguments
        
    Returns:
        Component instance or None
    """
    if registry is None:
        registry = get_registry()
    
    return registry.create_component(component_name, config, **kwargs)