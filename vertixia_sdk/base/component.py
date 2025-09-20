"""
Base AI Service Component for AI-OS

Provides the foundation for all AI-OS components with configuration-driven design,
auto-discovery, and integration with the existing Langflow ecosystem.
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path

from ..config.models import ComponentConfig, ComponentType, ExecutionMode
from ..config.validation import validate_environment_variables


logger = logging.getLogger(__name__)


class AIServiceComponent(ABC):
    """
    Base class for all AI-OS service components
    
    Features:
    - Configuration-driven design
    - Environment variable integration
    - Async/sync execution support
    - Auto-discovery and registration
    - Marketplace compatibility
    - Integration with Langflow ecosystem
    """
    
    def __init__(self, config: Union[ComponentConfig, str, Path, Dict[str, Any]]):
        """
        Initialize component with configuration
        
        Args:
            config: Component configuration (config object, file path, or dict)
        """
        self.config = self._load_config(config)
        self.logger = logging.getLogger(f"ai_os.{self.config.metadata.name}")
        
        # Component state
        self._initialized = False
        self._running = False
        self._last_execution = None
        self._execution_count = 0
        self._errors = []
        
        # Runtime parameters (resolved from config + environment)
        self.parameters = self._resolve_parameters()
        
        # Validate environment
        self._validate_environment()
        
        # Initialize component-specific setup
        self._setup()
    
    def _load_config(self, config: Union[ComponentConfig, str, Path, Dict[str, Any]]) -> ComponentConfig:
        """Load and validate component configuration"""
        if isinstance(config, ComponentConfig):
            return config
        elif isinstance(config, (str, Path)):
            from ..config.validation import validate_config_file
            is_valid, parsed_config, errors, warnings = validate_config_file(str(config))
            if not is_valid:
                raise ValueError(f"Invalid configuration: {errors}")
            if warnings:
                logger.warning(f"Configuration warnings: {warnings}")
            return parsed_config
        elif isinstance(config, dict):
            from ..config.models import create_config
            component_type = ComponentType(config.get('type', 'service'))
            return create_config(component_type, **config)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
    
    def _resolve_parameters(self) -> Dict[str, Any]:
        """Resolve parameters from config and environment variables"""
        params = {}
        
        for param in self.config.parameters:
            value = param.default
            
            # Check environment variable
            if param.environment_variable:
                env_value = os.environ.get(param.environment_variable)
                if env_value is not None:
                    # Type conversion
                    if param.type.value == "integer":
                        value = int(env_value)
                    elif param.type.value == "float":
                        value = float(env_value)
                    elif param.type.value == "boolean":
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif param.type.value == "array":
                        value = env_value.split(',')
                    else:
                        value = env_value
            
            params[param.name] = value
        
        return params
    
    def _validate_environment(self):
        """Validate environment setup"""
        missing_vars, warnings = validate_environment_variables(self.config)
        
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {missing_vars}")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
    
    def _setup(self):
        """Component-specific setup - override in subclasses"""
        pass
    
    @property
    def name(self) -> str:
        """Component name"""
        return self.config.metadata.name
    
    @property
    def type(self) -> ComponentType:
        """Component type"""
        return self.config.type
    
    @property
    def version(self) -> str:
        """Component version"""
        return self.config.metadata.version
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        return self._initialized
    
    @property
    def is_running(self) -> bool:
        """Check if component is currently running"""
        return self._running
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            "name": self.name,
            "type": self.type.value,
            "version": self.version,
            "initialized": self._initialized,
            "running": self._running,
            "last_execution": self._last_execution,
            "execution_count": self._execution_count,
            "error_count": len(self._errors),
            "config": self.config.dict() if hasattr(self.config, 'dict') else str(self.config)
        }
    
    def initialize(self) -> bool:
        """
        Initialize the component
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"Initializing component {self.name}")
            
            # Component-specific initialization
            self._initialize()
            
            self._initialized = True
            self.logger.info(f"Component {self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize component {self.name}: {e}")
            self._errors.append({"timestamp": datetime.now(), "error": str(e), "type": "initialization"})
            return False
    
    @abstractmethod
    def _initialize(self):
        """Component-specific initialization - implement in subclasses"""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the component
        
        Handles both sync and async execution based on configuration
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError(f"Component {self.name} failed to initialize")
        
        try:
            self._running = True
            self._last_execution = datetime.now()
            self._execution_count += 1
            
            self.logger.debug(f"Executing component {self.name}")
            
            if self.config.execution_mode == ExecutionMode.ASYNC:
                if asyncio.iscoroutinefunction(self._execute):
                    # Already in async context
                    result = self._execute(*args, **kwargs)
                else:
                    # Run sync function in async context
                    loop = asyncio.get_event_loop()
                    result = loop.run_in_executor(None, self._execute, *args, **kwargs)
            else:
                result = self._execute(*args, **kwargs)
            
            self.logger.debug(f"Component {self.name} execution completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Component {self.name} execution failed: {e}")
            self._errors.append({"timestamp": datetime.now(), "error": str(e), "type": "execution"})
            raise
        finally:
            self._running = False
    
    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """Component-specific execution logic - implement in subclasses"""
        pass
    
    async def execute_async(self, *args, **kwargs) -> Any:
        """Execute component asynchronously"""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError(f"Component {self.name} failed to initialize")
        
        try:
            self._running = True
            self._last_execution = datetime.now()
            self._execution_count += 1
            
            self.logger.debug(f"Executing component {self.name} async")
            
            if asyncio.iscoroutinefunction(self._execute):
                result = await self._execute(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._execute, *args, **kwargs)
            
            self.logger.debug(f"Component {self.name} async execution completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Component {self.name} async execution failed: {e}")
            self._errors.append({"timestamp": datetime.now(), "error": str(e), "type": "async_execution"})
            raise
        finally:
            self._running = False
    
    def shutdown(self):
        """Shutdown the component"""
        try:
            self.logger.info(f"Shutting down component {self.name}")
            self._shutdown()
            self._initialized = False
            self.logger.info(f"Component {self.name} shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during component {self.name} shutdown: {e}")
            self._errors.append({"timestamp": datetime.now(), "error": str(e), "type": "shutdown"})
    
    def _shutdown(self):
        """Component-specific shutdown logic - override in subclasses"""
        pass
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value"""
        return self.parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any):
        """Set a parameter value at runtime"""
        self.parameters[name] = value
        self.logger.debug(f"Parameter {name} set to {value}")
    
    def get_errors(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get component errors"""
        if limit:
            return self._errors[-limit:]
        return self._errors.copy()
    
    def clear_errors(self):
        """Clear component errors"""
        self._errors.clear()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform component health check"""
        try:
            health_result = self._health_check()
            return {
                "healthy": True,
                "status": self.status,
                "details": health_result
            }
        except Exception as e:
            return {
                "healthy": False,
                "status": self.status,
                "error": str(e)
            }
    
    def _health_check(self) -> Dict[str, Any]:
        """Component-specific health check - override in subclasses"""
        return {"status": "ok"}
    
    def to_marketplace_format(self) -> Dict[str, Any]:
        """Convert component to marketplace format"""
        return {
            "name": self.config.metadata.name,
            "display_name": self.config.metadata.display_name,
            "description": self.config.metadata.description,
            "version": self.config.metadata.version,
            "author": self.config.metadata.author,
            "license": self.config.metadata.license,
            "tags": self.config.metadata.tags,
            "category": self.config.metadata.category,
            "type": self.config.type.value,
            "execution_mode": self.config.execution_mode.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in self.config.parameters
            ],
            "dependencies": [
                {
                    "name": d.name,
                    "version": d.version,
                    "optional": d.optional
                }
                for d in self.config.dependencies
            ],
            "documentation_url": self.config.metadata.documentation_url,
            "source_url": self.config.metadata.source_url,
            "icon": self.config.metadata.icon,
            "ai_os_version": self.config.ai_os_version,
            "langflow_compatible": self.config.langflow_compatible
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, type={self.type.value}, version={self.version})>"