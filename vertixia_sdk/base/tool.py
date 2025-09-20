"""
AI Tool Component for AI-OS

Specialized component class for creating reusable AI tools that can be
used by agents, workflows, and other components.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from abc import abstractmethod

from .component import AIServiceComponent
from ..config.models import ComponentConfig, ComponentType


class AIToolComponent(AIServiceComponent):
    """
    AI Tool component for reusable functionality
    
    Features:
    - Input/output schema definition
    - Tool chaining capabilities
    - Error handling and validation
    - Usage tracking and metrics
    - Integration with agent systems
    """
    
    def __init__(self, config: Union[ComponentConfig, str, Dict[str, Any]]):
        """Initialize AI Tool with tool-specific configuration"""
        
        # Ensure we have the right type
        if isinstance(config, dict):
            config['type'] = ComponentType.TOOL
            from ..config.models import create_config
            config = create_config(ComponentType.TOOL, **config)
        elif isinstance(config, str):
            super().__init__(config)
            if self.config.type != ComponentType.TOOL:
                raise ValueError(f"Configuration type must be 'tool', got '{self.config.type}'")
            return
        
        super().__init__(config)
        
        # Tool-specific attributes
        self.input_schema = {}
        self.output_schema = {}
        self.usage_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        # Tool capabilities
        self.chainable = True
        self.cacheable = False
        self.cache = {}
    
    def _initialize(self):
        """Initialize tool-specific components"""
        self.logger.info(f"Initializing AI Tool: {self.name}")
        
        # Define input/output schemas
        self._define_schemas()
        
        # Tool-specific setup
        self._setup_tool()
    
    @abstractmethod
    def _define_schemas(self):
        """Define input and output schemas - implement in subclasses"""
        pass
    
    @abstractmethod
    def _setup_tool(self):
        """Tool-specific setup - implement in subclasses"""
        pass
    
    def _execute(self, *args, **kwargs) -> Any:
        """Execute tool with input validation"""
        self.usage_count += 1
        
        try:
            # Validate inputs
            validated_inputs = self._validate_inputs(*args, **kwargs)
            
            # Check cache if cacheable
            if self.cacheable:
                cache_key = self._generate_cache_key(validated_inputs)
                if cache_key in self.cache:
                    self.logger.debug(f"Tool {self.name} returning cached result")
                    return self.cache[cache_key]
            
            # Execute tool logic
            result = self._tool_execute(**validated_inputs)
            
            # Validate outputs
            validated_result = self._validate_outputs(result)
            
            # Cache result if cacheable
            if self.cacheable:
                cache_key = self._generate_cache_key(validated_inputs)
                self.cache[cache_key] = validated_result
            
            self.success_count += 1
            return validated_result
            
        except Exception as e:
            self.failure_count += 1
            self.logger.error(f"Tool {self.name} execution failed: {e}")
            raise
    
    def _validate_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """Validate tool inputs against schema"""
        # Basic validation - can be enhanced with jsonschema or pydantic
        validated = {}
        
        # Convert positional args to keyword args based on schema
        if args and 'parameters' in self.input_schema:
            param_names = list(self.input_schema['parameters'].keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    validated[param_names[i]] = arg
        
        # Add keyword arguments
        validated.update(kwargs)
        
        # Check required parameters
        if 'required' in self.input_schema:
            for param in self.input_schema['required']:
                if param not in validated:
                    raise ValueError(f"Required parameter '{param}' missing")
        
        return validated
    
    def _validate_outputs(self, result: Any) -> Any:
        """Validate tool outputs against schema"""
        # Basic validation - can be enhanced based on output schema
        return result
    
    @abstractmethod
    def _tool_execute(self, **kwargs) -> Any:
        """Tool-specific execution logic - implement in subclasses"""
        pass
    
    def _generate_cache_key(self, inputs: Dict[str, Any]) -> str:
        """Generate cache key from inputs"""
        import hashlib
        import json
        
        # Create a consistent string representation
        sorted_inputs = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(sorted_inputs.encode()).hexdigest()
    
    def set_input_schema(self, schema: Dict[str, Any]):
        """Set input schema for validation"""
        self.input_schema = schema
        self.logger.debug(f"Tool {self.name} input schema updated")
    
    def set_output_schema(self, schema: Dict[str, Any]):
        """Set output schema for validation"""
        self.output_schema = schema
        self.logger.debug(f"Tool {self.name} output schema updated")
    
    def enable_caching(self, max_size: int = 100):
        """Enable result caching"""
        self.cacheable = True
        self.cache = {}
        self._cache_max_size = max_size
        self.logger.debug(f"Tool {self.name} caching enabled (max_size={max_size})")
    
    def disable_caching(self):
        """Disable result caching"""
        self.cacheable = False
        self.cache.clear()
        self.logger.debug(f"Tool {self.name} caching disabled")
    
    def clear_cache(self):
        """Clear tool cache"""
        self.cache.clear()
        self.logger.debug(f"Tool {self.name} cache cleared")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        total_usage = self.usage_count
        success_rate = (self.success_count / total_usage * 100) if total_usage > 0 else 0
        
        return {
            "total_usage": total_usage,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 2),
            "cache_size": len(self.cache) if self.cacheable else 0,
            "cacheable": self.cacheable
        }
    
    def chain_with(self, other_tool: 'AIToolComponent', **kwargs) -> Any:
        """Chain this tool with another tool"""
        if not self.chainable:
            raise ValueError(f"Tool {self.name} is not chainable")
        
        self.logger.debug(f"Chaining tool {self.name} with {other_tool.name}")
        
        # Execute this tool first
        result = self.execute(**kwargs)
        
        # Pass result to next tool
        return other_tool.execute(input_data=result)
    
    def create_langchain_tool(self) -> Any:
        """Create a LangChain tool wrapper"""
        try:
            from langchain.tools import Tool
            
            return Tool(
                name=self.name,
                description=self.config.metadata.description,
                func=self.execute
            )
        except ImportError:
            self.logger.warning("LangChain not available for tool wrapper")
            return None
    
    def create_function_schema(self) -> Dict[str, Any]:
        """Create OpenAI function calling schema"""
        return {
            "name": self.name.replace("-", "_"),
            "description": self.config.metadata.description,
            "parameters": self.input_schema.get('parameters', {}),
            "required": self.input_schema.get('required', [])
        }
    
    def _health_check(self) -> Dict[str, Any]:
        """Tool-specific health check"""
        stats = self.get_usage_stats()
        
        return {
            "status": "ok",
            "chainable": self.chainable,
            "usage_stats": stats,
            "schemas_defined": {
                "input": bool(self.input_schema),
                "output": bool(self.output_schema)
            }
        }
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make tool callable directly"""
        return self.execute(*args, **kwargs)