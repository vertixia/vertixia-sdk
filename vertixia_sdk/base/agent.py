"""
AI Agent Component for AI-OS

Specialized component class for creating AI agents with goal-oriented behavior,
tool integration, and memory management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from abc import abstractmethod

from .component import AIServiceComponent
from ..config.models import AgentConfig, ComponentType


class AIAgentComponent(AIServiceComponent):
    """
    AI Agent component for goal-oriented autonomous behavior
    
    Features:
    - Goal and role-based behavior
    - Tool integration and management
    - Memory and context management
    - Step-by-step execution tracking
    - Delegation capabilities
    """
    
    def __init__(self, config: Union[AgentConfig, str, Dict[str, Any]]):
        """Initialize AI Agent with agent-specific configuration"""
        
        # Ensure we have an AgentConfig
        if isinstance(config, dict):
            config['type'] = ComponentType.AGENT
            from ..config.models import create_config
            config = create_config(ComponentType.AGENT, **config)
        elif isinstance(config, str):
            # Load from file and ensure it's agent type
            super().__init__(config)
            if self.config.type != ComponentType.AGENT:
                raise ValueError(f"Configuration type must be 'agent', got '{self.config.type}'")
            return
        
        super().__init__(config)
        
        # Agent-specific attributes
        self.goal = getattr(self.config, 'goal', None)
        self.role = getattr(self.config, 'role', None)
        self.backstory = getattr(self.config, 'backstory', None)
        self.verbose = getattr(self.config, 'verbose', False)
        self.delegation = getattr(self.config, 'delegation', True)
        
        # Tool management
        self.tools = []
        self.tool_registry = {}
        
        # Memory and context
        self.memory = []
        self.context = {}
        self.current_task = None
        
        # Execution tracking
        self.steps_executed = []
        self.current_step = None
    
    def _initialize(self):
        """Initialize agent-specific components"""
        self.logger.info(f"Initializing AI Agent: {self.name}")
        
        if self.goal:
            self.logger.info(f"Agent Goal: {self.goal}")
        if self.role:
            self.logger.info(f"Agent Role: {self.role}")
        
        # Load and initialize tools
        self._load_tools()
        
        # Initialize memory system
        self._initialize_memory()
        
        # Agent-specific setup
        self._setup_agent()
    
    def _load_tools(self):
        """Load and initialize agent tools"""
        tool_names = getattr(self.config, 'tools', [])
        
        for tool_name in tool_names:
            try:
                tool = self._create_tool(tool_name)
                if tool:
                    self.tools.append(tool)
                    self.tool_registry[tool_name] = tool
                    self.logger.debug(f"Loaded tool: {tool_name}")
            except Exception as e:
                self.logger.error(f"Failed to load tool {tool_name}: {e}")
    
    def _create_tool(self, tool_name: str) -> Optional[Any]:
        """Create a tool instance - override in subclasses"""
        # This would integrate with the actual tool system
        # For now, return a placeholder
        return {"name": tool_name, "type": "placeholder"}
    
    def _initialize_memory(self):
        """Initialize agent memory system"""
        if getattr(self.config, 'memory_enabled', True):
            self.memory = []
            self.logger.debug("Memory system initialized")
    
    @abstractmethod
    def _setup_agent(self):
        """Agent-specific setup - implement in subclasses"""
        pass
    
    def _execute(self, task: Union[str, Dict[str, Any]], **kwargs) -> Any:
        """Execute agent task"""
        self.current_task = task
        self.logger.info(f"Agent {self.name} executing task: {task}")
        
        try:
            # Plan the task
            plan = self._plan_task(task, **kwargs)
            
            # Execute the plan
            result = self._execute_plan(plan, **kwargs)
            
            # Update memory
            self._update_memory(task, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} task execution failed: {e}")
            raise
        finally:
            self.current_task = None
    
    @abstractmethod
    def _plan_task(self, task: Union[str, Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Plan how to execute the task
        
        Returns:
            List of execution steps
        """
        pass
    
    def _execute_plan(self, plan: List[Dict[str, Any]], **kwargs) -> Any:
        """Execute the planned steps"""
        results = []
        
        for i, step in enumerate(plan):
            self.current_step = step
            self.logger.debug(f"Executing step {i+1}/{len(plan)}: {step}")
            
            try:
                step_result = self._execute_step(step, **kwargs)
                results.append(step_result)
                
                self.steps_executed.append({
                    "step": step,
                    "result": step_result,
                    "success": True
                })
                
                if self.verbose:
                    self.logger.info(f"Step {i+1} completed: {step_result}")
                
            except Exception as e:
                self.logger.error(f"Step {i+1} failed: {e}")
                
                self.steps_executed.append({
                    "step": step,
                    "error": str(e),
                    "success": False
                })
                
                # Decide whether to continue or fail
                if not kwargs.get('continue_on_error', False):
                    raise
        
        self.current_step = None
        return self._synthesize_results(results)
    
    @abstractmethod
    def _execute_step(self, step: Dict[str, Any], **kwargs) -> Any:
        """Execute a single step - implement in subclasses"""
        pass
    
    def _synthesize_results(self, results: List[Any]) -> Any:
        """Synthesize step results into final result"""
        if len(results) == 1:
            return results[0]
        return results
    
    def _update_memory(self, task: Union[str, Dict[str, Any]], result: Any):
        """Update agent memory with task and result"""
        if getattr(self.config, 'memory_enabled', True):
            memory_entry = {
                "timestamp": self._last_execution,
                "task": task,
                "result": result,
                "steps_count": len(self.steps_executed),
                "success": True
            }
            self.memory.append(memory_entry)
            
            # Limit memory size
            max_memory = kwargs.get('max_memory_size', 100)
            if len(self.memory) > max_memory:
                self.memory = self.memory[-max_memory:]
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a specific tool by name"""
        return self.tool_registry.get(tool_name)
    
    def use_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Use a specific tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        self.logger.debug(f"Using tool {tool_name}")
        
        # This would call the actual tool
        # For now, return a placeholder
        return {"tool": tool_name, "args": args, "kwargs": kwargs}
    
    def delegate_task(self, task: Union[str, Dict[str, Any]], agent_name: str) -> Any:
        """Delegate a task to another agent"""
        if not self.delegation:
            raise ValueError("Delegation is disabled for this agent")
        
        self.logger.info(f"Delegating task to agent {agent_name}: {task}")
        
        # This would integrate with the agent management system
        # For now, return a placeholder
        return {"delegated_to": agent_name, "task": task}
    
    def get_memory(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get agent memory"""
        if limit:
            return self.memory[-limit:]
        return self.memory.copy()
    
    def clear_memory(self):
        """Clear agent memory"""
        self.memory.clear()
        self.logger.debug("Agent memory cleared")
    
    def get_context(self) -> Dict[str, Any]:
        """Get current agent context"""
        return {
            "goal": self.goal,
            "role": self.role,
            "backstory": self.backstory,
            "current_task": self.current_task,
            "current_step": self.current_step,
            "tools_available": list(self.tool_registry.keys()),
            "memory_size": len(self.memory),
            "steps_executed": len(self.steps_executed)
        }
    
    def set_context(self, context: Dict[str, Any]):
        """Update agent context"""
        self.context.update(context)
        self.logger.debug(f"Agent context updated: {list(context.keys())}")
    
    def _health_check(self) -> Dict[str, Any]:
        """Agent-specific health check"""
        return {
            "status": "ok",
            "goal_set": self.goal is not None,
            "tools_loaded": len(self.tools),
            "memory_entries": len(self.memory),
            "delegation_enabled": self.delegation,
            "current_task": self.current_task is not None
        }