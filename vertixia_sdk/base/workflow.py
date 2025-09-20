"""
AI Workflow Template Component for AI-OS

Specialized component class for creating reusable workflow templates that can be
composed, executed, and shared in the AI-OS ecosystem.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from abc import abstractmethod
from enum import Enum

from .component import AIServiceComponent
from ..config.models import WorkflowConfig, ComponentType


class WorkflowStepStatus(str, Enum):
    """Status of workflow steps"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep:
    """Individual workflow step"""
    
    def __init__(
        self,
        name: str,
        action: Union[str, Callable],
        inputs: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None
    ):
        self.name = name
        self.action = action
        self.inputs = inputs or {}
        self.condition = condition
        self.retry_count = retry_count
        self.timeout = timeout
        self.status = WorkflowStepStatus.PENDING
        self.result = None
        self.error = None
        self.execution_time = None


class AIWorkflowTemplate(AIServiceComponent):
    """
    AI Workflow Template component for orchestrating complex multi-step processes
    
    Features:
    - Step-by-step workflow definition
    - Conditional execution and branching
    - Parallel and sequential execution modes
    - Error handling and retry logic
    - Workflow composition and nesting
    - State management and persistence
    """
    
    def __init__(self, config: Union[WorkflowConfig, str, Dict[str, Any]]):
        """Initialize AI Workflow Template with workflow-specific configuration"""
        
        # Ensure we have the right type
        if isinstance(config, dict):
            config['type'] = ComponentType.WORKFLOW
            from ..config.models import create_config
            config = create_config(ComponentType.WORKFLOW, **config)
        elif isinstance(config, str):
            super().__init__(config)
            if self.config.type != ComponentType.WORKFLOW:
                raise ValueError(f"Configuration type must be 'workflow', got '{self.config.type}'")
            return
        
        super().__init__(config)
        
        # Workflow-specific attributes
        self.steps: List[WorkflowStep] = []
        self.step_registry: Dict[str, WorkflowStep] = {}
        self.execution_context = {}
        self.workflow_state = {}
        
        # Execution configuration
        self.parallel_execution = getattr(self.config, 'parallel_execution', False)
        self.max_workers = getattr(self.config, 'max_workers', 4)
        self.continue_on_error = getattr(self.config, 'continue_on_error', False)
        self.save_state = getattr(self.config, 'save_state', True)
        
        # Workflow metrics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0
    
    def _initialize(self):
        """Initialize workflow-specific components"""
        self.logger.info(f"Initializing AI Workflow Template: {self.name}")
        
        # Define workflow steps
        self._define_workflow()
        
        # Validate workflow structure
        self._validate_workflow()
        
        # Workflow-specific setup
        self._setup_workflow()
    
    @abstractmethod
    def _define_workflow(self):
        """Define workflow steps - implement in subclasses"""
        pass
    
    @abstractmethod
    def _setup_workflow(self):
        """Workflow-specific setup - implement in subclasses"""
        pass
    
    def _validate_workflow(self):
        """Validate workflow structure"""
        if not self.steps:
            raise ValueError("Workflow must have at least one step")
        
        # Check for duplicate step names
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Workflow steps must have unique names")
        
        # Validate parallel execution configuration
        if self.parallel_execution and self.max_workers < 1:
            raise ValueError("Parallel workflows must have max_workers >= 1")
    
    def add_step(
        self,
        name: str,
        action: Union[str, Callable],
        inputs: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None
    ) -> 'AIWorkflowTemplate':
        """Add a step to the workflow"""
        step = WorkflowStep(name, action, inputs, condition, retry_count, timeout)
        self.steps.append(step)
        self.step_registry[name] = step
        self.logger.debug(f"Added step '{name}' to workflow")
        return self
    
    def insert_step(
        self,
        index: int,
        name: str,
        action: Union[str, Callable],
        inputs: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None
    ) -> 'AIWorkflowTemplate':
        """Insert a step at a specific position"""
        step = WorkflowStep(name, action, inputs, condition, retry_count, timeout)
        self.steps.insert(index, step)
        self.step_registry[name] = step
        self.logger.debug(f"Inserted step '{name}' at position {index}")
        return self
    
    def remove_step(self, name: str) -> 'AIWorkflowTemplate':
        """Remove a step from the workflow"""
        step = self.step_registry.get(name)
        if step:
            self.steps.remove(step)
            del self.step_registry[name]
            self.logger.debug(f"Removed step '{name}' from workflow")
        return self
    
    def _execute(self, inputs: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute workflow with input data"""
        import time
        
        start_time = time.time()
        self.execution_count += 1
        
        # Initialize execution context
        self.execution_context = inputs or {}
        self.execution_context.update(kwargs)
        
        try:
            self.logger.info(f"Starting workflow execution: {self.name}")
            
            if self.parallel_execution:
                result = self._execute_parallel()
            else:
                result = self._execute_sequential()
            
            # Update metrics
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.success_count += 1
            
            self.logger.info(f"Workflow execution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.failure_count += 1
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            
            self.logger.error(f"Workflow execution failed after {execution_time:.2f}s: {e}")
            
            if not self.continue_on_error:
                raise
            
            return {"error": str(e), "partial_results": self._get_completed_results()}
    
    def _execute_sequential(self) -> Any:
        """Execute workflow steps sequentially"""
        results = {}
        
        for i, step in enumerate(self.steps):
            self.logger.debug(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            # Check step condition
            if step.condition and not step.condition(self.execution_context):
                step.status = WorkflowStepStatus.SKIPPED
                self.logger.debug(f"Step '{step.name}' skipped due to condition")
                continue
            
            try:
                step.status = WorkflowStepStatus.RUNNING
                result = self._execute_step(step)
                step.result = result
                step.status = WorkflowStepStatus.COMPLETED
                
                results[step.name] = result
                
                # Update execution context with step result
                self.execution_context[f"{step.name}_result"] = result
                
            except Exception as e:
                step.error = str(e)
                step.status = WorkflowStepStatus.FAILED
                
                self.logger.error(f"Step '{step.name}' failed: {e}")
                
                if not self.continue_on_error:
                    raise
                
                results[step.name] = {"error": str(e)}
        
        return self._synthesize_workflow_results(results)
    
    def _execute_parallel(self) -> Any:
        """Execute workflow steps in parallel"""
        import concurrent.futures
        import threading
        
        results = {}
        
        # Filter steps that should run (check conditions)
        executable_steps = []
        for step in self.steps:
            if not step.condition or step.condition(self.execution_context):
                executable_steps.append(step)
            else:
                step.status = WorkflowStepStatus.SKIPPED
                self.logger.debug(f"Step '{step.name}' skipped due to condition")
        
        # Execute steps in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all steps
            future_to_step = {
                executor.submit(self._execute_step_wrapper, step): step
                for step in executable_steps
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                
                try:
                    result = future.result()
                    step.result = result
                    step.status = WorkflowStepStatus.COMPLETED
                    results[step.name] = result
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = WorkflowStepStatus.FAILED
                    
                    self.logger.error(f"Step '{step.name}' failed: {e}")
                    
                    if not self.continue_on_error:
                        # Cancel remaining futures
                        for f in future_to_step:
                            f.cancel()
                        raise
                    
                    results[step.name] = {"error": str(e)}
        
        return self._synthesize_workflow_results(results)
    
    def _execute_step_wrapper(self, step: WorkflowStep) -> Any:
        """Wrapper for executing a step with status tracking"""
        step.status = WorkflowStepStatus.RUNNING
        return self._execute_step(step)
    
    def _execute_step(self, step: WorkflowStep) -> Any:
        """Execute a single workflow step"""
        import time
        
        start_time = time.time()
        
        try:
            # Handle retry logic
            last_error = None
            for attempt in range(step.retry_count + 1):
                try:
                    if attempt > 0:
                        self.logger.debug(f"Retrying step '{step.name}' (attempt {attempt + 1})")
                    
                    # Execute step action
                    if callable(step.action):
                        result = step.action(self.execution_context, **step.inputs)
                    else:
                        result = self._execute_step_action(step.action, step.inputs)
                    
                    step.execution_time = time.time() - start_time
                    return result
                    
                except Exception as e:
                    last_error = e
                    if attempt < step.retry_count:
                        wait_time = 2 ** attempt  # Exponential backoff
                        time.sleep(wait_time)
                        continue
                    else:
                        break
            
            raise last_error
            
        except Exception as e:
            step.execution_time = time.time() - start_time
            raise
    
    @abstractmethod
    def _execute_step_action(self, action: str, inputs: Dict[str, Any]) -> Any:
        """Execute a step action by name - implement in subclasses"""
        pass
    
    def _synthesize_workflow_results(self, results: Dict[str, Any]) -> Any:
        """Synthesize step results into final workflow result"""
        # Default implementation returns all results
        # Override in subclasses for custom result synthesis
        return results
    
    def _get_completed_results(self) -> Dict[str, Any]:
        """Get results from completed steps"""
        return {
            step.name: step.result
            for step in self.steps
            if step.status == WorkflowStepStatus.COMPLETED and step.result is not None
        }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow execution status"""
        status_counts = {}
        for status in WorkflowStepStatus:
            status_counts[status.value] = sum(1 for step in self.steps if step.status == status)
        
        return {
            "total_steps": len(self.steps),
            "status_counts": status_counts,
            "execution_mode": "parallel" if self.parallel_execution else "sequential",
            "continue_on_error": self.continue_on_error,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status.value,
                    "execution_time": step.execution_time,
                    "error": step.error
                }
                for step in self.steps
            ]
        }
    
    def reset_workflow(self):
        """Reset workflow to initial state"""
        for step in self.steps:
            step.status = WorkflowStepStatus.PENDING
            step.result = None
            step.error = None
            step.execution_time = None
        
        self.execution_context.clear()
        self.workflow_state.clear()
        
        self.logger.debug("Workflow reset to initial state")
    
    def save_workflow_state(self, filepath: Optional[str] = None) -> str:
        """Save workflow state to file"""
        import json
        import os
        
        if not filepath:
            filepath = f"{self.name}_state.json"
        
        state_data = {
            "workflow_name": self.name,
            "execution_context": self.execution_context,
            "workflow_state": self.workflow_state,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status.value,
                    "result": step.result,
                    "error": step.error,
                    "execution_time": step.execution_time
                }
                for step in self.steps
            ],
            "metrics": {
                "execution_count": self.execution_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "total_execution_time": self.total_execution_time
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        self.logger.info(f"Workflow state saved to {filepath}")
        return filepath
    
    def load_workflow_state(self, filepath: str):
        """Load workflow state from file"""
        import json
        
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Restore execution context and state
        self.execution_context = state_data.get("execution_context", {})
        self.workflow_state = state_data.get("workflow_state", {})
        
        # Restore step states
        step_states = {s["name"]: s for s in state_data.get("steps", [])}
        for step in self.steps:
            if step.name in step_states:
                step_state = step_states[step.name]
                step.status = WorkflowStepStatus(step_state["status"])
                step.result = step_state["result"]
                step.error = step_state["error"]
                step.execution_time = step_state["execution_time"]
        
        # Restore metrics
        metrics = state_data.get("metrics", {})
        self.execution_count = metrics.get("execution_count", 0)
        self.success_count = metrics.get("success_count", 0)
        self.failure_count = metrics.get("failure_count", 0)
        self.total_execution_time = metrics.get("total_execution_time", 0)
        
        self.logger.info(f"Workflow state loaded from {filepath}")
    
    def compose_with(self, other_workflow: 'AIWorkflowTemplate', **kwargs) -> 'AIWorkflowTemplate':
        """Compose this workflow with another workflow"""
        self.logger.debug(f"Composing workflow {self.name} with {other_workflow.name}")
        
        # Create a new workflow that executes both
        composed_name = f"{self.name}_composed_with_{other_workflow.name}"
        
        def composed_action(context, **inputs):
            # Execute this workflow first
            result1 = self.execute(**inputs)
            
            # Use result as input for second workflow
            combined_inputs = {**inputs, **result1} if isinstance(result1, dict) else {**inputs, "previous_result": result1}
            result2 = other_workflow.execute(**combined_inputs)
            
            return {"first_workflow": result1, "second_workflow": result2}
        
        # This would create a new workflow instance
        # For now, return self for method chaining
        return self
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count
            if self.execution_count > 0 else 0
        )
        
        success_rate = (
            self.success_count / self.execution_count * 100
            if self.execution_count > 0 else 0
        )
        
        return {
            "total_executions": self.execution_count,
            "successful_executions": self.success_count,
            "failed_executions": self.failure_count,
            "success_rate": round(success_rate, 2),
            "total_execution_time": round(self.total_execution_time, 2),
            "average_execution_time": round(avg_execution_time, 2),
            "total_steps": len(self.steps),
            "parallel_execution": self.parallel_execution
        }
    
    def _health_check(self) -> Dict[str, Any]:
        """Workflow-specific health check"""
        metrics = self.get_workflow_metrics()
        status = self.get_workflow_status()
        
        return {
            "status": "ok",
            "workflow_defined": len(self.steps) > 0,
            "execution_mode": "parallel" if self.parallel_execution else "sequential",
            "metrics": metrics,
            "current_status": status
        }
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make workflow callable directly"""
        return self.execute(*args, **kwargs)