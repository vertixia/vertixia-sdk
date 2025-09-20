"""
Configuration models for AI-OS components

Provides Pydantic models for component configuration with validation
and environment-based parameter management.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ComponentType(str, Enum):
    """Types of AI-OS components"""
    SERVICE = "service"
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    REASONING = "reasoning"
    AUTOMATION = "automation"
    INTELLIGENCE = "intelligence"
    INTEGRATION = "integration"


class ExecutionMode(str, Enum):
    """Component execution modes"""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"


class ParameterType(str, Enum):
    """Parameter types for dynamic configuration"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    SECRET = "secret"


class ComponentParameter(BaseModel):
    """Dynamic parameter definition"""
    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Optional[Any] = None
    validation: Optional[Dict[str, Any]] = None
    environment_variable: Optional[str] = None


class ComponentMetadata(BaseModel):
    """Component metadata for marketplace"""
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    license: Optional[str] = "MIT"
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    icon: Optional[str] = None


class ComponentDependency(BaseModel):
    """Component dependency specification"""
    name: str
    version: Optional[str] = None
    optional: bool = False
    environment_marker: Optional[str] = None


class ComponentConfig(BaseModel):
    """Base configuration for all AI-OS components"""
    model_config = ConfigDict(extra="allow")
    
    metadata: ComponentMetadata
    type: ComponentType
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    parameters: List[ComponentParameter] = Field(default_factory=list)
    dependencies: List[ComponentDependency] = Field(default_factory=list)
    
    # Runtime configuration
    timeout: Optional[int] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Integration settings
    ai_os_version: Optional[str] = None
    langflow_compatible: bool = True
    marketplace_id: Optional[str] = None


class AgentConfig(ComponentConfig):
    """Configuration for AI Agent components"""
    type: ComponentType = ComponentType.AGENT
    
    # Agent-specific settings
    goal: Optional[str] = None
    role: Optional[str] = None
    backstory: Optional[str] = None
    verbose: bool = False
    delegation: bool = True
    step_callback: Optional[str] = None
    
    # Tool integration
    tools: List[str] = Field(default_factory=list)
    max_execution_time: Optional[int] = None
    memory_enabled: bool = True


class WorkflowConfig(ComponentConfig):
    """Configuration for AI Workflow templates"""
    type: ComponentType = ComponentType.WORKFLOW
    
    # Workflow structure
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    error_handling: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution settings
    parallel_execution: bool = False
    max_workers: int = 4
    continue_on_error: bool = False
    
    # State management
    state_persistence: bool = False
    state_storage: Optional[str] = None


class ReasoningConfig(ComponentConfig):
    """Configuration for AI Reasoning components (like ITRS)"""
    type: ComponentType = ComponentType.REASONING
    
    # Reasoning parameters
    max_iterations: int = 10
    quality_threshold: float = 0.8
    reasoning_strategies: List[str] = Field(default_factory=list)
    
    # LLM configuration
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # Memory and context
    context_window: int = 4000
    memory_type: str = "short_term"
    knowledge_base: Optional[str] = None


class AutomationConfig(ComponentConfig):
    """Configuration for OS Automation components"""
    type: ComponentType = ComponentType.AUTOMATION
    
    # Automation settings
    schedule: Optional[str] = None  # Cron expression
    trigger_events: List[str] = Field(default_factory=list)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # System integration
    permissions: List[str] = Field(default_factory=list)
    system_access: Dict[str, bool] = Field(default_factory=dict)
    notification_channels: List[str] = Field(default_factory=list)


class IntelligenceConfig(ComponentConfig):
    """Configuration for AI Intelligence/Analysis components"""
    type: ComponentType = ComponentType.INTELLIGENCE
    
    # Analysis settings
    analysis_type: str = "pattern_detection"
    data_sources: List[str] = Field(default_factory=list)
    output_format: str = "json"
    
    # Processing parameters
    batch_size: int = 100
    processing_mode: str = "real_time"
    confidence_threshold: float = 0.75
    
    # ML/AI settings
    model_path: Optional[str] = None
    preprocessing_steps: List[str] = Field(default_factory=list)
    feature_extraction: Dict[str, Any] = Field(default_factory=dict)


# Factory function for creating appropriate config
def create_config(component_type: ComponentType, **kwargs) -> ComponentConfig:
    """Create appropriate configuration based on component type"""
    config_classes = {
        ComponentType.AGENT: AgentConfig,
        ComponentType.WORKFLOW: WorkflowConfig,
        ComponentType.REASONING: ReasoningConfig,
        ComponentType.AUTOMATION: AutomationConfig,
        ComponentType.INTELLIGENCE: IntelligenceConfig,
    }
    
    config_class = config_classes.get(component_type, ComponentConfig)
    return config_class(**kwargs)