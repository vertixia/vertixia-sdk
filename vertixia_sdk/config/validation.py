"""
Configuration validation utilities for AI-OS components
"""

import os
import yaml
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from pydantic import ValidationError

from .models import ComponentConfig, ComponentType, create_config


class ConfigValidator:
    """Validates AI-OS component configurations"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: ComponentConfig) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a component configuration
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Basic validation
        self._validate_metadata(config)
        self._validate_parameters(config)
        self._validate_dependencies(config)
        self._validate_type_specific(config)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_metadata(self, config: ComponentConfig):
        """Validate component metadata"""
        metadata = config.metadata
        
        if not metadata.name:
            self.errors.append("Component name is required")
        
        if not metadata.description:
            self.warnings.append("Component description is recommended")
        
        if not metadata.version:
            self.warnings.append("Component version is recommended")
        
        # Validate name format (alphanumeric, underscore, hyphen)
        if metadata.name and not metadata.name.replace("_", "").replace("-", "").isalnum():
            self.errors.append("Component name must be alphanumeric with underscores or hyphens")
    
    def _validate_parameters(self, config: ComponentConfig):
        """Validate component parameters"""
        param_names = set()
        
        for param in config.parameters:
            if param.name in param_names:
                self.errors.append(f"Duplicate parameter name: {param.name}")
            param_names.add(param.name)
            
            # Check environment variable conflicts
            if param.environment_variable:
                if not param.environment_variable.isupper():
                    self.warnings.append(f"Environment variable {param.environment_variable} should be uppercase")
    
    def _validate_dependencies(self, config: ComponentConfig):
        """Validate component dependencies"""
        dep_names = set()
        
        for dep in config.dependencies:
            if dep.name in dep_names:
                self.warnings.append(f"Duplicate dependency: {dep.name}")
            dep_names.add(dep.name)
    
    def _validate_type_specific(self, config: ComponentConfig):
        """Validate type-specific configuration"""
        if config.type == ComponentType.AGENT:
            self._validate_agent_config(config)
        elif config.type == ComponentType.WORKFLOW:
            self._validate_workflow_config(config)
        elif config.type == ComponentType.REASONING:
            self._validate_reasoning_config(config)
        elif config.type == ComponentType.AUTOMATION:
            self._validate_automation_config(config)
        elif config.type == ComponentType.INTELLIGENCE:
            self._validate_intelligence_config(config)
    
    def _validate_agent_config(self, config: ComponentConfig):
        """Validate agent-specific configuration"""
        if hasattr(config, 'goal') and not config.goal:
            self.warnings.append("Agent goal is recommended for better performance")
        
        if hasattr(config, 'tools') and config.tools:
            for tool in config.tools:
                if not isinstance(tool, str):
                    self.errors.append(f"Tool reference must be string: {tool}")
    
    def _validate_workflow_config(self, config: ComponentConfig):
        """Validate workflow-specific configuration"""
        if hasattr(config, 'steps') and not config.steps:
            self.errors.append("Workflow must have at least one step")
        
        if hasattr(config, 'parallel_execution') and config.parallel_execution:
            if not hasattr(config, 'max_workers') or config.max_workers < 1:
                self.errors.append("Parallel workflows must specify max_workers >= 1")
    
    def _validate_reasoning_config(self, config: ComponentConfig):
        """Validate reasoning-specific configuration"""
        if hasattr(config, 'max_iterations') and config.max_iterations < 1:
            self.errors.append("Reasoning max_iterations must be >= 1")
        
        if hasattr(config, 'quality_threshold'):
            if config.quality_threshold < 0 or config.quality_threshold > 1:
                self.errors.append("Quality threshold must be between 0 and 1")
    
    def _validate_automation_config(self, config: ComponentConfig):
        """Validate automation-specific configuration"""
        if hasattr(config, 'schedule') and config.schedule:
            # Basic cron validation
            try:
                import croniter
                croniter.croniter(config.schedule)
            except ImportError:
                self.warnings.append("Install croniter to validate cron expressions")
            except Exception:
                self.errors.append(f"Invalid cron expression: {config.schedule}")
    
    def _validate_intelligence_config(self, config: ComponentConfig):
        """Validate intelligence-specific configuration"""
        if hasattr(config, 'confidence_threshold'):
            if config.confidence_threshold < 0 or config.confidence_threshold > 1:
                self.errors.append("Confidence threshold must be between 0 and 1")


def validate_config_file(file_path: str) -> Tuple[bool, Optional[ComponentConfig], List[str], List[str]]:
    """
    Validate a component configuration file
    
    Args:
        file_path: Path to YAML or JSON configuration file
    
    Returns:
        Tuple of (is_valid, config_object, errors, warnings)
    """
    errors = []
    warnings = []
    config = None
    
    try:
        # Load file
        path = Path(file_path)
        if not path.exists():
            errors.append(f"Configuration file not found: {file_path}")
            return False, None, errors, warnings
        
        with open(path, 'r') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                errors.append(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")
                return False, None, errors, warnings
        
        # Parse configuration
        component_type = ComponentType(data.get('type', 'service'))
        config = create_config(component_type, **data)
        
        # Validate
        validator = ConfigValidator()
        is_valid, validation_errors, validation_warnings = validator.validate_config(config)
        
        errors.extend(validation_errors)
        warnings.extend(validation_warnings)
        
        return is_valid, config, errors, warnings
        
    except ValidationError as e:
        errors.append(f"Configuration validation error: {e}")
        return False, None, errors, warnings
    except Exception as e:
        errors.append(f"Error loading configuration: {e}")
        return False, None, errors, warnings


def validate_environment_variables(config: ComponentConfig) -> Tuple[List[str], List[str]]:
    """
    Validate that required environment variables are set
    
    Returns:
        Tuple of (missing_vars, warnings)
    """
    missing = []
    warnings = []
    
    for param in config.parameters:
        if param.environment_variable:
            if param.required and not os.environ.get(param.environment_variable):
                missing.append(param.environment_variable)
            elif not param.required and not os.environ.get(param.environment_variable):
                warnings.append(f"Optional environment variable not set: {param.environment_variable}")
    
    return missing, warnings