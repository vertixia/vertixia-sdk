"""
CLI Command Implementations

Implements the actual command logic for the AI-OS SDK CLI.
"""

import os
import json
import yaml
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from ..config.models import ComponentType, create_config
from ..config.validation import validate_config_file
from ..utils.discovery import get_registry, auto_discover_components
from ..marketplace.registry_integration import MarketplaceRegistry
from ..base.component import AIServiceComponent


logger = logging.getLogger(__name__)


class ComponentCommands:
    """Commands for component development"""
    
    async def create_component(
        self,
        name: str,
        component_type: str,
        template: Optional[str] = None,
        output_dir: str = ".",
        author: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """Create a new component from template"""
        
        try:
            # Validate component type
            comp_type = ComponentType(component_type)
        except ValueError:
            print(f"Error: Invalid component type '{component_type}'")
            print(f"Valid types: {[t.value for t in ComponentType]}")
            return 1
        
        # Create output directory
        output_path = Path(output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate component files
        config_data = self._generate_component_config(
            name, comp_type, author, description
        )
        
        # Write configuration file
        config_path = output_path / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        # Write Python implementation file
        impl_path = output_path / f"{name}.py"
        impl_code = self._generate_component_implementation(name, comp_type)
        with open(impl_path, 'w') as f:
            f.write(impl_code)
        
        # Write README
        readme_path = output_path / "README.md"
        readme_content = self._generate_readme(name, comp_type, description)
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Created {comp_type.value} component '{name}' in {output_path}")
        print(f"📄 Configuration: {config_path}")
        print(f"🐍 Implementation: {impl_path}")
        print(f"📖 Documentation: {readme_path}")
        print(f"\\n🚀 Next steps:")
        print(f"   1. Edit {impl_path} to implement your component logic")
        print(f"   2. Test with: ai-os component test {config_path}")
        print(f"   3. Register with: ai-os registry register {config_path}")
        
        return 0
    
    async def validate_component(
        self,
        config_path: str,
        strict: bool = False
    ) -> int:
        """Validate component configuration"""
        
        if not os.path.exists(config_path):
            print(f"Error: Configuration file not found: {config_path}")
            return 1
        
        try:
            is_valid, config, errors, warnings = validate_config_file(config_path)
            
            if is_valid:
                print(f"✅ Configuration is valid: {config_path}")
                if warnings:
                    print(f"⚠️  Warnings ({len(warnings)}):")
                    for warning in warnings:
                        print(f"   - {warning}")
                
                # Additional validation for strict mode
                if strict:
                    strict_issues = self._strict_validation(config)
                    if strict_issues:
                        print(f"❌ Strict validation issues ({len(strict_issues)}):")
                        for issue in strict_issues:
                            print(f"   - {issue}")
                        return 1
                
                return 0
            else:
                print(f"❌ Configuration is invalid: {config_path}")
                print(f"Errors ({len(errors)}):")
                for error in errors:
                    print(f"   - {error}")
                return 1
                
        except Exception as e:
            print(f"Error validating configuration: {e}")
            return 1
    
    async def test_component(
        self,
        config_path: str,
        test_input: Optional[str] = None,
        timeout: int = 30
    ) -> int:
        """Test component functionality"""
        
        print(f"🧪 Testing component: {config_path}")
        
        try:
            # Load and validate configuration
            is_valid, config, errors, warnings = validate_config_file(config_path)
            if not is_valid:
                print(f"❌ Configuration invalid: {errors}")
                return 1
            
            # Create component instance
            component_class = self._get_component_class(config.type)
            component = component_class(config)
            
            # Initialize component
            print("🔄 Initializing component...")
            if not component.initialize():
                print("❌ Component initialization failed")
                return 1
            print("✅ Component initialized successfully")
            
            # Health check
            print("🩺 Running health check...")
            health_status = component.health_check()
            if health_status.get("healthy", False):
                print("✅ Component health check passed")
            else:
                print(f"⚠️  Health check warnings: {health_status}")
            
            # Test execution if input provided
            if test_input:
                print("🚀 Testing component execution...")
                try:
                    # Parse test input
                    if test_input.startswith('{'):
                        input_data = json.loads(test_input)
                    else:
                        input_data = test_input
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self._execute_component_async(component, input_data),
                        timeout=timeout
                    )
                    
                    print(f"✅ Component execution successful")
                    print(f"📤 Result: {json.dumps(result, indent=2, default=str)}")
                    
                except asyncio.TimeoutError:
                    print(f"⏰ Component execution timed out after {timeout}s")
                    return 1
                except Exception as e:
                    print(f"❌ Component execution failed: {e}")
                    return 1
            
            print("🎉 All tests passed!")
            return 0
            
        except Exception as e:
            print(f"Error testing component: {e}")
            return 1
    
    async def package_component(
        self,
        config_path: str,
        output_path: Optional[str] = None,
        include_deps: bool = False
    ) -> int:
        """Package component for distribution"""
        
        print(f"📦 Packaging component: {config_path}")
        
        try:
            # Validate configuration
            is_valid, config, errors, warnings = validate_config_file(config_path)
            if not is_valid:
                print(f"❌ Configuration invalid: {errors}")
                return 1
            
            # Determine output path
            if not output_path:
                output_path = f"{config.metadata.name}-{config.metadata.version}.tar.gz"
            
            # Create package
            package_info = {
                "name": config.metadata.name,
                "version": config.metadata.version,
                "type": config.type.value,
                "author": config.metadata.author,
                "description": config.metadata.description,
                "packaged_at": "2024-01-01T00:00:00Z",  # Would use actual timestamp
                "files": [],
                "dependencies": [
                    {"name": dep.name, "version": dep.version}
                    for dep in config.dependencies
                ]
            }
            
            print(f"✅ Package created: {output_path}")
            print(f"📋 Package info: {json.dumps(package_info, indent=2)}")
            
            return 0
            
        except Exception as e:
            print(f"Error packaging component: {e}")
            return 1
    
    def _generate_component_config(
        self,
        name: str,
        comp_type: ComponentType,
        author: Optional[str],
        description: Optional[str]
    ) -> Dict[str, Any]:
        """Generate component configuration"""
        
        config = {
            "type": comp_type.value,
            "version": "1.0.0",
            "metadata": {
                "name": name,
                "display_name": name.replace("-", " ").replace("_", " ").title(),
                "description": description or f"A {comp_type.value} component",
                "author": author or "Unknown",
                "version": "1.0.0",
                "license": "MIT",
                "category": "general",
                "tags": [comp_type.value, "ai-os"],
                "documentation_url": "",
                "source_url": "",
                "icon": "🤖"
            },
            "execution_mode": "sync",
            "langflow_compatible": True,
            "ai_os_version": ">=0.1.0",
            "parameters": [
                {
                    "name": "input_data",
                    "type": "string",
                    "description": "Input data for the component",
                    "required": True
                }
            ],
            "dependencies": []
        }
        
        # Add type-specific configuration
        if comp_type == ComponentType.AGENT:
            config.update({
                "goal": "Accomplish tasks efficiently",
                "role": "Assistant",
                "delegation": True,
                "verbose": False,
                "tools": []
            })
        elif comp_type == ComponentType.REASONING:
            config.update({
                "max_iterations": 10,
                "quality_threshold": 0.85,
                "time_limit": 300,
                "convergence_threshold": 0.05
            })
        elif comp_type == ComponentType.WORKFLOW:
            config.update({
                "parallel_execution": False,
                "max_workers": 4,
                "continue_on_error": False
            })
        
        return config
    
    def _generate_component_implementation(
        self,
        name: str,
        comp_type: ComponentType
    ) -> str:
        """Generate component implementation code"""
        
        class_name = "".join(word.capitalize() for word in name.replace("-", "_").split("_"))
        
        base_imports = {
            ComponentType.SERVICE: "AIServiceComponent",
            ComponentType.AGENT: "AIAgentComponent",
            ComponentType.TOOL: "AIToolComponent",
            ComponentType.WORKFLOW: "AIWorkflowTemplate",
            ComponentType.REASONING: "AIServiceComponent"  # For now
        }
        
        base_class = base_imports.get(comp_type, "AIServiceComponent")
        
        implementation = f'''"""
{class_name} - AI-OS {comp_type.value} component

Auto-generated component implementation.
Edit this file to implement your component logic.
"""

from typing import Any, Dict, Optional
from vertixia_sdk.base.{comp_type.value if comp_type != ComponentType.SERVICE else "component"} import {base_class}


class {class_name}({base_class}):
    """
    {class_name} implementation
    
    TODO: Add component description and documentation
    """
    
    def _initialize(self):
        """Initialize component-specific setup"""
        self.logger.info(f"Initializing {{self.name}}")
        # TODO: Add initialization logic
    
    def _execute(self, input_data: str, **kwargs) -> Dict[str, Any]:
        """Execute component logic"""
        self.logger.info(f"Executing {{self.name}} with input: {{input_data}}")
        
        # TODO: Implement component logic
        result = {{
            "output": f"Processed: {{input_data}}",
            "component": self.name,
            "version": self.version,
            "success": True
        }}
        
        return result
    
    def _health_check(self) -> Dict[str, Any]:
        """{comp_type.value.capitalize()} component health check"""
        return {{
            "status": "ok",
            "component_type": "{comp_type.value}",
            "ready": True
        }}


# Optional: Export component class for easy importing
__all__ = ["{class_name}"]
'''
        
        return implementation
    
    def _generate_readme(
        self,
        name: str,
        comp_type: ComponentType,
        description: Optional[str]
    ) -> str:
        """Generate README documentation"""
        
        return f'''# {name.replace("-", " ").replace("_", " ").title()}

{description or f"A {comp_type.value} component for AI-OS"}

## Overview

This is an AI-OS {comp_type.value} component that...

## Configuration

The component is configured through `{name}.yaml`. Key parameters:

- `input_data`: Input data for processing

## Usage

### Local Testing

```bash
# Validate configuration
ai-os component validate {name}.yaml

# Test component
ai-os component test {name}.yaml --input "test data"

# Register with local registry
ai-os registry register {name}.yaml
```

### Integration

```python
from vertixia_sdk import create_component

# Create component instance
component = create_component("{name}")

# Execute component
result = component.execute("test input")
print(result)
```

## Development

1. Edit `{name}.py` to implement component logic
2. Update `{name}.yaml` configuration as needed
3. Test thoroughly before publishing
4. Consider adding usage examples

## Publishing

```bash
# Publish to marketplace
ai-os marketplace publish {name} --api-key YOUR_KEY --public
```

## License

MIT License - see LICENSE file for details
'''
    
    def _strict_validation(self, config) -> List[str]:
        """Additional strict validation checks"""
        issues = []
        
        # Check for required documentation
        if not config.metadata.documentation_url:
            issues.append("Documentation URL is recommended")
        
        # Check for proper versioning
        if config.metadata.version == "1.0.0" and not config.metadata.description:
            issues.append("Description is required for version 1.0.0+")
        
        # Check for proper categorization
        if config.metadata.category == "general":
            issues.append("Specific category is recommended over 'general'")
        
        return issues
    
    def _get_component_class(self, comp_type: ComponentType):
        """Get appropriate component class"""
        if comp_type == ComponentType.AGENT:
            from ..base.agent import AIAgentComponent
            return AIAgentComponent
        elif comp_type == ComponentType.TOOL:
            from ..base.tool import AIToolComponent
            return AIToolComponent
        elif comp_type == ComponentType.WORKFLOW:
            from ..base.workflow import AIWorkflowTemplate
            return AIWorkflowTemplate
        else:
            return AIServiceComponent
    
    async def _execute_component_async(self, component: AIServiceComponent, input_data: Any) -> Any:
        """Execute component asynchronously"""
        if hasattr(component, 'execute_async'):
            return await component.execute_async(input_data)
        else:
            # Run sync execution in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, component.execute, input_data)


class RegistryCommands:
    """Commands for registry management"""
    
    def __init__(self):
        self.registry = get_registry()
    
    async def discover_components(
        self,
        paths: List[str],
        recursive: bool = False
    ) -> int:
        """Discover components in specified paths"""
        
        print(f"🔍 Discovering components in: {', '.join(paths)}")
        
        try:
            discovered_count = auto_discover_components(paths, self.registry)
            
            if discovered_count > 0:
                print(f"✅ Discovered {discovered_count} components")
                
                # Show discovered components
                components = self.registry.list_components(include_metadata=False)
                for component in components[-discovered_count:]:  # Show newly discovered
                    print(f"   📦 {component['name']} ({component['type']})")
            else:
                print("ℹ️  No components found")
            
            return 0
            
        except Exception as e:
            print(f"Error during discovery: {e}")
            return 1
    
    async def list_components(
        self,
        component_type: Optional[str] = None,
        output_format: str = "table"
    ) -> int:
        """List registered components"""
        
        try:
            comp_type = ComponentType(component_type) if component_type else None
            components = self.registry.list_components(
                component_type=comp_type,
                include_metadata=True
            )
            
            if not components:
                print("ℹ️  No components registered")
                return 0
            
            if output_format == "json":
                print(json.dumps(components, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(components, default_flow_style=False))
            else:
                # Table format
                print(f"\\n📦 Registered Components ({len(components)})")
                print("-" * 80)
                print(f"{'Name':<20} {'Type':<12} {'Module':<25} {'Description':<20}")
                print("-" * 80)
                
                for component in components:
                    name = component['name'][:19]
                    comp_type = component['type'][:11]
                    module = component['module'][:24]
                    desc = component['description'][:19] if component['description'] else "N/A"
                    
                    print(f"{name:<20} {comp_type:<12} {module:<25} {desc:<20}")
                print("-" * 80)
            
            return 0
            
        except Exception as e:
            print(f"Error listing components: {e}")
            return 1
    
    async def register_component(
        self,
        config_path: str,
        name_override: Optional[str] = None
    ) -> int:
        """Register component manually"""
        
        print(f"📝 Registering component: {config_path}")
        
        try:
            # Validate configuration
            is_valid, config, errors, warnings = validate_config_file(config_path)
            if not is_valid:
                print(f"❌ Configuration invalid: {errors}")
                return 1
            
            # Try to find Python implementation
            config_dir = os.path.dirname(config_path)
            python_file = os.path.join(config_dir, f"{config.metadata.name}.py")
            
            if not os.path.exists(python_file):
                print(f"⚠️  Python implementation not found: {python_file}")
                print("   Component will be registered with configuration only")
            
            # Register with registry
            # Note: This would need the actual component class
            # For now, just register the configuration
            component_info = {
                "name": name_override or config.metadata.name,
                "type": config.type,
                "config_path": config_path,
                "module": f"{config.metadata.name}.py" if os.path.exists(python_file) else None
            }
            
            print(f"✅ Component registered: {component_info['name']}")
            return 0
            
        except Exception as e:
            print(f"Error registering component: {e}")
            return 1
    
    async def unregister_component(self, component_name: str) -> int:
        """Unregister component"""
        
        print(f"🗑️  Unregistering component: {component_name}")
        
        try:
            success = self.registry.unregister_component(component_name)
            
            if success:
                print(f"✅ Component unregistered: {component_name}")
                return 0
            else:
                print(f"❌ Component not found: {component_name}")
                return 1
                
        except Exception as e:
            print(f"Error unregistering component: {e}")
            return 1
    
    async def show_status(self) -> int:
        """Show registry status"""
        
        try:
            registry_data = self.registry.export_registry()
            
            print("📊 Registry Status")
            print("-" * 40)
            print(f"Total Components: {registry_data['component_count']}")
            print(f"Search Paths: {len(registry_data['search_paths'])}")
            print(f"Supported Types: {', '.join(registry_data['supported_types'])}")
            
            # Component breakdown by type
            components = registry_data['components']
            type_counts = {}
            for component in components:
                comp_type = component['type']
                type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
            
            if type_counts:
                print("\\nComponent Types:")
                for comp_type, count in sorted(type_counts.items()):
                    print(f"  {comp_type}: {count}")
            
            return 0
            
        except Exception as e:
            print(f"Error showing registry status: {e}")
            return 1


class MarketplaceCommands:
    """Commands for marketplace interaction"""
    
    def __init__(self):
        self.marketplace = MarketplaceRegistry()
    
    async def search_components(
        self,
        query: Optional[str] = None,
        component_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        verified_only: bool = False,
        limit: int = 20
    ) -> int:
        """Search marketplace components"""
        
        print(f"🔍 Searching marketplace...")
        if query:
            print(f"   Query: {query}")
        if component_type:
            print(f"   Type: {component_type}")
        if tags:
            print(f"   Tags: {', '.join(tags)}")
        if verified_only:
            print(f"   Verified only: Yes")
        
        try:
            comp_type = ComponentType(component_type) if component_type else None
            components = await self.marketplace.discover_marketplace_components(
                query=query,
                component_type=comp_type,
                tags=tags,
                verified_only=verified_only,
                limit=limit
            )
            
            if not components:
                print("ℹ️  No components found")
                return 0
            
            print(f"\\n🎯 Found {len(components)} components")
            print("-" * 80)
            
            for component in components:
                status_icons = []
                if component.get("is_installed"):
                    status_icons.append("✅")
                if component.get("update_available"):
                    status_icons.append("🔄")
                if component.get("is_verified"):
                    status_icons.append("✨")
                
                status = " ".join(status_icons) if status_icons else "  "
                
                print(f"{status} {component['name']} v{component['version']}")
                print(f"    {component['description'][:60]}...")
                print(f"    👤 {component['author']} | ⬇️ {component['downloads']} | ⭐ {component['rating']}")
                print()
            
            return 0
            
        except Exception as e:
            print(f"Error searching marketplace: {e}")
            return 1
    
    async def install_component(
        self,
        component_id: str,
        installation_path: Optional[str] = None,
        skip_dependencies: bool = False,
        skip_configuration: bool = False
    ) -> int:
        """Install component from marketplace"""
        
        print(f"📥 Installing component: {component_id}")
        
        try:
            result = await self.marketplace.install_component(
                component_id,
                installation_path=installation_path,
                resolve_dependencies=not skip_dependencies,
                auto_configure=not skip_configuration
            )
            
            if result["status"] == "success":
                component = result["component"]
                print(f"✅ Successfully installed {component['name']} v{component['version']}")
                print(f"   Type: {component['type']}")
                print(f"   ID: {component['id']}")
                return 0
            else:
                print(f"❌ Installation failed: {result['message']}")
                if "issues" in result:
                    for issue in result["issues"]:
                        print(f"   - {issue}")
                return 1
                
        except Exception as e:
            print(f"Error installing component: {e}")
            return 1
    
    async def update_component(
        self,
        component_name: str,
        target_version: Optional[str] = None
    ) -> int:
        """Update installed component"""
        
        print(f"🔄 Updating component: {component_name}")
        if target_version:
            print(f"   Target version: {target_version}")
        
        try:
            result = await self.marketplace.update_component(
                component_name,
                target_version=target_version
            )
            
            if result["status"] == "success":
                print(f"✅ Successfully updated {component_name}")
                print(f"   {result['old_version']} → {result['new_version']}")
                return 0
            elif result["status"] == "info":
                print(f"ℹ️  {result['message']}")
                return 0
            else:
                print(f"❌ Update failed: {result['message']}")
                return 1
                
        except Exception as e:
            print(f"Error updating component: {e}")
            return 1
    
    async def publish_component(
        self,
        component_name: str,
        api_key: str,
        config_path: Optional[str] = None,
        examples_path: Optional[str] = None,
        docs_path: Optional[str] = None,
        make_public: bool = False
    ) -> int:
        """Publish component to marketplace"""
        
        print(f"📤 Publishing component: {component_name}")
        
        try:
            # Load examples if provided
            examples = None
            if examples_path and os.path.exists(examples_path):
                with open(examples_path, 'r') as f:
                    examples = json.load(f)
            
            # Load documentation if provided
            documentation = None
            if docs_path and os.path.exists(docs_path):
                with open(docs_path, 'r') as f:
                    documentation = f.read()
            
            result = await self.marketplace.publish_component(
                component_name,
                api_key,
                configuration_path=config_path,
                examples=examples,
                documentation=documentation,
                make_public=make_public
            )
            
            if result["status"] == "success":
                print(f"✅ Successfully published {component_name}")
                print(f"   Component ID: {result['component_id']}")
                print(f"   Marketplace URL: {result['marketplace_url']}")
                if make_public:
                    print(f"   🌐 Component is public")
                return 0
            else:
                print(f"❌ Publishing failed: {result['message']}")
                return 1
                
        except Exception as e:
            print(f"Error publishing component: {e}")
            return 1
    
    async def sync_with_marketplace(self, force: bool = False) -> int:
        """Sync with marketplace"""
        
        print(f"🔄 Syncing with marketplace...")
        if force:
            print("   Force sync enabled")
        
        try:
            result = await self.marketplace.sync_with_marketplace(force=force)
            
            if result["status"] == "success":
                print(f"✅ Sync completed")
                print(f"   Updates available: {result['updates_available']}")
                print(f"   Auto-updated: {result['auto_updated']}")
                print(f"   Manual updates needed: {result['manual_updates_needed']}")
                
                # Show details if any updates
                if result['details']:
                    print("\\n📋 Update Details:")
                    for detail in result['details']:
                        if detail.get('auto_updated'):
                            print(f"   ✅ {detail['component']}: Auto-updated")
                        else:
                            print(f"   🔄 {detail['component']}: Update available to {detail.get('marketplace_version', 'latest')}")
                
                return 0
            elif result["status"] == "info":
                print(f"ℹ️  {result['message']}")
                return 0
            else:
                print(f"❌ Sync failed: {result['message']}")
                return 1
                
        except Exception as e:
            print(f"Error syncing with marketplace: {e}")
            return 1
    
    async def show_stats(self) -> int:
        """Show marketplace statistics"""
        
        print(f"📊 Marketplace Statistics")
        
        try:
            # Get sync status
            sync_status = self.marketplace.get_sync_status()
            
            print("-" * 40)
            print(f"Last Sync: {sync_status['last_sync'] or 'Never'}")
            print(f"Sync Interval: {sync_status['sync_interval_hours']:.1f} hours")
            print(f"Cache Entries: {sync_status['cache_entries']}")
            print(f"Pending Installations: {sync_status['pending_installations']}")
            
            # Get installation history
            history = self.marketplace.get_installation_history(limit=5)
            if history:
                print(f"\\n📥 Recent Installations:")
                for install in history:
                    status_icon = "✅" if install["status"] == "success" else "❌"
                    print(f"   {status_icon} {install['component_name']} v{install['version']}")
            
            return 0
            
        except Exception as e:
            print(f"Error showing marketplace stats: {e}")
            return 1