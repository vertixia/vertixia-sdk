"""
Extended Store Service for AI-OS Components

Extends the existing Langflow store service with AI-OS specific functionality
for component marketplace integration, discovery, and deployment.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import UUID
import httpx
from dataclasses import dataclass, asdict

from ..base.component import AIServiceComponent
from ..config.models import ComponentConfig, ComponentType
from ..utils.discovery import ComponentRegistry, get_registry


@dataclass
class AIComponentMetadata:
    """Enhanced metadata for AI-OS components"""
    component_type: ComponentType
    ai_os_version: str
    sdk_version: str
    configuration_schema: Dict[str, Any]
    execution_modes: List[str]
    dependencies: List[Dict[str, str]]
    integration_points: List[str]
    performance_metrics: Optional[Dict[str, Any]] = None
    compatibility_matrix: Optional[Dict[str, Any]] = None
    deployment_info: Optional[Dict[str, Any]] = None


@dataclass
class AIStoreComponent:
    """AI-OS specific component representation for marketplace"""
    id: Optional[str]
    name: str
    display_name: str
    description: str
    component_type: ComponentType
    version: str
    author: str
    license: str
    tags: List[str]
    category: str
    
    # AI-OS specific fields
    ai_metadata: AIComponentMetadata
    configuration: Dict[str, Any]
    usage_examples: List[Dict[str, Any]]
    integration_guide: Optional[str] = None
    api_documentation: Optional[str] = None
    
    # Marketplace fields
    downloads: int = 0
    likes: int = 0
    rating: float = 0.0
    is_verified: bool = False
    is_private: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AIStoreService:
    """
    Extended store service for AI-OS components
    
    Provides enhanced marketplace functionality specifically designed for
    AI-OS components with configuration-driven deployment and discovery.
    """
    
    def __init__(
        self,
        base_store_service: Optional[Any] = None,
        registry: Optional[ComponentRegistry] = None,
        ai_os_store_url: Optional[str] = None
    ):
        self.base_store_service = base_store_service
        self.registry = registry or get_registry()
        self.ai_os_store_url = ai_os_store_url or "https://marketplace.ai-os.com/api/v1"
        self.timeout = 30
        
        # AI-OS specific endpoints
        self.ai_components_url = f"{self.ai_os_store_url}/components"
        self.templates_url = f"{self.ai_os_store_url}/templates"
        self.workflows_url = f"{self.ai_os_store_url}/workflows"
        self.configurations_url = f"{self.ai_os_store_url}/configurations"
        
        # Enhanced search fields for AI-OS components
        self.ai_search_fields = [
            "id", "name", "display_name", "description", "component_type",
            "version", "author", "license", "tags", "category",
            "ai_metadata", "configuration", "usage_examples",
            "downloads", "likes", "rating", "is_verified", "is_private",
            "created_at", "updated_at"
        ]
    
    async def search_ai_components(
        self,
        query: Optional[str] = None,
        component_type: Optional[ComponentType] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        min_rating: Optional[float] = None,
        verified_only: bool = False,
        sort_by: str = "downloads",
        sort_order: str = "desc",
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search for AI-OS components in the marketplace
        
        Args:
            query: Text search query
            component_type: Filter by component type
            tags: Filter by tags
            author: Filter by author
            min_rating: Minimum rating filter
            verified_only: Show only verified components
            sort_by: Sort field (downloads, likes, rating, created_at)
            sort_order: Sort order (asc, desc)
            page: Page number
            limit: Results per page
            
        Returns:
            Search results with components and metadata
        """
        params = {
            "page": page,
            "limit": limit,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
        
        filters = {}
        if query:
            filters["search"] = query
        if component_type:
            filters["component_type"] = component_type.value
        if tags:
            filters["tags"] = tags
        if author:
            filters["author"] = author
        if min_rating:
            filters["min_rating"] = min_rating
        if verified_only:
            filters["verified_only"] = True
        
        if filters:
            params["filters"] = json.dumps(filters)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.ai_components_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
        return response.json()
    
    async def get_ai_component(self, component_id: str) -> AIStoreComponent:
        """Get detailed information about an AI-OS component"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ai_components_url}/{component_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
        component_data = response.json()
        return self._parse_ai_component(component_data)
    
    async def publish_ai_component(
        self,
        component: AIServiceComponent,
        api_key: str,
        configuration_path: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        documentation: Optional[str] = None
    ) -> str:
        """
        Publish an AI-OS component to the marketplace
        
        Args:
            component: The AI-OS component instance
            api_key: User's API key
            configuration_path: Path to component configuration file
            examples: Usage examples
            documentation: Component documentation
            
        Returns:
            Published component ID
        """
        # Extract component metadata
        component_data = self._extract_component_data(
            component, configuration_path, examples, documentation
        )
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ai_components_url,
                headers=headers,
                json=component_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
        result = response.json()
        return result["id"]
    
    async def update_ai_component(
        self,
        component_id: str,
        component: AIServiceComponent,
        api_key: str,
        configuration_path: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        documentation: Optional[str] = None
    ) -> bool:
        """Update an existing AI-OS component in the marketplace"""
        component_data = self._extract_component_data(
            component, configuration_path, examples, documentation
        )
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.ai_components_url}/{component_id}",
                headers=headers,
                json=component_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
        return True
    
    async def install_ai_component(
        self,
        component_id: str,
        installation_path: Optional[str] = None,
        auto_register: bool = True
    ) -> AIServiceComponent:
        """
        Install an AI-OS component from the marketplace
        
        Args:
            component_id: Component ID to install
            installation_path: Local installation path
            auto_register: Automatically register with local registry
            
        Returns:
            Installed component instance
        """
        # Download component metadata and configuration
        component_info = await self.get_ai_component(component_id)
        
        # Download component files
        download_data = await self._download_component_files(component_id)
        
        # Install component locally
        installed_component = await self._install_component_locally(
            component_info, download_data, installation_path
        )
        
        # Register with local registry if requested
        if auto_register:
            self.registry.register_component(
                installed_component.__class__,
                config_path=installation_path,
                metadata=asdict(component_info.ai_metadata)
            )
        
        return installed_component
    
    async def get_component_templates(
        self,
        component_type: Optional[ComponentType] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get available component templates"""
        params = {}
        if component_type:
            params["component_type"] = component_type.value
        if category:
            params["category"] = category
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.templates_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
        return response.json()["templates"]
    
    async def create_from_template(
        self,
        template_id: str,
        component_name: str,
        configuration: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> AIServiceComponent:
        """Create a new component from a template"""
        # Get template
        template_data = await self._get_template(template_id)
        
        # Customize with provided configuration
        customized_config = self._customize_template_config(
            template_data, component_name, configuration
        )
        
        # Create component instance
        component_class = self._get_component_class_from_template(template_data)
        component = component_class(customized_config)
        
        # Initialize component
        if not component.initialize():
            raise RuntimeError("Failed to initialize component from template")
        
        return component
    
    async def validate_component_compatibility(
        self,
        component_id: str,
        target_ai_os_version: str
    ) -> Dict[str, Any]:
        """Validate component compatibility with AI-OS version"""
        component_info = await self.get_ai_component(component_id)
        
        compatibility_check = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check AI-OS version compatibility
        required_version = component_info.ai_metadata.ai_os_version
        if not self._check_version_compatibility(required_version, target_ai_os_version):
            compatibility_check["compatible"] = False
            compatibility_check["issues"].append(
                f"Requires AI-OS {required_version}, but target is {target_ai_os_version}"
            )
        
        # Check dependencies
        for dep in component_info.ai_metadata.dependencies:
            dep_check = await self._check_dependency(dep)
            if not dep_check["available"]:
                compatibility_check["issues"].append(
                    f"Missing dependency: {dep['name']} {dep['version']}"
                )
        
        return compatibility_check
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ai_os_store_url}/stats",
                timeout=self.timeout
            )
            response.raise_for_status()
            
        return response.json()
    
    async def sync_with_registry(self) -> Dict[str, Any]:
        """Synchronize local registry with marketplace"""
        # Get local components
        local_components = self.registry.list_components(include_metadata=True)
        
        # Check for updates in marketplace
        updates_available = []
        for component in local_components:
            if component.get("metadata", {}).get("marketplace_id"):
                marketplace_id = component["metadata"]["marketplace_id"]
                try:
                    marketplace_component = await self.get_ai_component(marketplace_id)
                    local_version = component.get("version", "0.0.0")
                    marketplace_version = marketplace_component.version
                    
                    if self._is_newer_version(marketplace_version, local_version):
                        updates_available.append({
                            "component_name": component["name"],
                            "local_version": local_version,
                            "marketplace_version": marketplace_version,
                            "marketplace_id": marketplace_id
                        })
                except Exception:
                    # Component not found in marketplace or other error
                    continue
        
        return {
            "local_components": len(local_components),
            "updates_available": len(updates_available),
            "update_details": updates_available
        }
    
    def _extract_component_data(
        self,
        component: AIServiceComponent,
        configuration_path: Optional[str],
        examples: Optional[List[Dict[str, Any]]],
        documentation: Optional[str]
    ) -> Dict[str, Any]:
        """Extract component data for marketplace submission"""
        # Get marketplace format from component
        marketplace_data = component.to_marketplace_format()
        
        # Add AI-OS specific metadata
        ai_metadata = AIComponentMetadata(
            component_type=component.type,
            ai_os_version=marketplace_data.get("ai_os_version", ">=0.1.0"),
            sdk_version="0.1.0",  # Current SDK version
            configuration_schema=self._extract_config_schema(component.config),
            execution_modes=[marketplace_data.get("execution_mode", "sync")],
            dependencies=marketplace_data.get("dependencies", []),
            integration_points=self._get_integration_points(component),
            performance_metrics=self._get_performance_metrics(component),
            compatibility_matrix=self._get_compatibility_matrix(component)
        )
        
        return {
            **marketplace_data,
            "ai_metadata": asdict(ai_metadata),
            "usage_examples": examples or [],
            "integration_guide": documentation,
            "configuration_path": configuration_path
        }
    
    def _parse_ai_component(self, data: Dict[str, Any]) -> AIStoreComponent:
        """Parse marketplace data into AIStoreComponent"""
        ai_metadata_data = data.get("ai_metadata", {})
        ai_metadata = AIComponentMetadata(**ai_metadata_data)
        
        return AIStoreComponent(
            id=data.get("id"),
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            description=data["description"],
            component_type=ComponentType(data["component_type"]),
            version=data["version"],
            author=data["author"],
            license=data["license"],
            tags=data.get("tags", []),
            category=data.get("category", "general"),
            ai_metadata=ai_metadata,
            configuration=data.get("configuration", {}),
            usage_examples=data.get("usage_examples", []),
            integration_guide=data.get("integration_guide"),
            api_documentation=data.get("api_documentation"),
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            rating=data.get("rating", 0.0),
            is_verified=data.get("is_verified", False),
            is_private=data.get("is_private", False),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )
    
    async def _download_component_files(self, component_id: str) -> Dict[str, Any]:
        """Download component implementation files"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.ai_components_url}/{component_id}/download",
                timeout=self.timeout
            )
            response.raise_for_status()
            
        return response.json()
    
    async def _install_component_locally(
        self,
        component_info: AIStoreComponent,
        download_data: Dict[str, Any],
        installation_path: Optional[str]
    ) -> AIServiceComponent:
        """Install component files locally and create instance"""
        # This would handle file installation, dependency resolution, etc.
        # For now, create a mock component based on the metadata
        
        # Create configuration from marketplace data
        config_data = {
            "type": component_info.component_type.value,
            "metadata": {
                "name": component_info.name,
                "display_name": component_info.display_name,
                "description": component_info.description,
                "version": component_info.version,
                "author": component_info.author,
                "license": component_info.license,
                "category": component_info.category,
                "tags": component_info.tags
            },
            **component_info.configuration
        }
        
        # Create component instance based on type
        from ..config.models import create_config
        config = create_config(component_info.component_type, **config_data)
        
        # Import appropriate base class
        if component_info.component_type == ComponentType.AGENT:
            from ..base.agent import AIAgentComponent
            component = AIAgentComponent(config)
        elif component_info.component_type == ComponentType.TOOL:
            from ..base.tool import AIToolComponent
            component = AIToolComponent(config)
        elif component_info.component_type == ComponentType.WORKFLOW:
            from ..base.workflow import AIWorkflowTemplate
            component = AIWorkflowTemplate(config)
        else:
            # Default to base component
            component = AIServiceComponent(config)
        
        return component
    
    def _extract_config_schema(self, config: ComponentConfig) -> Dict[str, Any]:
        """Extract configuration schema from component config"""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in config.parameters:
            schema["properties"][param.name] = {
                "type": param.type.value,
                "description": param.description,
                "default": param.default
            }
            if param.required:
                schema["required"].append(param.name)
        
        return schema
    
    def _get_integration_points(self, component: AIServiceComponent) -> List[str]:
        """Get component integration points"""
        integration_points = ["ai_os_core"]
        
        if hasattr(component.config, 'langflow_compatible') and component.config.langflow_compatible:
            integration_points.append("langflow")
        
        # Add other integration points based on component capabilities
        return integration_points
    
    def _get_performance_metrics(self, component: AIServiceComponent) -> Dict[str, Any]:
        """Get component performance metrics"""
        return {
            "execution_count": getattr(component, '_execution_count', 0),
            "average_execution_time": 0.0,  # Would be calculated from real metrics
            "memory_usage": "unknown",
            "cpu_usage": "unknown"
        }
    
    def _get_compatibility_matrix(self, component: AIServiceComponent) -> Dict[str, Any]:
        """Get component compatibility matrix"""
        return {
            "python_versions": [">=3.8"],
            "operating_systems": ["linux", "macos", "windows"],
            "ai_os_versions": [">=0.1.0"],
            "langflow_versions": [">=1.0.0"] if getattr(component.config, 'langflow_compatible', False) else []
        }
    
    def _check_version_compatibility(self, required: str, target: str) -> bool:
        """Check if target version meets requirements"""
        # Simple version check - would use proper semver in production
        return True  # Simplified for now
    
    async def _check_dependency(self, dependency: Dict[str, str]) -> Dict[str, Any]:
        """Check if dependency is available"""
        # This would check if dependency is installed/available
        return {"available": True, "version": dependency.get("version", "unknown")}
    
    def _is_newer_version(self, marketplace_version: str, local_version: str) -> bool:
        """Check if marketplace version is newer than local version"""
        # Simple version comparison - would use proper semver in production
        return marketplace_version != local_version
    
    async def _get_template(self, template_id: str) -> Dict[str, Any]:
        """Get template data"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.templates_url}/{template_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
        return response.json()
    
    def _customize_template_config(
        self,
        template_data: Dict[str, Any],
        component_name: str,
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize template with user configuration"""
        base_config = template_data.get("configuration", {})
        
        # Merge configurations
        customized_config = {**base_config, **configuration}
        customized_config["metadata"]["name"] = component_name
        
        return customized_config
    
    def _get_component_class_from_template(self, template_data: Dict[str, Any]):
        """Get appropriate component class from template data"""
        component_type = ComponentType(template_data["component_type"])
        
        if component_type == ComponentType.AGENT:
            from ..base.agent import AIAgentComponent
            return AIAgentComponent
        elif component_type == ComponentType.TOOL:
            from ..base.tool import AIToolComponent
            return AIToolComponent
        elif component_type == ComponentType.WORKFLOW:
            from ..base.workflow import AIWorkflowTemplate
            return AIWorkflowTemplate
        else:
            return AIServiceComponent