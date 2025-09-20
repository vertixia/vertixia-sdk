"""
Marketplace Registry Integration

Connects the local component registry with the AI-OS marketplace,
enabling seamless discovery, installation, and publishing of components.
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import logging

from ..utils.discovery import ComponentRegistry, get_registry
from ..base.component import AIServiceComponent
from ..config.models import ComponentType
from .store_extension import AIStoreService


logger = logging.getLogger(__name__)


class MarketplaceRegistry:
    """
    Registry integration with AI-OS marketplace
    
    Provides high-level interface for marketplace operations including
    component discovery, installation, updates, and publishing.
    """
    
    def __init__(
        self,
        local_registry: Optional[ComponentRegistry] = None,
        store_service: Optional[AIStoreService] = None
    ):
        self.local_registry = local_registry or get_registry()
        self.store_service = store_service or AIStoreService()
        
        # Caching
        self._marketplace_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(hours=1)
        
        # Sync state
        self._last_sync: Optional[datetime] = None
        self._sync_interval = timedelta(hours=6)
        
        # Installation tracking
        self._pending_installations: Set[str] = set()
        self._installation_history: List[Dict[str, Any]] = []
    
    async def discover_marketplace_components(
        self,
        query: Optional[str] = None,
        component_type: Optional[ComponentType] = None,
        tags: Optional[List[str]] = None,
        verified_only: bool = False,
        limit: int = 50,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Discover components in the marketplace
        
        Args:
            query: Search query
            component_type: Filter by component type
            tags: Filter by tags
            verified_only: Show only verified components
            limit: Maximum results
            force_refresh: Force refresh cache
            
        Returns:
            List of available components
        """
        cache_key = f"discover_{query}_{component_type}_{str(tags)}_{verified_only}_{limit}"
        
        # Check cache
        if not force_refresh and self._is_cache_valid(cache_key):
            return self._marketplace_cache[cache_key]
        
        try:
            # Search marketplace
            results = await self.store_service.search_ai_components(
                query=query,
                component_type=component_type,
                tags=tags,
                verified_only=verified_only,
                limit=limit
            )
            
            components = results.get("components", [])
            
            # Enhance with local installation status
            enhanced_components = []
            for component in components:
                enhanced_component = component.copy()
                enhanced_component["is_installed"] = self._is_component_installed(component["id"])
                enhanced_component["local_version"] = self._get_local_version(component["name"])
                enhanced_component["update_available"] = self._has_update_available(component)
                enhanced_components.append(enhanced_component)
            
            # Cache results
            self._marketplace_cache[cache_key] = enhanced_components
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
            
            logger.info(f"Discovered {len(enhanced_components)} marketplace components")
            return enhanced_components
            
        except Exception as e:
            logger.error(f"Error discovering marketplace components: {e}")
            return []
    
    async def install_component(
        self,
        component_id: str,
        installation_path: Optional[str] = None,
        auto_configure: bool = True,
        resolve_dependencies: bool = True
    ) -> Dict[str, Any]:
        """
        Install a component from the marketplace
        
        Args:
            component_id: Component ID to install
            installation_path: Local installation path
            auto_configure: Automatically configure component
            resolve_dependencies: Resolve and install dependencies
            
        Returns:
            Installation result
        """
        if component_id in self._pending_installations:
            return {"status": "error", "message": "Installation already in progress"}
        
        self._pending_installations.add(component_id)
        
        try:
            logger.info(f"Installing component {component_id}")
            
            # Check compatibility
            compatibility = await self.store_service.validate_component_compatibility(
                component_id, "0.1.0"  # Current AI-OS version
            )
            
            if not compatibility["compatible"]:
                return {
                    "status": "error",
                    "message": "Component not compatible",
                    "issues": compatibility["issues"]
                }
            
            # Install dependencies if requested
            if resolve_dependencies:
                dep_results = await self._install_dependencies(component_id)
                if not dep_results["success"]:
                    return {
                        "status": "error",
                        "message": "Failed to install dependencies",
                        "details": dep_results
                    }
            
            # Install component
            component = await self.store_service.install_ai_component(
                component_id,
                installation_path=installation_path,
                auto_register=True  # Register with local registry
            )
            
            # Auto-configure if requested
            if auto_configure:
                config_result = await self._auto_configure_component(component)
                if not config_result["success"]:
                    logger.warning(f"Auto-configuration failed: {config_result['message']}")
            
            # Record installation
            installation_record = {
                "component_id": component_id,
                "component_name": component.name,
                "version": component.version,
                "installed_at": datetime.now(),
                "installation_path": installation_path,
                "status": "success"
            }
            self._installation_history.append(installation_record)
            
            # Clear cache
            self._clear_discovery_cache()
            
            logger.info(f"Successfully installed component {component.name}")
            return {
                "status": "success",
                "component": {
                    "id": component_id,
                    "name": component.name,
                    "version": component.version,
                    "type": component.type.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error installing component {component_id}: {e}")
            return {"status": "error", "message": str(e)}
            
        finally:
            self._pending_installations.discard(component_id)
    
    async def update_component(
        self,
        component_name: str,
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a locally installed component"""
        
        # Find component in local registry
        local_component = self.local_registry.get_component(component_name)
        if not local_component:
            return {"status": "error", "message": "Component not found locally"}
        
        # Get marketplace ID
        marketplace_id = local_component.get("metadata", {}).get("marketplace_id")
        if not marketplace_id:
            return {"status": "error", "message": "Component not linked to marketplace"}
        
        try:
            # Get marketplace component info
            marketplace_component = await self.store_service.get_ai_component(marketplace_id)
            
            # Check if update is needed
            local_version = local_component.get("version", "0.0.0")
            marketplace_version = target_version or marketplace_component.version
            
            if not self._is_newer_version(marketplace_version, local_version):
                return {
                    "status": "info",
                    "message": "Component is already up to date",
                    "local_version": local_version,
                    "marketplace_version": marketplace_version
                }
            
            # Perform update (reinstall)
            update_result = await self.install_component(
                marketplace_id,
                installation_path=local_component.get("file_path"),
                auto_configure=True
            )
            
            if update_result["status"] == "success":
                logger.info(f"Updated component {component_name} from {local_version} to {marketplace_version}")
                return {
                    "status": "success",
                    "message": "Component updated successfully",
                    "old_version": local_version,
                    "new_version": marketplace_version
                }
            else:
                return update_result
                
        except Exception as e:
            logger.error(f"Error updating component {component_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def publish_component(
        self,
        component_name: str,
        api_key: str,
        configuration_path: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        documentation: Optional[str] = None,
        make_public: bool = False
    ) -> Dict[str, Any]:
        """Publish a local component to the marketplace"""
        
        # Get component from local registry
        component_class = self.local_registry.get_component_class(component_name)
        if not component_class:
            return {"status": "error", "message": "Component not found in local registry"}
        
        try:
            # Create component instance
            component = self.local_registry.create_component(component_name)
            if not component:
                return {"status": "error", "message": "Failed to create component instance"}
            
            # Publish to marketplace
            component_id = await self.store_service.publish_ai_component(
                component,
                api_key,
                configuration_path=configuration_path,
                examples=examples,
                documentation=documentation
            )
            
            # Update local registry with marketplace ID
            component_info = self.local_registry.get_component(component_name)
            if component_info:
                component_info["metadata"]["marketplace_id"] = component_id
                component_info["metadata"]["published_at"] = datetime.now().isoformat()
                component_info["metadata"]["is_public"] = make_public
            
            logger.info(f"Published component {component_name} to marketplace as {component_id}")
            return {
                "status": "success",
                "component_id": component_id,
                "marketplace_url": f"https://marketplace.ai-os.com/components/{component_id}"
            }
            
        except Exception as e:
            logger.error(f"Error publishing component {component_name}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def sync_with_marketplace(self, force: bool = False) -> Dict[str, Any]:
        """Synchronize local registry with marketplace"""
        
        if not force and self._last_sync and datetime.now() - self._last_sync < self._sync_interval:
            return {"status": "info", "message": "Sync not needed yet"}
        
        try:
            # Get sync data from store service
            sync_result = await self.store_service.sync_with_registry()
            
            updates_available = sync_result.get("update_details", [])
            
            # Process available updates
            update_results = []
            for update in updates_available:
                component_name = update["component_name"]
                marketplace_version = update["marketplace_version"]
                
                # Check if auto-update is enabled for this component
                component_info = self.local_registry.get_component(component_name)
                auto_update = component_info.get("metadata", {}).get("auto_update", False)
                
                if auto_update:
                    update_result = await self.update_component(component_name, marketplace_version)
                    update_results.append({
                        "component": component_name,
                        "auto_updated": True,
                        "result": update_result
                    })
                else:
                    update_results.append({
                        "component": component_name,
                        "auto_updated": False,
                        "update_available": True,
                        "marketplace_version": marketplace_version
                    })
            
            self._last_sync = datetime.now()
            
            logger.info(f"Marketplace sync completed. {len(updates_available)} updates available.")
            return {
                "status": "success",
                "updates_available": len(updates_available),
                "auto_updated": len([r for r in update_results if r.get("auto_updated")]),
                "manual_updates_needed": len([r for r in update_results if not r.get("auto_updated")]),
                "details": update_results
            }
            
        except Exception as e:
            logger.error(f"Error syncing with marketplace: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_component_recommendations(
        self,
        based_on_installed: bool = True,
        component_type: Optional[ComponentType] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get component recommendations"""
        
        recommendations = []
        
        try:
            if based_on_installed:
                # Get recommendations based on installed components
                installed_components = self.local_registry.list_components()
                installed_tags = set()
                installed_types = set()
                
                for component in installed_components:
                    if "tags" in component.get("metadata", {}):
                        installed_tags.update(component["metadata"]["tags"])
                    installed_types.add(component["type"])
                
                # Search for similar components
                for tag in list(installed_tags)[:3]:  # Limit tags to avoid too broad search
                    similar_components = await self.discover_marketplace_components(
                        tags=[tag],
                        verified_only=True,
                        limit=5
                    )
                    for component in similar_components:
                        if not component["is_installed"]:
                            recommendations.append(component)
            
            else:
                # Get popular/trending components
                popular_components = await self.discover_marketplace_components(
                    component_type=component_type,
                    verified_only=True,
                    limit=limit
                )
                recommendations.extend([c for c in popular_components if not c["is_installed"]])
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec["id"] not in seen_ids:
                    seen_ids.add(rec["id"])
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= limit:
                        break
            
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_installation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get component installation history"""
        history = self._installation_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get marketplace sync status"""
        return {
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_interval_hours": self._sync_interval.total_seconds() / 3600,
            "cache_entries": len(self._marketplace_cache),
            "pending_installations": len(self._pending_installations)
        }
    
    # Private helper methods
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid"""
        if cache_key not in self._marketplace_cache:
            return False
        if cache_key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[cache_key]
    
    def _is_component_installed(self, component_id: str) -> bool:
        """Check if component is installed locally"""
        components = self.local_registry.list_components(include_metadata=True)
        for component in components:
            if component.get("metadata", {}).get("marketplace_id") == component_id:
                return True
        return False
    
    def _get_local_version(self, component_name: str) -> Optional[str]:
        """Get local version of component"""
        component = self.local_registry.get_component(component_name)
        return component.get("version") if component else None
    
    def _has_update_available(self, marketplace_component: Dict[str, Any]) -> bool:
        """Check if component has update available"""
        local_version = self._get_local_version(marketplace_component["name"])
        if not local_version:
            return False
        return self._is_newer_version(marketplace_component["version"], local_version)
    
    def _is_newer_version(self, version1: str, version2: str) -> bool:
        """Check if version1 is newer than version2"""
        # Simple version comparison - would use proper semver in production
        return version1 != version2
    
    def _clear_discovery_cache(self):
        """Clear discovery cache"""
        keys_to_remove = [k for k in self._marketplace_cache.keys() if k.startswith("discover_")]
        for key in keys_to_remove:
            del self._marketplace_cache[key]
            if key in self._cache_expiry:
                del self._cache_expiry[key]
    
    async def _install_dependencies(self, component_id: str) -> Dict[str, Any]:
        """Install component dependencies"""
        try:
            component_info = await self.store_service.get_ai_component(component_id)
            dependencies = component_info.ai_metadata.dependencies
            
            failed_deps = []
            for dep in dependencies:
                # This would integrate with pip, conda, or other package managers
                # For now, assume all dependencies are available
                logger.debug(f"Installing dependency: {dep['name']} {dep.get('version', 'latest')}")
            
            return {"success": len(failed_deps) == 0, "failed_dependencies": failed_deps}
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return {"success": False, "error": str(e)}
    
    async def _auto_configure_component(self, component: AIServiceComponent) -> Dict[str, Any]:
        """Auto-configure component with sensible defaults"""
        try:
            # Initialize component
            if not component.initialize():
                return {"success": False, "message": "Component initialization failed"}
            
            # Run health check
            health_status = component.health_check()
            if not health_status.get("healthy", False):
                return {"success": False, "message": "Component health check failed", "details": health_status}
            
            return {"success": True, "message": "Component configured successfully"}
            
        except Exception as e:
            logger.error(f"Error auto-configuring component: {e}")
            return {"success": False, "message": str(e)}