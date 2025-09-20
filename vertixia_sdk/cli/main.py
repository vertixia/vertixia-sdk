"""
AI-OS SDK Command Line Interface

Main CLI entry point for AI-OS SDK development tooling.
Provides commands for component development, marketplace interaction, and registry management.
"""

import sys
import asyncio
import argparse
from typing import List, Optional
import logging

from .commands import ComponentCommands, MarketplaceCommands, RegistryCommands
from ..utils.discovery import get_registry


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class AIOSCLI:
    """Main CLI class for AI-OS SDK"""
    
    def __init__(self):
        self.component_commands = ComponentCommands()
        self.marketplace_commands = MarketplaceCommands()
        self.registry_commands = RegistryCommands()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with all commands"""
        parser = argparse.ArgumentParser(
            prog='ai-os',
            description='AI-OS SDK development tooling',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  ai-os component create --name my-agent --type agent
  ai-os component validate ./my-component.yaml
  ai-os registry discover ./components/
  ai-os marketplace search reasoning
  ai-os marketplace install itrs-reasoning
  ai-os marketplace publish my-component --api-key YOUR_KEY
            """
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            help='Path to AI-OS SDK configuration file'
        )
        
        # Create subparsers for main command groups
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Component commands
        self._add_component_commands(subparsers)
        
        # Registry commands
        self._add_registry_commands(subparsers)
        
        # Marketplace commands
        self._add_marketplace_commands(subparsers)
        
        return parser
    
    def _add_component_commands(self, subparsers):
        """Add component-related commands"""
        component_parser = subparsers.add_parser(
            'component',
            help='Component development commands',
            aliases=['comp', 'c']
        )
        
        component_subparsers = component_parser.add_subparsers(
            dest='component_action',
            help='Component actions'
        )
        
        # Create component
        create_parser = component_subparsers.add_parser(
            'create',
            help='Create a new component from template'
        )
        create_parser.add_argument('--name', required=True, help='Component name')
        create_parser.add_argument('--type', required=True, 
                                 choices=['service', 'agent', 'tool', 'workflow', 'reasoning'],
                                 help='Component type')
        create_parser.add_argument('--template', help='Template to use')
        create_parser.add_argument('--output-dir', default='.', help='Output directory')
        create_parser.add_argument('--author', help='Component author')
        create_parser.add_argument('--description', help='Component description')
        
        # Validate component
        validate_parser = component_subparsers.add_parser(
            'validate',
            help='Validate component configuration'
        )
        validate_parser.add_argument('config_path', help='Path to component configuration')
        validate_parser.add_argument('--strict', action='store_true', help='Strict validation mode')
        
        # Test component
        test_parser = component_subparsers.add_parser(
            'test',
            help='Test component functionality'
        )
        test_parser.add_argument('config_path', help='Path to component configuration')
        test_parser.add_argument('--input', help='Test input data (JSON)')
        test_parser.add_argument('--timeout', type=int, default=30, help='Test timeout in seconds')
        
        # Package component
        package_parser = component_subparsers.add_parser(
            'package',
            help='Package component for distribution'
        )
        package_parser.add_argument('config_path', help='Path to component configuration')
        package_parser.add_argument('--output', help='Output package path')
        package_parser.add_argument('--include-deps', action='store_true', help='Include dependencies')
    
    def _add_registry_commands(self, subparsers):
        """Add registry-related commands"""
        registry_parser = subparsers.add_parser(
            'registry',
            help='Component registry commands',
            aliases=['reg', 'r']
        )
        
        registry_subparsers = registry_parser.add_subparsers(
            dest='registry_action',
            help='Registry actions'
        )
        
        # Discover components
        discover_parser = registry_subparsers.add_parser(
            'discover',
            help='Auto-discover components in directories'
        )
        discover_parser.add_argument('paths', nargs='*', default=['.'], help='Paths to search')
        discover_parser.add_argument('--recursive', action='store_true', help='Recursive search')
        
        # List components
        list_parser = registry_subparsers.add_parser(
            'list',
            help='List registered components'
        )
        list_parser.add_argument('--type', help='Filter by component type')
        list_parser.add_argument('--format', choices=['table', 'json', 'yaml'], default='table')
        
        # Register component
        register_parser = registry_subparsers.add_parser(
            'register',
            help='Register a component manually'
        )
        register_parser.add_argument('config_path', help='Path to component configuration')
        register_parser.add_argument('--name', help='Override component name')
        
        # Unregister component
        unregister_parser = registry_subparsers.add_parser(
            'unregister',
            help='Unregister a component'
        )
        unregister_parser.add_argument('name', help='Component name to unregister')
        
        # Registry status
        registry_subparsers.add_parser(
            'status',
            help='Show registry status'
        )
    
    def _add_marketplace_commands(self, subparsers):
        """Add marketplace-related commands"""
        marketplace_parser = subparsers.add_parser(
            'marketplace',
            help='Marketplace interaction commands',
            aliases=['market', 'm']
        )
        
        marketplace_subparsers = marketplace_parser.add_subparsers(
            dest='marketplace_action',
            help='Marketplace actions'
        )
        
        # Search marketplace
        search_parser = marketplace_subparsers.add_parser(
            'search',
            help='Search marketplace components'
        )
        search_parser.add_argument('query', nargs='?', help='Search query')
        search_parser.add_argument('--type', help='Filter by component type')
        search_parser.add_argument('--tags', nargs='*', help='Filter by tags')
        search_parser.add_argument('--author', help='Filter by author')
        search_parser.add_argument('--verified', action='store_true', help='Show only verified components')
        search_parser.add_argument('--limit', type=int, default=20, help='Maximum results')
        
        # Install component
        install_parser = marketplace_subparsers.add_parser(
            'install',
            help='Install component from marketplace'
        )
        install_parser.add_argument('component_id', help='Component ID or name to install')
        install_parser.add_argument('--path', help='Installation path')
        install_parser.add_argument('--no-deps', action='store_true', help='Skip dependency installation')
        install_parser.add_argument('--no-config', action='store_true', help='Skip auto-configuration')
        
        # Update component
        update_parser = marketplace_subparsers.add_parser(
            'update',
            help='Update installed component'
        )
        update_parser.add_argument('component_name', help='Component name to update')
        update_parser.add_argument('--version', help='Target version')
        
        # Publish component
        publish_parser = marketplace_subparsers.add_parser(
            'publish',
            help='Publish component to marketplace'
        )
        publish_parser.add_argument('component_name', help='Component name to publish')
        publish_parser.add_argument('--api-key', required=True, help='Marketplace API key')
        publish_parser.add_argument('--config', help='Component configuration path')
        publish_parser.add_argument('--examples', help='Path to usage examples (JSON)')
        publish_parser.add_argument('--docs', help='Path to documentation')
        publish_parser.add_argument('--public', action='store_true', help='Make component public')
        
        # Sync with marketplace
        sync_parser = marketplace_subparsers.add_parser(
            'sync',
            help='Sync with marketplace'
        )
        sync_parser.add_argument('--force', action='store_true', help='Force sync even if recent')
        
        # Show marketplace stats
        marketplace_subparsers.add_parser(
            'stats',
            help='Show marketplace statistics'
        )
    
    async def run(self, args: List[str]) -> int:
        """Run CLI with given arguments"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Setup logging
        setup_logging(parsed_args.verbose)
        
        # Handle commands
        try:
            if parsed_args.command in ['component', 'comp', 'c']:
                return await self._handle_component_command(parsed_args)
            elif parsed_args.command in ['registry', 'reg', 'r']:
                return await self._handle_registry_command(parsed_args)
            elif parsed_args.command in ['marketplace', 'market', 'm']:
                return await self._handle_marketplace_command(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            print("\\nOperation cancelled by user")
            return 130
        except Exception as e:
            print(f"Error: {e}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    async def _handle_component_command(self, args) -> int:
        """Handle component commands"""
        if args.component_action == 'create':
            return await self.component_commands.create_component(
                name=args.name,
                component_type=args.type,
                template=args.template,
                output_dir=args.output_dir,
                author=args.author,
                description=args.description
            )
        elif args.component_action == 'validate':
            return await self.component_commands.validate_component(
                config_path=args.config_path,
                strict=args.strict
            )
        elif args.component_action == 'test':
            return await self.component_commands.test_component(
                config_path=args.config_path,
                test_input=args.input,
                timeout=args.timeout
            )
        elif args.component_action == 'package':
            return await self.component_commands.package_component(
                config_path=args.config_path,
                output_path=args.output,
                include_deps=args.include_deps
            )
        else:
            print("Unknown component action")
            return 1
    
    async def _handle_registry_command(self, args) -> int:
        """Handle registry commands"""
        if args.registry_action == 'discover':
            return await self.registry_commands.discover_components(
                paths=args.paths,
                recursive=args.recursive
            )
        elif args.registry_action == 'list':
            return await self.registry_commands.list_components(
                component_type=args.type,
                output_format=args.format
            )
        elif args.registry_action == 'register':
            return await self.registry_commands.register_component(
                config_path=args.config_path,
                name_override=args.name
            )
        elif args.registry_action == 'unregister':
            return await self.registry_commands.unregister_component(
                component_name=args.name
            )
        elif args.registry_action == 'status':
            return await self.registry_commands.show_status()
        else:
            print("Unknown registry action")
            return 1
    
    async def _handle_marketplace_command(self, args) -> int:
        """Handle marketplace commands"""
        if args.marketplace_action == 'search':
            return await self.marketplace_commands.search_components(
                query=args.query,
                component_type=args.type,
                tags=args.tags,
                author=args.author,
                verified_only=args.verified,
                limit=args.limit
            )
        elif args.marketplace_action == 'install':
            return await self.marketplace_commands.install_component(
                component_id=args.component_id,
                installation_path=args.path,
                skip_dependencies=args.no_deps,
                skip_configuration=args.no_config
            )
        elif args.marketplace_action == 'update':
            return await self.marketplace_commands.update_component(
                component_name=args.component_name,
                target_version=args.version
            )
        elif args.marketplace_action == 'publish':
            return await self.marketplace_commands.publish_component(
                component_name=args.component_name,
                api_key=args.api_key,
                config_path=args.config,
                examples_path=args.examples,
                docs_path=args.docs,
                make_public=args.public
            )
        elif args.marketplace_action == 'sync':
            return await self.marketplace_commands.sync_with_marketplace(
                force=args.force
            )
        elif args.marketplace_action == 'stats':
            return await self.marketplace_commands.show_stats()
        else:
            print("Unknown marketplace action")
            return 1


# Global CLI instance
cli = AIOSCLI()


def main():
    """Main entry point for CLI"""
    try:
        return asyncio.run(cli.run(sys.argv[1:]))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())