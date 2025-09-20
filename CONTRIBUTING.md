# Contributing to Vertixia SDK

Thank you for your interest in contributing to the Vertixia SDK! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/vertixia-sdk.git
   cd vertixia-sdk
   ```

2. **Install dependencies with uv**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync --all-extras
   ```

3. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

4. **Verify installation**
   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy vertixia_sdk
   ```

## 📋 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the coding standards outlined below
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests and Linting
```bash
# Run the full test suite
uv run pytest

# Run code formatting
uv run black .
uv run ruff check --fix .

# Type checking
uv run mypy vertixia_sdk
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add new component type for workflow automation"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/modifications
- `refactor:` for code refactoring
- `chore:` for maintenance tasks

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 🎯 What to Contribute

### Priority Areas
- **Component Templates**: New AI component types and templates
- **Documentation**: Tutorials, examples, and API documentation
- **Testing**: Unit tests, integration tests, and test utilities
- **Performance**: Optimizations and performance improvements
- **Bug Fixes**: Issues reported in GitHub Issues

### Component Development
When adding new component types:

1. **Base Class**: Extend from appropriate base class
2. **Configuration**: Add proper configuration schema
3. **Validation**: Include input/output validation
4. **Documentation**: Add docstrings and examples
5. **Tests**: Comprehensive test coverage

Example component structure:
```python
from vertixia_sdk import AIToolComponent
from vertixia_sdk.config import ComponentConfig

class MyNewComponent(AIToolComponent):
    """Brief description of the component."""
    
    def __init__(self):
        config = ComponentConfig(
            name="my-new-component",
            description="Detailed description",
            input_schema={...},
            output_schema={...}
        )
        super().__init__(config)
    
    def execute(self, input_data: dict) -> dict:
        """Execute the component logic."""
        # Implementation here
        return result
```

## 📝 Coding Standards

### Python Code Style
- **Formatter**: Black (line length: 88)
- **Linter**: Ruff with standard configuration
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings

### Code Quality Guidelines
1. **Type Safety**: Use type hints consistently
2. **Error Handling**: Proper exception handling and logging
3. **Documentation**: Clear docstrings and inline comments
4. **Testing**: Minimum 80% code coverage
5. **Performance**: Consider performance implications

### Example Code Structure
```python
from typing import Dict, Any, Optional
from vertixia_sdk import AIToolComponent

class ExampleComponent(AIToolComponent):
    """Example component demonstrating best practices.
    
    This component shows how to properly structure a Vertixia SDK
    component with type hints, error handling, and documentation.
    
    Args:
        config: Component configuration
        
    Example:
        >>> component = ExampleComponent()
        >>> result = component.execute({"input": "data"})
        >>> print(result["output"])
    """
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component with proper error handling.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dictionary containing execution results
            
        Raises:
            ValueError: If input_data is invalid
            RuntimeError: If execution fails
        """
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Process data
            result = self._process_data(input_data)
            
            return {"status": "success", "output": result}
            
        except Exception as e:
            self.logger.error(f"Component execution failed: {e}")
            raise RuntimeError(f"Execution failed: {e}") from e
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/                   # Unit tests
│   ├── test_components.py
│   ├── test_config.py
│   └── test_utils.py
├── integration/            # Integration tests
│   ├── test_marketplace.py
│   └── test_discovery.py
└── fixtures/               # Test fixtures and data
    ├── sample_components/
    └── test_configs/
```

### Writing Tests
```python
import pytest
from vertixia_sdk import AIToolComponent

class TestExampleComponent:
    """Test suite for ExampleComponent."""
    
    @pytest.fixture
    def component(self):
        """Create component instance for testing."""
        return ExampleComponent()
    
    def test_execute_success(self, component):
        """Test successful execution."""
        input_data = {"input": "test"}
        result = component.execute(input_data)
        
        assert result["status"] == "success"
        assert "output" in result
    
    def test_execute_invalid_input(self, component):
        """Test execution with invalid input."""
        with pytest.raises(ValueError):
            component.execute({})
    
    @pytest.mark.asyncio
    async def test_async_operation(self, component):
        """Test asynchronous operations."""
        result = await component.async_execute({"input": "test"})
        assert result is not None
```

## 📚 Documentation

### Adding Documentation
1. **API Documentation**: Docstrings for all public functions/classes
2. **Usage Examples**: Practical examples in docstrings
3. **Tutorials**: Step-by-step guides for complex features
4. **README Updates**: Update README.md for significant changes

### Documentation Style
- Use Google-style docstrings
- Include practical examples
- Link to related concepts
- Keep examples up-to-date

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Vertixia SDK version
   - Operating system

2. **Reproduction Steps**
   - Minimal code example
   - Expected behavior
   - Actual behavior

3. **Error Messages**
   - Full stack trace
   - Log messages

## ✨ Feature Requests

For feature requests:

1. **Use Case**: Describe the problem you're solving
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other solutions you've considered
4. **Implementation**: Offer to help implement if possible

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the project's goals and vision

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Pull Requests**: Code contributions and reviews

## 📋 Pull Request Checklist

Before submitting a pull request:

- [ ] Code follows style guidelines
- [ ] Tests pass (`uv run pytest`)
- [ ] Code is properly formatted (`uv run black .`)
- [ ] No linting errors (`uv run ruff check .`)
- [ ] Type checking passes (`uv run mypy vertixia_sdk`)
- [ ] Documentation is updated
- [ ] Changelog is updated (for significant changes)
- [ ] Pull request description is clear and complete

## 📞 Getting Help

If you need help:

1. **Check Documentation**: Review existing docs and examples
2. **Search Issues**: Look for similar questions or problems
3. **GitHub Discussions**: Ask questions in the community
4. **Contact Maintainers**: Reach out for complex questions

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Vertixia SDK! 🚀