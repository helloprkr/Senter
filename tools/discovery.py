"""
Tool Discovery - Auto-discover capabilities.

Scans directories for tool definitions and registers them.
"""

from __future__ import annotations
import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class DiscoveredTool:
    """A discovered tool."""

    name: str
    description: str
    module_path: Path
    function_name: str
    triggers: List[str]
    handler: Optional[Callable] = None


class ToolDiscovery:
    """
    Discovers tools from Python files.

    Scans configured directories for tool definitions
    and makes them available for registration.
    """

    # Decorator names that mark functions as tools
    TOOL_DECORATORS = ["tool", "capability", "register_tool"]

    def __init__(self, sources: List[Dict[str, str]] = None):
        """
        Initialize tool discovery.

        Args:
            sources: List of {path, pattern} dictionaries
        """
        self.sources = sources or []
        self._discovered: Dict[str, DiscoveredTool] = {}

    def discover(self) -> List[DiscoveredTool]:
        """
        Discover tools from all configured sources.

        Returns:
            List of discovered tools
        """
        discovered = []

        for source in self.sources:
            path = Path(source.get("path", "."))
            pattern = source.get("pattern", "*.py")

            if path.exists():
                for file_path in path.glob(pattern):
                    tools = self._discover_from_file(file_path)
                    discovered.extend(tools)

        # Cache results
        for tool in discovered:
            self._discovered[tool.name] = tool

        return discovered

    def _discover_from_file(self, file_path: Path) -> List[DiscoveredTool]:
        """
        Discover tools from a single Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of discovered tools
        """
        tools = []

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    tool = self._analyze_function(node, file_path)
                    if tool:
                        tools.append(tool)

        except Exception:
            pass

        return tools

    def _analyze_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
    ) -> Optional[DiscoveredTool]:
        """
        Analyze a function to see if it's a tool.

        Args:
            node: AST function definition
            file_path: Source file path

        Returns:
            DiscoveredTool if it's a tool, None otherwise
        """
        # Check for tool decorator
        is_tool = False
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator)
            if decorator_name in self.TOOL_DECORATORS:
                is_tool = True
                break

        # Also consider functions with "tool" in name or docstring mentioning "capability"
        if not is_tool:
            if "tool" in node.name.lower() or "capability" in node.name.lower():
                is_tool = True

        if not is_tool:
            return None

        # Extract description from docstring
        description = ""
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            description = str(node.body[0].value.value).strip()

        # Generate triggers from function name
        triggers = self._generate_triggers(node.name, description)

        return DiscoveredTool(
            name=node.name,
            description=description[:200] if description else f"Tool: {node.name}",
            module_path=file_path,
            function_name=node.name,
            triggers=triggers,
        )

    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return ""

    def _generate_triggers(self, name: str, description: str) -> List[str]:
        """Generate trigger keywords from name and description."""
        triggers = []

        # Split function name by underscore
        parts = name.lower().replace("tool", "").replace("capability", "").split("_")
        triggers.extend([p for p in parts if len(p) > 2])

        # Extract keywords from description
        if description:
            words = description.lower().split()
            keywords = [
                w
                for w in words
                if len(w) > 3 and w not in ("the", "this", "that", "with", "from")
            ]
            triggers.extend(keywords[:5])

        return list(set(triggers))[:10]

    def load_tool(self, tool: DiscoveredTool) -> Optional[Callable]:
        """
        Load a discovered tool's handler function.

        Args:
            tool: The discovered tool

        Returns:
            Handler function or None
        """
        try:
            spec = importlib.util.spec_from_file_location(
                tool.module_path.stem,
                tool.module_path,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                handler = getattr(module, tool.function_name, None)
                if callable(handler):
                    tool.handler = handler
                    return handler

        except Exception:
            pass

        return None

    def get_discovered(self) -> Dict[str, DiscoveredTool]:
        """Get all discovered tools."""
        return self._discovered.copy()

    def get_tool(self, name: str) -> Optional[DiscoveredTool]:
        """Get a specific discovered tool."""
        return self._discovered.get(name)

    def refresh(self) -> List[DiscoveredTool]:
        """Refresh discovery (re-scan sources)."""
        self._discovered.clear()
        return self.discover()
