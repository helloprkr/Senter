"""
Senter 3.0 Tools Module

Available tools/capabilities:
- WebSearch: Search the web via DuckDuckGo
- FileOps: Safe file operations
- Discovery: Auto-discover capabilities
"""

from .web_search import WebSearch, search_web
from .file_ops import FileOps
from .discovery import ToolDiscovery

__all__ = [
    "WebSearch",
    "search_web",
    "FileOps",
    "ToolDiscovery",
]
