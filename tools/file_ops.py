"""
File Operations - Safe file system operations.

Provides safe read/write/search operations with path validation.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileOps:
    """
    Safe file operations.

    All operations are sandboxed to an allowed directory
    and validate paths to prevent directory traversal.
    """

    def __init__(
        self,
        allowed_paths: List[Path] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB default
    ):
        """
        Initialize file operations.

        Args:
            allowed_paths: List of allowed base paths (defaults to cwd)
            max_file_size: Maximum file size for read operations
        """
        self.allowed_paths = allowed_paths or [Path.cwd()]
        self.max_file_size = max_file_size

    def _validate_path(self, path: Path) -> bool:
        """Validate that path is within allowed directories."""
        path = path.resolve()

        for allowed in self.allowed_paths:
            allowed = allowed.resolve()
            try:
                path.relative_to(allowed)
                return True
            except ValueError:
                continue

        return False

    def _safe_path(self, path: str | Path) -> Optional[Path]:
        """Get safe, validated path."""
        path = Path(path).resolve()
        if self._validate_path(path):
            return path
        return None

    def read(self, path: str | Path) -> Optional[str]:
        """
        Read file contents.

        Args:
            path: File path

        Returns:
            File contents or None if invalid/error
        """
        safe_path = self._safe_path(path)
        if not safe_path:
            return None

        if not safe_path.exists():
            return None

        if not safe_path.is_file():
            return None

        # Check file size
        if safe_path.stat().st_size > self.max_file_size:
            return None

        try:
            return safe_path.read_text(encoding="utf-8")
        except Exception:
            return None

    def write(
        self,
        path: str | Path,
        content: str,
        create_dirs: bool = True,
    ) -> bool:
        """
        Write content to file.

        Args:
            path: File path
            content: Content to write
            create_dirs: Create parent directories if needed

        Returns:
            True if successful
        """
        safe_path = self._safe_path(path)
        if not safe_path:
            return False

        try:
            if create_dirs:
                safe_path.parent.mkdir(parents=True, exist_ok=True)

            safe_path.write_text(content, encoding="utf-8")
            return True
        except Exception:
            return False

    def append(self, path: str | Path, content: str) -> bool:
        """
        Append content to file.

        Args:
            path: File path
            content: Content to append

        Returns:
            True if successful
        """
        safe_path = self._safe_path(path)
        if not safe_path:
            return False

        try:
            with open(safe_path, "a", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            return False

    def exists(self, path: str | Path) -> bool:
        """Check if path exists."""
        safe_path = self._safe_path(path)
        if not safe_path:
            return False
        return safe_path.exists()

    def is_file(self, path: str | Path) -> bool:
        """Check if path is a file."""
        safe_path = self._safe_path(path)
        if not safe_path:
            return False
        return safe_path.is_file()

    def is_dir(self, path: str | Path) -> bool:
        """Check if path is a directory."""
        safe_path = self._safe_path(path)
        if not safe_path:
            return False
        return safe_path.is_dir()

    def list_dir(self, path: str | Path) -> List[str]:
        """
        List directory contents.

        Args:
            path: Directory path

        Returns:
            List of file/directory names
        """
        safe_path = self._safe_path(path)
        if not safe_path or not safe_path.is_dir():
            return []

        try:
            return [p.name for p in safe_path.iterdir()]
        except Exception:
            return []

    def search(
        self,
        path: str | Path,
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        """
        Search for files matching pattern.

        Args:
            path: Base directory
            pattern: Glob pattern
            recursive: Search recursively

        Returns:
            List of matching paths
        """
        safe_path = self._safe_path(path)
        if not safe_path or not safe_path.is_dir():
            return []

        try:
            if recursive:
                matches = list(safe_path.rglob(pattern))
            else:
                matches = list(safe_path.glob(pattern))

            # Validate all matches
            return [p for p in matches if self._validate_path(p)]
        except Exception:
            return []

    def delete(self, path: str | Path) -> bool:
        """
        Delete a file.

        Args:
            path: File path

        Returns:
            True if successful
        """
        safe_path = self._safe_path(path)
        if not safe_path or not safe_path.is_file():
            return False

        try:
            safe_path.unlink()
            return True
        except Exception:
            return False

    def mkdir(self, path: str | Path) -> bool:
        """
        Create a directory.

        Args:
            path: Directory path

        Returns:
            True if successful
        """
        safe_path = self._safe_path(path)
        if not safe_path:
            return False

        try:
            safe_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def get_info(self, path: str | Path) -> Optional[Dict[str, Any]]:
        """
        Get file/directory info.

        Args:
            path: Path to check

        Returns:
            Info dictionary or None
        """
        safe_path = self._safe_path(path)
        if not safe_path or not safe_path.exists():
            return None

        try:
            stat = safe_path.stat()
            return {
                "name": safe_path.name,
                "path": str(safe_path),
                "is_file": safe_path.is_file(),
                "is_dir": safe_path.is_dir(),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
            }
        except Exception:
            return None
