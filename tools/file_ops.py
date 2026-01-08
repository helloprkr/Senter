"""
File Operations - Safe file system operations.

Provides safe read/write/search operations with path validation.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# FileContent dataclass (US-016)
# =============================================================================


@dataclass
class FileContent:
    """Result of reading a file with metadata and summary."""
    path: str
    content: str
    metadata: Dict[str, Any]
    summary: str
    file_type: str  # "text", "code", "markdown", "unknown"
    language: Optional[str] = None  # For code files: "python", "javascript", etc.
    line_count: int = 0
    char_count: int = 0
    success: bool = True
    error: Optional[str] = None


# File extension to type mapping
FILE_TYPE_MAP = {
    # Code files
    ".py": ("code", "python"),
    ".js": ("code", "javascript"),
    ".ts": ("code", "typescript"),
    ".jsx": ("code", "javascript"),
    ".tsx": ("code", "typescript"),
    ".java": ("code", "java"),
    ".c": ("code", "c"),
    ".cpp": ("code", "cpp"),
    ".h": ("code", "c"),
    ".hpp": ("code", "cpp"),
    ".go": ("code", "go"),
    ".rs": ("code", "rust"),
    ".rb": ("code", "ruby"),
    ".php": ("code", "php"),
    ".swift": ("code", "swift"),
    ".kt": ("code", "kotlin"),
    ".scala": ("code", "scala"),
    ".cs": ("code", "csharp"),
    ".sh": ("code", "shell"),
    ".bash": ("code", "shell"),
    ".zsh": ("code", "shell"),
    ".sql": ("code", "sql"),
    ".r": ("code", "r"),
    ".R": ("code", "r"),
    ".html": ("code", "html"),
    ".css": ("code", "css"),
    ".scss": ("code", "scss"),
    ".less": ("code", "less"),
    ".json": ("code", "json"),
    ".xml": ("code", "xml"),
    ".yaml": ("code", "yaml"),
    ".yml": ("code", "yaml"),
    ".toml": ("code", "toml"),
    # Markdown
    ".md": ("markdown", None),
    ".markdown": ("markdown", None),
    ".mdx": ("markdown", None),
    # Text files
    ".txt": ("text", None),
    ".text": ("text", None),
    ".rst": ("text", None),
    ".log": ("text", None),
    ".csv": ("text", None),
    ".tsv": ("text", None),
}


def _detect_file_type(path: Path) -> tuple[str, Optional[str]]:
    """Detect file type and language from extension."""
    suffix = path.suffix.lower()
    return FILE_TYPE_MAP.get(suffix, ("text", None))


def _generate_summary(content: str, file_type: str, language: Optional[str]) -> str:
    """Generate a brief summary of file content."""
    lines = content.split('\n')
    line_count = len(lines)

    if file_type == "code":
        # For code, summarize structure
        imports = [line for line in lines[:50] if line.strip().startswith(('import ', 'from ', '#include', 'using ', 'require'))]
        definitions = [line for line in lines if line.strip().startswith(('class ', 'def ', 'function ', 'const ', 'let ', 'var '))]

        parts = [f"{line_count} lines of {language or 'code'}"]
        if imports:
            parts.append(f"{len(imports)} imports")
        if definitions:
            parts.append(f"{len(definitions)} definitions")
        return ". ".join(parts)

    elif file_type == "markdown":
        # For markdown, extract headers
        headers = [line for line in lines if line.strip().startswith('#')]
        if headers:
            return f"{line_count} lines. Headers: {headers[0][:50]}..."
        return f"{line_count} lines of markdown"

    else:
        # For text, show first line preview
        first_line = lines[0][:80] if lines else ""
        return f"{line_count} lines. Preview: {first_line}..."


def read_file(path: str | Path, max_size: int = 10 * 1024 * 1024) -> FileContent:
    """
    Read a file and return structured FileContent.

    Args:
        path: Path to the file
        max_size: Maximum file size in bytes (default 10MB)

    Returns:
        FileContent with content, metadata, and summary
    """
    path_obj = Path(path).resolve()

    # Check if file exists
    if not path_obj.exists():
        return FileContent(
            path=str(path),
            content="",
            metadata={},
            summary="",
            file_type="unknown",
            success=False,
            error=f"File not found: {path}"
        )

    # Check if it's a file (not directory)
    if not path_obj.is_file():
        return FileContent(
            path=str(path),
            content="",
            metadata={},
            summary="",
            file_type="unknown",
            success=False,
            error=f"Path is not a file: {path}"
        )

    # Check file size
    try:
        file_stat = path_obj.stat()
        if file_stat.st_size > max_size:
            return FileContent(
                path=str(path),
                content="",
                metadata={"size": file_stat.st_size},
                summary="",
                file_type="unknown",
                success=False,
                error=f"File too large: {file_stat.st_size} bytes (max {max_size})"
            )
    except OSError as e:
        return FileContent(
            path=str(path),
            content="",
            metadata={},
            summary="",
            file_type="unknown",
            success=False,
            error=f"Cannot stat file: {e}"
        )

    # Detect file type
    file_type, language = _detect_file_type(path_obj)

    # Read content
    try:
        content = path_obj.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        try:
            content = path_obj.read_text(encoding="latin-1")
        except Exception as e:
            return FileContent(
                path=str(path),
                content="",
                metadata={},
                summary="",
                file_type=file_type,
                language=language,
                success=False,
                error=f"Cannot decode file: {e}"
            )
    except Exception as e:
        return FileContent(
            path=str(path),
            content="",
            metadata={},
            summary="",
            file_type=file_type,
            language=language,
            success=False,
            error=f"Cannot read file: {e}"
        )

    # Build metadata
    metadata = {
        "name": path_obj.name,
        "extension": path_obj.suffix,
        "size": file_stat.st_size,
        "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
        "parent": str(path_obj.parent),
    }

    # Generate summary
    summary = _generate_summary(content, file_type, language)

    # Count lines and chars
    line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
    char_count = len(content)

    return FileContent(
        path=str(path_obj),
        content=content,
        metadata=metadata,
        summary=summary,
        file_type=file_type,
        language=language,
        line_count=line_count,
        char_count=char_count,
        success=True,
        error=None
    )


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
