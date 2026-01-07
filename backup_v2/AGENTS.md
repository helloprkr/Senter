# Senter Development Guidelines

## Commands
- **Run App**: `python scripts/senter_main.py` (Orchestrator) or `python scripts/senter.py` (TUI)
- **Test**: `python scripts/test_components.py` (Component validation)
- **Lint**: `ruff check .` (Recommended)
- **Install**: `pip install -r requirements.txt` (or via `uv`)

## Code Style & Conventions
- **Language**: Python 3.10+ with strict type hints (`typing.List`, `typing.Optional`, etc.).
- **Formatting**: PEP 8 standards, 4 spaces indentation.
- **Documentation**: Mandatory docstrings for all modules, classes, and public methods.
- **Imports**: Grouped: Standard Library -> Third Party -> Local. Use `pathlib.Path` over `os.path`.
- **Naming**: `PascalCase` for classes (`SenterOrchestrator`), `snake_case` for functions/vars. 
- **Error Handling**: Use specific exceptions (e.g., `ImportError`) rather than bare `except:`.
- **UI Framework**: Textual (TUI). Core logic integrated with TUI components.
- **Architecture**: Follow "JSON Agents" pattern. Agents defined in `Agents/*.json`.
- **State**: Use `SENTER.md` files in `Topics/` for persistent agent knowledge/state.
