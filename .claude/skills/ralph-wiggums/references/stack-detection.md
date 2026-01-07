# Stack Detection Reference

Auto-detect project stack by examining marker files and configurations.

## Detection Order

Check files in this order (first match wins for each category):

### Language Detection

| Marker File | Language | Notes |
|-------------|----------|-------|
| `package.json` | JavaScript/TypeScript | Check for `"type": "module"` |
| `tsconfig.json` | TypeScript | Confirms TS over JS |
| `requirements.txt` | Python | pip dependencies |
| `pyproject.toml` | Python | Modern Python projects |
| `Pipfile` | Python | pipenv projects |
| `go.mod` | Go | Go modules |
| `Cargo.toml` | Rust | Cargo projects |
| `Gemfile` | Ruby | Bundler projects |
| `pom.xml` | Java | Maven projects |
| `build.gradle` | Java/Kotlin | Gradle projects |

### Framework Detection

| Marker | Framework | Verification Command |
|--------|-----------|---------------------|
| `next.config.*` | Next.js | `npm run build` |
| `nuxt.config.*` | Nuxt | `npm run build` |
| `vite.config.*` | Vite | `npm run build` |
| `angular.json` | Angular | `ng build` |
| `svelte.config.*` | SvelteKit | `npm run build` |
| `remix.config.*` | Remix | `npm run build` |
| `astro.config.*` | Astro | `npm run build` |
| `manage.py` | Django | `python manage.py check` |
| `app.py` or `wsgi.py` | Flask | `flask --version` |
| `main.go` | Go (standard) | `go build` |

### Test Runner Detection

| Marker | Test Runner | Run Command |
|--------|-------------|-------------|
| `vitest.config.*` | Vitest | `npm test` or `npx vitest` |
| `jest.config.*` | Jest | `npm test` or `npx jest` |
| `playwright.config.*` | Playwright | `npx playwright test` |
| `cypress.config.*` | Cypress | `npx cypress run` |
| `pytest.ini` or `conftest.py` | pytest | `pytest` |
| `*_test.go` | Go testing | `go test ./...` |
| `*_test.rs` | Rust testing | `cargo test` |

### Typecheck Detection

| Stack | Command |
|-------|---------|
| TypeScript | `npx tsc --noEmit` or `npm run typecheck` |
| Python (typed) | `mypy .` or `pyright` |
| Go | `go vet ./...` |
| Rust | `cargo check` |

## Detection Script

```bash
#!/bin/bash
# detect-stack.sh - Output detected stack as JSON

detect_stack() {
  local lang="unknown"
  local framework="none"
  local test_runner="none"
  local typecheck="none"
  local lint="none"

  # Language
  if [[ -f "tsconfig.json" ]]; then
    lang="typescript"
    typecheck="npx tsc --noEmit"
  elif [[ -f "package.json" ]]; then
    lang="javascript"
  elif [[ -f "pyproject.toml" ]] || [[ -f "requirements.txt" ]]; then
    lang="python"
    if command -v mypy &> /dev/null; then
      typecheck="mypy ."
    fi
  elif [[ -f "go.mod" ]]; then
    lang="go"
    typecheck="go vet ./..."
  elif [[ -f "Cargo.toml" ]]; then
    lang="rust"
    typecheck="cargo check"
  fi

  # Framework (Node.js)
  if [[ -f "next.config.js" ]] || [[ -f "next.config.mjs" ]] || [[ -f "next.config.ts" ]]; then
    framework="nextjs"
  elif [[ -f "vite.config.js" ]] || [[ -f "vite.config.ts" ]]; then
    framework="vite"
  elif [[ -f "nuxt.config.ts" ]]; then
    framework="nuxt"
  fi

  # Framework (Python)
  if [[ -f "manage.py" ]]; then
    framework="django"
  elif grep -q "flask" requirements.txt 2>/dev/null || grep -q "flask" pyproject.toml 2>/dev/null; then
    framework="flask"
  elif grep -q "fastapi" requirements.txt 2>/dev/null || grep -q "fastapi" pyproject.toml 2>/dev/null; then
    framework="fastapi"
  fi

  # Test runner
  if [[ -f "vitest.config.js" ]] || [[ -f "vitest.config.ts" ]]; then
    test_runner="vitest"
  elif [[ -f "jest.config.js" ]] || [[ -f "jest.config.ts" ]]; then
    test_runner="jest"
  elif [[ -f "pytest.ini" ]] || [[ -f "conftest.py" ]] || [[ -d "tests" ]]; then
    test_runner="pytest"
  elif [[ "$lang" == "go" ]]; then
    test_runner="go test"
  elif [[ "$lang" == "rust" ]]; then
    test_runner="cargo test"
  fi

  # Lint
  if [[ -f ".eslintrc.js" ]] || [[ -f ".eslintrc.json" ]] || [[ -f "eslint.config.js" ]]; then
    lint="eslint"
  elif [[ -f ".flake8" ]] || [[ -f "setup.cfg" ]]; then
    lint="flake8"
  elif [[ -f "ruff.toml" ]] || [[ -f ".ruff.toml" ]]; then
    lint="ruff"
  fi

  # Output JSON
  cat << EOF
{
  "language": "$lang",
  "framework": "$framework",
  "testRunner": "$test_runner",
  "typecheck": "$typecheck",
  "lint": "$lint"
}
EOF
}

detect_stack
```

## Verification Commands by Stack

### TypeScript/Next.js
```json
{
  "typecheck": "npm run typecheck || npx tsc --noEmit",
  "unitTest": "npm test",
  "integrationTest": "npm run test:e2e",
  "lint": "npm run lint"
}
```

### Python/FastAPI
```json
{
  "typecheck": "mypy . || pyright",
  "unitTest": "pytest tests/unit",
  "integrationTest": "pytest tests/integration",
  "lint": "ruff check . || flake8"
}
```

### Go
```json
{
  "typecheck": "go vet ./...",
  "unitTest": "go test ./...",
  "integrationTest": "go test -tags=integration ./...",
  "lint": "golangci-lint run"
}
```

### Rust
```json
{
  "typecheck": "cargo check",
  "unitTest": "cargo test",
  "integrationTest": "cargo test --test '*'",
  "lint": "cargo clippy"
}
```

## Fallback Behavior

If stack cannot be detected:
1. Check README.md for setup instructions
2. Check package.json scripts section
3. Ask user to specify stack in requirements
4. Default to most common patterns:
   - `npm test` for Node projects
   - `pytest` for Python projects
   - `go test ./...` for Go projects
