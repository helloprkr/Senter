#!/bin/bash
# detect-stack.sh - Auto-detect project tech stack
# Outputs JSON with detected stack information

detect_stack() {
  local lang="unknown"
  local framework="none"
  local test_runner="none"
  local typecheck="echo 'No typecheck configured'"
  local lint="echo 'No lint configured'"
  local unit_test="echo 'No test runner configured'"
  local integration_test="echo 'No integration tests configured'"

  # === Language Detection ===
  if [[ -f "tsconfig.json" ]]; then
    lang="typescript"
    typecheck="npx tsc --noEmit"
  elif [[ -f "package.json" ]]; then
    lang="javascript"
  elif [[ -f "pyproject.toml" ]] || [[ -f "requirements.txt" ]] || [[ -f "Pipfile" ]]; then
    lang="python"
    if command -v mypy &> /dev/null; then
      typecheck="mypy ."
    elif command -v pyright &> /dev/null; then
      typecheck="pyright"
    fi
  elif [[ -f "go.mod" ]]; then
    lang="go"
    typecheck="go vet ./..."
    unit_test="go test ./..."
    integration_test="go test -tags=integration ./..."
  elif [[ -f "Cargo.toml" ]]; then
    lang="rust"
    typecheck="cargo check"
    unit_test="cargo test"
    lint="cargo clippy"
  elif [[ -f "Gemfile" ]]; then
    lang="ruby"
  fi

  # === Framework Detection (Node.js) ===
  if [[ -f "next.config.js" ]] || [[ -f "next.config.mjs" ]] || [[ -f "next.config.ts" ]]; then
    framework="nextjs"
  elif [[ -f "vite.config.js" ]] || [[ -f "vite.config.ts" ]] || [[ -f "vite.config.mjs" ]]; then
    framework="vite"
  elif [[ -f "nuxt.config.ts" ]] || [[ -f "nuxt.config.js" ]]; then
    framework="nuxt"
  elif [[ -f "angular.json" ]]; then
    framework="angular"
  elif [[ -f "svelte.config.js" ]]; then
    framework="sveltekit"
  elif [[ -f "remix.config.js" ]]; then
    framework="remix"
  elif [[ -f "astro.config.mjs" ]]; then
    framework="astro"
  fi

  # === Framework Detection (Python) ===
  if [[ -f "manage.py" ]]; then
    framework="django"
  elif grep -q "fastapi" requirements.txt 2>/dev/null || grep -q "fastapi" pyproject.toml 2>/dev/null; then
    framework="fastapi"
  elif grep -q "flask" requirements.txt 2>/dev/null || grep -q "flask" pyproject.toml 2>/dev/null; then
    framework="flask"
  fi

  # === Test Runner Detection ===
  if [[ -f "vitest.config.js" ]] || [[ -f "vitest.config.ts" ]] || [[ -f "vitest.config.mts" ]]; then
    test_runner="vitest"
    unit_test="npx vitest run"
    integration_test="npx vitest run --config vitest.integration.config.ts"
  elif [[ -f "jest.config.js" ]] || [[ -f "jest.config.ts" ]] || [[ -f "jest.config.json" ]]; then
    test_runner="jest"
    unit_test="npx jest"
    integration_test="npx jest --config jest.integration.config.js"
  elif [[ -f "playwright.config.ts" ]] || [[ -f "playwright.config.js" ]]; then
    test_runner="playwright"
    integration_test="npx playwright test"
  elif [[ -f "cypress.config.ts" ]] || [[ -f "cypress.config.js" ]]; then
    test_runner="cypress"
    integration_test="npx cypress run"
  elif [[ -f "pytest.ini" ]] || [[ -f "conftest.py" ]] || [[ -d "tests" && "$lang" == "python" ]]; then
    test_runner="pytest"
    unit_test="pytest tests/unit -v"
    integration_test="pytest tests/integration -v"
  fi

  # === Lint Detection ===
  if [[ -f ".eslintrc.js" ]] || [[ -f ".eslintrc.json" ]] || [[ -f "eslint.config.js" ]] || [[ -f "eslint.config.mjs" ]]; then
    lint="npx eslint ."
  elif [[ -f "biome.json" ]]; then
    lint="npx biome check ."
  elif [[ -f ".flake8" ]] || grep -q "flake8" requirements.txt 2>/dev/null; then
    lint="flake8 ."
  elif [[ -f "ruff.toml" ]] || [[ -f ".ruff.toml" ]] || grep -q "ruff" pyproject.toml 2>/dev/null; then
    lint="ruff check ."
  elif [[ "$lang" == "go" ]]; then
    lint="golangci-lint run 2>/dev/null || go vet ./..."
  fi

  # === Check package.json scripts ===
  if [[ -f "package.json" ]]; then
    # Check for common script names
    if grep -q '"typecheck"' package.json 2>/dev/null; then
      typecheck="npm run typecheck"
    elif grep -q '"type-check"' package.json 2>/dev/null; then
      typecheck="npm run type-check"
    fi
    
    if grep -q '"test"' package.json 2>/dev/null; then
      unit_test="npm test"
    fi
    
    if grep -q '"test:e2e"' package.json 2>/dev/null; then
      integration_test="npm run test:e2e"
    elif grep -q '"test:integration"' package.json 2>/dev/null; then
      integration_test="npm run test:integration"
    fi
    
    if grep -q '"lint"' package.json 2>/dev/null; then
      lint="npm run lint"
    fi
  fi

  # Output JSON
  cat << EOF
{
  "language": "$lang",
  "framework": "$framework",
  "testRunner": "$test_runner",
  "verification": {
    "typecheck": "$typecheck",
    "unitTest": "$unit_test",
    "integrationTest": "$integration_test",
    "lint": "$lint"
  }
}
EOF
}

# Run detection
detect_stack
