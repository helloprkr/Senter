# Test Strategies Reference

Testing patterns and strategies by stack, plus solutions for common testing challenges.

## Test Pyramid

```
         /\
        /  \      E2E/Visual (few, slow, high confidence)
       /----\
      /      \    Integration (some, medium speed)
     /--------\
    /          \  Unit (many, fast, focused)
   --------------
```

**Ralph prioritizes**: Unit tests for fast feedback, integration for critical paths, visual for UI.

## Stack-Specific Patterns

### TypeScript/Node.js (Vitest)

```typescript
// Unit test pattern
import { describe, it, expect, vi } from 'vitest'
import { calculateTotal } from './cart'

describe('calculateTotal', () => {
  it('sums item prices correctly', () => {
    const items = [
      { price: 10, quantity: 2 },
      { price: 5, quantity: 1 }
    ]
    expect(calculateTotal(items)).toBe(25)
  })

  it('returns 0 for empty cart', () => {
    expect(calculateTotal([])).toBe(0)
  })

  it('handles decimal prices', () => {
    const items = [{ price: 10.99, quantity: 1 }]
    expect(calculateTotal(items)).toBeCloseTo(10.99)
  })
})
```

```typescript
// Integration test pattern (API)
import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { createServer } from '../server'

describe('POST /api/users', () => {
  let server: ReturnType<typeof createServer>

  beforeAll(async () => {
    server = createServer()
    await server.listen(0)
  })

  afterAll(async () => {
    await server.close()
  })

  it('creates user with valid data', async () => {
    const res = await fetch(`${server.url}/api/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'test@example.com', name: 'Test' })
    })
    
    expect(res.status).toBe(201)
    const user = await res.json()
    expect(user.email).toBe('test@example.com')
  })

  it('rejects invalid email', async () => {
    const res = await fetch(`${server.url}/api/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: 'invalid', name: 'Test' })
    })
    
    expect(res.status).toBe(400)
  })
})
```

### TypeScript/React (Vitest + Testing Library)

```typescript
// Component test pattern
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { LoginForm } from './LoginForm'

describe('LoginForm', () => {
  it('renders email and password fields', () => {
    render(<LoginForm onSubmit={vi.fn()} />)
    
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
  })

  it('disables submit when fields empty', () => {
    render(<LoginForm onSubmit={vi.fn()} />)
    
    const button = screen.getByRole('button', { name: /submit/i })
    expect(button).toBeDisabled()
  })

  it('calls onSubmit with form data', async () => {
    const onSubmit = vi.fn()
    render(<LoginForm onSubmit={onSubmit} />)
    
    fireEvent.change(screen.getByLabelText(/email/i), {
      target: { value: 'test@example.com' }
    })
    fireEvent.change(screen.getByLabelText(/password/i), {
      target: { value: 'password123' }
    })
    fireEvent.click(screen.getByRole('button', { name: /submit/i }))
    
    expect(onSubmit).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'password123'
    })
  })
})
```

### Python (pytest)

```python
# Unit test pattern
import pytest
from cart import calculate_total, Item

class TestCalculateTotal:
    def test_sums_item_prices(self):
        items = [Item(price=10, quantity=2), Item(price=5, quantity=1)]
        assert calculate_total(items) == 25

    def test_empty_cart_returns_zero(self):
        assert calculate_total([]) == 0

    def test_handles_decimal_prices(self):
        items = [Item(price=10.99, quantity=1)]
        assert calculate_total(items) == pytest.approx(10.99)

# Fixture pattern
@pytest.fixture
def sample_user():
    return {"email": "test@example.com", "name": "Test User"}

@pytest.fixture
def db_session():
    session = create_test_session()
    yield session
    session.rollback()
    session.close()
```

```python
# Integration test pattern (FastAPI)
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestUserEndpoints:
    def test_create_user(self, client):
        response = client.post("/api/users", json={
            "email": "test@example.com",
            "name": "Test"
        })
        assert response.status_code == 201
        assert response.json()["email"] == "test@example.com"

    def test_create_user_invalid_email(self, client):
        response = client.post("/api/users", json={
            "email": "invalid",
            "name": "Test"
        })
        assert response.status_code == 422
```

### Go

```go
// Unit test pattern
package cart

import (
    "testing"
)

func TestCalculateTotal(t *testing.T) {
    tests := []struct {
        name     string
        items    []Item
        expected float64
    }{
        {
            name:     "sums item prices",
            items:    []Item{{Price: 10, Quantity: 2}, {Price: 5, Quantity: 1}},
            expected: 25,
        },
        {
            name:     "empty cart",
            items:    []Item{},
            expected: 0,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got := CalculateTotal(tt.items)
            if got != tt.expected {
                t.Errorf("CalculateTotal() = %v, want %v", got, tt.expected)
            }
        })
    }
}
```

```go
// Integration test pattern
package api

import (
    "bytes"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"
)

func TestCreateUser(t *testing.T) {
    server := NewServer()
    
    body := bytes.NewBuffer([]byte(`{"email":"test@example.com","name":"Test"}`))
    req := httptest.NewRequest(http.MethodPost, "/api/users", body)
    req.Header.Set("Content-Type", "application/json")
    
    rec := httptest.NewRecorder()
    server.ServeHTTP(rec, req)
    
    if rec.Code != http.StatusCreated {
        t.Errorf("expected status 201, got %d", rec.Code)
    }
}
```

## Dealing with Flaky Tests

### Common Causes and Solutions

| Cause | Solution |
|-------|----------|
| Timing/async issues | Use explicit waits, not sleeps |
| Shared state | Isolate tests, use fresh fixtures |
| Network dependencies | Mock external services |
| Date/time dependencies | Mock time functions |
| Random data | Use deterministic seeds or fixed data |

### Async Wait Patterns

```typescript
// Bad: Fixed sleep
await new Promise(r => setTimeout(r, 1000))

// Good: Wait for condition
await waitFor(() => {
  expect(screen.getByText('Loaded')).toBeInTheDocument()
})

// Good: Poll with timeout
const result = await pollUntil(
  () => fetchStatus(),
  (status) => status === 'complete',
  { timeout: 5000, interval: 100 }
)
```

### Test Isolation

```typescript
// Use beforeEach for clean state
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
})

// Use unique identifiers
const testId = `test-${Date.now()}-${Math.random()}`
const user = await createUser({ email: `${testId}@example.com` })
```

### Mocking External Services

```typescript
// Mock fetch globally
vi.mock('node-fetch', () => ({
  default: vi.fn(() => Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ data: 'mocked' })
  }))
}))

// Or use MSW for realistic mocking
import { setupServer } from 'msw/node'
import { rest } from 'msw'

const server = setupServer(
  rest.get('/api/users', (req, res, ctx) => {
    return res(ctx.json([{ id: 1, name: 'Test' }]))
  })
)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
```

## Alternative Test Strategies

When standard approaches fail, try these alternatives:

### 1. Snapshot Testing
For complex output where manual assertions are brittle:

```typescript
it('renders user profile correctly', () => {
  const { container } = render(<UserProfile user={testUser} />)
  expect(container).toMatchSnapshot()
})
```

### 2. Property-Based Testing
For functions with many edge cases:

```typescript
import { fc } from 'fast-check'

it('calculate always returns non-negative', () => {
  fc.assert(
    fc.property(fc.array(fc.record({ price: fc.float(), quantity: fc.nat() })), (items) => {
      return calculateTotal(items) >= 0
    })
  )
})
```

### 3. Contract Testing
For API boundaries:

```typescript
it('fulfills user contract', async () => {
  const user = await fetchUser(1)
  
  // Contract: must have these fields with these types
  expect(user).toMatchObject({
    id: expect.any(Number),
    email: expect.stringMatching(/@/),
    name: expect.any(String)
  })
})
```

### 4. Smoke Testing
For basic sanity checks when full testing is blocked:

```typescript
it('app starts without crashing', () => {
  expect(() => render(<App />)).not.toThrow()
})

it('critical path is accessible', async () => {
  render(<App />)
  expect(screen.getByRole('navigation')).toBeInTheDocument()
})
```

## Visual Verification

When `visualVerification: true`:

```typescript
// Playwright screenshot comparison
import { test, expect } from '@playwright/test'

test('login page looks correct', async ({ page }) => {
  await page.goto('/login')
  await expect(page).toHaveScreenshot('login-page.png')
})

// Or manual screenshot for review
test('capture current state', async ({ page }) => {
  await page.goto('/dashboard')
  await page.screenshot({ path: 'tmp/dashboard.png', fullPage: true })
  console.log('Screenshot saved to tmp/dashboard.png - verify manually')
})
```

## Test Generation Checklist

For each story, generate tests that verify:

- [ ] Happy path (expected input → expected output)
- [ ] Invalid input handling (validation errors)
- [ ] Edge cases (empty, null, boundary values)
- [ ] Error states (network failure, timeout)
- [ ] Loading states (for async operations)
- [ ] Accessibility (if UI component)
