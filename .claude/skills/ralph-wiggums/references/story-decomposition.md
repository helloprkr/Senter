# Story Decomposition Reference

How to break requirements into atomic, implementable user stories.

## The Atomic Story Rule

**A story is atomic when it can be completed in ONE context window iteration.**

Signs a story is too big:
- Requires changes to more than 3-5 files
- Has more than 5 acceptance criteria
- Contains words like "entire", "complete", "full"
- Touches multiple architectural layers simultaneously

## Decomposition Strategies

### 1. Vertical Slicing (Preferred)

Slice through all layers for ONE piece of functionality.

❌ **Horizontal (bad)**:
- Story 1: Build all database models
- Story 2: Build all API endpoints
- Story 3: Build all UI components

✅ **Vertical (good)**:
- Story 1: User can see list of todos (DB + API + UI for LIST)
- Story 2: User can create a todo (DB + API + UI for CREATE)
- Story 3: User can delete a todo (DB + API + UI for DELETE)

### 2. By User Action

Each story = one user action with observable outcome.

**Feature**: Shopping cart

✅ **Decomposed**:
1. User can view empty cart state
2. User can add item to cart
3. User can see item count badge update
4. User can remove item from cart
5. User can update item quantity
6. User can see cart total price
7. User can proceed to checkout

### 3. By State Transition

Each story = one state change.

**Feature**: Order status

✅ **Decomposed**:
1. Order created → pending state
2. Payment received → processing state
3. Order shipped → shipped state
4. Order delivered → completed state
5. Order cancelled → cancelled state

### 4. By Validation Rule

Each validation = separate story.

**Feature**: Registration form

✅ **Decomposed**:
1. Email field with format validation
2. Password field with strength validation
3. Password confirmation with match validation
4. Username with uniqueness check
5. Terms checkbox required validation
6. Form submission with all validations passing

### 5. By Error Path

Happy path first, then error paths.

**Feature**: Login

✅ **Decomposed**:
1. Successful login redirects to dashboard
2. Invalid email shows format error
3. Wrong password shows auth error
4. Too many attempts shows rate limit error
5. Network error shows retry option

## Story Template

```json
{
  "id": "US-XXX",
  "title": "[Verb] [specific outcome]",
  "acceptanceCriteria": [
    "Given [context], when [action], then [outcome]",
    "Specific UI element exists/renders",
    "Specific data transformation occurs",
    "typecheck passes",
    "relevant tests pass"
  ],
  "testRequirements": {
    "unit": ["test_[function]_[scenario]"],
    "integration": ["test_[flow]_[outcome]"],
    "visual": true/false
  },
  "priority": 1,
  "passes": false,
  "notes": "Dependencies: US-XXX, US-YYY"
}
```

## Title Patterns

Good titles are imperative and specific:

| Pattern | Example |
|---------|---------|
| Add [component] | Add login form |
| Implement [feature] | Implement email validation |
| Create [resource] | Create user model |
| Display [content] | Display error messages |
| Handle [scenario] | Handle network timeout |
| Update [element] when [trigger] | Update cart badge when item added |

## Acceptance Criteria Patterns

### Given-When-Then (Behavior)
```
Given user is on login page
When user enters valid credentials and clicks submit
Then user is redirected to dashboard
```

### Checklist (Implementation)
```
- Login form renders with email and password fields
- Submit button is disabled when fields are empty
- Loading spinner appears during submission
- Error message displays on failure
- typecheck passes
- unit tests pass
```

### Technical Constraints
```
- Response time < 200ms
- Input sanitized against XSS
- Passwords hashed with bcrypt
- Session expires after 24 hours
```

## Dependency Ordering

Stories should be ordered so dependencies come first:

```
Priority 1: Create User model (no deps)
Priority 2: Add user creation endpoint (depends on model)
Priority 3: Add registration form (depends on endpoint)
Priority 4: Add email verification (depends on user creation)
```

Use the `notes` field to document dependencies:
```json
{
  "id": "US-004",
  "title": "Add email verification flow",
  "notes": "Depends on: US-001 (User model), US-002 (creation endpoint)"
}
```

## Test Requirements by Story Type

### Data Model Story
```json
"testRequirements": {
  "unit": [
    "test_model_creation",
    "test_model_validation",
    "test_model_relationships"
  ],
  "integration": [],
  "visual": false
}
```

### API Endpoint Story
```json
"testRequirements": {
  "unit": [
    "test_handler_valid_input",
    "test_handler_invalid_input",
    "test_handler_auth_required"
  ],
  "integration": [
    "test_endpoint_returns_correct_status",
    "test_endpoint_persists_data"
  ],
  "visual": false
}
```

### UI Component Story
```json
"testRequirements": {
  "unit": [
    "test_component_renders",
    "test_component_handles_click",
    "test_component_displays_loading"
  ],
  "integration": [],
  "visual": true
}
```

### Full Feature Story (if unavoidable)
```json
"testRequirements": {
  "unit": [
    "test_model",
    "test_handler",
    "test_component"
  ],
  "integration": [
    "test_e2e_flow"
  ],
  "visual": true
}
```

## Common Decomposition Mistakes

### Too Vague
❌ "Implement user authentication"
✅ Split into: login form, validation, auth endpoint, session handling, logout

### Too Technical
❌ "Refactor database queries to use joins"
✅ "Improve user list load time by fetching related data efficiently"

### Hidden Dependencies
❌ "Add payment processing" (assumes user, cart, products exist)
✅ Ensure prerequisite stories are defined and prioritized first

### Mixing Concerns
❌ "Add login with Google OAuth and email verification"
✅ Split: "Add email login" + "Add Google OAuth" + "Add email verification"

## Decomposition Checklist

Before finalizing stories, verify:

- [ ] Each story has ≤5 acceptance criteria
- [ ] Each story changes ≤5 files
- [ ] Each story has clear pass/fail verification
- [ ] Dependencies are documented and ordered
- [ ] Stories follow vertical slicing when possible
- [ ] Test requirements are specific, not generic
- [ ] No story contains "and" joining unrelated features
