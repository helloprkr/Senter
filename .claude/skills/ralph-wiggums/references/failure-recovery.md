# Failure Recovery Reference

Detailed protocols for handling failures during Ralph Wiggums execution.

## Failure Classification

| Type | Symptoms | Recovery Strategy |
|------|----------|-------------------|
| **Compilation** | typecheck fails, syntax errors | Fix errors, re-run |
| **Test** | Tests fail after implementation | Debug, fix, or adjust tests |
| **Flaky** | Tests pass/fail inconsistently | Replace with stable alternative |
| **Stuck** | Same story fails 3+ times | Decompose into sub-stories |
| **Blocked** | External dependency unavailable | Document, skip, flag for human |
| **Scope Creep** | Story requires unexpected changes | Expand scope OR decompose |

## Protocol 1: Story Fails 3+ Times

### Detection
Track failure count in progress.txt:

```markdown
## [DATE] - US-005 (Attempt 3 - STUCK)
- Attempted: Add email validation
- Failure: Test "validates email format" fails
- Error: Regex pattern not matching edge cases
```

### Recovery Steps

1. **Analyze Failure Pattern**
   - What specifically is failing?
   - Is it the implementation or the test?
   - Is the acceptance criteria unclear?

2. **Decompose the Story**
   Split into 2-3 smaller stories:
   
   **Original**:
   ```json
   {
     "id": "US-005",
     "title": "Add email validation",
     "acceptanceCriteria": [
       "Validates email format",
       "Shows error for invalid email",
       "Prevents form submission"
     ]
   }
   ```
   
   **Decomposed**:
   ```json
   [
     {
       "id": "US-005a",
       "title": "Add email format regex validator",
       "acceptanceCriteria": [
         "Function validates basic email format (user@domain.tld)",
         "Returns true for valid, false for invalid",
         "Unit test passes"
       ],
       "priority": 5.1
     },
     {
       "id": "US-005b",
       "title": "Integrate validator with form field",
       "acceptanceCriteria": [
         "Form field calls validator on blur",
         "Error state displayed when invalid",
         "Unit test passes"
       ],
       "priority": 5.2,
       "notes": "Depends on US-005a"
     },
     {
       "id": "US-005c",
       "title": "Block form submission on validation error",
       "acceptanceCriteria": [
         "Submit button disabled when validation errors exist",
         "Integration test passes"
       ],
       "priority": 5.3,
       "notes": "Depends on US-005b"
     }
   ]
   ```

3. **Update prd.json**
   - Mark original story as `"passes": true` with note: "Decomposed into US-005a/b/c"
   - Insert sub-stories with higher priority than remaining stories

4. **Log in progress.txt**
   ```markdown
   ## DECOMPOSITION: US-005 → US-005a, US-005b, US-005c
   Original story was too broad. Split into:
   - US-005a: Core validator function
   - US-005b: UI integration
   - US-005c: Form submission blocking
   
   **Learning**: Email validation has edge cases; isolate regex logic.
   ```

5. **Continue with First Sub-Story**

## Protocol 2: Flaky Tests

### Detection
Test passes sometimes, fails other times:
- Timing-dependent assertions
- Shared state between tests
- Network/external dependencies

### Recovery Steps

1. **Identify Flaky Test**
   ```markdown
   ## FLAKY TEST DETECTED: test_api_response_time
   - Passed: 3 times
   - Failed: 2 times
   - Cause: Network latency variation
   ```

2. **Choose Alternative Strategy**

   | Cause | Alternative |
   |-------|-------------|
   | Timing | Mock the async operation |
   | Network | Use recorded responses (MSW, VCR) |
   | Shared state | Add isolation/cleanup |
   | Random data | Use fixed seed or deterministic data |
   | Date/time | Mock system clock |

3. **Implement Alternative**
   ```typescript
   // Before (flaky)
   it('responds within 200ms', async () => {
     const start = Date.now()
     await fetch('/api/data')
     expect(Date.now() - start).toBeLessThan(200)
   })
   
   // After (stable)
   it('returns valid response structure', async () => {
     const response = await fetch('/api/data')
     expect(response.ok).toBe(true)
     expect(await response.json()).toHaveProperty('data')
   })
   ```

4. **Log the Change**
   ```markdown
   ## TEST REPLACEMENT: test_api_response_time
   - Original: Timing-based assertion (flaky due to network variance)
   - Replacement: Response structure assertion (deterministic)
   - **Learning**: Avoid timing assertions; test behavior, not performance
   ```

## Protocol 3: Stuck (No Progress After 5 Iterations)

### Detection
Same story attempted 5+ times with different approaches, all failing.

### Recovery Steps

1. **Output Stuck Signal**
   ```
   <promise>STUCK</promise>
   ```

2. **Document Thoroughly in progress.txt**
   ```markdown
   ## STUCK: US-007 - Implement OAuth flow
   
   ### Attempts
   1. Used passport.js - session handling conflicts
   2. Used next-auth - version incompatibility
   3. Custom implementation - CORS issues
   4. Tried different redirect URLs - still CORS
   5. Added proxy - broke other endpoints
   
   ### Blocking Issue
   OAuth provider requires callback URL on same domain,
   but dev server runs on different port than expected.
   
   ### Suggested Alternatives
   1. Configure reverse proxy for dev environment
   2. Use provider's sandbox/test mode
   3. Mock OAuth for development, real for production
   4. Defer to human: requires infrastructure change
   
   ### Files Modified (may need rollback)
   - lib/auth.ts
   - pages/api/auth/[...nextauth].ts
   - next.config.js
   ```

3. **Loop Continues**
   The loop continues with remaining stories. Human should review and intervene.

## Protocol 4: Scope Creep Detection

### Symptoms
Implementing story requires changes outside expected scope:
- Modifying unrelated files
- Adding unexpected dependencies
- Breaking existing functionality

### Decision Tree

```
Story requires unexpected changes?
│
├── Changes are necessary for story to work?
│   ├── YES → Expand scope, document in notes
│   └── NO → Investigate alternative approach
│
└── Changes would break existing functionality?
    ├── YES → STOP, decompose story
    └── NO → Proceed with caution, add tests
```

### Recovery: Expand Scope
```markdown
## US-008: Add user avatar
**Scope Expansion**: Also required migration for avatar_url column

### Additional Changes
- db/migrations/add_avatar_url.sql
- Updated User type definition
- Updated user creation endpoint

### Justification
Avatar display requires storing URL; no alternative approach viable.
```

### Recovery: Decompose
```markdown
## SCOPE CREEP: US-008 decomposed

Original story touched too many concerns:
- Database schema change
- File upload handling
- Image processing
- UI display

Split into:
- US-008a: Add avatar_url field to user schema
- US-008b: Implement file upload endpoint
- US-008c: Add image processing (resize/optimize)
- US-008d: Display avatar in UI
```

## Protocol 5: External Dependency Blocked

### Examples
- API rate limited
- Service unavailable
- Credentials missing
- Network issues

### Recovery Steps

1. **Document Blocker**
   ```markdown
   ## BLOCKED: US-010 - Integrate payment provider
   - Blocker: Stripe API keys not configured
   - Required: STRIPE_SECRET_KEY environment variable
   - Status: Waiting for human to provide credentials
   ```

2. **Skip Story**
   Set `"passes": false` but add `"blocked": true`:
   ```json
   {
     "id": "US-010",
     "passes": false,
     "blocked": true,
     "blockedReason": "Missing STRIPE_SECRET_KEY"
   }
   ```

3. **Continue with Other Stories**

4. **Flag for Human Review**
   ```
   ⚠️ BLOCKED STORIES: US-010 (missing credentials)
   Review scripts/ralph/progress.txt for details.
   ```

## Logging Best Practices

### Always Log
- What was attempted
- Why it failed
- What was tried differently
- The outcome

### Progress.txt Structure During Recovery
```markdown
## [DATE] - [Story ID] (Attempt N)
**Status**: [FAILED/STUCK/BLOCKED/DECOMPOSED]
**Approach**: What was tried
**Result**: What happened
**Error**: Specific error message if applicable
**Next**: What to try next OR decomposition plan

---
```

### Update Codebase Patterns
When recovery reveals reusable knowledge:
```markdown
## Codebase Patterns
- OAuth: Use next-auth v4, not v5 (breaking changes)
- Migrations: Always use IF NOT EXISTS
- Tests: Mock Date.now() for time-dependent logic
```

## Emergency Recovery

If everything is broken:

1. **Check git status**
   ```bash
   git status
   git log --oneline -10
   ```

2. **Find last good commit**
   ```bash
   git log --oneline | head -20
   ```

3. **Reset if necessary**
   ```bash
   git reset --hard <last-good-commit>
   ```

4. **Update prd.json**
   Mark affected stories as `"passes": false`

5. **Log incident in progress.txt**
   ```markdown
   ## EMERGENCY RESET: [DATE]
   Reset to commit abc123 due to cascading failures.
   Stories US-005 through US-008 marked incomplete.
   ```

6. **Create .ralph-pause**
   Allow human to review before continuing.
