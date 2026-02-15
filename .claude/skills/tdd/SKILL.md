---
name: tdd
description: >
  Test-driven development workflow. Apply when implementing any feature,
  fix, or refactor. Enforces red-green-refactor cycle with mandatory
  edge case and security test coverage.
---

# Test-Driven Development Skill

## The Cycle

Every implementation follows this exact sequence. No exceptions.

### Step 1: RED — Write the test first
```
Create or update the test file BEFORE touching source code.
Write tests that define the expected behavior.
Run tests — they MUST FAIL.
If tests pass before you've written implementation, your tests are wrong.
```

### Step 2: GREEN — Make it pass
```
Write the MINIMUM code needed to make tests pass.
No extra features, no optimization, no "while I'm here" changes.
Run tests — they MUST PASS.
```

### Step 3: REFACTOR — Clean up
```
Improve code quality without changing behavior.
Run tests after EVERY refactoring step.
Tests must stay green throughout.
Extract helpers, rename for clarity, reduce duplication.
```

### Step 4: COMMIT
```
Commit after each complete cycle.
Format: test(scope): description | feat(scope): description
```

## Test Categories — All Must Be Covered

| Category | What to test | Example |
|----------|-------------|---------|
| Happy path | Normal expected input/output | `test_parse_valid_document()` |
| Error path | Invalid inputs, failures | `test_parse_returns_error_on_malformed_xml()` |
| Edge cases | Empty, null, boundary values | `test_parse_empty_document_returns_empty_list()` |
| Security | Auth, injection, traversal | `test_parse_rejects_path_traversal_in_filename()` |

## Test Naming Convention

```python
# Python: test_{function}_{scenario}_{expected_result}
def test_chunk_document_with_empty_input_returns_empty_list():
def test_chunk_document_with_oversized_input_raises_value_error():
```

```javascript
// JS/TS: describe(unit) → it('should {behavior} when {condition}')
describe('chunkDocument', () => {
  it('should return empty array when input is empty', () => {});
  it('should throw ValueError when input exceeds max size', () => {});
});
```

## Coverage Targets

- New code: 90% line coverage minimum
- Changed files: coverage must not decrease
- Overall project: 80% minimum
