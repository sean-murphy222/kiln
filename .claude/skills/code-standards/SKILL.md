---
name: code-standards
description: >
  Project-specific coding standards. Apply when writing or reviewing code.
  Covers naming, typing, documentation, error handling, and architecture
  patterns. This skill grows over time through the compounding loop.
---

# Code Standards Skill

## This Skill Grows Over Time

Every time Claude makes a mistake that gets corrected, the pattern should be
added here as a "DO" / "DON'T" pair. This is the compounding loop in action.
Run `/retrospective` at the end of each sprint to identify new patterns.

## Universal Standards

### Naming
- Functions: `verb_noun` (Python: `parse_document`, JS: `parseDocument`)
- Classes: PascalCase descriptive nouns (`DocumentParser`, not `Parser`)
- Constants: SCREAMING_SNAKE_CASE (`MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- Booleans: prefix with is/has/can/should (`is_valid`, `has_content`)
- Avoid single-letter variables except in comprehensions and lambdas

### Documentation
- Every module: module-level docstring explaining purpose
- Every public function: docstring with params, returns, raises
- Every class: class-level docstring explaining responsibility
- Inline comments: explain WHY, not WHAT (the code shows what)

### Type Hints (Python) / Types (TypeScript)
- All function parameters and return types must be annotated
- Use Optional[] for nullable values, never bare None
- Use Union[] sparingly — prefer specific types
- Complex types get a TypeAlias

### Error Handling
- Never catch bare `except:` or `except Exception:`
- Catch specific exceptions
- Log the full error, return a safe message
- Use custom exception classes for domain errors
- Always include context in error messages

### Functions
- Maximum 50 lines per function
- Single responsibility — does one thing
- Maximum 4 parameters — use a config object/dataclass if more
- Pure functions preferred — minimize side effects
- Return early to avoid deep nesting

### Imports
- Standard library first
- Third-party second
- Project imports third
- Blank line between groups
- No wildcard imports (`from x import *`)

## Project-Specific Patterns

<!-- ADD YOUR PROJECT PATTERNS HERE -->
<!-- Example: -->
<!-- ### Database Access -->
<!-- - Always use the connection pool from db.pool, never create direct connections -->
<!-- - All queries go through the repository pattern in src/repositories/ -->
<!-- - Never expose SQLAlchemy models outside the repository layer -->

## Common Mistakes Catalog

<!-- THIS SECTION COMPOUNDS OVER TIME -->
<!-- After every correction, add entries like: -->
<!-- - DON'T: use datetime.now() → DO: use datetime.utcnow() or datetime.now(tz=UTC) -->
<!-- - DON'T: return dict from API → DO: return Pydantic model for validation -->
<!-- - DON'T: put business logic in route handlers → DO: extract to service layer -->
