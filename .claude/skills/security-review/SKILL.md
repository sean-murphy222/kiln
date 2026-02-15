---
name: security-review
description: >
  Security analysis patterns for code review. Apply when reviewing code
  changes, adding dependencies, or implementing authentication, data
  handling, or API endpoints.
---

# Security Review Skill

## Dangerous Patterns to Scan For

### Injection — CRITICAL
```python
# BAD: SQL injection
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
cursor.execute("SELECT * FROM users WHERE id = " + user_id)

# GOOD: Parameterized query
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

```python
# BAD: Command injection
os.system(f"convert {filename}")
subprocess.call(f"ls {user_input}", shell=True)

# GOOD: Safe subprocess
subprocess.run(["convert", filename], check=True)
```

```python
# BAD: eval/exec
eval(user_expression)
exec(config_string)

# GOOD: Never use eval/exec with any external input. Period.
```

### Secrets — CRITICAL
```python
# BAD: Hardcoded secrets
API_KEY = "sk-1234567890abcdef"
password = "hunter2"
token = "ghp_xxxxxxxxxxxxxxxxxxxx"

# GOOD: Environment variables
API_KEY = os.environ.get("API_KEY")
# Better: Secret manager
API_KEY = secret_client.get_secret("api-key")
```

### Path Traversal — HIGH
```python
# BAD: Unsanitized path
filepath = os.path.join(UPLOAD_DIR, user_filename)
open(filepath)

# GOOD: Validate path doesn't escape
filepath = os.path.join(UPLOAD_DIR, user_filename)
if not os.path.realpath(filepath).startswith(os.path.realpath(UPLOAD_DIR)):
    raise ValueError("Invalid path")
```

### Error Handling — MEDIUM
```python
# BAD: Leaks internals
except Exception as e:
    return {"error": str(e)}  # May contain DB schema, paths, etc.

# GOOD: Safe error messages
except Exception as e:
    logger.error(f"Operation failed: {e}")  # Log full error
    return {"error": "Operation failed. Contact support."}  # Return safe message
```

## Dependency Audit
When adding new dependencies:
1. Check if stdlib can do it first
2. Verify no critical CVEs: `npm audit` / `pip-audit`
3. Pin to specific version (not ranges)
4. Check last update date — abandoned packages are a risk
