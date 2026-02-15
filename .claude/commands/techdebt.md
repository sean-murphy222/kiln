# /techdebt — End-of-Session Technical Debt Scan

Boris Cherny runs this at the end of every session. Scan the codebase for:

```bash
echo "=== TECH DEBT SCAN ==="

echo ""
echo "--- TODO/FIXME/HACK Comments ---"
grep -rn "TODO\|FIXME\|HACK\|XXX\|WORKAROUND" src/ --include="*.py" --include="*.js" --include="*.ts" --include="*.jsx" --include="*.tsx" 2>/dev/null | head -20

echo ""
echo "--- Dead Imports ---"
# Python: imported but unused (basic check)
for f in $(find src/ -name "*.py" 2>/dev/null); do
    ruff check --select F401 "$f" 2>/dev/null
done

echo ""
echo "--- Long Functions (>50 lines) ---"
# This is a rough heuristic
for f in $(find src/ -name "*.py" 2>/dev/null); do
    awk '/^def /{name=$0; start=NR} /^def |^class /{if(NR-start>50 && start>0) print FILENAME":"start" ("NR-start" lines) "name; start=NR}' "$f" 2>/dev/null
done

echo ""
echo "--- Duplicated Code Patterns ---"
# Check for similar function signatures that might be duplicated
grep -rn "^def \|^function \|^const .* = " src/ --include="*.py" --include="*.js" --include="*.ts" 2>/dev/null | awk -F: '{print $3}' | sort | uniq -d | head -10

echo ""
echo "--- Missing Docstrings ---"
for f in $(find src/ -name "*.py" 2>/dev/null); do
    python3 -c "
import ast, sys
with open('$f') as fh:
    tree = ast.parse(fh.read())
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        if not ast.get_docstring(node):
            print(f'$f:{node.lineno} {node.name} - missing docstring')
" 2>/dev/null
done | head -20
```

After scanning:
1. Create new tasks in the BACKLOG.md Icebox for significant tech debt items
2. Group related items (e.g., all missing docstrings = one task)
3. Estimate story points for each new task
4. Note any tech debt that's actively harmful vs. merely untidy
