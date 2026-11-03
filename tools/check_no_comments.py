import ast
import io
import sys
import tokenize
from pathlib import Path

EXCLUDED_PARTS = {
    "legacy",
    ".venv",
    "venv",
    "data",
    "results",
    ".git",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    ".ruff_cache",
    ".pytest_cache",
}
LINE_COMMENT_SUFFIXES = {".yaml", ".yml", ".toml", ".sh", ".cfg", ".ini"}
LINE_COMMENT_NAMES = {"Makefile"}
DEFAULT_ROOTS = ["src", "tests", "pipeline", "tools", "config", "Makefile", "pyproject.toml"]


def iter_files(roots):
    for root in roots:
        path = Path(root)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for candidate in candidates:
            if candidate.is_file() and not EXCLUDED_PARTS.intersection(candidate.parts):
                yield candidate


def has_docstring(node):
    body = node.body
    return (
        bool(body)
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    )


def python_violations(path):
    source = path.read_text(encoding="utf-8")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        tree = ast.parse(source)
    except (SyntaxError, tokenize.TokenError) as error:
        yield 0, f"unparseable: {error}"
        return
    for token in tokens:
        is_shebang = token.start[0] == 1 and token.string.startswith("#!")
        if token.type == tokenize.COMMENT and not is_shebang:
            yield token.start[0], "comment"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if has_docstring(node):
                yield node.body[0].lineno, "docstring"


def line_comment_violations(path):
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        is_shebang = number == 1 and line.startswith("#!")
        if line.lstrip().startswith("#") and not is_shebang:
            yield number, "comment"


def violations_for(path):
    if path.suffix == ".py":
        return python_violations(path)
    if path.suffix in LINE_COMMENT_SUFFIXES or path.name in LINE_COMMENT_NAMES:
        return line_comment_violations(path)
    return iter(())


def main(argv):
    roots = argv or DEFAULT_ROOTS
    found = 0
    for path in iter_files(roots):
        for line, kind in violations_for(path):
            print(f"{path}:{line}: {kind}")
            found += 1
    if found:
        print(f"{found} violation(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
