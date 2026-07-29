#!/usr/bin/env python3
"""Post-process pybind11-stubgen output before it gets auto-committed.

Wraps implicit-Optional defaults and hard-fails on stub syntax errors
(e.g. a defaulted arg placed before a required one) instead of letting
them silently reach PyPI.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

IMPLICIT_OPTIONAL_RE = re.compile(
    r"(?P<name>\w+)\s*:\s*(?P<type>[\w\.\[\], ]+?)\s*=\s*None(?=[,\)])"
)


def fix_implicit_optional(source: str) -> str:
    def _replace(m: re.Match[str]) -> str:
        type_ = m.group("type").strip()
        if type_ in ("None", "Any") or "| None" in type_ or "Optional[" in type_:
            return m.group(0)
        return f"{m.group('name')}: {type_} | None = None"

    return IMPLICIT_OPTIONAL_RE.sub(_replace, source)


def check_parses(path: Path, source: str) -> str | None:
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return f"{path}: {e.msg} at line {e.lineno}"
    return None


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("nnengine")
    errors: list[str] = []

    for pyi_path in sorted(root.glob("*.pyi")):
        original = pyi_path.read_text()
        fixed = fix_implicit_optional(original)

        parse_error = check_parses(pyi_path, fixed)
        if parse_error:
            errors.append(parse_error)
            continue

        if fixed != original:
            pyi_path.write_text(fixed)
            print(f"fixed implicit-Optional defaults in {pyi_path}")

    if errors:
        print("\nSTUB VALIDATION FAILED -- refusing to commit broken stubs:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        print(
            "\nThis is almost always a py::arg(...) ordering issue in the "
            "pybind11 binding. Fix the .def(...) call in src/bindings/*.cpp, "
            "not this script.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))