"""Sandboxed test execution for coding candidates."""

from __future__ import annotations

import ast
from typing import Any

from hillclimb.core.types import Evaluation
from hillclimb.software.sift_coding.task import CodingTask, run_test


ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def validate_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Imports are not allowed in sandboxed code"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"exec", "eval", "open", "__import__", "compile"}:
                return f"Disallowed call: {node.func.id}()"
    return None


def load_function(code: str, function_name: str) -> tuple[Any | None, str | None]:
    syntax_error = validate_code(code)
    if syntax_error:
        return None, syntax_error

    namespace: dict[str, Any] = {"__builtins__": ALLOWED_BUILTINS}
    try:
        exec(code, namespace)  # noqa: S102 — intentional sandboxed execution
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    fn = namespace.get(function_name)
    if fn is None or not callable(fn):
        return None, f"Function {function_name!r} not found"
    return fn, None


def evaluate_code(code: str, task: CodingTask) -> Evaluation:
    fn, load_error = load_function(code, task.function_name)
    if load_error:
        return Evaluation(
            score=0.0,
            diagnostics={"pass_rate": 0.0, "failed_tests": [], "load_error": load_error},
            passed=False,
            error=load_error,
        )

    passed_count = 0
    failed_tests: list[dict[str, Any]] = []
    for case in task.test_cases:
        ok, actual, message = run_test(fn, case)
        if ok:
            passed_count += 1
        else:
            failed_tests.append(
                {
                    "name": case.name,
                    "args": case.args,
                    "expected": case.expected,
                    "actual": actual,
                    "error": message,
                }
            )

    total = len(task.test_cases)
    pass_rate = passed_count / total if total else 0.0
    error_types = sorted(
        {entry["error"].split(":")[0] for entry in failed_tests if entry.get("error")}
    )

    return Evaluation(
        score=pass_rate,
        diagnostics={
            "pass_rate": pass_rate,
            "passed": passed_count,
            "total": total,
            "failed_tests": failed_tests,
            "error_types": error_types,
        },
        passed=pass_rate == 1.0,
    )
