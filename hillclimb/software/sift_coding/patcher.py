"""Diagnostic-aware code patch proposer."""

from __future__ import annotations

import random
import re
from typing import Any

from hillclimb.software.sift_coding.task import CodingTask


def _failed_names(diagnostics: dict[str, Any]) -> set[str]:
    return {entry["name"] for entry in diagnostics.get("failed_tests", [])}


def _patch_reverse_string(code: str, diagnostics: dict[str, Any]) -> str:
    failed = _failed_names(diagnostics)
    if "basic" in failed or "unicode" in failed:
        if "return s[::-1]" not in code:
            return re.sub(r"return\s+s\s*$", "return s[::-1]", code, flags=re.MULTILINE)
    if "empty" in failed and "if not s" not in code:
        return code.replace(
            "def reverse_string(s):",
            "def reverse_string(s):\n    if not s:\n        return s",
        )
    return code.replace("return s\n", "return s[::-1]\n")


def _patch_sort_list(code: str, diagnostics: dict[str, Any]) -> str:
    if "[:-1]" in code:
        return code.replace("sorted(items)[:-1]", "sorted(items)")
    failed = _failed_names(diagnostics)
    if "empty" in failed and "if not items" not in code:
        return code.replace(
            "def sort_list(items):",
            "def sort_list(items):\n    if not items:\n        return []",
        )
    return code.replace("sorted(items)[:-1]", "sorted(items)")


def _patch_count_vowels(code: str, diagnostics: dict[str, Any]) -> str:
    failed = _failed_names(diagnostics)
    if failed & {"upper", "mixed"}:
        if ".lower()" not in code:
            return re.sub(
                r"for c in text",
                "for c in text.lower()",
                code,
            )
        if '"aeiou"' in code and "AEIOU" not in code:
            return code.replace('"aeiou"', '"aeiouAEIOU"')
    return code.replace(
        'if c in "aeiou"',
        'if c.lower() in "aeiou"',
    )


def _patch_sum_even(code: str, diagnostics: dict[str, Any]) -> str:
    if "sum(numbers)" in code and "n % 2" not in code:
        return code.replace(
            "return sum(numbers)",
            "return sum(n for n in numbers if n % 2 == 0)",
        )
    failed = _failed_names(diagnostics)
    if "empty" in failed and "if not numbers" not in code:
        return code.replace(
            "def sum_even(numbers):",
            "def sum_even(numbers):\n    if not numbers:\n        return 0",
        )
    return code


_PATCHERS = {
    "reverse_string": _patch_reverse_string,
    "sort_list": _patch_sort_list,
    "count_vowels": _patch_count_vowels,
    "sum_even": _patch_sum_even,
}


def patch_code(code: str, task: CodingTask, diagnostics: dict[str, Any]) -> str:
    patcher = _PATCHERS.get(task.name)
    if patcher is None:
        return code
    patched = patcher(code, diagnostics)
    return patched if patched != code else _fallback_patch(code, task)


def _fallback_patch(code: str, task: CodingTask) -> str:
    """Small exploratory mutation when targeted rules stall."""
    lines = code.splitlines()
    if not lines:
        return code
    idx = random.randint(0, len(lines) - 1)
    line = lines[idx]
    if "return" in line and task.name == "reverse_string":
        lines[idx] = "    return s[::-1]"
    elif "return" in line and task.name == "sort_list":
        lines[idx] = "    return sorted(items)"
    return "\n".join(lines) + ("\n" if code.endswith("\n") else "")


def generate_candidates(
    code: str, task: CodingTask, diagnostics: dict[str, Any], *, n: int = 3
) -> list[str]:
    """Generate multiple patch variants for judge pre-filtering."""
    primary = patch_code(code, task, diagnostics)
    candidates = [primary]
    for _ in range(n - 1):
        variant = _fallback_patch(primary, task)
        if variant not in candidates:
            candidates.append(variant)
    return candidates
