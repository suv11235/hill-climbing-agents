"""Heuristic LLM-as-judge simulation — no external API required."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from hillclimb.software.sift_coding.task import CodingTask


@dataclass(frozen=True)
class JudgeVerdict:
    score: float
    reasons: tuple[str, ...]
    accepted: bool


def _code_length_score(code: str) -> tuple[float, str]:
    lines = [line for line in code.splitlines() if line.strip() and not line.strip().startswith("#")]
    n_lines = len(lines)
    if n_lines <= 6:
        return 1.0, "concise implementation"
    if n_lines <= 12:
        return 0.7, "moderate length"
    return 0.4, "verbose implementation"


def _complexity_score(code: str) -> tuple[float, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0, "syntax error"

    loops = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
    nested = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.While, ast.If))
        and any(isinstance(child, (ast.For, ast.While)) for child in ast.iter_child_nodes(node))
    )
    if loops == 0:
        return 1.0, "low complexity"
    if loops <= 2 and nested == 0:
        return 0.8, "acceptable complexity"
    return 0.5, "high complexity"


def _style_score(code: str, task: CodingTask) -> tuple[float, str]:
    if f"def {task.function_name}" not in code:
        return 0.0, "missing target function"
    if re.search(r"return\s+None\b", code) and task.name != "sort_list":
        return 0.5, "returns None explicitly"
    return 1.0, "function present"


def judge_code(
    code: str,
    task: CodingTask,
    *,
    test_pass_rate: float | None = None,
    min_accept_score: float = 0.35,
) -> JudgeVerdict:
    """
    Simulate an LLM judge with cheap heuristics.

    Weights mirror SIFT's sample-efficient pre-filter: correctness signal first,
    then brevity and complexity penalties.
    """
    reasons: list[str] = []
    style, style_reason = _style_score(code, task)
    reasons.append(style_reason)

    length, length_reason = _code_length_score(code)
    reasons.append(length_reason)

    complexity, complexity_reason = _complexity_score(code)
    reasons.append(complexity_reason)

    pass_component = test_pass_rate if test_pass_rate is not None else 0.5
    if test_pass_rate is not None:
        reasons.append(f"test pass rate {test_pass_rate:.0%}")

    score = 0.55 * pass_component + 0.20 * style + 0.15 * length + 0.10 * complexity
    accepted = score >= min_accept_score and style > 0

    return JudgeVerdict(score=score, reasons=tuple(reasons), accepted=accepted)
