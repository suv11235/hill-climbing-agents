"""Tests for SIFT coding self-improvement prototype."""

from __future__ import annotations

import pytest

from hillclimb.software.sift_coding.evaluator import evaluate_code, validate_code
from hillclimb.software.sift_coding.judge import judge_code
from hillclimb.software.sift_coding.patcher import patch_code
from hillclimb.software.sift_coding.run import run_climb
from hillclimb.software.sift_coding.task import get_task


@pytest.mark.parametrize(
    "task_name",
    ["reverse_string", "sort_list", "count_vowels", "sum_even"],
)
def test_starter_code_fails_some_tests(task_name: str) -> None:
    task = get_task(task_name)
    result = evaluate_code(task.starter_code, task)
    assert result.diagnostics["pass_rate"] < 1.0


def test_evaluator_passes_correct_implementation() -> None:
    task = get_task("reverse_string")
    code = "def reverse_string(s):\n    return s[::-1]\n"
    result = evaluate_code(code, task)
    assert result.passed
    assert result.score == 1.0


def test_sandbox_blocks_imports() -> None:
    code = "import os\ndef reverse_string(s):\n    return s[::-1]\n"
    assert validate_code(code) is not None


def test_judge_prefers_passing_code() -> None:
    task = get_task("reverse_string")
    buggy = task.starter_code
    fixed = "def reverse_string(s):\n    return s[::-1]\n"
    buggy_verdict = judge_code(buggy, task, test_pass_rate=0.0)
    fixed_verdict = judge_code(fixed, task, test_pass_rate=1.0)
    assert fixed_verdict.score > buggy_verdict.score
    assert fixed_verdict.accepted


def test_patcher_improves_reverse_string() -> None:
    task = get_task("reverse_string")
    initial = evaluate_code(task.starter_code, task)
    patched = patch_code(task.starter_code, task, initial.diagnostics)
    improved = evaluate_code(patched, task)
    assert improved.diagnostics["pass_rate"] > initial.diagnostics["pass_rate"]


def test_run_climb_reaches_full_pass_rate() -> None:
    summary = run_climb("reverse_string", max_rounds=5)
    assert summary["best_pass_rate"] == 1.0
    assert summary["best_pass_rate"] >= summary["initial_pass_rate"]


def test_run_climb_sort_list() -> None:
    summary = run_climb("sort_list", max_rounds=5)
    assert summary["best_pass_rate"] == 1.0
