"""Simple coding tasks with starter implementations and test cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TestCase:
    name: str
    args: tuple[Any, ...]
    expected: Any


@dataclass(frozen=True)
class CodingTask:
    name: str
    description: str
    function_name: str
    starter_code: str
    test_cases: tuple[TestCase, ...]


TASKS: dict[str, CodingTask] = {
    "reverse_string": CodingTask(
        name="reverse_string",
        description="Return the reverse of a string.",
        function_name="reverse_string",
        starter_code=(
            "def reverse_string(s):\n"
            "    # Bug: returns input unchanged\n"
            "    return s\n"
        ),
        test_cases=(
            TestCase("basic", ("hello",), "olleh"),
            TestCase("empty", ("",), ""),
            TestCase("palindrome", ("aba",), "aba"),
            TestCase("unicode", ("café",), "éfac"),
        ),
    ),
    "sort_list": CodingTask(
        name="sort_list",
        description="Return a sorted copy of a list of integers.",
        function_name="sort_list",
        starter_code=(
            "def sort_list(items):\n"
            "    # Bug: drops the last element\n"
            "    return sorted(items)[:-1]\n"
        ),
        test_cases=(
            TestCase("basic", ([3, 1, 2],), [1, 2, 3]),
            TestCase("empty", ([],), []),
            TestCase("single", ([7],), [7]),
            TestCase("duplicates", ([2, 2, 1],), [1, 2, 2]),
        ),
    ),
    "count_vowels": CodingTask(
        name="count_vowels",
        description="Count vowels in a string (case-insensitive).",
        function_name="count_vowels",
        starter_code=(
            "def count_vowels(text):\n"
            "    # Bug: only lowercase vowels\n"
            '    return sum(1 for c in text if c in "aeiou")\n'
        ),
        test_cases=(
            TestCase("lower", ("hello",), 2),
            TestCase("upper", ("HELLO",), 2),
            TestCase("mixed", ("HeLLo",), 2),
            TestCase("none", ("xyz",), 0),
        ),
    ),
    "sum_even": CodingTask(
        name="sum_even",
        description="Sum only the even integers in a list.",
        function_name="sum_even",
        starter_code=(
            "def sum_even(numbers):\n"
            "    # Bug: sums all numbers\n"
            "    return sum(numbers)\n"
        ),
        test_cases=(
            TestCase("mixed", ([1, 2, 3, 4],), 6),
            TestCase("all_odd", ([1, 3, 5],), 0),
            TestCase("all_even", ([2, 4, 6],), 12),
            TestCase("empty", ([],), 0),
        ),
    ),
}


def get_task(name: str = "reverse_string") -> CodingTask:
    if name not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task {name!r}. Available: {available}")
    return TASKS[name]


def run_test(fn: Callable[..., Any], case: TestCase) -> tuple[bool, Any, str | None]:
    try:
        actual = fn(*case.args)
        if actual == case.expected:
            return True, actual, None
        return False, actual, f"expected {case.expected!r}, got {actual!r}"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
