from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExprKind(str, Enum):
    VAR = "var"
    ZERO = "zero"
    SUCC = "succ"
    ADD = "add"
    MUL = "mul"
    LIT = "lit"


@dataclass(frozen=True)
class Expr:
    """Simple natural-number expression AST."""

    kind: ExprKind
    name: str = ""
    left: Expr | None = None
    right: Expr | None = None
    child: Expr | None = None
    value: int | None = None

    def __str__(self) -> str:
        if self.kind == ExprKind.VAR:
            return self.name
        if self.kind == ExprKind.ZERO:
            return "0"
        if self.kind == ExprKind.LIT:
            return str(self.value)
        if self.kind == ExprKind.SUCC:
            return f"suc({self.child})"
        if self.kind == ExprKind.ADD:
            return f"({self.left} + {self.right})"
        if self.kind == ExprKind.MUL:
            return f"({self.left} * {self.right})"
        return "?"

    def eval(self, env: dict[str, int] | None = None) -> int:
        env = env or {}
        if self.kind == ExprKind.ZERO:
            return 0
        if self.kind == ExprKind.LIT:
            return int(self.value)
        if self.kind == ExprKind.VAR:
            return env[self.name]
        if self.kind == ExprKind.SUCC:
            assert self.child is not None
            return self.child.eval(env) + 1
        if self.kind == ExprKind.ADD:
            assert self.left and self.right
            return self.left.eval(env) + self.right.eval(env)
        if self.kind == ExprKind.MUL:
            assert self.left and self.right
            return self.left.eval(env) * self.right.eval(env)
        raise ValueError(f"Cannot evaluate {self}")

    @staticmethod
    def lit(n: int) -> Expr:
        return Expr(kind=ExprKind.LIT, value=n)

    @staticmethod
    def var(name: str) -> Expr:
        return Expr(kind=ExprKind.VAR, name=name)

    @staticmethod
    def add(a: Expr, b: Expr) -> Expr:
        return Expr(kind=ExprKind.ADD, left=a, right=b)

    @staticmethod
    def mul(a: Expr, b: Expr) -> Expr:
        return Expr(kind=ExprKind.MUL, left=a, right=b)

    @staticmethod
    def succ(n: Expr) -> Expr:
        return Expr(kind=ExprKind.SUCC, child=n)


@dataclass(frozen=True)
class Goal:
    """Prove lhs = rhs (optionally under a variable binding)."""

    lhs: Expr
    rhs: Expr
    binding: str | None = None

    def __str__(self) -> str:
        if self.binding:
            return f"∀ {self.binding}, {self.lhs} = {self.rhs}"
        return f"{self.lhs} = {self.rhs}"

    def is_solved(self) -> bool:
        if self.binding:
            return False
        try:
            return self.lhs.eval() == self.rhs.eval()
        except (ValueError, KeyError):
            return str(self.lhs) == str(self.rhs)


@dataclass
class FormalProblem:
    """Toy formal problem with a known proof script."""

    name: str
    goal: Goal
    axioms: list[str] = field(default_factory=list)
    reference_proof: list[str] = field(default_factory=list)
    difficulty: int = 1


def all_problems() -> list[FormalProblem]:
    n = Expr.var("n")
    zero = Expr.lit(0)
    two = Expr.lit(2)
    four = Expr.lit(4)

    return [
        FormalProblem(
            name="add_zero_right",
            goal=Goal(lhs=Expr.add(n, zero), rhs=n, binding="n"),
            axioms=["add_zero", "add_succ"],
            reference_proof=["intro n", "induction n", "rewrite add_zero", "rewrite add_zero"],
            difficulty=2,
        ),
        FormalProblem(
            name="add_zero_left",
            goal=Goal(lhs=Expr.add(zero, n), rhs=n, binding="n"),
            axioms=["add_zero", "add_succ"],
            reference_proof=["intro n", "rewrite add_zero"],
            difficulty=1,
        ),
        FormalProblem(
            name="two_plus_two",
            goal=Goal(lhs=Expr.add(two, two), rhs=four),
            axioms=["add_zero", "add_succ"],
            reference_proof=["rfl"],
            difficulty=1,
        ),
        FormalProblem(
            name="mul_zero",
            goal=Goal(lhs=Expr.mul(n, zero), rhs=zero, binding="n"),
            axioms=["mul_zero", "mul_succ"],
            reference_proof=["intro n", "induction n", "rewrite mul_zero", "rewrite mul_zero"],
            difficulty=2,
        ),
    ]


def get_problem(name: str) -> FormalProblem:
    for problem in all_problems():
        if problem.name == name:
            return problem
    raise KeyError(f"Unknown problem: {name}")
