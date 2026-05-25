from __future__ import annotations

from dataclasses import dataclass

from hillclimb.science.lean_prover.problems import FormalProblem, Goal
from hillclimb.science.lean_prover.tactics import ProofState, Tactic, TacticName, TacticStep


@dataclass
class VerificationResult:
    ok: bool
    state: ProofState
    error: str | None = None
    message: str = ""


class MockLeanVerifier:
    """
    Mock Lean 4 verifier: validates tactic steps against simple arithmetic rules.

    No Lean installation required; simulates type-checking and goal reduction.
    """

    def __init__(self, problem: FormalProblem) -> None:
        self.problem = problem
        self.rules = Tactic.RULES

    def initial_state(self) -> ProofState:
        goal_str = str(self.problem.goal)
        return ProofState(goals=[goal_str], context=[])

    def verify_script(self, steps: list[TacticStep]) -> VerificationResult:
        state = self.initial_state()
        for step in steps:
            result = self.apply_tactic(state, step)
            if not result.ok:
                return result
            state = result.state
        if not state.solved:
            return VerificationResult(
                ok=False,
                state=state,
                error="incomplete",
                message="Proof incomplete: goals remain.",
            )
        return VerificationResult(ok=True, state=state, message="QED")

    def apply_tactic(self, state: ProofState, step: TacticStep) -> VerificationResult:
        if state.solved:
            return VerificationResult(
                ok=False,
                state=state,
                error="already_solved",
                message="No goals left to solve.",
            )

        goal = state.primary_goal()
        if goal is None:
            return VerificationResult(
                ok=False,
                state=state,
                error="no_goal",
                message="No active goal.",
            )

        new_state = ProofState(
            goals=list(state.goals),
            context=list(state.context),
            solved=state.solved,
            steps_applied=list(state.steps_applied) + [step],
        )

        if step.name == TacticName.INTRO:
            return self._intro(new_state, step, goal)
        if step.name == TacticName.REWRITE:
            return self._rewrite(new_state, step, goal)
        if step.name == TacticName.APPLY:
            return self._apply(new_state, step, goal)
        if step.name == TacticName.INDUCTION:
            return self._induction(new_state, step, goal)
        if step.name == TacticName.CASES:
            return self._cases(new_state, step, goal)
        if step.name == TacticName.REFLEXIVITY:
            return self._reflexivity(new_state, goal)

        return VerificationResult(
            ok=False,
            state=state,
            error="unknown_tactic",
            message=f"Unknown tactic: {step.name}",
        )

    def _intro(self, state: ProofState, step: TacticStep, goal: str) -> VerificationResult:
        var = step.arg or "n"
        prefix = f"∀ {var}, "
        if not goal.startswith(prefix):
            return VerificationResult(
                ok=False,
                state=state,
                error="intro_failed",
                message=f"Goal is not universal: {goal}",
            )
        state.context.append(var)
        state.goals[0] = goal[len(prefix) :]
        return VerificationResult(ok=True, state=state, message=f"Introduced {var}.")

    def _rewrite(self, state: ProofState, step: TacticStep, goal: str) -> VerificationResult:
        rule = step.arg
        if rule not in self.problem.axioms:
            return VerificationResult(
                ok=False,
                state=state,
                error="unknown_axiom",
                message=f"Axiom `{rule}` not in scope.",
            )

        transformed = self._rewrite_goal(goal, rule)
        if transformed is None:
            return VerificationResult(
                ok=False,
                state=state,
                error="rewrite_failed",
                message=f"Cannot rewrite `{goal}` with `{rule}`.",
            )

        state.goals[0] = transformed
        if self._goal_closed(state.goals[0]):
            state.goals.pop(0)
            if not state.goals:
                state.solved = True
        return VerificationResult(ok=True, state=state, message=f"Rewrote with {rule}.")

    def _apply(self, state: ProofState, step: TacticStep, goal: str) -> VerificationResult:
        rule = step.arg
        if rule not in self.problem.axioms:
            return VerificationResult(
                ok=False,
                state=state,
                error="unknown_axiom",
                message=f"Axiom `{rule}` not in scope.",
            )
        rule_str = self.rules[rule]
        if self._matches_rule(goal, rule_str):
            state.goals.pop(0)
            if not state.goals:
                state.solved = True
            return VerificationResult(ok=True, state=state, message=f"Applied {rule}.")
        return VerificationResult(
            ok=False,
            state=state,
            error="apply_failed",
            message=f"Lemma `{rule}` does not apply to `{goal}`.",
        )

    def _induction(self, state: ProofState, step: TacticStep, goal: str) -> VerificationResult:
        var = step.arg or "n"
        if var not in state.context:
            return VerificationResult(
                ok=False,
                state=state,
                error="induction_failed",
                message=f"Variable `{var}` not in context.",
            )
        base = goal.replace(var, "0")
        step_case = goal.replace(var, f"suc({var})")
        state.goals = [base, step_case]
        return VerificationResult(ok=True, state=state, message=f"Induction on {var}.")

    def _cases(self, state: ProofState, step: TacticStep, goal: str) -> VerificationResult:
        var = step.arg or "n"
        if var not in state.context:
            return VerificationResult(
                ok=False,
                state=state,
                error="cases_failed",
                message=f"Variable `{var}` not in context.",
            )
        state.goals = [goal.replace(var, "0"), goal.replace(var, f"suc({var})")]
        return VerificationResult(ok=True, state=state, message=f"Cases on {var}.")

    def _reflexivity(self, state: ProofState, goal: str) -> VerificationResult:
        if not self._is_reflexive(goal):
            return VerificationResult(
                ok=False,
                state=state,
                error="rfl_failed",
                message=f"Goal `{goal}` is not reflexive.",
            )
        state.goals.pop(0)
        if not state.goals:
            state.solved = True
        return VerificationResult(ok=True, state=state, message="Closed by rfl.")

    def _rewrite_goal(self, goal: str, rule: str) -> str | None:
        lhs, rhs = goal.split(" = ", 1)
        lhs_inner = lhs.strip()
        if lhs_inner.startswith("(") and lhs_inner.endswith(")"):
            lhs_inner = lhs_inner[1:-1]

        if rule == "add_zero":
            if lhs_inner.startswith("0 + "):
                return f"{lhs_inner[4:]} = {rhs}"
            if lhs_inner.endswith(" + 0"):
                return f"{lhs_inner[:-4]} = {rhs}"
            if rhs.strip().endswith(" + 0"):
                inner_rhs = rhs.strip()
                if inner_rhs.startswith("(") and inner_rhs.endswith(")"):
                    inner_rhs = inner_rhs[1:-1]
                return f"{lhs} = {inner_rhs[:-4]}"
        if rule == "add_succ":
            for side, other in ((lhs_inner, rhs), (rhs, lhs_inner)):
                token = " + suc("
                if token in side:
                    idx = side.index(token)
                    base = side[:idx]
                    rest = side[idx + len(token) :]
                    if rest.endswith(")"):
                        m = rest[:-1]
                        new_side = f"suc({base} + {m})"
                        if side is lhs_inner:
                            return f"{new_side} = {rhs}"
                        return f"{lhs} = {new_side}"
        if rule == "mul_zero":
            if " * 0" in lhs_inner:
                if lhs_inner.strip().endswith("* 0") or lhs_inner.endswith(" * 0"):
                    return f"0 = {rhs}"
        if rule == "mul_succ":
            token = " * suc("
            if token in lhs_inner:
                idx = lhs_inner.index(token)
                base = lhs_inner[:idx]
                rest = lhs_inner[idx + len(token) :]
                if rest.endswith(")"):
                    m = rest[:-1]
                    return f"({base} * {m}) + {base} = {rhs}"
        return None

    @staticmethod
    def _goal_closed(goal: str) -> bool:
        if "=" not in goal:
            return False
        lhs, rhs = goal.split(" = ", 1)
        if lhs.strip() == rhs.strip():
            return True
        lval = MockLeanVerifier._eval_numeric(lhs)
        rval = MockLeanVerifier._eval_numeric(rhs)
        return lval is not None and lval == rval

    @staticmethod
    def _is_reflexive(goal: str) -> bool:
        return MockLeanVerifier._goal_closed(goal)

    @staticmethod
    def _eval_numeric(expr: str) -> int | None:
        cleaned = expr.strip().replace(" ", "")
        allowed = set("0123456789()+*")
        if not cleaned or not all(c in allowed for c in cleaned):
            return None
        try:
            return int(eval(cleaned, {"__builtins__": {}}, {}))
        except Exception:
            return None

    @staticmethod
    def _matches_rule(goal: str, rule: str) -> bool:
        if "=" not in goal or "=" not in rule:
            return False
        g_lhs, g_rhs = goal.split(" = ", 1)
        r_lhs, r_rhs = rule.split(" = ", 1)

        def pattern_match(pat: str, val: str) -> bool:
            pat = pat.strip()
            val = val.strip()
            if pat == val:
                return True
            if pat in {"n", "m"}:
                return True
            if "suc(" in pat and "suc(" in val:
                return True
            return False

        return pattern_match(r_lhs, g_lhs) and pattern_match(r_rhs, g_rhs)

    def score_state(self, state: ProofState, reference_len: int) -> float:
        if state.solved:
            return 1.0
        remaining = len(state.goals)
        progress = max(0, reference_len - remaining) / max(reference_len, 1)
        step_bonus = len(state.steps_applied) / max(reference_len * 2, 1)
        return min(0.95, 0.3 * progress + 0.4 * step_bonus + 0.1)
