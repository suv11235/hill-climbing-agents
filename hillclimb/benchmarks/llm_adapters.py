"""LLM-enhanced proposers for hill-climbing experiments."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

from hillclimb.core.llm import LLMClient
from hillclimb.core.types import Candidate, Evaluation


@dataclass
class LLMOCRRefiner:
    """LLM proposes regex/preprocessing fixes from OCR field diagnostics."""

    llm: LLMClient
    fallback: Any

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        missed = diagnostics.get("field_diagnostics", [])
        if not missed:
            return self.fallback.propose(current, history)

        config = current.state
        system = (
            "You improve OCR parsing configs for invoices. Common failure: regex for "
            "'total' matches 'Subtotal' — use (?<!Sub)Total pattern. "
            "Return JSON with keys: field_patterns (dict field->regex), scale, contrast, "
            "threshold (int), sharpen (bool), invert (bool)."
        )
        user = json.dumps(
            {
                "current_patterns": config.field_patterns,
                "preprocess": {
                    "scale": config.scale,
                    "contrast": config.contrast,
                    "threshold": config.threshold,
                    "sharpen": config.sharpen,
                    "invert": config.invert,
                },
                "failed_fields": missed[:5],
            },
            indent=2,
        )
        try:
            patch = self.llm.complete_json(system, user)
            new_config = copy.deepcopy(config)
            for field, pattern in patch.get("field_patterns", {}).items():
                if isinstance(pattern, str):
                    new_config.field_patterns[field] = pattern
            for key in ("scale", "contrast", "threshold", "sharpen", "invert"):
                if key in patch:
                    setattr(new_config, key, patch[key])
            return Candidate(state=new_config, metadata={"source": "llm_ocr_refiner"})
        except Exception:
            return self.fallback.propose(current, history)


@dataclass
class LLMSiftProposer:
    """LLM proposes code patches from test failure diagnostics."""

    llm: LLMClient
    task: Any
    fallback: Any

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        failed = diagnostics.get("failed_tests", [])
        if not failed:
            return self.fallback.propose(current, history)

        system = (
            "You fix Python functions to pass unit tests. "
            "Return JSON: {\"code\": \"full corrected function code\"}. "
            "Only standard library. Keep the same function signature."
        )
        user = json.dumps(
            {
                "task": self.task.description,
                "current_code": current.state,
                "failed_tests": failed,
                "test_cases": [
                    {"name": t.name, "args": t.args, "expected": t.expected}
                    for t in self.task.test_cases
                ],
            },
            indent=2,
        )
        try:
            result = self.llm.complete_json(system, user)
            code = result.get("code", "")
            if code and "def " in code:
                return Candidate(
                    state=code,
                    metadata={"source": "llm_sift_proposer"},
                )
        except Exception:
            pass
        return self.fallback.propose(current, history)


@dataclass
class LLMFinanceProposer:
    """LLM refines trading strategy hypotheses from backtest diagnostics."""

    llm: LLMClient
    researcher: Any
    fallback: Any

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        hypothesis = current.state

        system = (
            "You are a quantitative researcher. Given a strategy hypothesis and backtest "
            "diagnostics, propose an improved hypothesis. Return JSON with keys: "
            "kind (momentum|mean_reversion|vol_target), lookback (int 5-60), "
            "vol_target (float 0.05-0.25), leverage (float 0.5-2.0), "
            "reasoning (short string)."
        )
        user = json.dumps(
            {
                "current": hypothesis.to_dict(),
                "diagnostics": {
                    k: diagnostics.get(k)
                    for k in ("sharpe", "max_drawdown", "turnover", "reflection")
                },
            },
            indent=2,
        )
        try:
            patch = self.llm.complete_json(system, user)
            from hillclimb.science.finance_research.hypothesis import StrategyHypothesis
            from hillclimb.science.finance_research.strategies import StrategyKind

            kind_str = patch.get("kind", hypothesis.kind.value)
            try:
                kind = StrategyKind(kind_str)
            except ValueError:
                kind = hypothesis.kind
            refined = StrategyHypothesis(
                kind=kind,
                lookback=int(patch.get("lookback", hypothesis.lookback)),
                target_vol=float(patch.get("vol_target", patch.get("target_vol", hypothesis.target_vol))),
                max_leverage=float(
                    patch.get("leverage", patch.get("max_leverage", hypothesis.max_leverage))
                ),
            )
            return Candidate(
                state=refined,
                metadata={"source": "llm_finance", "reasoning": patch.get("reasoning", "")},
            )
        except Exception:
            return self.fallback.propose(current, history)


@dataclass
class LLMProofProposer:
    """LLM proposes next Lean tactic from goal state and verifier error."""

    llm: LLMClient
    problem: Any
    verifier: Any
    fallback: Any

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        from hillclimb.science.lean_prover.proposer import ProofCandidate
        from hillclimb.science.lean_prover.tactics import TacticStep

        proof: ProofCandidate = current.state
        diagnostics = history[-1][1].diagnostics if history else {}

        system = (
            "You are a Lean 4 proof assistant. Given a goal and available axioms/tactics, "
            "propose ONE next tactic line. Return JSON: {\"tactic\": \"intro n\"} "
            "Valid tactics: intro <var>, rewrite <axiom>, apply <axiom>, rfl, cases <var>, induction <var>."
        )
        user = json.dumps(
            {
                "goal": self.problem.goal,
                "axioms": list(self.problem.axioms),
                "current_script": proof.script(),
                "error": diagnostics.get("error"),
                "goal_stack": diagnostics.get("goals", []),
            },
            indent=2,
        )
        try:
            result = self.llm.complete_json(system, user)
            tactic_str = result.get("tactic", "").strip()
            if tactic_str:
                step = TacticStep.parse(tactic_str)
                new_steps = list(proof.steps) + [step]
                return Candidate(
                    state=ProofCandidate(steps=new_steps),
                    metadata={"source": "llm_proof", "tactic": tactic_str},
                )
        except Exception:
            pass
        return self.fallback.propose(current, history)


@dataclass
class LLMHypothesisProposer:
    """LLM evolves research hypotheses via tournament feedback."""

    llm: LLMClient
    fallback: Any
    population_size: int = 4

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis

        state = current.state
        diagnostics = history[-1][1].diagnostics if history else {}
        top_hyps = diagnostics.get("top_hypotheses", [])
        if not top_hyps and state.hypotheses:
            top_hyps = [
                {
                    "claim": h.claim,
                    "mechanism": h.mechanism,
                    "prediction": h.testable_prediction,
                }
                for h in state.hypotheses[:2]
            ]
        if diagnostics.get("best_claim"):
            top_hyps = [{"claim": diagnostics["best_claim"]}] + top_hyps

        system = (
            "You evolve scientific hypotheses for a research question. "
            "Return JSON: {\"hypotheses\": [{\"claim\": \"...\", \"mechanism\": \"...\", "
            "\"testable_prediction\": \"...\"}]} with exactly "
            f"{self.population_size} improved hypotheses."
        )
        user = json.dumps(
            {
                "research_question": state.research_question,
                "evidence_context": state.evidence_context,
                "current_best": top_hyps,
                "score": diagnostics.get("composite_score"),
            },
            indent=2,
        )
        try:
            result = self.llm.complete_json(system, user)
            hyps = []
            for i, h in enumerate(result.get("hypotheses", [])[: self.population_size]):
                hyps.append(
                    Hypothesis(
                        id=f"llm_{state.tournament_round}_{i}",
                        claim=h.get("claim", ""),
                        mechanism=h.get("mechanism", ""),
                        testable_prediction=h.get("testable_prediction", ""),
                    )
                )
            if hyps:
                from hillclimb.science.hypothesis_tournament.run import TournamentState

                new_state = TournamentState(
                    research_question=state.research_question,
                    hypotheses=hyps,
                    evidence_context=state.evidence_context,
                    tournament_round=state.tournament_round + 1,
                )
                return Candidate(state=new_state, metadata={"source": "llm_hypothesis"})
        except Exception:
            pass
        return self.fallback.propose(current, history)


@dataclass
class LLMConfigProposer:
    """LLM proposes sklearn hyperparameter changes from CV diagnostics."""

    llm: LLMClient
    fallback: Any

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        diagnostics = history[-1][1].diagnostics if history else {}
        config = current.state

        system = (
            "You tune RandomForest hyperparameters. Return JSON with any of: "
            "n_estimators (int 10-300), max_depth (int 2-30 or null), "
            "min_samples_split (int 2-20), min_samples_leaf (int 1-10), "
            "max_features (sqrt|log2|null). Improve cross-validation accuracy."
        )
        user = json.dumps({"current": config, "diagnostics": diagnostics}, indent=2)
        try:
            patch = self.llm.complete_json(system, user)
            new_config = dict(config)
            new_config.update({k: v for k, v in patch.items() if k in config or k in patch})
            return Candidate(state=new_config, metadata={"source": "llm_config"})
        except Exception:
            return self.fallback.propose(current, history)


@dataclass
class LLMRLInterfaceProposer:
    """LLM proposes observation indices and reward shaping for RL interface."""

    llm: LLMClient
    fallback: Any
    hard_mode: bool = False

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        from hillclimb.software.rl_interface.interface import RLInterface

        interface: RLInterface = current.state
        diagnostics = history[-1][1].diagnostics if history else {}

        n_features = 6 if self.hard_mode else 4
        hard_hint = (
            "HARD MODE: features 0-1 are distractors; agent position is at indices 2 (x) and 3 (y). "
            "Use feature_indices [2, 3] with distance shaping enabled."
            if self.hard_mode
            else "Easy mode: position at indices 0, 1."
        )
        system = (
            f"You design RL interfaces for a gridworld. Raw state has {n_features} features. "
            f"{hard_hint} "
            "Return JSON: {\"feature_indices\": [int,...], \"num_bins\": int 2-8, "
            "\"use_distance_shaping\": bool, \"goal_reward\": float 5-20, "
            "\"step_penalty\": float -1 to 0, \"distance_coef\": float 0-1}."
        )
        user = json.dumps(
            {
                "current": {
                    "feature_indices": interface.observation.feature_indices,
                    "num_bins": interface.observation.num_bins,
                    "use_distance_shaping": interface.reward.use_distance_shaping,
                    "goal_reward": interface.reward.goal_reward,
                    "step_penalty": interface.reward.step_penalty,
                    "distance_coef": interface.reward.distance_coef,
                },
                "diagnostics": diagnostics,
                "hard_mode_hint": "True position may NOT be in first two features" if self.hard_mode else "",
            },
            indent=2,
        )
        try:
            patch = self.llm.complete_json(system, user)
            new_iface = copy.deepcopy(interface)
            obs_idx = patch.get("obs_indices") or patch.get("feature_indices")
            if obs_idx:
                new_iface.observation.feature_indices = [int(i) for i in obs_idx]
            bins = patch.get("discretize_bins") or patch.get("num_bins")
            if bins is not None:
                new_iface.observation.num_bins = int(bins)
            if "use_distance_shaping" in patch or "distance_shaping" in patch:
                new_iface.reward.use_distance_shaping = bool(
                    patch.get("use_distance_shaping", patch.get("distance_shaping"))
                )
            if "goal_reward" in patch or "goal_bonus" in patch:
                new_iface.reward.goal_reward = float(
                    patch.get("goal_reward", patch.get("goal_bonus"))
                )
            if "step_penalty" in patch:
                new_iface.reward.step_penalty = float(patch["step_penalty"])
            if "distance_coef" in patch:
                new_iface.reward.distance_coef = float(patch["distance_coef"])
            return Candidate(state=new_iface, metadata={"source": "llm_rl_interface"})
        except Exception:
            return self.fallback.propose(current, history)


@dataclass
class LLMPortfolioProposer:
    """LLM adjusts regime-aware portfolio weights from Sharpe diagnostics."""

    llm: LLMClient
    fallback: Any

    def propose(
        self, current: Candidate, history: list[tuple[Candidate, Evaluation]]
    ) -> Candidate:
        params = current.state
        diagnostics = history[-1][1].diagnostics if history else {}

        system = (
            "You optimize portfolio allocation across bull/bear/sideways regimes. "
            "Three assets. Weights must sum to 1.0 per regime. "
            "Return JSON: {\"bull_weights\": [f,f,f], \"bear_weights\": [f,f,f], "
            "\"sideways_weights\": [f,f,f], \"turnover_penalty\": float 0-0.5}."
        )
        user = json.dumps(
            {
                "current": {
                    "bull_weights": list(params.bull_weights),
                    "bear_weights": list(params.bear_weights),
                    "sideways_weights": list(params.sideways_weights),
                    "turnover_penalty": params.turnover_penalty,
                },
                "diagnostics": diagnostics,
            },
            indent=2,
        )
        try:
            import numpy as np
            from hillclimb.science.portfolio_optimizer.allocator import AllocationParams

            patch = self.llm.complete_json(system, user)

            def _norm(w: list[float]) -> tuple[float, float, float]:
                arr = np.array(w, dtype=float)
                arr = np.clip(arr, 0, 1)
                s = arr.sum()
                if s <= 0:
                    return (1 / 3, 1 / 3, 1 / 3)
                arr /= s
                return (float(arr[0]), float(arr[1]), float(arr[2]))

            new_params = AllocationParams(
                bull_weights=_norm(patch.get("bull_weights", list(params.bull_weights))),
                bear_weights=_norm(patch.get("bear_weights", list(params.bear_weights))),
                sideways_weights=_norm(
                    patch.get("sideways_weights", list(params.sideways_weights))
                ),
                turnover_penalty=float(
                    patch.get("turnover_penalty", params.turnover_penalty)
                ),
            )
            return Candidate(state=new_params, metadata={"source": "llm_portfolio"})
        except Exception:
            return self.fallback.propose(current, history)
