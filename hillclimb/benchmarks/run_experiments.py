"""Run all hill-climbing experiments with baseline and LLM-enhanced proposers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from hillclimb.core.harness import HillClimber
from hillclimb.core.llm import LLMClient, get_llm
from hillclimb.core.types import AcceptPolicy, Candidate
from hillclimb.benchmarks.hybrid import LLMEnhancedProposer
from hillclimb.benchmarks.llm_adapters import (
    LLMConfigProposer,
    LLMFinanceProposer,
    LLMHypothesisProposer,
    LLMOCRRefiner,
    LLMPortfolioProposer,
    LLMProofProposer,
    LLMRLInterfaceProposer,
    LLMSiftProposer,
)


@dataclass
class ExperimentResult:
    name: str
    domain: str
    mode: str  # "baseline" | "llm"
    initial_score: float
    best_score: float
    improvement: float
    rounds: int
    converged: bool
    target_met: bool
    details: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


def _score_from_history(history: list) -> tuple[float, float]:
    if not history:
        return 0.0, 0.0
    initial = history[0][1].score
    best = max(ev.score for _, ev in history)
    return initial, best


def run_rl_interface(*, llm: LLMClient | None, hard: bool = True, max_rounds: int = 8) -> ExperimentResult:
    from hillclimb.software.rl_interface.env import GridWorldConfig
    from hillclimb.software.rl_interface.evaluator import QLearningConfig, RLInterfaceEvaluator
    from hillclimb.software.rl_interface.interface import baseline_interface
    from hillclimb.software.rl_interface.proposer import MutationMode, RLInterfaceProposer

    seed = 7
    mode_label = "llm" if llm else "baseline"
    fallback = RLInterfaceProposer(mode=MutationMode.JOINT, hard_mode=hard, seed=seed)
    proposer = (
        LLMEnhancedProposer(
            llm=LLMRLInterfaceProposer(llm, fallback, hard_mode=hard),
            fallback=fallback,
            name="rl_interface",
        )
        if llm
        else fallback
    )

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=RLInterfaceEvaluator(
            env_config=GridWorldConfig(hard_mode=hard, seed=seed),
            ql_config=QLearningConfig(seed=seed, episodes=80),
        ),
        max_rounds=max_rounds,
        early_stop_patience=4,
        accept_policy=AcceptPolicy.GREEDY,
    )
    result = climber.climb(Candidate(state=baseline_interface(hard_mode=hard)))
    initial, best = _score_from_history(result.history)

    return ExperimentResult(
        name="rl_interface",
        domain="software",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best >= 0.8,
        details={"hard_mode": hard},
        duration_s=time.time() - t0,
    )


def run_ocr_self_iterate(*, llm: LLMClient | None, max_rounds: int = 8) -> ExperimentResult:
    from hillclimb.software.ocr_self_iterate.parser import baseline_parser_config
    from hillclimb.software.ocr_self_iterate.refiner import OCRRefiner
    from hillclimb.software.ocr_self_iterate.run import OCRBatchEvaluator
    from hillclimb.software.ocr_self_iterate.synthetic import generate_invoice

    seed = 42
    batch = [(generate_invoice(seed=seed + i).image,
              generate_invoice(seed=seed + i).ground_truth,
              generate_invoice(seed=seed + i).rendered_text) for i in range(4)]

    fallback = OCRRefiner(seed=seed)
    proposer = (
        LLMEnhancedProposer(
            llm=LLMOCRRefiner(llm, fallback),
            fallback=fallback,
            name="ocr",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=OCRBatchEvaluator(batch, seed=seed),
        max_rounds=max_rounds,
        early_stop_patience=3,
        accept_policy=AcceptPolicy.GREEDY,
    )
    result = climber.climb(Candidate(state=baseline_parser_config()))
    initial, best = _score_from_history(result.history)

    return ExperimentResult(
        name="ocr_self_iterate",
        domain="software",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best >= 0.95,
        duration_s=time.time() - t0,
    )


def run_sift_coding(*, llm: LLMClient | None, task_name: str = "reverse_string") -> ExperimentResult:
    from hillclimb.software.sift_coding.run import SiftCodingEvaluator, SiftCodingProposer
    from hillclimb.software.sift_coding.task import get_task

    task = get_task(task_name)
    fallback = SiftCodingProposer(task=task)
    proposer = (
        LLMEnhancedProposer(
            llm=LLMSiftProposer(llm, task, fallback),
            fallback=fallback,
            name="sift",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=SiftCodingEvaluator(task=task),
        max_rounds=8,
        early_stop_patience=3,
        accept_policy=AcceptPolicy.GREEDY,
    )
    result = climber.climb(Candidate(state=task.starter_code, metadata={"source": "starter"}))
    initial_pass = result.history[0][1].diagnostics.get("pass_rate", 0.0)
    best_pass = max(ev.diagnostics.get("pass_rate", 0.0) for _, ev in result.history)

    return ExperimentResult(
        name="sift_coding",
        domain="software",
        mode=mode_label,
        initial_score=initial_pass,
        best_score=best_pass,
        improvement=best_pass - initial_pass,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best_pass >= 1.0,
        details={"task": task_name},
        duration_s=time.time() - t0,
    )


def run_config_discovery(*, llm: LLMClient | None, seed: int = 42) -> ExperimentResult:
    import random

    from hillclimb.software.config_discovery.evaluator import ConfigEvaluator
    from hillclimb.software.config_discovery.proposer import ConfigProposer
    from hillclimb.software.config_discovery.search_space import random_config

    fallback = ConfigProposer()
    proposer = (
        LLMEnhancedProposer(
            llm=LLMConfigProposer(llm, fallback),
            fallback=fallback,
            name="config",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=ConfigEvaluator(),
        max_rounds=12,
        early_stop_patience=4,
        accept_policy=AcceptPolicy.GREEDY,
    )
    result = climber.climb(Candidate(state=random_config(random.Random(seed))))
    initial, best = _score_from_history(result.history)

    return ExperimentResult(
        name="config_discovery",
        domain="software",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best >= initial,
        duration_s=time.time() - t0,
    )


def run_finance_research(*, llm: LLMClient | None, seed: int = 42) -> ExperimentResult:
    import random

    from hillclimb.science.finance_research.data import generate_synthetic_prices
    from hillclimb.science.finance_research.researcher import (
        FinanceEvaluator,
        FinanceProposer,
        FinanceResearcher,
    )

    market = generate_synthetic_prices(seed=seed)
    researcher = FinanceResearcher(market=market, rng=random.Random(seed))
    fallback = FinanceProposer(researcher)
    proposer = (
        LLMEnhancedProposer(
            llm=LLMFinanceProposer(llm, researcher, fallback),
            fallback=fallback,
            name="finance",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=FinanceEvaluator(researcher),
        max_rounds=10,
        early_stop_patience=4,
    )
    result = climber.climb(Candidate(state=researcher.propose_initial()))
    initial, best = _score_from_history(result.history)

    return ExperimentResult(
        name="finance_research",
        domain="science",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best > initial,
        duration_s=time.time() - t0,
    )


def run_lean_prover(*, llm: LLMClient | None, problem: str = "two_plus_two") -> ExperimentResult:
    import random

    from hillclimb.science.lean_prover.evaluator import ProofEvaluator
    from hillclimb.science.lean_prover.mock_lean import MockLeanVerifier
    from hillclimb.science.lean_prover.problems import get_problem
    from hillclimb.science.lean_prover.proposer import ProofCandidate, ProofProposer

    prob = get_problem(problem)
    verifier = MockLeanVerifier(prob)
    fallback = ProofProposer(prob, verifier, rng=random.Random(0))
    proposer = (
        LLMEnhancedProposer(
            llm=LLMProofProposer(llm, prob, verifier, fallback),
            fallback=fallback,
            name="lean",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=ProofEvaluator(prob, verifier),
        max_rounds=8,
        early_stop_patience=3,
    )
    result = climber.climb(Candidate(state=ProofCandidate()))
    initial, best = _score_from_history(result.history)
    solved = result.history[-1][1].diagnostics.get("solved", False)

    return ExperimentResult(
        name="lean_prover",
        domain="science",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=solved or best >= 1.0,
        details={"problem": problem, "solved": solved},
        duration_s=time.time() - t0,
    )


def run_hypothesis_tournament(*, llm: LLMClient | None) -> ExperimentResult:
    import random

    from hillclimb.science.hypothesis_tournament.agents import GenerationAgent
    from hillclimb.science.hypothesis_tournament.run import (
        TournamentPopulationEvaluator,
        TournamentProposer,
        TournamentState,
    )

    question = "What causes resistance to drug X?"
    rng = random.Random(42)
    initial_hyps = GenerationAgent(rng).generate(question, count=4)
    initial_state = TournamentState(research_question=question, hypotheses=initial_hyps)

    fallback = TournamentProposer(population_size=4, rng=rng)
    proposer = (
        LLMEnhancedProposer(
            llm=LLMHypothesisProposer(llm, fallback, population_size=4),
            fallback=fallback,
            name="hypothesis",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=TournamentPopulationEvaluator(question),
        max_rounds=6,
        early_stop_patience=3,
    )
    result = climber.climb(Candidate(state=initial_state))
    initial, best = _score_from_history(result.history)

    return ExperimentResult(
        name="hypothesis_tournament",
        domain="science",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best > initial,
        details={"question": question},
        duration_s=time.time() - t0,
    )


def run_portfolio_optimizer(*, llm: LLMClient | None, seed: int = 42) -> ExperimentResult:
    import numpy as np

    from hillclimb.science.portfolio_optimizer.allocator import AllocationParams
    from hillclimb.science.portfolio_optimizer.data import generate_synthetic_returns
    from hillclimb.science.portfolio_optimizer.evaluator import PortfolioEvaluator
    from hillclimb.science.portfolio_optimizer.proposer import PortfolioProposer
    from hillclimb.science.portfolio_optimizer.regime import detect_regimes

    data = generate_synthetic_returns(504, seed=seed)
    returns = data["returns"]
    true_labels = data["regime_labels"]
    detected = detect_regimes(returns)

    fallback = PortfolioProposer(rng=np.random.default_rng(seed))
    proposer = (
        LLMEnhancedProposer(
            llm=LLMPortfolioProposer(llm, fallback),
            fallback=fallback,
            name="portfolio",
        )
        if llm
        else fallback
    )
    mode_label = "llm" if llm else "baseline"

    t0 = time.time()
    climber = HillClimber(
        proposer=proposer,
        evaluator=PortfolioEvaluator(returns, true_labels, use_detected_regimes=True),
        max_rounds=10,
        early_stop_patience=3,
    )
    result = climber.climb(Candidate(state=AllocationParams()))
    initial, best = _score_from_history(result.history)

    return ExperimentResult(
        name="portfolio_optimizer",
        domain="science",
        mode=mode_label,
        initial_score=initial,
        best_score=best,
        improvement=best - initial,
        rounds=result.rounds,
        converged=result.converged,
        target_met=best >= initial,
        duration_s=time.time() - t0,
    )


EXPERIMENTS: list[tuple[str, Callable[..., ExperimentResult]]] = [
    ("rl_interface", run_rl_interface),
    ("ocr_self_iterate", run_ocr_self_iterate),
    ("sift_coding", run_sift_coding),
    ("config_discovery", run_config_discovery),
    ("finance_research", run_finance_research),
    ("lean_prover", run_lean_prover),
    ("hypothesis_tournament", run_hypothesis_tournament),
    ("portfolio_optimizer", run_portfolio_optimizer),
]


def run_all(*, use_llm: bool = True, model: str = "gpt-4o-mini") -> dict[str, Any]:
    llm = get_llm(model) if use_llm else None
    if use_llm and llm is None:
        raise RuntimeError("OPENAI_API_KEY required for LLM experiments")

    results: list[ExperimentResult] = []
    for name, fn in EXPERIMENTS:
        print(f"\n{'='*60}\n  {name} ({'LLM' if llm else 'baseline'})\n{'='*60}")
        for mode_label, client in [("baseline", None), ("llm", llm)] if llm else [("baseline", None)]:
            if mode_label == "llm" and client is None:
                continue
            r = fn(llm=client)
            results.append(r)
            status = "PASS" if r.target_met else "FAIL"
            print(
                f"  [{status}] {r.mode}: {r.initial_score:.4f} → {r.best_score:.4f} "
                f"(Δ{r.improvement:+.4f}, {r.rounds} rounds, {r.duration_s:.1f}s)"
            )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model if llm else None,
        "experiments": [asdict(r) for r in results],
        "all_targets_met": all(r.target_met for r in results),
        "baseline_pass": sum(1 for r in results if r.mode == "baseline" and r.target_met),
        "llm_pass": sum(1 for r in results if r.mode == "llm" and r.target_met),
    }
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run hill-climbing LLM benchmark suite")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    summary = run_all(use_llm=not args.baseline_only, model=args.model)

    out = Path(args.output)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*60}")
    print(f"Results written to {out}")
    print(f"All targets met: {summary['all_targets_met']}")
    print(f"Baseline pass: {summary['baseline_pass']}/8")
    if summary.get("model"):
        print(f"LLM pass: {summary['llm_pass']}/8")


if __name__ == "__main__":
    main()
