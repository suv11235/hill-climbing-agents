from __future__ import annotations

import random

from hillclimb.science.hypothesis_tournament.agents import (
    EvolutionAgent,
    GenerationAgent,
    RankingAgent,
    ReflectionAgent,
)
from hillclimb.science.hypothesis_tournament.evaluator import (
    composite_score,
    score_hypothesis,
)
from hillclimb.science.hypothesis_tournament.hypothesis import Hypothesis
from hillclimb.science.hypothesis_tournament.run import run_tournament_evolution
from hillclimb.science.hypothesis_tournament.tournament import (
    run_pairwise_tournament,
    update_elo,
)


def test_hypothesis_dataclass_id():
    hyp = Hypothesis(
        claim="Drug X resistance is efflux-driven",
        mechanism="ABC transporter upregulation",
        testable_prediction="Knockout restores sensitivity",
    )
    assert hyp.id.startswith("h0_")
    assert "efflux" in hyp.claim.lower() or "Drug" in hyp.claim


def test_generation_agent_produces_population():
    agent = GenerationAgent()
    hyps = agent.generate("What causes resistance to drug X?", count=4)
    assert len(hyps) == 4
    for h in hyps:
        assert h.claim and h.mechanism and h.testable_prediction


def test_reflection_agent_refines_hypothesis():
    parent = Hypothesis(
        claim="Resistance emerges",
        mechanism="pump activity",
        testable_prediction="Compare IC50 in resistant lines",
    )
    child = ReflectionAgent().reflect(parent, "What causes resistance to drug X?")
    assert child.parent_id == parent.id
    assert child.generation == parent.generation + 1
    assert len(child.mechanism) >= len(parent.mechanism)


def test_evaluator_scores_in_unit_interval():
    hyp = Hypothesis(
        claim="Drug X resistance via ABCB1 efflux mutation",
        mechanism="Efflux pump overexpression reduces intracellular drug via transporter activity",
        testable_prediction="Measure IC50 shift after transporter knockout in resistant cohort",
    )
    scores = score_hypothesis(hyp, research_question="What causes resistance to drug X?")
    assert 0.0 <= scores["novelty"] <= 1.0
    assert 0.0 <= scores["testability"] <= 1.0
    assert 0.0 <= scores["evidence_alignment"] <= 1.0
    assert composite_score(scores) > 0.0


def test_tournament_elo_ranking_orders_by_quality():
    strong = Hypothesis(
        claim="Drug X resistance via ABCB1 efflux and target mutation",
        mechanism="Dual efflux and target-site binding loss through ABC transporter pathway",
        testable_prediction="Measure IC50 shift after knockout in randomized resistant cohort assay",
    )
    weak = Hypothesis(
        claim="Maybe something",
        mechanism="unknown",
        testable_prediction="look at cells",
    )
    ranked = run_pairwise_tournament(
        [weak, strong],
        research_question="What causes resistance to drug X?",
        rounds=10,
        rng=random.Random(7),
    )
    assert ranked[0].id == strong.id
    assert ranked[0].elo > ranked[1].elo


def test_elo_update_increases_winner_rating():
    a = Hypothesis(claim="a", mechanism="m1", testable_prediction="p1", elo=1500.0)
    b = Hypothesis(claim="b", mechanism="m2", testable_prediction="p2", elo=1500.0)
    update_elo(a, b)
    assert a.elo > 1500.0
    assert b.elo < 1500.0


def test_evolution_agent_builds_next_generation():
    gen = GenerationAgent()
    initial = gen.generate("What causes resistance to drug X?", count=4)
    ranked = RankingAgent().rank(initial, research_question="What causes resistance to drug X?")
    next_gen = EvolutionAgent().evolve(ranked, "What causes resistance to drug X?", population_size=4)
    assert len(next_gen) == 4
    assert any(h.generation >= 1 for h in next_gen)


def test_run_tournament_evolution_completes():
    outcome = run_tournament_evolution(
        "What causes resistance to drug X?",
        max_rounds=5,
        seed=0,
    )
    assert outcome["best_score"] > 0.0
    assert outcome["rounds"] >= 1
    assert outcome["best_hypothesis"] is not None
    assert len(outcome["final_population"]) >= 1
