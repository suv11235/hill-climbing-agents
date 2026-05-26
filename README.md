# Hill Climbing AI Agents

Experimental harnesses for **hill climbing** as the central optimization mechanism for AI agents in **scientific discovery** and **software development**.

Based on recent research on the topic until May, 2026 (see [docs/RESEARCH.md](docs/RESEARCH.md)). Key finding: **greedy hill climbing with early stopping is the strong default** when LLMs serve as proposal generators (Li et al., arXiv:2603.27415).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Proposer   │────▶│  Evaluator   │────▶│   Acceptor  │
│  (LLM/rule) │     │  (metric)    │     │  (greedy/SA)│
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                                         │
       └─────────── diagnostics ─────────────────┘
```

All prototypes implement this loop via `HillClimber` in `hillclimb/core/harness.py`.

## Prototypes

### Software Development (4)

| # | Prototype | Inspired By | Run Command |
|---|-----------|-------------|-------------|
| 1 | **RL Interface Discovery** | LIMEN (arXiv:2605.03408) | `python -m hillclimb.software.rl_interface.run --compare --hard` |
| 2 | **OCR Self-Iterate** | MinerU Judge-and-Refine, OCR-Agent | `python -m hillclimb.software.ocr_self_iterate.run --rounds 10` |
| 2b | **OCR Workflow** (orchestrator + sub-agents) | Agentic workflow hill climb | `python -m hillclimb.software.ocr_self_iterate.workflow.run --rounds 10` |
| 2c | **OCR Offline Train** (feedback → stable workflow) | Backward propagation + registry | `python -m hillclimb.software.ocr_self_iterate.workflow.offline_train_run --epochs 6` |
| 3 | **SIFT Coding Self-Improvement** | SIFT, SICA | `python -m hillclimb.software.sift_coding.run --task reverse_string` |
| 4 | **Config Discovery** | CliffSearch, Li et al. | `python -m hillclimb.software.config_discovery.run` |

### Science Discovery (4)

| # | Prototype | Inspired By | Run Command |
|---|-----------|-------------|-------------|
| 5 | **Financial Strategy Deep Research** | Co-Scientist, TiMi, Self-Driving Portfolio | `python -m hillclimb.science.finance_research.run` |
| 6 | **Lean Theorem Proving** | AlphaProof Nexus | `python -m hillclimb.science.lean_prover.run` |
| 7 | **Hypothesis Tournament** | Co-Scientist tournament evolution | `python -m hillclimb.science.hypothesis_tournament.run` |
| 8 | **Regime-Aware Portfolio Optimizer** | Regime-aware agentic portfolio (Springer 2026) | `python -m hillclimb.science.portfolio_optimizer.run` |

## Quick Start

```bash
pip install -e ".[dev,llm]"
pytest tests/ -v

# Run all demos
hillclimb demo --all

# Run baseline + OpenAI LLM benchmark (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
hillclimb benchmark --output benchmark_results.json

# Run a specific prototype
hillclimb run rl_interface -- --compare --hard
hillclimb run ocr_self_iterate -- --rounds 10
hillclimb run lean_prover
```

## LLM Benchmark Results (gpt-4o-mini, verified May 2026)

All 8 prototypes pass targets in both **baseline** (rule-based proposer) and **LLM-enhanced** (OpenAI + fallback) modes:

| Prototype | Baseline Δ | LLM Δ | Highlight |
|-----------|-----------|-------|-----------|
| RL Interface (hard) | +0.58 → 100% | +0.58 → 100% | Joint obs+reward discovery |
| OCR Self-Iterate | +0.20 → 100% | +0.20 → 100% | Regex fix via judge diagnostics |
| SIFT Coding | +0.50 → 100% | +0.50 → 100% | Test-driven patch hill climb |
| Config Discovery | +0.007 CV acc | maintained | RandomForest hyperparams |
| Finance Research | +0.40 Sharpe | **+1.01 Sharpe** | LLM outperforms on strategy refinement |
| Lean Prover | +0.90 → QED | +0.90 → QED | Tactic search with mock Lean |
| Hypothesis Tournament | +0.26 score | +0.12 score | Elo evolution on drug resistance |
| Portfolio Optimizer | +0.20 Sharpe | +0.20 Sharpe | Regime-aware allocation |

Run `hillclimb benchmark` to reproduce. LLM proposers use hybrid fallback — if the model returns an unchanged state, rule-based proposers take over (guaranteeing baseline progress).

## Key Results (from demos)

- **RL Interface**: Joint observation+reward evolution reaches ~100% success vs ~42% reward-only on hard gridworld (validates LIMEN finding)
- **OCR Self-Iterate**: Climbs from 80% → 100% field accuracy by fixing regex patterns via judge diagnostics
- **SIFT Coding**: Buggy starter code reaches 100% test pass rate in 4 greedy rounds
- **Config Discovery**: RandomForest CV accuracy improves via diagnostic-aware hyperparameter mutations
- **Lean Prover**: Finds complete proofs for `2+2=4` and `0+n=n` via tactic hill climbing
- **Hypothesis Tournament**: Elo-ranked evolution improves composite hypothesis scores over generations
- **Portfolio Optimizer**: Regime-aware weight hill climbing improves net Sharpe minus turnover penalty

## Project Structure

```
hillclimb/
├── core/           # Universal harness (HillClimber, types, accept policies)
├── software/       # Software development prototypes
│   ├── rl_interface/
│   ├── ocr_self_iterate/
│   ├── sift_coding/
│   └── config_discovery/
└── science/        # Science discovery prototypes
    ├── finance_research/
    ├── lean_prover/
    ├── hypothesis_tournament/
    └── portfolio_optimizer/
docs/
└── RESEARCH.md     # Literature synthesis (Nov 2025 – May 2026)
tests/              # 59 tests across all prototypes
```

## Extending

To add a new prototype:

1. Create `proposer.py` and `evaluator.py` implementing the protocols in `hillclimb/core/harness.py`
2. Wire them into `HillClimber(proposer=..., evaluator=..., accept_policy=AcceptPolicy.GREEDY)`
3. Add a `run.py` CLI demo and tests

See existing prototypes for patterns. Greedy acceptance + 3-round early stopping is the recommended default.

## References

See [docs/RESEARCH.md](docs/RESEARCH.md) for full bibliography including Li et al. (2026), LIMEN, Co-Scientist (Nature 2026), AlphaProof Nexus, CliffSearch, and MinerU2.5-Pro.
