# Hill Climbing with AI Agents: Research Synthesis (Nov 2025 – May 2026)

This document synthesizes recent work on using hill climbing and related iterative optimization as the central mechanism for AI agent improvement in **scientific discovery** and **software development**.

## Core Finding: Greedy Hill Climbing Is a Strong Default

The most actionable recent result comes from **"Greedy Is a Strong Default: Agents as Iterative Optimizers"** (Li et al., arXiv:2603.27415, Mar 2026). When an LLM replaces random perturbation as the proposal generator, classical optimization machinery (simulated annealing, parallel investigators, multi-model proposals) provides **no meaningful benefit** over greedy hill climbing with early stopping—while requiring 2–3× more evaluations.

**Practical recipe:**
1. LLM proposes candidate based on evaluation diagnostics
2. Evaluate candidate against objective metric
3. Accept if improved (greedy); stop after 2–3 rounds of no improvement
4. Round 1 alone delivers the majority of gains

This framework produced interpretable artifacts across rule discovery, hyperparameter tuning, LoRA fine-tuning, and XGBoost optimization.

## Software Development Domain

### Self-Improving Coding Agents

| System | Mechanism | Key Insight |
|--------|-----------|-------------|
| **Darwin Gödel Machine** (Zhang et al., arXiv:2505.22954) | Archive-based open-ended evolution; parent selection by performance | Stepping stones matter—archive beats pure hill climbing |
| **HyperAgents** (arXiv:2603.19461) | Self-referential meta-level evolution | The improvement mechanism itself can evolve |
| **Polaris** (arXiv:2603.23129) | Experience-abstracted policy repair | Conservative, auditable patches for small models |
| **SIFT** (OpenReview wZMNXHPYcO) | LLM-as-judge before expensive benchmark eval | Sample-efficient greedy tree search |
| **SICA** (arXiv:2504.15228) | Agent edits own codebase | Self-referential meta-programming is viable today |

### RL for Bespoke Environments

**LIMEN** (Learning Interfaces via MDP-guided EvolutioN, arXiv:2605.03408, May 2026) is the standout result:
- Jointly evolves **observation mappings** and **reward functions** as executable Python programs
- Uses PPO training as fitness evaluator; MAP-Elites for diversity (not plain hill climbing)
- Reward-only collapses on hard gridworld (7–19%); observation-only fails on Panda (0%)
- **Joint evolution is the only configuration that avoids catastrophic failure across all domains**

**RLVR** (Reinforcement Learning with Verifiable Rewards) became dominant in 2025–2026 for coding agents: test suites provide unambiguous reward signals. Cursor's targeted textual feedback and RULER (OpenPipe ART) extend this to non-verifiable tasks via LLM-as-judge relative ranking.

### OCR Self-Iteration

| System | Mechanism | Result |
|--------|-----------|--------|
| **MinerU2.5-Pro** (arXiv:2604.04771) | Render-then-verify Judge-and-Refine loop | 95.69 on OmniDocBench v1.6 |
| **OCR-Agent** (arXiv:2602.21053) | Capability + Memory Reflection | +2.0 on OCRBench v2 English |
| **MOCR** (arXiv:2603.13032) | Self-improving data curation via render verification | Judge-based evaluation loops |

Common pattern: **propose → render/verify → diagnose errors → refine → repeat** until convergence or budget exhausted.

### Scientific Algorithm Discovery

**CliffSearch** (arXiv:2604.01210, Apr 2026): evolutionary framework where nodes are scientific artifacts (theory+code), with correctness/originality as selection gates alongside benchmark metrics. Separates exploration mutation from correction mutation.

**OR-Agent** (arXiv:2602.13769): tree-structured research workflow combining evolutionary search with deep local investigation and memory-based reflection for operations research.

## Science Discovery Domain

### Multi-Agent Research Systems

| System | Mechanism | Domain |
|--------|-----------|--------|
| **Co-Scientist** (Nature, May 2026) | Tournament evolution + debate + reflection agents | Biomedical hypothesis generation |
| **AAR** (Anthropic, Jan 2026) | Parallel sandbox hill-climbing with shared findings | Alignment research automation |
| **CliffSearch** | Evolutionary artifact discovery with correctness gates | ML algorithm discovery |

Co-Scientist's key insight: **scaling test-time compute** via tournament ranking and evolution agents continuously improves hypothesis quality. The majority of computation goes to verification, not generation.

### Financial Strategy Research

| System | Mechanism | Result |
|--------|-----------|--------|
| **Regime-Aware Agentic Portfolio** (Springer, Feb 2026) | LLM signals + regime inference + constrained RL controller | Sharpe +0.373 net of costs |
| **Self-Driving Portfolio** (arXiv:2604.02279) | ~50 agents with peer review, Borda voting, meta-agent code rewrite | Novel PC methods discovered |
| **TiMi** (arXiv:2510.04787) | Decouple strategy dev from deployment; reflection agents | Closed-loop optimization |
| **MASS** (OpenReview NNpE9iiPNR) | Multi-agent scaling simulation with backward optimization | Scaling effect: more agents → higher excess returns |

Pattern: **sense → infer regime → propose strategy → backtest → accept if Sharpe improves → iterate**.

### Lean Theorem Proving

| System | Mechanism | Result |
|--------|-----------|--------|
| **AlphaProof Nexus** (DeepMind, May 2026) | LLM + Lean verify loop + evolutionary proof sketch population | 9/353 Erdős problems solved |
| **nanoproof** (GitHub) | MCTS + RL on LeanTree transitions | 38.5% MiniF2F |
| **Hierarchical RL for Lean** (OpenReview SIyJ5JizgA) | Lemma-based decomposition + online RL | Overcomes whole-proof generation plateaus |

Core loop: **propose proof sketch → Lean compiler verifies → accept/reject → hill climb on verified partial proofs**. The compiler is the ground-truth evaluator—no hallucination possible.

## Design Patterns for Our Prototypes

### Universal Hill-Climbing Harness

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Proposer   │────▶│  Evaluator   │────▶│   Acceptor  │
│  (LLM/rule) │     │  (metric)    │     │  (greedy/SA)│
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                                         │
       └─────────── diagnostics ─────────────────┘
```

### Acceptance Policies (from literature)

1. **Greedy** (default): accept only if score improves
2. **Simulated Annealing**: occasionally accept worse moves (limited benefit with LLM proposers)
3. **Archive/MAP-Elites**: maintain diverse population (LIMEN, DGM)
4. **Tournament**: pairwise comparison ranking (Co-Scientist)

### Evaluation Discipline

- **Verifiable domains** (Lean, tests, backtests): use compiler/execution as ground truth
- **Non-verifiable domains** (OCR quality, research hypotheses): use LLM-as-judge with render-verify loops
- **Anti-reward-hacking**: layered verification, holdout sets, correctness gates (AAR, CliffSearch)

## References

1. Li et al. "Greedy Is a Strong Default: Agents as Iterative Optimizers." arXiv:2603.27415, 2026.
2. LIMEN authors. "Discovering RL Interfaces with LLMs." arXiv:2605.03408, 2026.
3. Gottweis et al. "Accelerating scientific discovery with Co-Scientist." Nature, 2026.
4. Anthropic. "Automated Weak-to-Strong Researcher." alignment.anthropic.com, 2026.
5. DeepMind. "AlphaProof Nexus." arXiv:2605.22763, 2026.
6. CliffSearch authors. arXiv:2604.01210, 2026.
7. MinerU team. "MinerU2.5-Pro." arXiv:2604.04771, 2026.
8. Zhang et al. "Darwin Gödel Machine." arXiv:2505.22954, 2025.
