"""iTRADER online stack: the spec's live loop (section 10.2), built as new files only.

    data -> MarketFramer -> CandidateTasks (+ WAIT) -> DecisionGate -> Portfolio -> RiskEngine
         -> Executor (paper / live) -> DecisionRecord -> CapabilityProfile

Everything here is deterministic and inspectable. Learned components (the repo's solvers, the
LLM proposers) can propose and explain; they can never place an order or move a risk limit.

Entry points:  python -m itrader.online --help
"""

__version__ = "0.1.0"
