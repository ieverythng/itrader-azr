# Research

Start with the [iTrader Research Dashboard](itrader_research_dashboard.md)
([HTML](itrader_research_dashboard.html)). It is the canonical registry of
research wings, controls, prerequisites, evidence status, and open decisions.
It catalogs research and ablations only; core phase implementation remains in
the masterplan and is linked only where an experiment depends on it.

Use the generated [Experiment Browser](itrader_experiment_browser.html) to
search concrete runs, members, retained checkpoints, and comparison factors.
Its [registry](experiment_registry.json) contains provenance and factor
metadata; metrics are loaded from the linked authoritative summaries during
generation rather than copied into another hand-maintained ledger. Direct
comparability also requires authoritative dataset, panel, and metric-contract
identities; missing provenance stays unknown rather than being inferred from a
matching display label.
The contract is fail-closed: undeclared factors or metric units, duplicate JSON
keys, non-finite values, malformed action fractions, incomplete records, and
artifact/hash disagreement stop generation. Historical paired replay artifacts
without producer IDs are shown as artifact parity, not independent-process
proof.

The dashboard does not replace the
[canonical masterplan](../plans/itrader_masterplan.md), which owns cross-phase
order and promotion gates, or the active architecture documents, which own
system boundaries.

## Focused design notes

- [Recurrent Solver + FiLM Ablation](recurrent-solver-ablation.md)
  ([HTML](recurrent-solver-ablation.html)) — R-series identity, placement, and
  lifecycle design provenance.
- [Sharpe-Aware Reward Design](sharpe-reward-design.md)
  ([HTML](sharpe-reward-design.html)) — independent reward-axis proposal and
  promotion gates.

## Evidence and provenance

Dated investigations, experiment receipts, evidence directories, and imported
reference material live under the
[research artifact catalog](../artifacts/research/README.md). Those files support
claims; they do not become current architecture merely by existing.
