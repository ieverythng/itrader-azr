from __future__ import annotations

from typing import Any

from .base import BaseProposer
from .rule_based import RuleBasedProposer
from .ollama import OllamaProposer



def build_proposer(backend: str = "rule_based", **kwargs: Any) -> BaseProposer:
    key = str(backend).strip().lower().replace("-", "_")
    if key in {"rule", "rule_based", "random"}:
        return RuleBasedProposer(**kwargs)
    if key in {"ollama", "ollama_local"}:
        return OllamaProposer(**kwargs)
    if key in {"llm", "transformers"}:
        # Lazy import to avoid hard dependency on transformers for local/dev usage.
        from itrader.llm.proposer_llm import ProposerLLM

        return ProposerLLM(**kwargs)
    raise ValueError(f"Unknown proposer backend: {backend}")
