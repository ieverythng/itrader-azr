from __future__ import annotations

from typing import Any

from .base import BaseProposer
from .ollama import OllamaProposer
from .openai_compat import OpenAICompatProposer
from .rule_based import RuleBasedProposer


_PROPOSER_FACTORIES = {
    "rule": RuleBasedProposer,
    "rule_based": RuleBasedProposer,
    "random": RuleBasedProposer,
    "ollama": OllamaProposer,
    "ollama_local": OllamaProposer,
    "openai": OpenAICompatProposer,
    "openai_compat": OpenAICompatProposer,
    "openai_compatible": OpenAICompatProposer,
    "llama_cpp": OpenAICompatProposer,
    "llamacpp": OpenAICompatProposer,
    "vllm": OpenAICompatProposer,
    "sglang": OpenAICompatProposer,
}


def build_proposer(backend: str = "rule_based", **kwargs: Any) -> BaseProposer:
    key = str(backend).strip().lower().replace("-", "_")
    factory = _PROPOSER_FACTORIES.get(key)
    if factory is not None:
        return factory(**kwargs)
    if key in {"llm", "transformers"}:
        # Lazy import to avoid hard dependency on transformers for local/dev usage.
        from itrader.llm.proposer_llm import ProposerLLM

        return ProposerLLM(**kwargs)
    raise ValueError(f"Unknown proposer backend: {backend}")
