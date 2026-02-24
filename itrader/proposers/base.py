from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseProposer(ABC):
    """Backend-agnostic task proposer contract for Phase B integration."""

    backend_name = "base"

    @abstractmethod
    def propose_tasks(self, market_summary: str, n: int, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return up to `n` raw task dict candidates."""
        raise NotImplementedError
