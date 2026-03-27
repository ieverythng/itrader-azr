from __future__ import annotations

import json
from typing import Any, Dict, List
from urllib import error, request

from ._shared import normalize_count, parse_json_with_repair, schema_hint
from .base import BaseProposer


class OllamaProposer(BaseProposer):
    """Ollama-backed proposer for local/cloud chat models."""

    backend_name = "ollama"

    def __init__(
        self,
        *,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        num_predict: int = 768,
    ):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)

    def _schema_hint(self) -> str:
        return schema_hint()

    def _build_prompt(self, market_summary: str, n: int) -> str:
        return (
            f"You are a trading task generator. Generate {int(n)} diverse task specs.\n\n"
            f"Market summary:\n{market_summary}\n\n"
            f"{self._schema_hint()}\n"
            "Output: JSON array only; no markdown; no prose."
        )

    def _generate_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        req = request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama HTTP {exc.code} from {self.base_url}/api/generate: {details[:400]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"Ollama connection error at {self.base_url}: {exc}") from exc
        parsed = json.loads(body)
        return str(parsed.get("response", "")).strip()

    def _parse_json_with_repair(self, text: str, expected_n: int) -> List[Dict[str, Any]]:
        return parse_json_with_repair(text, expected_n)

    def propose_tasks(self, market_summary: str, n: int, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        count = normalize_count(n)
        if count == 0:
            return []
        prompt = self._build_prompt(market_summary=market_summary, n=count)
        raw = self._generate_text(prompt)
        return self._parse_json_with_repair(raw, expected_n=count)
