from __future__ import annotations

import json
import re
from typing import Any, Dict, List
from urllib import error, request

from .base import BaseProposer


class OpenAICompatProposer(BaseProposer):
    """OpenAI-compatible proposer for vLLM/SGLang style chat endpoints."""

    backend_name = "openai_compat"

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3.5-9B",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key) if api_key else None
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def _schema_hint(self) -> str:
        return (
            "Return ONLY valid JSON as an array of task objects with fields: "
            "family_id(str), price_id(str), horizon(int 64-512), reward_id(str), "
            "reward_params(dict), seed(int), obs_dim(int), action_dim(int), start_idx(null|int)."
        )

    def _messages(self, market_summary: str, n: int) -> List[Dict[str, str]]:
        system = (
            "You are a trading task generator. "
            "Generate diverse, valid tasks and output strict JSON only."
        )
        user = (
            f"Generate {int(n)} task specs.\n\n"
            f"Market summary:\n{market_summary}\n\n"
            f"{self._schema_hint()}\n"
            "Output: JSON array only; no markdown; no prose."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _generate_text(self, market_summary: str, n: int) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(market_summary=market_summary, n=n),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible HTTP {exc.code} from {self.base_url}/chat/completions: {details[:400]}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible connection error at {self.base_url}: {exc}") from exc

        parsed = json.loads(body)
        choices = parsed.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return str(msg.get("content", "")).strip()

    def _parse_json_with_repair(self, text: str, expected_n: int) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                tasks.extend([x for x in parsed if isinstance(x, dict)])
            elif isinstance(parsed, dict):
                tasks.append(parsed)
            return tasks[:expected_n]
        except Exception:
            pass

        patterns = [
            r"\[[\s\S]*\]",
            r"\{[\s\S]*\}",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        tasks.extend([x for x in parsed if isinstance(x, dict)])
                    elif isinstance(parsed, dict):
                        tasks.append(parsed)
                except Exception:
                    continue
                if len(tasks) >= expected_n:
                    break
            if len(tasks) >= expected_n:
                break
        return tasks[:expected_n]

    def propose_tasks(self, market_summary: str, n: int, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        count = max(0, int(n))
        if count == 0:
            return []
        raw = self._generate_text(market_summary=market_summary, n=count)
        return self._parse_json_with_repair(raw, expected_n=count)
