from __future__ import annotations

import json
import re
from typing import Any, Dict, List


_JSON_PATTERNS = (r"\[[\s\S]*\]", r"\{[\s\S]*\}")
_TASK_SCHEMA_HINT = (
    "Return ONLY valid JSON as an array of task objects. "
    "Use canonical task-contract fields when possible: "
    "task_family|family_id, bar_resolution, horizon_bars|horizon, "
    "reward_config or reward_id+reward_params, seed, obs_dim, action_dim, start_idx. "
    "Optional forward-compatible fields include regime_tags, difficulty_target, and metadata."
)


def schema_hint() -> str:
    return _TASK_SCHEMA_HINT


def normalize_count(n: int) -> int:
    return max(0, int(n))


def _task_dicts(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def parse_json_with_repair(text: str, expected_n: int) -> List[Dict[str, Any]]:
    count = normalize_count(expected_n)
    try:
        return _task_dicts(json.loads(text))[:count]
    except Exception:
        pass

    tasks: List[Dict[str, Any]] = []
    for pattern in _JSON_PATTERNS:
        for match in re.findall(pattern, text):
            try:
                tasks.extend(_task_dicts(json.loads(match)))
            except Exception:
                continue
            if len(tasks) >= count:
                return tasks[:count]
    return tasks[:count]
