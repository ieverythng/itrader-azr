"""Settings: one TOML file, loaded once, read everywhere.

`load_settings()` reads the packaged `settings.toml`, then overlays the file named by
ITRADER_ONLINE_SETTINGS (or an explicit path) key by key. Every decision the stack makes
logs the settings snapshot hash, so a DecisionRecord can always be traced to the thresholds
that produced it. Secrets are never in settings; they are env var *names* that the code reads.
"""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "settings.toml"
ENV_VAR = "ITRADER_ONLINE_SETTINGS"


class Section:
    """Attribute access over a dict, recursively. Missing keys raise a clear error."""

    def __init__(self, data: Dict[str, Any], path: str = "") -> None:
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "_path", path)

    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"settings: no key '{object.__getattribute__(self, '_path')}.{key}'".lstrip("."))
        v = data[key]
        return Section(v, f"{self._path}.{key}".strip(".")) if isinstance(v, dict) else v

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("settings are read only; edit settings.toml")

    def get(self, key: str, default: Any = None) -> Any:
        v = self._data.get(key, default)
        return Section(v, f"{self._path}.{key}".strip(".")) if isinstance(v, dict) else v

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def __repr__(self) -> str:
        return f"Section({self._path or 'root'}: {sorted(self._data)})"


class Settings(Section):
    def __init__(self, data: Dict[str, Any], sources: list) -> None:
        super().__init__(data, "")
        object.__setattr__(self, "sources", list(sources))

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self._data, sort_keys=True).encode()).hexdigest()[:12]

    def path(self, key_path: str, default: Any = None) -> Any:
        cur: Any = self._data
        for k in key_path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def resolve(self, rel: str, root: Optional[Path] = None) -> Path:
        """Paths in settings are relative to the repo root unless absolute."""
        p = Path(rel).expanduser()
        if p.is_absolute():
            return p
        return (root or repo_root()) / p


def repo_root() -> Path:
    # itrader/online/settings.py -> repo root is three levels up
    return HERE.parent.parent


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Settings:
    with DEFAULT_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    sources = [str(DEFAULT_PATH)]
    user = path or os.environ.get(ENV_VAR)
    if user:
        up = Path(user).expanduser()
        if not up.exists():
            raise FileNotFoundError(f"settings file not found: {up}")
        with up.open("rb") as fh:
            data = _deep_merge(data, tomllib.load(fh))
        sources.append(str(up))
    if overrides:
        data = _deep_merge(data, overrides)
        sources.append("<overrides>")
    return Settings(data, sources)


_CACHE: Optional[Settings] = None


def settings() -> Settings:
    """Process-wide default settings (loaded on first use)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_settings()
    return _CACHE
