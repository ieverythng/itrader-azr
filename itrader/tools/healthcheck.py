#!/usr/bin/env python3
"""
Lightweight healthcheck for local/dev CI usage.

This does not run training. It only validates imports, basic runtime info,
and exits non-zero if critical dependencies are missing.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys


def _import_version(pkg: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(pkg)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    required = ["numpy", "pandas", "torch", "ray", "gymnasium", "duckdb"]
    optional = ["llama_cpp", "transformers"]
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "required": {},
        "optional": {},
    }

    ok = True
    for pkg in required:
        is_ok, v = _import_version(pkg)
        report["required"][pkg] = {"ok": is_ok, "version_or_error": v}
        ok = ok and is_ok

    for pkg in optional:
        is_ok, v = _import_version(pkg)
        report["optional"][pkg] = {"ok": is_ok, "version_or_error": v}

    try:
        import torch

        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
