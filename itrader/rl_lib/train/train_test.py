"""
Deprecated training file kept as a compatibility shim.

Use the canonical entrypoint instead:
    python -m itrader.rl_lib.train.train_solo_V2 ...
"""

from __future__ import annotations

import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    sys.argv = ["python -m itrader.rl_lib.train.train_solo_V2", *(argv or [])]
    runpy.run_module("itrader.rl_lib.train.train_solo_V2", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
