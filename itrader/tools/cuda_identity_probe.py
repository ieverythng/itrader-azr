"""Emit standard-library interpreter identity for the CUDA test launcher."""

from __future__ import annotations

import json
import platform
import struct
import sys
from pathlib import Path


def main() -> int:
    print(
        json.dumps(
            {
                "implementation": platform.python_implementation(),
                "machine": platform.machine(),
                "pointer_bits": struct.calcsize("P") * 8,
                "version": [sys.version_info.major, sys.version_info.minor],
                "executable": str(Path(sys.executable).resolve()),
                "prefix": str(Path(sys.prefix).resolve()),
                "base_prefix": str(Path(sys.base_prefix).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
