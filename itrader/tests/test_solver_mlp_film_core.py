from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pytorch_solver_core_import_does_not_import_ray():
    repository_root = Path(__file__).resolve().parents[2]
    script = r"""
import builtins
import sys

real_import = builtins.__import__

def reject_ray_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "ray" or name.startswith("ray."):
        raise AssertionError(f"solver core attempted Ray import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_ray_import
import itrader.models.solver_mlp_film_core
assert not any(name == "ray" or name.startswith("ray.") for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
