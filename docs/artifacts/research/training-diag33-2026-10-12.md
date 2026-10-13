# TRAINING-DIAG33 diagnostics receipt - 2026-10-12

Scope: diagnose the missing leaf error from the closed CUDA32 R4 attempt, without rerunning pytest or CUDA. Started at 23:00:05Z and closed at 23:03:44Z. Branch/HEAD remained `agent/Watson-dev` / `8f9fcf94e028b51d883a1a484c2c9466a3d6b547`. No peer or upward message was sent.

The earlier R4 wrapper returned exit 1 in 1.203 seconds with no JUnit or result directories. Its `run_case.py` passed `stdout=subprocess.DEVNULL` and `stderr=subprocess.DEVNULL` to `subprocess.run`, so the failure text was discarded. The exact failing leaf and whether it reached a CUDA operation remain unknown.

In the isolated `_local/training-diag33/` copy, only that stream routing changed: leaf stdout and stderr inherit into the existing supervisor-owned log. Original `training-cuda32/run_case.py` SHA256: `73ad9d9ec810f73391f7e131bf0c2aa01c24b6496f5e3af52477988241005f74`; diagnostic copy SHA256: `cd257f23f1c57f6d2cbb49d26f25891e52c5a9aa7d25b26243d549e86b8661be`. Copied `supervise.py` and `native_scope.py` are byte-identical to CUDA32 (`35f138604a2bb95c0f1fd1915761d3e7c99fa254c850abd6baa588e8f52d1195`, `19adeb4662eab808ce68b8b63e3064f951ecc1533abb0592f2e12be1df710089`). The CUDA32 reviewed packet remains `b5cd901d8bfe1114314380a56d2e5f19808b92a79305dca7c5514870f2a56c8c`.

The transport check ran a standard-library child that wrote `diag33-stdout-marker` and `diag33-stderr-marker`, then exited 17. Both markers appear in `transport/supervisor-receipt.log`; `transport/runner-inventory.json` retains `child_exit_code: 17`; the supervisor receipt retains exit 17. Job cleanup reports active 0, errors `[]`, survivors `[]`.

Two forwarded CLI probes used the original public `itrader/tools/invoke_cuda_tested.ps1` launcher and standard-library modules only:

- `-B -m platform` exited 0 and printed `Windows-10-10.0.19045-SP0`.
- `-B -m site` exited 0 and showed `%LOCALAPPDATA%\iTrader\envs\cuda-tested-cu128` and its `Lib\site-packages` path.

These results show that the public launcher accepts the dash-argument shape, validates the isolated interpreter, and runs a standard-library module. They do not diagnose the pytest-specific command. No pytest collection, Torch/Ray import, CUDA query/compute, or model run occurred. Each leaf completed in under 0.6 seconds; total supervisor-owned time was 3.483 seconds. RAM minima were 8.582 GiB and disk remained at least 46.425 GiB; pre/post native PowerShell inventories passed with no eligible Python/Ray processes or known leftovers. Scope caps were child 60 seconds and total 180 seconds.

Minimal next diagnostic fix: in a separately authorized reproduction, use the validated stream-inheriting transport to retain the actual leaf failure, then diagnose that specific failure. Do not blame CUDA based on the current evidence. This receipt grants no pytest/CUDA retry or follow-on execution.
