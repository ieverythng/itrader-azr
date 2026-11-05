# CUDA35 launcher and CUDA readiness receipt

Closed 2026-10-12 23:56 UTC without activating an actual test cell. This work
was limited to the TRAINING board, ignored `_local/training-cuda35/` evidence,
and this receipt. Product files, environments, drivers, and Git state were not
changed.

The root `.venv` contains CPU-only PyTorch. A separate installed profile exists
at `%LOCALAPPDATA%\iTrader\envs\cuda-tested-cu128`, with recorded Torch
`2.9.0.dev20250801+cu128` and Ray `2.58.0`. Therefore CUDA was not disabled
everywhere in the project. This scope did not import PyTorch or query CUDA; it
does not certify that the isolated profile currently initializes or completes
the iTrader lifecycle.

The local typed PowerShell adapter now passes an explicit string argument array
to the existing public launcher and preserves its exit code. The local runner
inherits stdout/stderr so the supervisor log captures leaf failures. Exact
standard-library fake probes passed the full `-B -m` argv tail, both R4 node
IDs, one concatenated `--junitxml=<path>` item, isolated interpreter, both
output streams, and clean Windows Job teardown. A nonzero fake child exit 17
was retained. An empty-root negative control exited 64 before launcher use,
logged the rejection, produced no probe markers or observed file, and cleaned
the Job.

Independent review found and prompted correction of two packet defects: the
actual lifecycle payloads initially split the JUnit option and filename, and
the adapter initially lacked explicit rejection of blank/invalid repository
roots. Both payloads and the adapter were corrected. The fresh exactG probe
passed before the root-validation fix; the empty-root negative control passed
after it. A fresh exactI review then stopped at preflight with 1.808 GiB free
RAM (below the unchanged 4 GiB floor), before probing or source review. A later
read found 1.338 GiB free. Disk remained above 46 GiB, and no eligible
Python/Ray/pytest/uv process was found. No apps or processes were closed.

The final candidate packet has 284 identities with zero hash mismatches:
manifest SHA256 `f61d1b6730fbe86fa7e2f365881008c3c1c5cc803595dcff8f22e1d0d48eb5d4`;
review diff SHA256 `e15f22e610d3a2c1afbbd34d54b9b6392d9758ac4ac1838d9e2ab5d56910e7b3`.
It has no approvals: the prior fresh reviewer rejected the root-validation gap,
which was fixed; the next reviewer could not start; a second final approval was
not obtained. The hard preparation/review stop is 2026-10-13 00:00 UTC, so no
CUDA lifecycle run is authorized in this window. Any continuation needs a new
GRILL scope and fresh review/resource gates.
