# TRAINING-FIX36 launcher readiness receipt

As of 2026-10-13 00:20 UTC, this is a readiness-only handoff on branch agent/Watson-dev, last recorded HEAD 8f9fcf94e028b51d883a1a484c2c9466a3d6b547. No product source or tests were changed, and no tests, probes, Python/Torch/Ray imports, GPU query, or CUDA job were run. A separate GRILL lease is required before any lifecycle execution.

The DIAG34 full-tail reproduction supplied -B -c <stdlib argv printer> -q -x -p no:cacheprovider --junitxml=<single path> <r0-r4-state node> <r1-r4-state node> after the public -File invoke_cuda_tested.ps1 entrypoint. PowerShell attempted to bind -B as a wrapper parameter and returned exactly: A parameter cannot be found that matches parameter name 'B'. Python and pytest never started. This locates the defect at launcher argument binding, not CUDA execution. DIAG34 receipt SHA256: 3175ea50b98aa1c61c237eececd5a5e37fa5927bbbfcac1d806ab80d455f6e391.

Source and contract hashes rechecked at 00:20 UTC:
- itrader/tools/run_repo_python.py: 939c94a1d17f802d5d5e1ac8339fe44589b6b0497ed39eac1a695ecfdca53e7a
- itrader/tools/cuda_identity_probe.py: 4899bd74a3888f3293233ad0c669e1a45fd17ee23dcc95f661af6d7e62cdcbd2
- itrader/tools/invoke_cuda_tested.ps1: 13f2db230f4e7fc21ab5875d0a7ff26885303c41612c7685f8e7283d711ff966
- itrader/tests/test_run_repo_python.py: 5169d4ab18c147a7e403bcee18e8c631caf53817bee66bc541b4de7653ad5096
- itrader/tests/test_cuda_test_launcher.py: 55168f0fd58e791a572a84f7c7385630f8414c52be31352c8890ac2cf8525ab7
- itrader/scripts/experiments/run_ablation.py: 07fa3dcfaf93ade6c49850186bd12945979e8f5d8e62421fb63ba526f217ca23
- docs/environment_strategy.md: 45eedc876a01bc80e87d4571cffa9f7235a665b3301d28847a1790c3012607c1

These still match the DIAG34 baseline. DEV's board showed FIX36 active at 00:18 UTC; no repaired-source hashes have been received in this readiness snapshot.

The repo .venv is CPU-only (torch 2.3.1+cpu), but CUDA was not disabled in every project environment. The isolated cuda-tested-cu128 environment contains torch 2.9.0.dev20250801+cu128; CUDA28 previously passed a tensor forward/backward smoke on the RTX 5070 Ti. That does not qualify RLlib/PPO or the current launcher path. run_repo_python.py currently falls back to the CPU .venv for a missing ITRADER_PYTHON_BIN; DEV owns a strict, fail-closed CUDA selection repair.

The latest recorded helper/Job reconciliation was 00:09 UTC: 13 prior CUDA35 Job receipts each reported active 0, errors [], survivors []; their supervisor roots and PID 33776 were absent. The later human-reported ~11-GiB PowerShell PID is unknown, so causation is unproven. TRAINING stopped no process; no newer process query was made in FIX36.

For a separately authorized future run, the retained historical recipe is 600 synthetic rows, 12 steps, one PPO update, one SGD epoch, CPU/thread count 1, workers 0, learner GPU 1; two native argv-list calls cover four tests in order: the R4 pair [r0-r4-state, r1-r4-state], then state-off pair [r0-state-off, r1-state-off]. Each call should use the strictly selected CUDA python.exe, -B -m pytest -q -x -p no:cacheprovider, one unique --junitxml=<path> item, both matching node IDs, and subprocess.run(argv, cwd=repo, env=env, shell=False) in the existing list-style pattern from run_ablation.py. This is not runnable authority; await DEV's bound repair plus GRILL review, resource gates, and a new explicit lease.

tools/knowledge/status.sh was deferred because it invokes Git, which FIX36 prohibits. No Git check or execution validation was performed.
