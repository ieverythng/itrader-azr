# Ray 2.58 Migration Audit and Acceptance

Date: 2026-09-14

Branch: `feat(R)/Ray_upgrade`

Status: dependency migration and legacy-stack parity accepted; native RLlib
RLModule/EnvRunner port remains a separately gated implementation.

## Outcome

The repository's authoritative runtime is migrated from Ray/RLlib 2.9.0 to
2.58.0. The upgrade also moves Gymnasium from 0.28.1 to 1.2.2 and PyArrow from
13.0.0 to 17.0.0. PyArrow 17 is not optional drift: resolving
`ray[rllib]==2.58.0` with PyArrow 13 fails because the newer RLlib distribution
requires PyArrow 17 or newer. NumPy 1.26.4, pandas 2.1.4, DuckDB 0.10.2, and
PyTorch 2.3.x remain unchanged because the accepted environment resolved and
passed without widening those seams.

All primary dependency declarations now agree:

| Surface | Accepted pins |
|---|---|
| `pyproject.toml` / `uv.lock` | Ray 2.58.0, Gymnasium 1.2.2, PyArrow 17.0.0, pandas 2.1.4 |
| `itrader/environment.yml` | same application versions; PyTorch 2.3.x |
| `env.lock.yml` | same direct-version snapshot |
| `itrader/constraints.txt` | same Ray/Gymnasium/PyArrow/pandas constraints |
| `itrader/infra/cluster.yaml` | `ray[rllib]` rather than the insufficient `ray[air]`, with the same pins |

The isolated project under `itrader/compat/ray258/` was used to resolve and
exercise the upgrade before the root lock changed. It remains as a reproducible
migration fixture; the root lock is now authoritative.

## Official compatibility basis

- Ray 2.58.0 is an official published release: <https://github.com/ray-project/ray/releases/tag/ray-2.58.0>
- Its package metadata is published on PyPI: <https://pypi.org/project/ray/2.58.0/>
- Ray's migration guide makes RLModule/Learner and EnvRunner/ConnectorV2 the new
  stack and maps older ModelV2, Policy/RolloutWorker, ViewRequirement, and
  Connector concepts to it:
  <https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html>
- The current RLModule contract separates exploration, inference, and training
  forwards: <https://docs.ray.io/en/latest/rllib/rl-modules.html>
- Tune's PBT guide uses `tune.get_checkpoint()` and checkpoint-bearing
  `tune.report()` calls, and recommends matching checkpoint availability to the
  perturbation cadence: <https://docs.ray.io/en/latest/tune/examples/pbt_guide.html>
- Ray documents Windows support as beta and does not support multi-node Windows
  clusters as a tested deployment surface:
  <https://docs.ray.io/en/latest/ray-overview/installation.html#windows-support>

## Dependency and API seam map

| Cross-layer seam | Ray 2.9 implementation | Ray 2.58 migration state | Decision |
|---|---|---|---|
| Solver module | `SolverMLPFiLM` through `TorchModelV2`/`ModelCatalog` | Legacy API remains available when explicitly selected; RLModule is the destination | Keep R0 behavior fixed for the build migration. Port to an RLModule only with tensor/logit/value and training parity tests. Do not introduce LSTM work here. |
| Sampling | `RolloutWorker`, `num_workers`, `evaluation_num_workers` | EnvRunner is the new abstraction; count/resource keys are renamed | Runtime bridges use the supported Ray 2.58 evaluation and GPU EnvRunner keys while the R0 stack is explicitly disabled from switching implicitly. |
| Learning | Policy optimizer and legacy PPO execution operators | Learner/LearnerGroup owns updates on the new stack | Not mixed into this parity run. The native-stack port must translate minibatch/epoch settings and compare update outputs before promotion. |
| Callbacks/metrics | `EpisodeV2.custom_metrics`, worker/base-env callbacks | `SingleAgentEpisode` plus `MetricsLogger` on EnvRunner callbacks | Existing callbacks are accepted only on the explicit legacy stack. A native callback adapter must preserve `EpisodeEvidence` names and aggregation exactly. |
| Evaluation | `evaluation_workers`, manual policy `explore=False`, worker weight sync | evaluation EnvRunnerGroup and new result nesting | Compatibility helpers select current group/property names, force deterministic post-sync scoring, and read `env_runners/custom_metrics`. |
| Checkpoints | `Algorithm.save/restore`, AIR/session checkpoint handoff | `Checkpointable` direction plus Tune `get_checkpoint/report` | Tune 2.58 APIs are used where available; immutable repository-owned copies and restore-aware evidence remain authoritative. |
| Tune storage | `local_dir`, `Trial.logdir`, `ray.air.session` | `storage_path`, `Trial.local_path`, `tune.get_context/get_checkpoint/report` | Runtime-selected compatibility helpers cover both shapes; the authoritative root now exercises the 2.58 side. |
| PBT | scheduler on held-out net return, aligned checkpoints | same scheduler contract, but donor availability remains timing-sensitive | Six iterations were required for a two-member asynchronous smoke to reach a non-terminal exploit. The accepted run records a real restore at iteration 5 from an iteration-4 donor. |
| Observation filters/connectors | `MeanStdFilter` worker state can diverge on restore | ConnectorV2 owns preprocessing/state | PBT remains fail-closed on `NoFilter` plus deterministic `market_relative_v1`; connector migration may not silently add stateful normalization. |
| Windows/WSL2 | native Windows was the established test surface | Windows still works locally but emits child teardown faults and is officially beta | Native Windows is accepted for local parity. Use Linux/WSL2 or the pinned Linux container for multi-node/long-running deployment validation. Do not claim WSL2 was tested in this workstream. |

## Implemented compatibility changes

- The legacy RLlib stack is explicitly selected on Ray versions exposing
  `PPOConfig.api_stack`; this prevents a dependency update from silently changing
  the solver architecture or episode semantics.
- Removed `UnifiedLogger` is optional rather than an import-time failure.
- Tune context, incoming checkpoint, report, storage, and trial-output APIs are
  selected from the current runtime.
- Evaluation worker/group access and metric result nesting support Ray 2.58.
- Checkpoint-consistent evaluation still forces `explore=False` after restore,
  synchronizes the exact learner weights being persisted, and replaces any
  pre-sync evaluation payload.
- `run_ablation` now emits explicit `--no-auto_plot_on_end` for false Boolean
  optional arguments instead of accidentally reverting to the parser default.
- A fresh-process replay command validates immutable checkpoints against the
  held-out report row:

```text
python -m itrader.tools.replay_ray_checkpoint <stable-trial-dir> --iteration 6
```

## Acceptance evidence

The canonical smoke config is
`itrader/scripts/experiments/configs/ray258_legacy_acceptance_20260914.json`.
The retained compact receipts are under `itrader/compat/ray258/evidence/`.

- Root environment healthcheck: passed on Python 3.11.13, Ray 2.58.0,
  Gymnasium 1.2.2, PyArrow 17.0.0, pandas 2.1.4, and PyTorch 2.3.1 CPU.
- Focused post-lock tests: 52 passed.
- Full repository suite: 106 passed; all pre-commit hooks passed on the second
  run after its formatting hooks normalized existing evidence files.
- Acceptance control: one iteration completed with real held-out metrics.
- Acceptance PBT: two members completed six iterations each with checkpoint and
  perturbation cadence 2. A recorded exploit copied member `00001` into member
  `00000`; the target restored the donor's iteration-4 checkpoint and completed
  iterations 5 and 6.
- Fresh-process replay of the post-exploit iteration-6 checkpoint reproduced
  optimization reward, gross return, net return, and net Sharpe with absolute
  delta `0.0` at tolerance `1e-12`.
- A final one-iteration live smoke on the exact staged code passed with the Ray
  2.58 `evaluation_num_env_runners` key after the last deprecation cleanup.

This is execution and parity evidence. The small panel/population is not evidence
that the solver is profitable or promoted.

## Known failures and cautions

1. The first isolated lock attempt retained PyArrow 13 and was unsatisfiable.
   PyArrow 17 resolved the declared Ray 2.58 RLlib requirements.
2. Removed `ray.tune.logger.UnifiedLogger`, deprecated `local_dir`, renamed
   GPU/evaluation worker keys, moved Tune checkpoint/session helpers, renamed
   trial output properties, and changed evaluation result nesting each caused or
   would cause a concrete compatibility failure; all are covered by focused tests
   or the live smoke.
3. A four-iteration asynchronous PBT smoke ended before a live donor could be
   exploited. Extending the bounded smoke to six iterations produced a real
   non-terminal restore. A completed population without a PBT policy log is not
   sufficient restore evidence.
4. Native Windows repeatedly logged `Windows fatal exception: access violation`
   while Ray child processes disconnected. The driver, trials, checkpoints, and
   replay still completed. This is a platform warning, not a hidden pass for
   multi-node or unattended production operation.

## Native new-stack gate

The package migration is complete; the RLlib architecture migration is not being
misrepresented as complete. The new stack must be promoted in this order:

1. Implement an R0 `RLModule` that shares or exactly reproduces
   `SolverMLPFiLM` preprocessing, task dropout, FiLM equation, logits, and value.
2. Prove tensor parity and one-update parity against the accepted R0 control.
3. Port `PSMetrics` to `SingleAgentEpisode`/`MetricsLogger` without changing the
   `EpisodeEvidence` vocabulary or economic/optimization separation.
4. Replace worker/filter operations with EnvRunnerGroup/ConnectorV2 state and
   repeat deterministic evaluation, checkpoint replay, and real PBT restore.
5. Only then remove the explicit legacy `api_stack(False, False)` bridge.

This sequence preserves `TaskSpec`, `ITraderEnv`, `RewardEngine`, proposer
validation, and the future deterministic online shell. It neither adds nor
claims a recurrent/LSTM solver.
