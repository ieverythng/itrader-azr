# Retained Ray 2.58 optimizer and restore audit

Date: 2026-10-05. Read-only inspection; no training, replay, edits or deletion
of historical artifacts. Ten terminal checkpoints plus three G0 checkpoints
were inspected sequentially after verifying full checkpoint SHA-256 against
each retained index. Policy files were approximately 2.18 MB each.

## Findings and limits

All ten terminal checkpoints identify Ray 2.58.0 and persist native
`num_epochs=30`, `minibatch_size=128`. Their declared configs supplied legacy
aliases, not these native fields. All five exploited terminal targets have
policy/optimizer settings differing from the intended algorithm configuration;
their reported LR follows the intended value rather than saved Adam LR.

The unexploited causal member `00002` retains its expected lower LR, but has
the same epoch mismatch and six self-resume entropy spikes. “Unexploited” does
not mean “never restored.” This limits recipe and optimization-history claims;
it does not invalidate the observed returns of the policies actually produced,
nor prove that the defects caused every flat basin or transient gain.

The audit is a bounded sample, not certification of all past runs. Saved
metadata is not an unavailable per-step optimizer trace. Do not retrospectively
rewrite histories, infer corrected economic outcomes, or discard checkpoints.

## Artifact roots and method

- `A`: `D:/iTRADER-offload/active/ray_results/v2r0-ray258-parity-20260924/_stable_metrics`
- `B`: `D:/iTRADER-offload/active/ray_results/phase-a-r0-plus-r4-v1-actor-robustness-20260929/_stable_metrics`

For each checkpoint, read `algorithm_state.pkl`,
`policies/default_policy/policy_state.pkl` and `rllib_checkpoint.json` only after
digest verification. Intended settings came from the index's `config_snapshot`;
reported settings came from `report_rows.csv` and `metrics_full.jsonl`.
These are trusted local serialized artifacts, not arbitrary downloaded pickles.

| ID | Root / trial directory | Checkpoint under trial | Target steps |
|---|---|---|---:|
| R0-0 | A / `training_fn_ITraderEnv_63247_00000_0_seed=958639_2026-09-25_00-15-02` | `checkpoints/iter_000012_222110438339` | 58,592 |
| R0-1 | A / `training_fn_ITraderEnv_63247_00001_1_seed=1908464_2026-09-25_00-15-02` | `checkpoints/iter_000012_222114361318` | 58,648 |
| Exact-0 | B / `training_fn_ITraderEnv_1ea45_00000_0_seed=4822157_2026-09-29_07-46-30` | `checkpoints/iter_000040_062723649190` | 194,944 |
| Exact-1 | B / `training_fn_ITraderEnv_1ea45_00001_1_seed=5104182_2026-09-29_07-46-30` | `checkpoints/iter_000040_062935266133` | 194,912 |
| Exact-2 | B / `training_fn_ITraderEnv_1ea45_00002_2_seed=5259741_2026-09-29_07-46-30` | `checkpoints/iter_000040_062718940067` | 194,360 |
| Exact-3 | B / `training_fn_ITraderEnv_1ea45_00003_3_seed=4568774_2026-09-29_07-46-30` | `checkpoints/iter_000040_062930732297` | 194,840 |
| Causal-0 | B / `training_fn_ITraderEnv_dc4cf_00000_0_seed=4822157_2026-09-29_06-54-32` | `checkpoints/iter_000041_053539606657` | 195,320 |
| Causal-1 | B / `training_fn_ITraderEnv_dc4cf_00001_1_seed=5104182_2026-09-29_06-54-32` | `checkpoints/iter_000040_053725792156` | 194,456 |
| Causal-2 | B / `training_fn_ITraderEnv_dc4cf_00002_2_seed=5259741_2026-09-29_06-54-32` | `checkpoints/iter_000040_053721474237` | 194,368 |
| Causal-3 | B / `training_fn_ITraderEnv_dc4cf_00003_3_seed=4568774_2026-09-29_06-54-32` | `checkpoints/iter_000041_053543966654` | 195,560 |

## Requested versus persisted effective settings

Every row below actually used native 30 epochs / minibatch 128.

| ID | Intended epochs / minibatch | Intended LR | Saved optimizer LR |
|---|---:|---:|---:|
| R0-0 | 20 / 256 | 1.3577370939916343e-5 | same |
| R0-1 | 30 / 512 | 1.0861896751933076e-5 | 1.3577370939916343e-5 |
| Exact-0 | 20 / 512 | 6.144123266966957e-6 | 7.680154083708696e-6 |
| Exact-1 | 30 / 256 | 6.503092969927453e-6 | same |
| Exact-2 | 20 / 128 | 7.151674865214668e-6 | same |
| Exact-3 | 30 / 512 | 7.680154083708696e-6 | same |
| Causal-0 | 20 / 512 | 6.865607870606081e-6 | 7.151674865214668e-6 |
| Causal-1 | 20 / 128 | 8.582009838257601e-6 | 7.151674865214668e-6 |
| Causal-2 | 20 / 128 | 7.151674865214668e-6 | same |
| Causal-3 | 30 / 256 | 8.582009838257601e-6 | 7.151674865214668e-6 |

| Exploited target | Additional intended -> saved-policy differences |
|---|---|
| R0-1 | gamma .98 -> .995; lambda .90 -> .95; VF coefficient 2 -> .5 |
| Exact-0 | gamma .995 -> .99; clip .3 -> .2; VF clip 10 -> 5; VF coefficient 1 -> .5 |
| Causal-0 | lambda .97 -> .90; clip .2 -> .1 |
| Causal-1 | VF clip 10 -> 20; VF coefficient 1 -> 2 |
| Causal-3 | gamma .98 -> .99; lambda .95 -> .90 |

## Member 00002 self-resume confound

Causal-2's configured entropy schedule is
`[[0, .005], [96000, .002], [240000, .0005]]`. Its six receipts are
`tune_self_resume`, not PBT exploitation. The retained learner metrics show
approximately `.005` on each first update after restore. Expected values below
evaluate the unchanged schedule at that update's starting target progress.

| Receipt under Causal-2 `restore_receipts/` | Restore steps | Next update / report steps | Schedule at start | Recorded coefficient |
|---|---:|---|---:|---:|
| `self_resume_0000_050505533999.json` | 48,624 | 11 / 53,552 | .0034805 | .005 |
| `self_resume_0000_051030511764.json` | 72,904 | 16 / 77,832 | .00272175 | .005 |
| `self_resume_0000_051906192191.json` | 102,040 | 22 / 106,968 | .001937083333 | .005 |
| `self_resume_0000_052151742250.json` | 126,320 | 27 / 131,248 | .001684166667 | .005 |
| `self_resume_0000_052748339468.json` | 150,600 | 32 / 155,528 | .00143125 | .005 |
| `self_resume_0000_053553362442.json` | 179,736 | 38 / 184,664 | .00112775 | .005 |

Three inspected G0 points all retain LR `7.151674865214668e-6`, gamma `.99`,
lambda `.9`, clip `.1`, VF coefficient `2`, VF clip `20`, native 30/128.

| Steps | Checkpoint under Causal-2 | Adam steps | Adaptive KL | Reported update entropy |
|---:|---|---:|---:|---:|
| 77,832 | `checkpoints/iter_000016_051100507146` | 18,000 | 3.0517578125e-6 | .005, immediately after resume |
| 102,040 | `checkpoints/iter_000021_051327487556` | 23,580 | 9.5367431640625e-8 | .001987666667 |
| 106,968 | `checkpoints/iter_000022_051936217482` | 24,720 | 4.76837158203125e-8 | .005, immediately after resume |

Native `global_timestep` matches listed target steps for these points.
Causal-2 terminal Adam count **44,910** equals
`sum(floor(env_steps_this_update / 128) * 30)` over 40 retained reports.
The intended 20-epoch recipe would yield **29,940** for the same batch lengths.
This independently corroborates the effective native settings.

## Verified checkpoint digests

```text
R0-0     d04c1981d7491922d7bcca0aeee50e1ca1d63873e2440b8d3a5c152263c2a269
R0-1     077bf062cb797f477d3aa4f541dcdbb9e6746def47a594ab43a1a5b468b4b97e
Exact-0  c40930df52c4f6288c373e4155a9b1ee2ded22cb7ae475a006f80c427bc12f66
Exact-1  21f4900c6da36ac722edb757a9f9353de1b207632e07e0cfee67d36e6c76106f
Exact-2  2e726472aea9e901e32edad96894b363bed7fddf23e0be9c0ae34adc5c12f562
Exact-3  a95adf46a48f8df0c5f4a3dc1a7ff5a97b2d46f4e682f52d0e31f871a6583fdd
Causal-0 83e8854880f696a0d55c2676de82e05cf11ea4f0e633aa7f97df6bdaeda7a751
Causal-1 2046822ca9c3de76970b79e44dcc3396c1e1cef4936bc3d2183172f54e0dd38b
Causal-2 28afc955e1c7a41a223a941345ccc38c52654c843e26109f1f82b421837bbd63
Causal-3 f26bc1d353d24188368e3015a5f59e57265357a351433b1d2540fc8ff998527a
77832    08d3a5842e8ffaa7c41902be6ee1108ebe476e58619582b247a4fc271cb026d2
102040   74d9798b01daefce2da695c4bdfde6d1015ac01ec6e267feb2dbee4f5c207cce
106968   2566ab85cc1524c7f5d339df2f6e61c1d797840d059f740acb21dffa1a26f83b
```

## Consequence for the next control

Preserve the old policies as historical observed outcomes. A fresh run must
name its effective epoch/minibatch recipe and repaired schedule/restore behavior.
For lowest-confound representation attribution, use explicit native 30/128
across the fresh R0/R4 pair and later R1 pair; call any intentional 20-epoch
change a separate optimizer factor. Do not deliberately recreate entropy-reset
spikes as normal training. This recommendation does not authorize training or
claim a matched full-history reproduction.
