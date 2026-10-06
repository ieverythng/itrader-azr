# Terminal replay closeout review — 2026-10-05

## Verdict

**ACCEPTED TERMINAL REPLAY EVIDENCE: 8/8 cells. ECONOMIC PROMOTION: 0/8.**

Cells 4–7 are in `study_attempt04_terminal_replay_jobscope20261005`; cells 1–3 and the accepted cell 8 are in `study_attempt04_terminal_replay_jobscope_retry01_20261005`. The earlier cell 8 under the first root is a preserved resource-gate abort, not an accepted replay and not a replacement for retry cell 8.

## Independent checks

For each of the eight accepted cells, read-only checks confirmed: strict complete receipt and valid replay contract; exactly 32 episode rows; episode artifact SHA/row-count and reconstructed payload SHA match the receipt, checkpoint index, and acceptance; producer invocation/launch IDs and PID plus process-creation-time witnesses link to the recorded launcher; and launch returned 0 without timeout or stop reason, with zero active processes, survivors, or cleanup errors. Producer invocation IDs and launch IDs are distinct across the eight accepted cells.

Recomputed episode summaries match all 20 corresponding fields in each saved `acceptance.json`. The raw net return equals the indexed `evaluation/net_return_mean` selection score and both receipt selected/replayed scores for every cell; all three receipt parity flags are literal `true` and replay delta is zero. Ten evaluation metrics shared with each original terminal `metrics_full.jsonl` row also match to 1e-12 (net/gross return, optimization reward, Sharpe, and six evaluation action-fraction fields). The raw replay turnover summary matches acceptance, but `evaluation/turnover_mean` is not recorded in those training metric rows, so no direct training-row turnover comparison is claimed.

All eight checkpoint directories were rehashed and match the declaration, index, and replay-contract digest. Canonical config and effective environment-config hashes, observation schema, and panel hash recomputed from the frozen task list match their indexed/resolved/replay-contract declarations. Dataset hash declarations agree across resolved config, checkpoint index, and replay contract; **dataset bytes were not rehashed or loaded**. The frozen evaluator fingerprint is `944d7d308654b93b090fec83a8097728157405671ae8aff858d2d00658fcf233` (104 Python files); the separate launcher-source fingerprint is `a8bf027a2f2f0f7e412c5f0f5fcf921642f0a29d0baa3d83039ec816fe9b6321` (105 files) and matches the launcher-source stamp. These are distinct source identities by design.

The current external report driver is stamped `fd20bae59d3f16be7580a6deff3bcca9363ff86f63155ded46d4902b10b7a466`; its acceptance construction recomputes `summary` from episode rows (`summarize_control_episodes(episodes)`). Cells 5–7 carry the preceding patched-driver stamp (`efb5a8b7…`); retry cells 1–3 and 8 carry the current stamp. Cell 4 retains its original older operator stamp (`139f759d…`) and its acceptance artifact was not edited. Its stored summary nevertheless matches the independent raw reconstruction. This review does not infer that the older operator resisted a tampered receipt summary; the patched-driver tampering probe was handled separately.

## Raw reconstructed terminal metrics

| Cell | Net return | Gross return | Net Sharpe / √bar | Turnover | Short | Flat | Long |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 2 | -0.000002303047 | 0.000003946953 | -0.001758110101 | 0.0625 | 0 | 0.9998046875 | 0.0001953125 |
| 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 4 | 0.003927611730 | 0.004221361730 | 0.018891505246 | 2.9375 | 0.9508463542 | 0 | 0.0491536458 |
| 5 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 6 | -0.001710679514 | 0.000295570486 | -0.059884073985 | 20.0625 | 0.177734375 | 0.765234375 | 0.05703125 |
| 7 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |

## Preserved resource abort and scope limits

The first-root cell 8 stopped with `stop_reason=memory_reserve`, return code 71, observed minimum free memory 1,286,672,384 bytes (about 1.198 GiB) below the 1,342,177,280-byte (1.25 GiB) reserve; cleanup reported zero active processes, survivors, and errors. The later retry-root cell 8 is the accepted 32-episode receipt above. No Ray job was launched during this review; no full dataset was loaded, and no original acceptance artifact was modified.

## Review contract status — scoped evidence changes

Executed read-only Python probes against all eight accepted receipts using the frozen evaluator helpers: `verify_receipt`, `_validate_episode_replay_contract`, `summarize_control_episodes`, payload reconstruction, checkpoint rehash, index/metric lookup, config/panel digest recomputation, and selection-score comparison. Also inspected the saved launch/declaration/acceptance/failure artifacts and the current operator's summary construction. No pytest or full-data check was run in this closeout pass. The earlier failed cell 8 is explicitly excluded from the accepted set; its retry is included.

No open findings in the scoped receipt/evidence closeout. The old cell 8 resource abort, turnover logger omission, and cell 4's pre-patch operator stamp are recorded scope limitations, not hidden acceptance claims. Dataset bytes were not rehashed; only their stored declarations were cross-checked.

1. Separation of concerns — OK. Receipt/producer validation, indexed checkpoint replay and scoring, episode summary derivation, and external closeout reporting remain distinct in the reviewed path.
2. Programming by intention — OK. The accepted state denotes terminal replay evidence only; each saved acceptance explicitly keeps `economic_promotion` false, and unknown or failed evidence is not treated as promotion.
3. Encapsulation — OK. Validation, provenance hashing, payload summarization, and checkpoint metric calculation use their dedicated helpers; the closeout does not replace the original acceptance artifacts.
4. High cohesion — OK. Each receipt is bound to one producer, one checkpoint contract, and one episode artifact; the closeout record is limited to those evidence claims and their explicit exclusions.
5. Low coupling — OK. The frozen evaluator fingerprint and the separately stamped launcher/operator identities are independently recorded and checked rather than conflated.

VERDICT: APPROVE
