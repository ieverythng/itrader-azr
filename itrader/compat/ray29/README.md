# Ray 2.9 reproduction profile

This profile preserves the direct dependency versions used by the historical
dataset-v1 and dataset-v2 R0 evidence. It is reproduction-only; new R-series
experiments use the root Ray 2.58 lock.

Create a separate Python 3.11 environment and install with
`pip install -c constraints.txt "ray[rllib]==2.9.0"`. Do not mix this profile
with the root environment or compare stochastic PPO trajectories bit-for-bit
across RLlib versions.
