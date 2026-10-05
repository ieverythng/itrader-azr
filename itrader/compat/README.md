# Compatibility profiles

Compatibility profiles are explicit and non-default. They are not alternate
branches inside the active trainer.

- `rllib_legacy/` contains Ray ModelV2/TorchPolicy-era adapters and historical
  Ray 2.58 recipes for parity investigations. The active trainer rejects their
  stack selector.
- `ray258/` is the isolated Ray 2.58 dependency-resolution fixture used during
  migration acceptance. It does not define where legacy source belongs.
- `ray29/` is the historical Ray 2.9 runtime reproduction profile. Keep its
  constraints separate from the root Ray 2.58 lock.

For exact historical runs, prefer the source revision and dependency versions
recorded in that run's manifest. Do not use a compatibility profile to select a
legacy implementation from the active Ray trainer.
