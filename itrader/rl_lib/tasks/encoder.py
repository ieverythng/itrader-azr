from __future__ import annotations
import hashlib
import json

import numpy as np

DEFAULT_T_EMBED_DIM = 32

class TaskEncoder:
    """Produces a fixed-length embedding for a TaskSpec.

    Strategy: if spec.t_embed is provided, use it; otherwise create a deterministic embedding from
    categorical fields and scaled numerics. This keeps env/model stable across processes.
    """
    @staticmethod
    def _hash_to_vec(text: str, dim: int) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # repeat to ensure length
        bytes_needed = dim
        buf = (h * ((bytes_needed // len(h)) + 1))[:bytes_needed]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return arr

    @staticmethod
    def embed(
        spec,
        dim: int = DEFAULT_T_EMBED_DIM,
        *,
        version: str = "legacy_v1",
    ) -> np.ndarray:
        if getattr(spec, "t_embed", None) is not None:
            vec = np.array(spec.t_embed, dtype=np.float32)
            if vec.shape[0] != dim:
                # pad or trim deterministically
                if vec.shape[0] < dim:
                    pad = np.zeros(dim - vec.shape[0], dtype=np.float32)
                    vec = np.concatenate([vec, pad])
                else:
                    vec = vec[:dim]
            return vec
        if version == "semantic_v2":
            payload = {
                "family_id": spec.family_id,
                "price_id": spec.price_id,
                "horizon": int(spec.horizon),
                "reward_id": spec.reward_id,
                "reward_params": dict(spec.reward_params),
                "obs_dim": int(spec.obs_dim),
                "action_dim": int(spec.action_dim),
                "bar_resolution": getattr(spec, "bar_resolution", None),
                "regime_tags": sorted(getattr(spec, "regime_tags", []) or []),
                "difficulty_target": getattr(spec, "difficulty_target", {}) or {},
                "metadata": getattr(spec, "metadata", {}) or {},
            }
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            return TaskEncoder._hash_to_vec(canonical, dim).astype(np.float32)
        if version != "legacy_v1":
            raise ValueError(f"Unknown task encoder version: {version!r}")

        # Legacy checkpoint-compatible derivation.
        cats = [spec.family_id, spec.price_id, spec.reward_id]
        num = np.array([
            float(spec.horizon),
            float(spec.obs_dim),
            float(spec.action_dim),
            float(spec.seed % 997),  # bounded seed feature
        ], dtype=np.float32)
        cat_vec = sum((TaskEncoder._hash_to_vec(c, dim) for c in cats)) / len(cats)
        # Fixed scaling: normalize numeric features to reasonable ranges instead of per-array stats
        # This prevents numeric features from dominating due to meaningless normalization
        scales = np.array([1000.0, 5000.0, 10.0, 997.0], dtype=np.float32)
        num_vec = (num / scales)  # scale to [0, ~1] range
        num_vec = (num_vec - 0.5) * 2.0  # center and scale to [-1, 1]
        # mix
        mix = cat_vec.copy()
        mix[: min(len(num_vec), dim)] += num_vec[: min(len(num_vec), dim)]
        # final norm (keep for stability)
        mix = (mix - mix.mean()) / (mix.std() + 1e-8)
        return mix.astype(np.float32)
