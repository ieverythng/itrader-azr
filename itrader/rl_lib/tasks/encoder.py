from __future__ import annotations
from typing import Sequence
import numpy as np
import hashlib

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
    def embed(spec, dim: int = DEFAULT_T_EMBED_DIM) -> np.ndarray:
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
        # derive
        cats = [spec.family_id, spec.price_id, spec.reward_id]
        num = np.array([
            float(spec.horizon),
            float(spec.obs_dim),
            float(spec.action_dim),
            float(spec.seed % 997),  # bounded seed feature
        ], dtype=np.float32)
        cat_vec = sum((TaskEncoder._hash_to_vec(c, dim) for c in cats)) / len(cats)
        num_vec = (num - num.mean()) / (num.std() + 1e-8)
        # mix
        mix = cat_vec.copy()
        mix[: min(len(num_vec), dim)] += num_vec[: min(len(num_vec), dim)]
        # final norm
        mix = (mix - mix.mean()) / (mix.std() + 1e-8)
        return mix.astype(np.float32)