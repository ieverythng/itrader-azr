from copy import deepcopy

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete


def test_selector_reaches_untrained_control_and_rejects_legacy_lstm_alias():
    from itrader.rl_lib.evaluation.policy_baselines import UntrainedFiLMPolicy

    spaces = (Box(-1, 1, (332,), dtype=np.float32), Discrete(3))
    for selector in ("unknown", "solver_lstm_v2", "solver_window_lstm_film"):
        with pytest.raises((ValueError, NotImplementedError)):
            UntrainedFiLMPolicy(*spaces, seed=1, model_config={"custom_model": selector})


def test_selector_and_nested_model_must_agree():
    from itrader.models.registry import model_identity

    with pytest.raises(ValueError, match="conflict"):
        model_identity({"solver_model": "solver_window_lstm_film",
                        "model": {"custom_model": "solver_mlp_film"}})


def test_fixed_recipe_best_params_cannot_override_architecture_atomically():
    from itrader.rl_lib.train.train_solo_V2 import _apply_best_params

    config = {"representation_profile": "r0_r1_v1", "lr": 1e-5,
              "model": {"custom_model": "solver_mlp_film", "custom_model_config": {"t_embed_dim": 32}}}
    before = deepcopy(config)
    with pytest.raises(ValueError, match="architecture"):
        _apply_best_params(config, {"lr": 0.1, "model": {"custom_model_config": {"t_embed_dim": 64}}})
    assert config == before
