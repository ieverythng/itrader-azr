# itrader/models/my_mlp.py
from ray.rllib.core import RLModule
import torch.nn as nn

class MyMLP(RLModule):
    def __init__(self, observation_space, action_space,
                 inference_only=False, model_config=None,
                 catalog_class=None):
        super().__init__(observation_space, action_space,
                         inference_only, model_config, catalog_class)

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        # add policy & value heads here …

    # forward_train / forward_inference / etc.
