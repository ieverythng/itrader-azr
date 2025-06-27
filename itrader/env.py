import gymnasium as gym
import numpy as np

class iTraderEnv(gym.Env):
    """Vectorised back-test environment for the AZR trader."""
    metadata = {"render_modes": ["none"]}

    def __init__(self, df, window=30):
        super().__init__()
        self.df = df
        self.window = window
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # position size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, df.shape[1]), dtype=np.float32
        )
    # TODO: implement reset(), step(), render()