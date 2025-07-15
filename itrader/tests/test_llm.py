# # tests/test_llm.py
# from itrader.llm.engine import ask
# def test_llm_runs():
#     assert "trading" in ask("Say one word connected to trading:").lower()

from ray.rllib.algorithms.ppo import PPOConfig
cfg = PPOConfig()
print("Has postprocess_inputs:", hasattr(cfg, "postprocess_inputs"))