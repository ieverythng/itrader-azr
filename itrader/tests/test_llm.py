# # tests/test_llm.py
# from itrader.llm.engine import ask
# def test_llm_runs():
#     assert "trading" in ask("Say one word connected to trading:").lower()

import torch, ray, os; ray.init(num_gpus=1); print(torch.cuda.is_available(), ray.available_resources())