# # tests/test_llm.py
# from itrader.llm.engine import ask
# def test_llm_runs():
#     assert "trading" in ask("Say one word connected to trading:").lower()

import torch
print(torch.version.cuda)               # e.g. “12.1”
print(torch.cuda.get_arch_list())       # should include “12.0”
print(torch.cuda.get_device_name(0))