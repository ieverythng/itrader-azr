#!/usr/bin/env python
"""
Quick environment sanity check.

Run with:  python env_check.py
"""

import importlib
import textwrap
import sys

# ––––– packages you expect ––––– #
pkgs = [
    "torch",
    "ray",
    "duckdb",
    "gymnasium",
    "llama_cpp",
    "numpy",
]

print("▇ PACKAGE VERSIONS ▇")
for name in pkgs:
    mod = importlib.import_module(name)
    print(f"{name:12s} : {getattr(mod, '__version__', 'unknown')}")

print("\n▇ PYTORCH DETAILS ▇")
import torch

print("torch.version.cuda :", torch.version.cuda)
print("compiled w/ CUDA   :", torch.backends.cuda.is_built())   # was it *built* with CUDA?
print("cuda.available     :", torch.cuda.is_available())        # runtime detection
if torch.cuda.is_available():
    print("gpu count          :", torch.cuda.device_count())
    print("gpu name           :", torch.cuda.get_device_name(0))
else:
    print("NOTE ➜ torch was installed as **CPU-only** "
          "(or no compatible driver/device visible)")

print("\n▇ RAY QUICK TEST ▇")
import ray

try:
    ray.init(num_cpus=2, ignore_reinit_error=True, logging_level="ERROR")
    @ray.remote
    def plus(a, b): return a + b
    print("ray   plus(1,2)    :", ray.get(plus.remote(1, 2)))
finally:
    ray.shutdown()

print("\n▇ DUCKDB TEST ▇")
import duckdb
con = duckdb.connect(":memory:")
print("duckdb SQL         :", con.execute("select 2+2").fetchone()[0])

print("\n▇ GYMNASIUM TEST ▇")
import gymnasium as gym
env = gym.make("CartPole-v1", render_mode=None)
obs, _ = env.reset(seed=0)
action = env.action_space.sample()
obs2, reward, term, trunc, _ = env.step(action)
print("CartPole step      :", dict(reward=reward, terminated=term, truncated=trunc))
env.close()

print("\nDone ✓")
