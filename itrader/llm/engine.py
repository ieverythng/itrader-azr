# engine.py
from functools import lru_cache
from llama_cpp import Llama
import pathlib

MODEL_FILE = pathlib.Path("models/deepseek-llm-6b-q4_k_m.gguf").resolve()

@lru_cache
def get_llm():
    return Llama(model_path=str(MODEL_FILE), n_gpu_layers=-1, n_ctx=4096)

def ask(prompt: str, **gen):
    llm = get_llm()
    out = llm(prompt, **dict(max_tokens=128, temperature=0.7) | gen)
    return out["choices"][0]["text"].strip()
