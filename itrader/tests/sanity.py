# check_env.py
import importlib, pkg_resources, torch, sys
pkgs = ["llama_cpp_python", "torch", "ray", "duckdb", "gymnasium"]
for p in pkgs:
    mod = importlib.import_module(p.split("_python")[0])
    ver = getattr(mod, "__version__", "?")
    print(f"{p:<16} {ver}")
print("CUDA visible? ", torch.cuda.is_available())

model_path = r"C:\Users\Admin\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\Documents\iTRADER\itrader-azr\itrader\models\Deepseek-6.7B-Q4_K_M\deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

from llama_cpp import Llama
with Llama(model_path,
           n_ctx=512, n_gpu_layers= -1) as llm:
    out = llm("Hello!", max_tokens=10)
    print("Sample out:", out["choices"][0]["text"].strip())
