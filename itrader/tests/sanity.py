
import importlib, pkg_resources, torch, sys
pkgs = ["llama_cpp_python", "torch", "ray", "duckdb", "gymnasium"]
for p in pkgs:
    mod = importlib.import_module(p.split("_python")[0])
    ver = getattr(mod, "__version__", "?")
    print(f"{p:<16} {ver}")
print("CUDA visible? ", torch.cuda.is_available())

model_path = r"C:\Users\Admin\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\Documents\iTRADER\itrader-azr\itrader\models\Deepseek-6.7B-Q4_K_M\deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

from llama_cpp import Llama

llm = Llama(
    model_path,  # adjust if you used a different file name
    n_gpu_layers=-1,          # off-load all layers to GPU
    n_ctx=4096,               # enough for trading prompts
    seed=42,
)

##Response within -3s -- GOOD
print(llm("Give me a one-sentence trading tip:", max_tokens=50)["choices"][0]["text"])