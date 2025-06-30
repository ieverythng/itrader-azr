from pathlib import Path
from llama_cpp import Llama
import torch

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]     # repo root (itrader-azr)
MODEL_DIR = ROOT / "models" / "Deepseek-6.7B-Q4_K_M"
MODEL_FILE = next(MODEL_DIR.glob("*.gguf"))

print("Using model:", MODEL_FILE)

# ---------- GPU parameters ----------
# n_gpu_layers: how many layers to keep on GPU VRAM (0 = CPU only).
# For RTX 4060 8 GB, a quant-4_K_M model fits fully:
N_GPU_LAYERS = 99      # “big number” → all
MAIN_GPU      = 0       # CUDA device ID

llm = Llama(
    model_path=str(MODEL_FILE),
    n_ctx=4096,              # max context tokens
    n_gpu_layers=N_GPU_LAYERS,
    main_gpu=MAIN_GPU,
    logits_all=False,
    verbose=False
)

#print("Loaded! CUDA-aware =", llm.cuda)
prompt = "Write a one-sentence summary of Newton's first law of motion."
out = llm(prompt, max_tokens=64, stop=[".</s>"])
print("\nAssistant:", out["choices"][0]["text"].strip())
