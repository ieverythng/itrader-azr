#!/usr/bin/env python
"""
Quick CPU / CUDA inference with llama-cpp-python.
    python scripts/infer.py --model-path models/deepseek-6b-q4_0.gguf \
                            --prompt "Tell me a dad joke."
"""
import argparse, time, llama_cpp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=35, help="use GPU vram")
    args = ap.parse_args()

    t0 = time.perf_counter()
    llm = llama_cpp.Llama(
        model_path=args.model_path,
        n_ctx=4096,
        n_gpu_layers=args.n_gpu_layers,   # 0 = CPU-only
        chat_format="llama-2",            # deepseek follows llama-2 system
    )
    print(f"🦙 loaded in {time.perf_counter()-t0:.1f}s")

    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": args.prompt}],
        stream=False,
        max_tokens=200,
    )
    print("---- answer ----")
    print(out["choices"][0]["message"]["content"].strip())

if __name__ == "__main__":
    main()
