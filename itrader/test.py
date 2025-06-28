import ray, torch, duckdb, gymnasium
print("Ray-OK:", ray.__version__)
print("Torch:", torch.__version__, "| CUDA?", torch.cuda.is_available())
print("duckdb:", duckdb.__version__)
print("gymnasium:", gymnasium.__version__)