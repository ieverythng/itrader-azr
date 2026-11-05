"""Fail-closed GPU tensor forward/backward smoke for the isolated CUDA profile."""

from __future__ import annotations

import json
import math
import platform
import struct
import sys

EXPECTED_TORCH = "2.9.0.dev20250801+cu128"
EXPECTED_CUDA_RUNTIME = "12.8"


def main() -> int:
    import torch

    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(
            f"Expected torch {EXPECTED_TORCH}; observed {torch.__version__}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The cuda-tested profile has no available CUDA device; no CPU fallback."
        )
    if torch.version.cuda != EXPECTED_CUDA_RUNTIME:
        raise RuntimeError(
            f"Expected CUDA runtime {EXPECTED_CUDA_RUNTIME}; observed {torch.version.cuda}."
        )

    device = torch.device("cuda:0")
    torch.manual_seed(271828)
    model = torch.nn.Linear(8, 4, bias=True).to(device=device, dtype=torch.float32)
    values = torch.arange(32, dtype=torch.float32, device=device).reshape(4, 8) / 31.0
    prediction = model(values)
    loss = prediction.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    gradients = [parameter.grad for parameter in model.parameters()]
    if not torch.isfinite(prediction).all().item() or not math.isfinite(loss.item()):
        raise RuntimeError("GPU forward produced non-finite values.")
    if not all(
        gradient is not None and torch.isfinite(gradient).all().item()
        for gradient in gradients
    ):
        raise RuntimeError("GPU backward produced missing or non-finite gradients.")
    if any(gradient.device.type != "cuda" for gradient in gradients):
        raise RuntimeError("Backward gradients did not remain on CUDA.")

    major, minor = torch.cuda.get_device_capability(device)
    print(
        json.dumps(
            {
                "gate": "cuda_tensor_forward_backward",
                "status": "passed",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": torch.cuda.get_device_name(device),
                "compute_capability": [major, minor],
                "python": {
                    "implementation": platform.python_implementation(),
                    "machine": platform.machine(),
                    "pointer_bits": struct.calcsize("P") * 8,
                    "version": [sys.version_info.major, sys.version_info.minor],
                },
                "output_shape": list(prediction.shape),
                "loss": loss.item(),
                "gradient_shapes": [list(gradient.shape) for gradient in gradients],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
