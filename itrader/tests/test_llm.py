import torch
print(torch.cuda.is_available())          # Expect True
print(torch.cuda.get_device_name(0))      # NVIDIA GeForce RTX 5070 Ti
print(torch.cuda.get_arch_list()) 