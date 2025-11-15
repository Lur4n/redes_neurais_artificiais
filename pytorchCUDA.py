import torch
import time

# CPU
x_cpu = torch.rand((6000, 6000))
start = time.time()
y_cpu = x_cpu @ x_cpu
print("CPU:", time.time() - start, "s")

# GPU
x_gpu = torch.rand((6000, 6000), device="cuda")
torch.cuda.synchronize()  # garante início real
start = time.time()
y_gpu = x_gpu @ x_gpu
torch.cuda.synchronize()
print("GPU:", time.time() - start, "s")
