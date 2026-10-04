import torch
from torch.profiler import profile, ProfilerActivity
dev = torch.device("cuda")
ma = torch.randn(4096,4096, device=dev, dtype=torch.float16)
mb = torch.randn(4096,4096, device=dev, dtype=torch.float16)
for _ in range(2): torch.mm(ma, mb)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): torch.mm(ma, mb)
    torch.cuda.synchronize()
for e in sorted(prof.key_averages(), key=lambda e:-e.self_device_time_total):
    if e.self_device_time_total>0:
        print(f"{e.count:3d}x  {e.self_device_time_total/1000.0:8.3f} ms  {e.key}")
