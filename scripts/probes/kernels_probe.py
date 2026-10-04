import torch, json
from torch.profiler import profile, ProfilerActivity

dev = torch.device("cuda")
MIB = 1024*1024
n = 64*MIB
x = torch.randn(n, device=dev); y = torch.empty(n, device=dev)
ma = torch.randn(2048,2048, device=dev, dtype=torch.float16)
mb = torch.randn(2048,2048, device=dev, dtype=torch.float16)

for _ in range(3):
    y.copy_(x); torch.mm(ma, mb)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        y.copy_(x)
    for _ in range(3):
        torch.mm(ma, mb)
    torch.cuda.synchronize()

rows=[]
for e in prof.key_averages():
    if e.device_type.name == "CUDA" or e.self_device_time_total>0:
        rows.append((e.key, e.count, e.self_device_time_total/1000.0))
for k,c,t in sorted(rows, key=lambda r:-r[2]):
    if t>0: print(f"{c:3d}x  {t:8.3f} ms  {k}")
print("---- copy bytes", x.numel()*x.element_size(), "mm flop", 2*2048**3)
