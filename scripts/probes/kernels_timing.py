import torch, time
dev = torch.device("cuda")
ma = torch.randn(4096,4096, device=dev, dtype=torch.float16)
mb = torch.randn(4096,4096, device=dev, dtype=torch.float16)
x = torch.randn(64*1024*1024, device=dev); y = torch.empty_like(x)
for _ in range(2): torch.mm(ma,mb); y.copy_(x)
torch.cuda.synchronize()

def bench(fn, reps=12):
    fn(); torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/reps

mm_ms = bench(lambda: torch.mm(ma,mb))
cp_ms = bench(lambda: y.copy_(x))
print(f"matmul 4096^3 fp16 : {mm_ms:.3f} ms/call -> {2*4096**3/(mm_ms/1000)/1e12:.1f} TFLOP/s")
print(f"copy   256MiB DtoD : {cp_ms:.3f} ms/call -> {2*x.numel()*4/(cp_ms/1000)/1e9:.1f} GB/s")
