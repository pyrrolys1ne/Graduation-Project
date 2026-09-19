import subprocess, sys, threading, time
sys.path.insert(0, "/home/lyon/projects/bishe")
import torch
a = torch.randn(4096,4096,device="cuda",dtype=torch.float16)
b = torch.randn(4096,4096,device="cuda",dtype=torch.float16)
stop = False
def spin():
    while not stop:
        torch.mm(a,b)
t = threading.Thread(target=spin, daemon=True); t.start()
for i in range(20):
    out = subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu",
                          "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
    print(f"  t={i*2:>2}s  {out}", flush=True)
    time.sleep(2)
stop = True
