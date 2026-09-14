"""探查：TPC 区间互斥时，残余互相拖慢来自哪里？

已知事实：两个并发请求即使拿到**完全互斥**的 TPC 集合，仍有约 2.5 倍的互相拖慢
（见 `docs/实验记录.md`）。本脚本用来定位这部分争用的来源，从而决定研究方向。

做法：让两个线程各持一半 TPC（互斥），分别跑两种资源画像截然不同的负载：

- **计算受限**：大矩阵乘，算术强度高，理论上只吃 SM/TPC；
- **访存受限**：大张量逐元素运算，几乎不吃算力，吃 L2 与显存带宽。

然后比较三类配对的 slowdown：

| 配对 | 若争用只在访存侧 | 若争用来自驱动/发射队列 |
|---|---|---|
| 计算 + 计算 | slowdown ≈ 1 | slowdown 高 |
| 访存 + 访存 | slowdown 高 | slowdown 高 |
| 计算 + 访存 | 介于两者之间 | slowdown 高 |

**注意**：GPU 时钟会随负载漂移，所以每个测量都在同一进程内、交替进行；
绝对值可能整体偏移，**要看的是三类配对之间的相对关系**。

用法::

    export LIBSMCTRL_PATH=/path/to/libsmctrl.so    # 需打过补丁（含 set_thread_mask）
    python scripts/probe_contention_source.py --repeats 5
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import concurrent.futures
import json
import statistics
import threading

from encoder_sched.libsmctrl_adapter import LibSmCtrlAdapter, enabled_tpcs_to_native_mask


def workloads(torch):
    """返回 {名称: (构造张量, 每次调用的函数)}。"""
    compute_x = torch.randn(3072, 3072, device="cuda")
    mem_x = torch.randn(64 * 1024 * 1024, device="cuda")   # 256 MB
    mem_y = torch.randn(64 * 1024 * 1024, device="cuda")

    def compute():
        return compute_x @ compute_x

    def memory():
        return torch.add(mem_x, mem_y)          # 纯逐元素，算术强度极低

    return {"compute": compute, "memory": memory}


def run_solo(torch, fn, iters: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="定位 TPC 互斥下的残余争用来源")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10, help="每个测量段内的调用次数")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA 设备")

    adapter = LibSmCtrlAdapter(cuda_device=args.cuda_device)
    probe = adapter.probe()
    if not probe.get("available"):
        raise SystemExit(
            f"本环境无法真实施加 TPC 掩码，拒绝运行。\n原因: {probe.get('reason')}\n"
            "请用 scripts/build_libsmctrl.sh 构建补丁版库并设置 LIBSMCTRL_PATH。"
        )

    total = probe["total_tpcs"]
    half = total // 2
    # 两个互斥的 TPC 半区
    spans = [(0, half), (half, total - half)]
    masks = [enabled_tpcs_to_native_mask(start, count) for start, count in spans]
    print(f"设备 TPC 总数 {total}；线程 A 用 TPC {spans[0][0]}..{spans[0][0]+spans[0][1]-1}，"
          f"线程 B 用 TPC {spans[1][0]}..{spans[1][0]+spans[1][1]-1}（互斥）\n")

    fns = workloads(torch)
    names = list(fns)

    def solo_with_mask(kind: str, mask: int, handle: int) -> float:
        """独跑基准：**用与并发时完全相同的掩码**。

        这一点是必须的。若基准用满配额、并发用半卡，测出的"slowdown"就把
        "TPC 减半的固有代价"也算了进去——而计算受限负载在 1/4 配额下本就会慢近 4 倍
        （见 docs/实验记录.md 的配额曲线）。那样得到的数字无法解释为"争用"。
        """
        adapter.apply_quota(handle, 1.0)
        adapter._ensure_library().libsmctrl_set_thread_mask(
            __import__("ctypes").c_uint64(mask)
        )
        value = run_solo(torch, fns[kind], args.iters)
        adapter.release()
        return value

    def solo_times() -> dict[str, dict[str, float]]:
        """{负载类型: {'a': 用半区A独跑, 'b': 用半区B独跑}}"""
        out: dict[str, dict[str, float]] = {name: {} for name in names}
        for name in names:
            out[name]["a"] = solo_with_mask(name, masks[0], 0x9000)
            out[name]["b"] = solo_with_mask(name, masks[1], 0x9001)
        return out

    def concurrent(kind_a: str, kind_b: str) -> tuple[float, float]:
        """两个线程各持互斥半卡，同时跑。"""
        results: dict[str, float] = {}
        ready = threading.Barrier(2, timeout=60)

        def worker(tag: str, kind: str, mask: int, handle: int) -> None:
            adapter.apply_quota(handle, 1.0)          # 先建立本线程的分配
            # 直接用原生接口覆盖成指定的互斥掩码，保证半区精确
            adapter._ensure_library().libsmctrl_set_thread_mask(
                __import__("ctypes").c_uint64(mask)
            )
            ready.wait()                              # 两个线程同时开始
            results[tag] = run_solo(torch, fns[kind], args.iters)

        threads = [
            threading.Thread(target=worker, args=("a", kind_a, masks[0], 0xA000)),
            threading.Thread(target=worker, args=("b", kind_b, masks[1], 0xB000)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results["a"], results["b"]

    samples: dict[str, dict] = {}
    for repeat in range(args.repeats):
        solo = solo_times()
        for kind_a in names:
            for kind_b in names:
                a_ms, b_ms = concurrent(kind_a, kind_b)
                key = f"{kind_a}+{kind_b}"
                entry = samples.setdefault(key, {"a": [], "b": [], "solo_a": [], "solo_b": []})
                entry["a"].append(a_ms)
                entry["b"].append(b_ms)
                # 基准取"该负载用同一个线程号独跑同掩码"的值
                entry["solo_a"].append(solo[kind_a]["a"])
                entry["solo_b"].append(solo[kind_b]["b"])
        print(f"  repeat {repeat + 1}/{args.repeats} 完成", flush=True)

    print("\n" + "=" * 88)
    print("结果：并发 vs 独跑（两个线程各持互斥半卡）")
    print("=" * 88)
    print(f"  {'配对':22s} {'独跑A':>9s} {'并发A':>9s} {'slowA':>7s} "
          f"{'独跑B':>9s} {'并发B':>9s} {'slowB':>7s} {'均slow':>7s}")
    print("  " + "-" * 86)
    summary = {}
    for key, entry in samples.items():
        solo_a, solo_b = statistics.median(entry["solo_a"]), statistics.median(entry["solo_b"])
        conc_a, conc_b = statistics.median(entry["a"]), statistics.median(entry["b"])
        sa, sb = conc_a / solo_a, conc_b / solo_b
        summary[key] = {"slow_a": sa, "slow_b": sb, "mean_slow": (sa + sb) / 2}
        print(f"  {key:22s} {solo_a:9.2f} {conc_a:9.2f} {sa:6.2f}x "
              f"{solo_b:9.2f} {conc_b:9.2f} {sb:6.2f}x {(sa+sb)/2:6.2f}x")

    print()
    print("判读：")
    print("  compute+compute 与 memory+memory 的 mean_slow 差距大 -> 争用主要在访存侧，")
    print("    TPC 分区对计算受限请求有效、对访存受限请求无效。")
    print("  三类配对的 mean_slow 都接近且都 > 1 -> 争用与资源类型无关，")
    print("    更可能来自驱动/发射队列等 TPC 之外的因素。")
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
