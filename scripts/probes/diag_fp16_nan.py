"""诊断：fp16 下 `ClipEncoderBackend.encode_batch` 从第 2 次调用起返回 NaN 嵌入。

**这是 2026-09-26 在排查图回放流水线时偶然撞上的、与本课题主线无关的既有缺陷**
（`ClipEncoderBackend.encode_batch` 自初始提交起未改动）。它不影响延迟/吞吐类
指标——那些指标不读嵌入值——但**凡是引用嵌入数值的结论都不可信**。

已确认的事实（本脚本的核心就是前两条）
────────────────────────────────────

1. `dtype: float16`（本项目 `config.yaml` 的默认值）下，对**同一个请求**连续调用
   `encode_batch`，第 1 次给出有限值，第 2 次起 `embedding_norm` 恒为 `nan`：

       float16   [9.1445, nan, nan, nan, nan, nan]
       float32   [9.1529, 9.1529, 9.1529, 9.1529, 9.1529]

   同一个输入、同一份权重，fp32 逐位稳定，fp16 只在第 1 次对。

2. 手工拆开 fp16 前向（`vision_model.embeddings → encoder → post_layernorm →
   visual_projection`）逐段检查，**三段全无 NaN**，且连续 3 次逐位一致。
   也就是说 NaN 不是"某层溢出"这么简单，触发点在 `encode_batch` 这条具体调用路径上。

3. 把 `encode_batch` 的每一步手工复刻（含 CUDA 事件、资源后端调用、求模），
   结果是**不确定的**：有时 `[0, 512, 0, 0]`、有时 6 次全正常。因此触发条件
   **尚未隔离**，只能确定它发生在 `encode_batch` 而不是"CLIP 前向本身"。

2026-09-27 追加的定位结果（仍未找到根因，但把范围收窄了）
────────────────────────────────────────────────────────────

**已排除**（每条都有实测）：

| 候选 | 排除方式 |
|---|---|
| fp32→fp16 的**转换式 H2D**（`_input` 的 `non_blocking=True` + 分页源） | 三种拷贝方式（分页异步 / 分页同步 / 锁页异步）各 15 次 × 2 轮共 90 次前向，**全部干净** |
| `torch.cat` | 单独加上只让第 2 次 NaN，不产生持续性 |
| 两个 CUDA 事件与 `elapsed_time` | 加上后形态不变 |
| 配额调用、`int(stream.cuda_stream)` | 加上后形态不变 |
| BLAS 后端选择 | `preferred_blas_library("cublas")` 与 `"cublaslt"` 结果完全相同 |
| 显存分配历史（粗粒度） | 前置分配 0 / 300 MB / 800 MB 再释放，形态不变 |
| `encoder.py` 被其他会话改动 | `inspect.getsource()` 与文件逐行一致 |

**收窄到的一条**：**同一段代码、同一输入、同一模型，在不同脚本上下文里给出不同结果。**

- 最小脚本（`python -c`，前置分配极少）：`encode_batch` 第 1 次有限（恒为 9.1445），第 2 次起 NaN。
- 在带前置分配（锁页缓冲 + 一个 fp32 张量）的脚本里：**第 1 次就全 NaN**。
- 把 `encode_batch` 的主体逐行抄进那个脚本（"完全复刻"）：形态又变成 `[0, 512, 512, 512, 0, …]`
  ——**能自己恢复**。

**因此这是"进程状态相关"的失效，而不是数值溢出**——溢出不会关心分配历史。
最可能的机制是**读到了未被写入（或已被复用）的显存**，
但**未定位到具体是哪个张量/哪个 kernel**。要往下走需要 compute-sanitizer 一类工具，
当前尚未完成该层诊断。

**可用的绕过**：**fp32 在全部测试里一次都没出现过 NaN**。凡需要引用嵌入数值的场合改用
float32 即可；而本课题要求的五个指标（吞吐率、平均延迟、P99、GPU 利用率、SLO 违约率）
**都不读嵌入值，因此不受本缺陷影响**。

复现
────

    .venv/bin/python scripts/probes/diag_fp16_nan.py

退出码：若 fp16 出现 NaN 而 fp32 全有限（即缺陷复现）则返回 1，否则返回 0。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encoder_sched.config import load_config  # noqa: E402
from encoder_sched.encoder import ClipEncoderBackend  # noqa: E402
from encoder_sched.models import EncodeJob  # noqa: E402
from encoder_sched.resource import ProxyResourceBackend  # noqa: E402

#: 连续调用次数。第 2 次起就应当出现 NaN，取 6 足以看清形态。
CALLS = 6
SEED = 4242
SIZE = (224, 224)


def probe(dtype: str) -> list[str]:
    import torch

    cfg = load_config("config.yaml")
    enc = ClipEncoderBackend(replace(cfg.model, dtype=dtype), ProxyResourceBackend(), 2)
    job = EncodeJob(request_id=f"nan-{dtype}", seed=SEED, width=SIZE[0], height=SIZE[1],
                    deadline_ms=100_000)
    job.sm_fraction = 1.0
    values: list[str] = []
    for _ in range(CALLS):
        norm = enc.encode_batch([job], 0)[0].embedding_norm
        values.append("nan" if norm != norm else f"{norm:.4f}")
    del enc
    torch.cuda.empty_cache()
    return values


def main() -> None:
    fp16 = probe("float16")
    fp32 = probe("float32")
    print(f"float16  encode_batch 连续 {CALLS} 次: {fp16}")
    print(f"float32  encode_batch 连续 {CALLS} 次: {fp32}")

    fp16_bad = fp16[0] != "nan" and any(v == "nan" for v in fp16[1:])
    fp32_ok = all(v != "nan" for v in fp32)
    if fp16_bad and fp32_ok:
        print("\n→ 缺陷复现：fp16 从第 2 次调用起返回 NaN，fp32 全程有限。")
        print("  延迟/吞吐类指标不受影响（它们不读嵌入值），但引用嵌入数值的结论不可信。")
        raise SystemExit(1)
    if fp16_bad:
        print("\n→ fp16 出现 NaN，但 fp32 也有 NaN：更像环境或输入问题，不是本缺陷的形态。")
        raise SystemExit(1)
    print("\n→ 本次未复现。该缺陷的触发条件尚未隔离（见模块 docstring 第 3 条），"
          "单次不复现不能作为'不存在'的证据。")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
