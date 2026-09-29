"""运行时并发度控制：把"同时几个请求共驻"变成可变量。

本模块的定位
────────────

本课题此前的全部实验里，并发度是 **启动时定死的配置常量**
（``executor.streams``，默认 2），见 ``service.py::start()``。也就是说：
四个基线、DACC、分区实验、批处理实验——**都是在"并发度这一维被焊死"的条件下
测的**。而本课题读到的每一篇并发机制文献（Bless 的"空闲时用整卡、有竞争就收缩"、
FineST 的"剩余请求率能否饱和"、Bullet 的相位棘轮）**都是并发度的控制回路**。

因此本模块补上的不是"又一个评分函数"，而是那条一直缺失的**控制轴**。

为什么控制信号取自队列而不是 NVML
──────────────────────────────────

``demo_bubble_probe.py`` 读的是 ``nvmlDeviceGetUtilizationRates().gpu``，那是
**区间内是否有 kernel 活跃**的单标量（该脚本自己的注释也承认这是"宽松上界"，
并引用 DNN-occu 的 NVML 90% 对真实 occupancy 45%）。它**区分不了"算力忙"与
"带宽忙"**——而本机的实测事实恰恰是：12 TPC 里 **6 个就吃满显存带宽**
（``docs/实验记录.md`` §17.3）。

在无 root、无 ncu、无 DCGM 的条件下（本课题三条硬约束），能拿到的最干净信号
就是**队列自身的状态**：积压深度、到达率、以及**等待请求的尺寸混合**。

控制律：为什么是"尺寸混合"而不是"负载高低"
────────────────────────────────────────────

朴素的弹性池会随负载增删 worker，那等于"竞争越多共驻越多"——**方向与 Bless 相反**。
本机真正决定收益的是**尺寸混合**，依据来自本课题自己的剖析表
（``data/profiles/default.csv``，1/4 配额相对满配额的比值）：

    224×224  1.34×      336×336  1.56×
    448×448  2.45×      672×672  2.12×

即**小请求几乎不随配额缩放（受访存与固定开销限制），大请求显著缩放**。
推论：一个正在执行的**小**请求内部有大量 SM 空转（``demo_wave_quant.py`` 解析
证实：224 的平均 SM 空闲比 72.5%，672 只有 25.0%），**这些空转可以被共驻请求
免费使用**；而大请求自己就要吃满 SM，再塞一个进去只会互相拖慢。

因此控制律是**尺寸混合感知**的：

    - 等待队列以**小请求**为主 → 提高共驻度（填空转）
    - 等待队列以**大请求**为主 → 降低共驻度（避免带宽/算力争用）
    - 与队列有多长**无关**：长度决定排序，不决定共驻度

这与 "分区被不分区并发支配" 的实测自洽：本机制**不做任何 SM 划分**，
只调整共驻数量。

可证伪的判据
────────────

对照臂：固定并发度扫描 ``static_1/2/3/4/6/8``。
判据：``dynamic`` 的吞吐或 P99 稳定优于 ``max(static_*)``。
死亡：动态被最佳静态支配 → 说明"尺寸混合"不构成有效控制信号（同样可写）。
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field

#: 尺寸分档的 patch 数上界。CLIP ViT-B/32 的 patch 是 32×32，
#: 故 224→49、336→100、448→196、672→441（与 data/profiles/*.csv 一致）。
#: 分档依据是剖析表里的配额敏感性：49/100 约为 1.3–1.6×（弱缩放），
#: 196/441 约为 2.1–2.5×（强缩放）。
SMALL_MAX_PATCHES = 100
LARGE_MIN_PATCHES = 196


@dataclass(frozen=True)
class ConcurrencyConfig:
    """控制器参数。

    默认值是**保守起点**，不是调优结果——按本课题纪律，参数必须由静态扫描
    对照实验确定，不能凭直觉写死（见 ``demos/demo_dual_freedom.py`` 的教训：
    一个基于"稳定数值"的判据在同会话重跑后翻转）。
    """

    #: 共驻度下界。1 = 退化为串行。
    min_concurrency: int = 1
    #: 共驻度上界。上限取 8 是因为单流执行约 6.4 ms、双流约 6.8 ms，
    #: 再往上收益递减；真正的上界应由静态扫描给出。
    max_concurrency: int = 8
    #: 小请求占多数时使用的共驻度。
    small_mix_concurrency: int = 6
    #: 大请求占多数时使用的共驻度。
    large_mix_concurrency: int = 2
    #: "占多数"的判定阈值：小请求占比 ≥ 该值即视为小混合。
    small_mix_ratio: float = 0.5
    #: 到达率统计窗口（秒）。取 0.5 s 是因为秒级实验里更短的窗口样本不足，
    #: 而 ``demo_sts_switch.py`` 曾因 50 ms 窗口恒为空让整个判据退化。
    arrival_window_s: float = 0.5
    #: 是否启用"观测/预测偏离"回退（本课题特色项）。
    #: 当实测执行时间显著超过标定表预测时，说明该尺寸的表不可信，
    #: 此时**停止依赖尺寸混合做决策**，退回到保守共驻度。
    enable_drift_fallback: bool = True
    #: 触发回退的偏离倍率：实测中位 / 预测 > 该值即认为表失真。
    drift_threshold: float = 1.5
    #: 判定偏离所需的最小样本数（样本太少时倍率不可信）。
    drift_min_samples: int = 8


@dataclass
class _Observation:
    """一次执行的观测记录，用于漂移回退判定。"""

    patches: int
    observed_ms: float
    predicted_ms: float


@dataclass
class ConcurrencyController:
    """按队列状态给出**此刻**应允许多少请求共驻。

    本类是**纯决策逻辑**：不碰 GPU、不起线程、不读 NVML。输入是队列状态，
    输出是一个整数。因此它可以在无 GPU 环境下用单元测试完整覆盖
    （见 ``tests/test_concurrency.py``），这也让"判据本身的重复性"成为
    可验证的属性——而不是像 ``demo_dual_freedom.py`` 那样把一个落进噪声带的
    阈值当成通过条件。
    """

    config: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    #: 最近到达时刻（单调时钟，秒）。用 deque 做滑动窗口。
    _arrivals: deque[float] = field(default_factory=deque, repr=False)
    #: 最近的执行观测，用于漂移检测。
    _observations: deque[_Observation] = field(default_factory=deque, repr=False)
    #: 当前共驻请求数，由 service 在请求开始/结束时增减。
    _resident: int = 0
    #: 最近一次决策的说明，写入 job.metadata 供事后归因。
    _last_reason: str = ""

    # ── 状态更新（由 service 调用） ────────────────────────────────

    def note_arrival(self, now: float | None = None) -> None:
        """记录一次请求到达。"""
        self._arrivals.append(time.monotonic() if now is None else now)

    def note_start(self) -> None:
        self._resident += 1

    def note_finish(self, patches: int, observed_ms: float, predicted_ms: float) -> None:
        """记录一次执行完成。``predicted_ms`` 来自标定表，``observed_ms`` 是实测。"""
        self._resident = max(0, self._resident - 1)
        if predicted_ms > 0:
            self._observations.append(_Observation(patches, observed_ms, predicted_ms))

    def resident(self) -> int:
        return self._resident

    def last_reason(self) -> str:
        return self._last_reason

    # ── 决策 ──────────────────────────────────────────────────────

    def target_concurrency(self, pending_patches: list[int], now: float | None = None) -> int:
        """返回此刻允许的最大共驻数。

        ``pending_patches`` 是**等待中**请求的 patch 数列表（不含正在执行的）。
        传空列表表示队列已清空。
        """
        cfg = self.config

        # ① 漂移回退优先于一切：若标定表在这些尺寸上已被证明不可信，
        #    则"大请求/小请求"这个区分本身就不可靠，不再据此决策。
        if cfg.enable_drift_fallback and self._table_is_drifting():
            ratio = self._observed_drift_ratio()
            self._last_reason = f"drift_fallback(observed/predicted={ratio:.2f})"
            return max(cfg.min_concurrency, min(cfg.large_mix_concurrency, cfg.max_concurrency))

        # ② 队列为空：没有竞争者，用整卡——这正是 Bless 的"空闲时用整卡"。
        #    注意这里返回的是**上界**，实际共驻数受队列里有多少请求限制。
        if not pending_patches:
            self._last_reason = "idle_use_full_gpu"
            return cfg.max_concurrency

        # ③ 尺寸混合决定共驻度。
        small = sum(1 for p in pending_patches if p <= SMALL_MAX_PATCHES)
        ratio = small / len(pending_patches)
        if ratio >= cfg.small_mix_ratio:
            target = cfg.small_mix_concurrency
            self._last_reason = f"small_mix({ratio:.2f})"
        else:
            target = cfg.large_mix_concurrency
            self._last_reason = f"large_mix(small_ratio={ratio:.2f})"

        # ④ 夹到配置区间。下界至少 1（否则会死锁）。
        return max(cfg.min_concurrency, min(target, cfg.max_concurrency))

    def arrival_rate(self, now: float | None = None) -> float:
        """最近 ``arrival_window_s`` 内的到达率（req/s）。"""
        current = time.monotonic() if now is None else now
        cutoff = current - self.config.arrival_window_s
        while self._arrivals and self._arrivals[0] < cutoff:
            self._arrivals.popleft()
        if not self._arrivals:
            return 0.0
        span = max(current - self._arrivals[0], 1e-9)
        return len(self._arrivals) / span

    # ── 漂移检测 ──────────────────────────────────────────────────

    def _observed_drift_ratio(self) -> float:
        """最近观测的 实测中位 / 预测中位。

        按尺寸分档统计后再取最差档，因为本课题的核心发现正是
        **可复现性有尺寸依赖**（224 跨轮极差 1.936×，672 仅 1.001×），
        把不同尺寸混在一起取中位会把小尺寸的失真平均掉。
        """
        by_size: dict[int, list[tuple[float, float]]] = {}
        for obs in self._observations:
            by_size.setdefault(obs.patches, []).append((obs.observed_ms, obs.predicted_ms))
        worst = 0.0
        for pairs in by_size.values():
            if len(pairs) < self.config.drift_min_samples:
                continue
            observed = statistics.median(pair[0] for pair in pairs)
            predicted = statistics.median(pair[1] for pair in pairs)
            if predicted > 0:
                worst = max(worst, observed / predicted)
        return worst

    def _table_is_drifting(self) -> bool:
        worst = self._observed_drift_ratio()
        return worst > self.config.drift_threshold
