"""``encoder_sched.concurrency`` 的单元测试。

本模块是纯决策逻辑，因此可以完整覆盖——**这正是它相对于离线 demo 的优点**：
``demo_dual_freedom.py`` 的教训是"判据本身落在噪声带内会让结论在同会话重跑后翻转"，
而纯函数式的判据可以用确定性测试钉死，不依赖 GPU 会话状态。
"""

from __future__ import annotations

from encoder_sched.concurrency import (
    LARGE_MIN_PATCHES,
    SMALL_MAX_PATCHES,
    ConcurrencyConfig,
    ConcurrencyController,
)

# 与 data/profiles/default.csv 一致的 patch 数
P_224 = 49
P_336 = 100
P_448 = 196
P_672 = 441


def test_size_buckets_match_profiling_table():
    """分档边界必须与剖析表的实际尺寸对齐，否则混合判据会分错档。"""
    assert P_224 <= SMALL_MAX_PATCHES
    assert P_336 <= SMALL_MAX_PATCHES
    assert P_448 >= LARGE_MIN_PATCHES
    assert P_672 >= LARGE_MIN_PATCHES


def test_empty_queue_uses_full_gpu():
    """队列为空 = 无竞争者，应允许最高共驻度（Bless 的"空闲时用整卡"）。"""
    ctrl = ConcurrencyController()
    target = ctrl.target_concurrency([])
    assert target == ctrl.config.max_concurrency
    assert "idle" in ctrl.last_reason()


def test_small_mix_raises_concurrency():
    """等待队列以小请求为主 → 提高共驻度去填 SM 空转。"""
    ctrl = ConcurrencyController()
    target = ctrl.target_concurrency([P_224, P_224, P_336, P_224])
    assert target == ctrl.config.small_mix_concurrency
    assert "small_mix" in ctrl.last_reason()


def test_large_mix_lowers_concurrency():
    """等待队列以大请求为主 → 降低共驻度，避免争用。"""
    ctrl = ConcurrencyController()
    target = ctrl.target_concurrency([P_672, P_672, P_448, P_672])
    assert target == ctrl.config.large_mix_concurrency
    assert "large_mix" in ctrl.last_reason()


def test_mixed_queue_honours_ratio_threshold():
    """恰好一半小请求时按下界阈值判定。"""
    ctrl = ConcurrencyController()
    # 2 小 2 大 → ratio = 0.5 ≥ 0.5 → 小混合
    assert ctrl.target_concurrency([P_224, P_336, P_448, P_672]) == ctrl.config.small_mix_concurrency
    # 1 小 3 大 → ratio = 0.25 < 0.5 → 大混合
    assert ctrl.target_concurrency([P_224, P_448, P_672, P_672]) == ctrl.config.large_mix_concurrency


def test_target_is_clamped_to_config_bounds():
    """目标值必须夹在 [min, max] 内，且下界至少 1（否则死锁）。"""
    cfg = ConcurrencyConfig(min_concurrency=1, max_concurrency=3, small_mix_concurrency=99)
    ctrl = ConcurrencyController(cfg)
    assert ctrl.target_concurrency([P_224]) == 3


def test_concurrency_never_below_one():
    """下界必须 ≥1；返回 0 会让 worker 永远等不到许可而死锁。"""
    cfg = ConcurrencyConfig(min_concurrency=1, large_mix_concurrency=0)
    ctrl = ConcurrencyController(cfg)
    assert ctrl.target_concurrency([P_672]) >= 1


def test_resident_counter_tracks_starts_and_finishes():
    ctrl = ConcurrencyController()
    assert ctrl.resident() == 0
    ctrl.note_start()
    ctrl.note_start()
    assert ctrl.resident() == 2
    ctrl.note_finish(P_224, observed_ms=5.0, predicted_ms=4.9)
    assert ctrl.resident() == 1


def test_resident_never_negative():
    """重复 note_finish 不应把计数压到负数——那会让准入闸门永远放行。"""
    ctrl = ConcurrencyController()
    ctrl.note_finish(P_224, observed_ms=5.0, predicted_ms=4.9)
    assert ctrl.resident() == 0


def test_arrival_rate_respects_window():
    """到达率只在窗口内计数；窗口取空时必须返回 0 而不是抛异常。

    ``demo_sts_switch.py`` 曾因窗口恒为空让判据退化成平凡对照，
    因此这里显式钉死该行为。
    """
    cfg = ConcurrencyConfig(arrival_window_s=1.0)
    ctrl = ConcurrencyController(cfg)
    for t in (0.0, 0.1, 0.2, 0.3):
        ctrl.note_arrival(now=t)
    # now=0.5：cutoff=-0.5，四个都还在窗口内
    assert ctrl.arrival_rate(now=0.5) > 0
    # now=10.0：cutoff=9.0，全部过期 → 0
    assert ctrl.arrival_rate(now=10.0) == 0.0


def test_drift_fallback_disabled_by_default_does_not_trigger_early():
    """样本不足时不得触发回退——倍率在样本少时不可信。"""
    cfg = ConcurrencyConfig(drift_min_samples=8)
    ctrl = ConcurrencyController(cfg)
    for _ in range(3):
        ctrl.note_finish(P_224, observed_ms=30.0, predicted_ms=5.0)  # 6× 偏离
    # 只有 3 个样本 < 8，不应回退
    assert not ctrl._table_is_drifting()
    assert ctrl.target_concurrency([P_224]) == ctrl.config.small_mix_concurrency


def test_drift_fallback_triggers_and_is_conservative():
    """实测远超预测时，停止依赖尺寸混合，退回保守共驻度。

    这是本课题特色项：标定表在小请求上不可信（§11–13），
    因此**不查表**必须是显式的一条路径，而不是静默地用一张坏表决策。
    """
    cfg = ConcurrencyConfig(drift_min_samples=8, drift_threshold=1.5)
    ctrl = ConcurrencyController(cfg)
    for _ in range(10):
        ctrl.note_finish(P_224, observed_ms=30.0, predicted_ms=5.0)  # 6× > 1.5
    assert ctrl._table_is_drifting()
    target = ctrl.target_concurrency([P_224, P_224, P_224, P_224])
    # 关键：即使队列全是小请求，也不再用 small_mix_concurrency
    assert target != cfg.small_mix_concurrency
    assert target == cfg.large_mix_concurrency
    assert "drift_fallback" in ctrl.last_reason()


def test_drift_is_evaluated_per_size_not_pooled():
    """漂移必须按尺寸分档统计后取最差档。

    本课题核心发现是**可复现性有尺寸依赖**（224 极差 1.936×、672 仅 1.001×）。
    若把尺寸混在一起取中位，小尺寸的失真会被大尺寸的好数据平均掉，
    回退永远不触发——这会让特色项静默失效。
    """
    cfg = ConcurrencyConfig(drift_min_samples=8, drift_threshold=1.5)
    ctrl = ConcurrencyController(cfg)
    # 224 严重失真
    for _ in range(10):
        ctrl.note_finish(P_224, observed_ms=30.0, predicted_ms=5.0)
    # 672 完全准确，且样本更多——若池化会把总体中位拉回正常
    for _ in range(40):
        ctrl.note_finish(P_672, observed_ms=9.5, predicted_ms=9.5)
    assert ctrl._table_is_drifting(), "按尺寸分档应能抓住 224 的失真"
