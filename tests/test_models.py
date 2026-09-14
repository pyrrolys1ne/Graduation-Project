"""请求模型与剖析表的契约测试。

`EncodeJob.patches` 是性能模型、调度器与 DACC 共用的核心特征。它必须与
`data/profiles/*.csv` 里的 patches 列一致——否则性能模型会查到错误的延迟桶，
而所有基于预测的调度决策都会静默失真。

这类错误不会抛异常、不会报错，只会让预测值偏掉，所以用契约测试钉住。
（2026-09-14 确实发生过一次：`// 32` 被改成 `// 16`，224×224 的预测从 9.00 ms
变成 33.00 ms。）
"""

import csv
from pathlib import Path

import pytest

from encoder_sched.models import EncodeJob
from encoder_sched.performance import PerformanceModel

PROFILE_TABLE = Path(__file__).resolve().parents[1] / "data" / "profiles" / "default.csv"


def make(width: int, height: int) -> EncodeJob:
    return EncodeJob("contract", seed=0, width=width, height=height, deadline_ms=1000)


@pytest.mark.parametrize("size,expected", [(224, 49), (336, 100), (448, 196), (672, 441)])
def test_patches_match_clip_vit_b32_grid(size, expected):
    """CLIP ViT-B/32 的 patch 是 32×32。"""
    assert make(size, size).patches == expected


def test_patches_agree_with_profile_table():
    """剖析表的 patches 列必须能由 EncodeJob.patches 原样复现。

    这是最强的一条约束：表是实验数据，模型是代码，两者一旦不一致，
    预测就会指向错误的桶。
    """
    with PROFILE_TABLE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "剖析表为空"
    mismatches = [
        (row["width"], row["height"], row["patches"], make(int(row["width"]), int(row["height"])).patches)
        for row in rows
        if make(int(row["width"]), int(row["height"])).patches != int(row["patches"])
    ]
    assert not mismatches, f"这些行的 patches 与 EncodeJob.patches 不一致（宽,高,表值,模型值）: {mismatches}"


def test_predictions_reproduce_the_profile_table():
    """预测值必须原样复现剖析表——这是调度器实际依据的数字。

    不断言绝对值贴近实测：`default.csv` 是**先验估计**而非实测（672×672 @1.0 表里
    写 50 ms，本机实测仅 6.62 ms，差 7.5 倍）。表本身是否可信是另一个问题，
    这里只保证"模型读表读对了"。
    """
    model = PerformanceModel(PROFILE_TABLE)
    with PROFILE_TABLE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        patches = int(row["patches"])
        fraction = float(row["sm_fraction"])
        expected = float(row["latency_ms"])
        assert model.predict(patches, fraction) == pytest.approx(expected), (
            f"patches={patches} sm={fraction} 应预测 {expected}，实际 {model.predict(patches, fraction)}"
        )


def test_more_tpcs_never_predicts_slower():
    """配额越高，预测延迟不应变大（表若有倒挂，排序逻辑会失真）。"""
    model = PerformanceModel(PROFILE_TABLE)
    for patches in (49, 100, 196, 441):
        values = [model.predict(patches, q) for q in (0.25, 0.5, 0.75, 1.0)]
        assert values == sorted(values, reverse=True), f"patches={patches} 的配额—延迟关系非单调递减: {values}"


def test_patches_property_is_32_based_source():
    """直接锚定除数，避免有人"顺手"改成 16 之类的值。"""
    assert make(64, 64).patches == 4, "64//32 = 2, 2*2 = 4；若为 16 会得到 16"
