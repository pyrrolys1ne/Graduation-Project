from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class PerformanceModel:
    REQUIRED_COLUMNS = {"patches", "sm_fraction", "latency_ms"}

    def __init__(self, table_path: str | Path):
        self.table_path = Path(table_path)
        self.table = pd.read_csv(self.table_path)
        missing = self.REQUIRED_COLUMNS - set(self.table.columns)
        if missing:
            raise ValueError(f"剖析表缺少列: {sorted(missing)}")
        if self.table.empty or (self.table["latency_ms"] <= 0).any():
            raise ValueError("剖析表必须包含正延迟样本")
        self._cache: dict[tuple[int, float], float] = {}
        self._curves = {
            float(level): subset.sort_values("patches")[["patches", "latency_ms"]].to_numpy(dtype=float)
            for level, subset in self.table.groupby("sm_fraction")
        }

    @property
    def quota_levels(self) -> tuple[float, ...]:
        return tuple(sorted(float(value) for value in self.table["sm_fraction"].unique()))

    def predict(self, patches: int, sm_fraction: float) -> float:
        key = (int(patches), round(float(sm_fraction), 6))
        if key in self._cache:
            return self._cache[key]
        quotas = np.asarray(self.quota_levels, dtype=float)
        quota = float(np.clip(sm_fraction, quotas.min(), quotas.max()))
        quota_predictions = []
        for level in quotas:
            subset = self._curves[float(level)]
            predicted = np.interp(
                patches,
                subset[:, 0],
                subset[:, 1],
            )
            quota_predictions.append(float(predicted))
        result = float(np.interp(quota, quotas, np.asarray(quota_predictions)))
        self._cache[key] = result
        return result
