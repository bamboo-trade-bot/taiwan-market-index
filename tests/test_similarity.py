import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "market-index"))

import similarity as sim  # noqa: E402


def _panel(closes: list[list[float]], opens: list[float]) -> sim.Panel:
    arr = np.array(closes, dtype=float)
    return sim.Panel(
        days=pd.date_range("2024-01-01", periods=arr.shape[0], freq="B"),
        closes=arr,
        opens=np.array(opens, dtype=float),
        minutes=np.arange(arr.shape[1]) * 15.0,
    )


def test_pct_is_relative_to_real_open():
    panel = _panel([[101.0, 102.0, 103.0]], [100.0])
    # 原型用第一根 bar 收盤當基準，第一根永遠是 0；這裡要保留開盤後那段
    assert panel.pct()[0].tolist() == [1.0, 2.0, 3.0]


def test_bars_done_counts_only_finished_bars():
    panel = _panel([[1.0] * 19], [1.0])
    assert panel.bars_done_at("09:00") == 0
    assert panel.bars_done_at("09:15") == 1   # 09:00 那根剛走完
    assert panel.bars_done_at("10:30") == 6
    assert panel.bars_done_at("13:30") == 18


def test_neighbors_prefers_similar_shape_and_dedupes_adjacent_days():
    rng = np.random.default_rng(0)
    rows = [[100 + a, 100 + b, 100.0] for a, b in rng.normal(0, 1.0, size=(300, 2))]
    rows += [[100.3, 100.6, 100.0]] * 3      # 相鄰三天與 query 完全相同
    rows += [[100.3, 100.6, 100.0]]          # query
    panel = _panel(rows, [100.0] * len(rows))
    pos = len(rows) - 1

    idx, dist = sim.neighbors(panel, pos, 2, sim.MatchConfig(top_n=3, dedupe_gap=0))
    assert set(idx) == {pos - 1, pos - 2, pos - 3} and dist.max() < 1e-9

    idx, _ = sim.neighbors(panel, pos, 2, sim.MatchConfig(top_n=3, dedupe_gap=5))
    assert len(idx) == 3
    assert all(abs(a - b) > 5 for i, a in enumerate(idx) for b in idx[i + 1:])
    assert all(i < pos for i in idx)          # 絕不看到當天或未來


def test_forecast_is_anchored_to_today_and_keeps_actual_path():
    panel = _panel([[100.5, 101.0, 102.0], [100.1, 100.2, 100.3]], [100.0, 100.0])
    paths = sim.forecast_paths(panel, pos=1, rows=np.array([0]), bars_done=2)
    today_pct = panel.pct()[1]
    assert paths[0, :2].tolist() == today_pct[:2].tolist()      # 截止前照今天實際走勢
    # 歷史日第 3 根比第 2 根多 1.0%（102 vs 101），所以預估 = 今天現值 + 0.99%
    assert paths[0, 2] == pytest.approx(today_pct[1] + (102.0 / 101.0 - 1) * 100, abs=0.02)


def test_dtw_matches_naive_implementation():
    rng = np.random.default_rng(1)
    q, h = rng.normal(size=7), rng.normal(size=(3, 7))

    def naive(a, b):
        d = np.full((len(a) + 1, len(b) + 1), np.inf)
        d[0, 0] = 0
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i, j] = abs(a[i - 1] - b[j - 1]) + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
        return d[-1, -1] / len(a)

    got = sim.dtw(q, h)
    assert got == pytest.approx([naive(q, row) for row in h])
