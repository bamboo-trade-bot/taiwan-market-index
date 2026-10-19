"""隔日預測的特徵／目標檢查，重點是「不能用到未來資料」。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "market-index"))

import backtest_daily as bt  # noqa: E402
import daily_features as feat  # noqa: E402


def test_targets_look_one_day_ahead():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    daily = pd.DataFrame({"_open": [10.0, 11.0, 12.0, 13.0], "_close": [10.5, 11.5, 12.5, 13.5]}, index=dates)

    y_cc = (daily["_close"].shift(-1) / daily["_close"] - 1) * 100
    assert y_cc.iloc[0] == pytest.approx((11.5 / 10.5 - 1) * 100)   # T 的目標來自 T+1
    assert np.isnan(y_cc.iloc[-1])                                   # 最後一天沒有答案


def test_feature_names_excludes_targets_and_private_columns():
    frame = pd.DataFrame(columns=["ret1", "_close", "_open", "y_cc", "y_oc", "y_gap"])
    assert feat.feature_names(frame) == ["ret1"]


def test_streak_counts_consecutive_days():
    signs = pd.Series([1, 1, 1, -1, -1, 0, 1], index=pd.date_range("2024-01-01", periods=7))
    assert feat._streak(signs).tolist() == [1, 2, 3, -1, -2, 0, 1]


def test_ridge_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = 3 * x[:, 0] - 2 * x[:, 1]
    pred = bt.ridge_predict(x, y, np.array([1.0, 0.0]), alpha=1e-6)
    assert pred == pytest.approx(3.0, abs=0.05)


def test_knn_uses_only_the_given_history():
    x_train = np.array([[0.0], [1.0], [10.0]])
    y_train = np.array([1.0, 1.0, -99.0])
    # 最近的兩天報酬都是 +1，距離很遠的那天不該被選進來
    assert bt.knn_predict(x_train, y_train, np.array([0.5]), k=2) == pytest.approx(1.0)


def test_walk_forward_never_trains_on_the_predicted_day():
    rng = np.random.default_rng(1)
    n = bt.MIN_HISTORY + 20
    dates = pd.date_range("2016-01-01", periods=n, freq="B")
    frame = pd.DataFrame({"ret1": rng.normal(size=n), "other": rng.normal(size=n)}, index=dates)
    frame["y_cc"] = rng.normal(size=n)
    # 把最後一天的答案換成極端值：若模型偷看當天，預測也會被帶向極端
    frame.loc[dates[-1], "y_cc"] = 1e6

    preds = bt.walk_forward(frame, "y_cc", ["ret1", "other"])
    assert preds.index[-1] == dates[-1]
    assert abs(preds["ridge"].iloc[-1]) < 100
    assert abs(preds["knn"].iloc[-1]) < 100
    assert abs(preds["base"].iloc[-1]) < 100
