r"""收盤後預測隔日：特徵與目標。

每一列代表一個交易日 T，特徵只用 T 日收盤（含）之前就知道的資訊，目標是 T+1 的走勢。
所有「相對水位」類的特徵（量比、均線距離）都用「T 日之前」的滾動窗口，不含 T 自己，
避免用到當天才知道的統計量。

資料來源：daily（開高低收量額）、margin（融資融券）、1min（盤中分段強弱）。

目標（%）：
    y_cc  T+1 收盤 / T 收盤 - 1      最常用的「明天漲跌」
    y_oc  T+1 收盤 / T+1 開盤 - 1    開盤進、收盤出，比較接近可執行的交易
    y_gap T+1 開盤 / T 收盤 - 1      跳空
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import store

TARGETS = ["y_cc", "y_oc", "y_gap"]


def build_features(market: str, start: str | pd.Timestamp = "2016-10-18") -> pd.DataFrame:
    daily = _daily(market)
    frame = pd.concat([daily, _margin(market), _intraday(market)], axis=1)

    # 目標：T+1 相對 T 的走勢
    close, open_ = daily["_close"], daily["_open"]
    frame["y_cc"] = (close.shift(-1) / close - 1) * 100
    frame["y_oc"] = (close.shift(-1) / open_.shift(-1) - 1) * 100
    frame["y_gap"] = (open_.shift(-1) / close - 1) * 100

    frame = frame.loc[pd.Timestamp(start):]
    # 最後一天沒有 T+1，保留給「今天收盤後要預測明天」用；回測時自己 dropna
    return frame


def feature_names(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in TARGETS and not c.startswith("_")]


def _daily(market: str) -> pd.DataFrame:
    df = store.read(market, "daily").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    prev_close = close.shift()
    ret1 = (close / prev_close - 1) * 100

    out = pd.DataFrame(index=df.index)
    # 底線開頭＝不是特徵，只給 build_features 算目標用（feature_names 會排除）
    out["_open"], out["_close"] = open_, close
    out["ret1"] = ret1
    out["ret5"] = (close / close.shift(5) - 1) * 100
    out["ret20"] = (close / close.shift(20) - 1) * 100
    out["range_pct"] = (high - low) / prev_close * 100
    # 收盤位置：1 = 收在當日最高，0 = 收在最低
    span = (high - low).replace(0, np.nan)
    out["close_loc"] = ((close - low) / span).fillna(0.5)
    out["gap_pct"] = (open_ / prev_close - 1) * 100
    out["body_pct"] = (close / open_ - 1) * 100
    out["vol20"] = ret1.rolling(20).std()
    out["ma20_dist"] = (close / close.rolling(20).mean() - 1) * 100
    out["ma60_dist"] = (close / close.rolling(60).mean() - 1) * 100
    # 量比：當日成交金額 / 前 20 日均額（不含當日）
    out["turnover_ratio"] = df["turnover"] / df["turnover"].shift().rolling(20).mean()
    out["turnover_chg"] = (df["turnover"] / df["turnover"].shift() - 1) * 100
    out["streak"] = _streak(np.sign(ret1))
    return out


def _margin(market: str) -> pd.DataFrame:
    df = store.read(market, "margin").copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    out = pd.DataFrame(index=df.index)
    # 用交易所自己的前日餘額換算成 %，不要用前後兩列相減（那會在重設日出現假跳動）
    prev = df["margin_prev"].replace(0, np.nan)
    out["margin_chg_pct"] = df["margin_change"] / prev * 100
    out["margin_chg5_pct"] = (df["margin_change"].rolling(5).sum() / prev) * 100
    out["margin_chg20_pct"] = (df["margin_change"].rolling(20).sum() / prev) * 100
    short_prev = df["short_prev"].replace(0, np.nan)
    out["short_chg_pct"] = df["short_change"] / short_prev * 100
    # 融資餘額相對自己近一年的位置（0~1），看籌碼是偏高還偏低
    balance = df["margin_balance"]
    roll = balance.rolling(240, min_periods=60)
    out["margin_level"] = (balance - roll.min()) / (roll.max() - roll.min())
    out["short_ratio"] = df["short_balance"] / balance * 100
    return out


def _intraday(market: str) -> pd.DataFrame:
    """盤中分段強弱：上午、下午、尾盤各走多少，下午成交金額占比。"""
    df = store.read(market, "1min")
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("datetime").copy()
    df["day"] = df["datetime"].dt.normalize()
    # 從午夜起算的分鐘數：09:00 = 540、12:00 = 720、13:00 = 780
    df["minute"] = (df["datetime"] - df["day"]).dt.total_seconds() / 60

    opens = df.groupby("day")["open"].first()
    closes = df.groupby("day")["close"].last()
    noon = df[df["minute"] == 12 * 60].groupby("day")["close"].last()
    one = df[df["minute"] == 13 * 60].groupby("day")["close"].last()
    pm_turnover = df[df["minute"] >= 12 * 60].groupby("day")["turnover"].sum()
    all_turnover = df.groupby("day")["turnover"].sum()

    out = pd.DataFrame(index=closes.index)
    out["am_ret"] = (noon / opens - 1) * 100
    out["pm_ret"] = (closes / noon - 1) * 100
    out["last30_ret"] = (closes / one - 1) * 100
    out["pm_turnover_share"] = pm_turnover / all_turnover
    return out


def _streak(signs: pd.Series) -> pd.Series:
    """連續上漲（正）或下跌（負）天數。"""
    out = np.zeros(len(signs))
    run = 0.0
    for i, s in enumerate(signs.to_numpy()):
        if np.isnan(s) or s == 0:
            run = 0.0
        elif run != 0 and np.sign(run) == s:
            run += s
        else:
            run = s
        out[i] = run
    return pd.Series(out, index=signs.index)
