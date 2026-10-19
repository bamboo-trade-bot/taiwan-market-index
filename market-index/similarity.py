r"""盤中走勢相似度比對（可回測版）。

比對邏輯：把每天的 1 分 K 聚合成固定長度的 bar（預設 15 分鐘），換算成相對當天
開盤的漲跌幅，再和歷史上「同樣走到這個時間點」的日子比距離，取最近的幾天。

與原型相比修掉四件事：
  1. 標準化基準是當天真正的開盤價，不是第一根 bar 的收盤價。原型的做法會讓第一根
     bar 永遠是 0，開盤後那段走勢完全不進比對，09:15 那次更會讓所有日子距離都是 0。
  2. 只用「已走完」的 bar。盤中當下那根還沒結束，不能拿去和歷史的完整 bar 比。
  3. 預估路徑錨定在今天現價：pred(t) = 今日現價 + (歷史(t) - 歷史(截止點))。
     原型直接畫歷史日的絕對位置，和今天的現價接不起來。
  4. 相似日去重：相鄰交易日是同一段行情，只留其中最相似的一天，避免「8 天裡 6 天漲」
     其實只是同一週重複計算。

所有函式都只吃 query 日之前的資料，方便 walk-forward 回測。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import store

OPEN_TIME = pd.Timedelta(hours=9)
CLOSE_TIME = pd.Timedelta(hours=13, minutes=30)
FULL_DAY_BARS = 271


@dataclass(frozen=True)
class Panel:
    """逐日對齊的盤中走勢矩陣。

    days    : 交易日（遞增）
    closes  : (天數, bar 數) 各 bar 的收盤指數；最後一欄是 13:30 收盤
    opens   : (天數,) 當天開盤指數
    minutes : (bar 數,) 各 bar 起點距 09:00 的分鐘數
    """

    days: pd.DatetimeIndex
    closes: np.ndarray
    opens: np.ndarray
    minutes: np.ndarray

    @property
    def n_bars(self) -> int:
        return self.closes.shape[1]

    def bars_done_at(self, clock: str) -> int:
        """09:00 到 clock（不含）之間已走完幾根 bar。"""
        offset = (pd.Timedelta(f"{clock}:00") - OPEN_TIME).total_seconds() / 60
        return int((self.minutes < offset).sum())

    def pct(self) -> np.ndarray:
        """各 bar 收盤相對當天開盤的漲跌幅（%）。"""
        return (self.closes - self.opens[:, None]) / self.opens[:, None] * 100.0


def build_panel(market: str, bar: str = "15min", min_bars: int = FULL_DAY_BARS,
                start: str | pd.Timestamp | None = None) -> Panel:
    """從 1 分 K 建立 Panel；分 K 不完整的日子直接排除（避免早盤缺漏污染比對）。

    start 之前的日子完全不載入，連「歷史相似日」也不會用到——研究範圍要一致。
    """
    raw = store.read(market, "1min")
    if raw.empty:
        raise RuntimeError(f"{market} 還沒有分 K，請先執行 fetch_fubon.py / backfill_official.py")
    raw = raw.set_index("datetime").sort_index()
    if start is not None:
        raw = raw[raw.index >= pd.Timestamp(start)]

    day_index = raw.index.normalize()
    sizes = day_index.value_counts()
    keep = sizes[sizes >= min_bars].index
    raw = raw[day_index.isin(keep)]

    day = raw.index.normalize()
    time_of_day = raw.index.floor(bar) - day          # 當日時刻，不是絕對時間
    closes = raw.groupby([day, time_of_day])["close"].last().unstack()
    opens = raw.groupby(day)["open"].first()

    minutes = np.array([(t - OPEN_TIME).total_seconds() / 60 for t in closes.columns])
    order = np.argsort(minutes)
    return Panel(
        days=pd.DatetimeIndex(closes.index),
        closes=closes.to_numpy()[:, order],
        opens=opens.to_numpy(),
        minutes=minutes[order],
    )


# ---------------------------------------------------------------- 距離
def _shape(rows: np.ndarray, mode: str) -> np.ndarray:
    """level = 原始漲跌幅；shape = 再除以自身標準差，只比形狀不比波動大小。"""
    if mode == "level":
        return rows
    scale = rows.std(axis=-1, keepdims=True)
    return rows / np.where(scale > 1e-9, scale, 1.0)


def euclid(query: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """逐點歐氏距離（除以 bar 數，方便不同時間點互相比較）。"""
    return np.sqrt(((hist - query) ** 2).mean(axis=1))


def dtw(query: np.ndarray, hist: np.ndarray, window: int = 0) -> np.ndarray:
    """對所有歷史日一次算完的精確 DTW（成本 = 絕對差），window>0 為 Sakoe-Chiba 帶寬。

    序列很短（幾十根 bar），直接做完整 DP 比 fastdtw 的近似更準也夠快。
    """
    n, m = len(query), hist.shape[1]
    big = np.inf
    prev = np.full((hist.shape[0], m + 1), big)
    prev[:, 0] = 0.0
    for i in range(1, n + 1):
        cur = np.full((hist.shape[0], m + 1), big)
        lo, hi = (max(1, i - window), min(m, i + window)) if window else (1, m)
        for j in range(lo, hi + 1):
            cost = np.abs(query[i - 1] - hist[:, j - 1])
            cur[:, j] = cost + np.minimum(np.minimum(prev[:, j], cur[:, j - 1]), prev[:, j - 1])
        prev = cur
        prev[:, 0] = big
    return prev[:, m] / max(n, m)


# ---------------------------------------------------------------- 挑相似日
@dataclass(frozen=True)
class MatchConfig:
    top_n: int = 8
    metric: str = "euclid"      # euclid / dtw
    normalize: str = "level"    # level / shape
    dtw_window: int = 2
    dedupe_gap: int = 5         # 相鄰幾個交易日內只留最相似的一天；0 = 不去重
    min_history: int = 250      # 歷史不足這麼多天就不比


def neighbors(panel: Panel, pos: int, bars_done: int, config: MatchConfig = MatchConfig()):
    """找出 pos 這天「前 bars_done 根 bar」最相似的歷史日。

    回傳 (歷史日的位置索引, 距離)；只會用到 pos 之前的日子。
    """
    if pos < config.min_history or bars_done < 1:
        return np.empty(0, dtype=int), np.empty(0)

    pct = panel.pct()
    query = _shape(pct[pos, :bars_done], config.normalize)
    hist = _shape(pct[:pos, :bars_done], config.normalize)

    if config.metric == "dtw":
        dist = dtw(query, hist, window=config.dtw_window)
    elif config.metric == "euclid":
        dist = euclid(query, hist)
    else:
        raise ValueError(f"未知的 metric {config.metric!r}")

    order = np.argsort(dist, kind="stable")
    if not config.dedupe_gap:
        picked = order[: config.top_n]
    else:
        picked = []
        for cand in order:
            if all(abs(cand - kept) > config.dedupe_gap for kept in picked):
                picked.append(cand)
                if len(picked) == config.top_n:
                    break
        picked = np.array(picked, dtype=int)
    return picked, dist[picked]


def remainder_pct(panel: Panel, rows: np.ndarray, bars_done: int) -> np.ndarray:
    """這些日子從「第 bars_done 根 bar 收盤」到當天收盤的漲跌幅（%）。"""
    cutoff = panel.closes[rows, bars_done - 1]
    return (panel.closes[rows, -1] / cutoff - 1) * 100.0


def forecast_paths(panel: Panel, pos: int, rows: np.ndarray, bars_done: int) -> np.ndarray:
    """錨定在今天現價的預估路徑（相對今天開盤的 %），shape = (相似日數, bar 數)。

    pred(t) = 今日目前漲跌幅 + (歷史(t) - 歷史(截止點))
    截止點之前的部分直接填今天自己的實際走勢，畫圖時才不會出現兩條線。
    """
    pct = panel.pct()
    now = pct[pos, bars_done - 1]
    shifted = pct[rows, :] - pct[rows, bars_done - 1][:, None] + now
    shifted[:, :bars_done] = pct[pos, :bars_done]
    return shifted
