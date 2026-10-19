r"""找出「當天江波走勢和今天相似」的歷史日，列出它們隔天的江波走勢。

    ..\.venv\Scripts\python.exe similar_days.py --market otc --chart
    ..\.venv\Scripts\python.exe similar_days.py --market otc --shape-weight 0.8 --explain
    ..\.venv\Scripts\python.exe similar_days.py --market otc --weights 量能=2 融資=2 位階=0

比重分配：
    形狀（預設 60%）當天的江波走勢。1 分 K 聚合成 5 分鐘一點、換算成相對開盤 %，再算曲線距離；
                    A 轉、V 轉與轉折幅度都在形狀裡，不需要另外定義。
    數據（預設 40%）分成四組，組間可調、組內等權（--weights 調整，0 = 不看）：
                    量能（量比、成交金額變化）、融資（當日與 20 日增減）、
                    波動（20 日波動率、日內振幅）、位階（均線距離、5 日漲跌、跳空）。

    形狀與每一組都先用歷史分布標準化後才加權相加，否則尺度不同、權重形同虛設。
    漲跌幅、實體、上午／下午漲跌沒有放進數據組：那些資訊江波形狀裡已經有了，
    放進去等於同一件事算兩次（它們和漲跌幅的相關係數高達 0.95~0.99）。

距離算法（--metric）：
    euclid 逐點比較，對「同一時間在做什麼」敏感（預設）。
    dtw    允許時間前後錯開，形狀像但時間點不同也能配對（例如都 V 轉但一個早一小時）。

其他：
    * 相鄰交易日視為同一段行情，預設 5 日內只留最相似的一天。
    * 只會用到查詢日之前的資料（標準化用的平均與標準差也是）。
    * 報表會標出每天的轉折型態與幅度：最高／最低出現的時間、從高點回落多少、從低點反彈多少。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

import daily_features as feat
import similarity as sim
import store

DATA_GROUPS: dict[str, list[str]] = {
    "量能": ["turnover_ratio", "turnover_chg"],
    "融資": ["margin_chg_pct", "margin_chg20_pct"],
    "波動": ["vol20", "range_pct"],
    "位階": ["ma20_dist", "ret5", "gap_pct"],
}
DEFAULT_GROUP_WEIGHTS = {name: 1.0 for name in DATA_GROUPS}
DEFAULT_SHAPE_WEIGHT = 0.6
MIN_HISTORY = 250
DEDUPE_GAP = 5
SHAPE_BAR = "5min"


@dataclass(frozen=True)
class Profile:
    """一天的江波走勢摘要。"""

    turn: str          # 型態：V 轉 / A 轉 / 趨勢上 / 趨勢下 / 盤整
    close_pct: float   # 收盤相對開盤
    high_pct: float    # 最高相對開盤
    low_pct: float     # 最低相對開盤
    high_at: str       # 最高出現時間
    low_at: str        # 最低出現時間
    rebound: float     # 從最低點反彈到收盤（V 轉幅度）
    pullback: float    # 從最高點回落到收盤（A 轉幅度）

    def describe(self) -> str:
        return (f"{self.turn} 收{self.close_pct:+.2f}% "
                f"高{self.high_pct:+.2f}%({self.high_at}) 低{self.low_pct:+.2f}%({self.low_at}) "
                f"反彈{self.rebound:.2f}% 回落{self.pullback:.2f}%")


@dataclass(frozen=True)
class Match:
    date_t: pd.Timestamp
    date_t1: pd.Timestamp
    score: float
    profile_t: Profile
    profile_t1: Profile
    gap_t1: float      # T+1 開盤相對 T 收盤
    ret_t1: float      # T+1 收盤相對 T 收盤
    ret_t: float       # T 收盤相對前一日收盤
    turnover_ratio_t: float


def profile_day(bars: pd.DataFrame) -> Profile:
    """從一天的 1 分 K 算出江波型態。

    轉折判斷：最低點落在盤中且之後反彈夠多 → V 轉；最高點落在盤中且之後回落夠多 → A 轉。
    「夠多」定義為當日振幅的一半以上，避免把小波動也叫成轉折。
    """
    open_price = float(bars["open"].iloc[0])
    close = bars["close"].to_numpy(float)
    times = bars["datetime"].dt.strftime("%H:%M").to_numpy()
    pct = (close - open_price) / open_price * 100

    hi, lo = int(pct.argmax()), int(pct.argmin())
    span = pct[hi] - pct[lo]
    rebound = pct[-1] - pct[lo]
    pullback = pct[hi] - pct[-1]
    # 盤中：不在開盤或收盤前 30 分鐘
    mid = range(30, len(pct) - 30)
    v_turn = lo in mid and rebound >= span * 0.5
    a_turn = hi in mid and pullback >= span * 0.5

    if v_turn and not a_turn:
        turn = "V 轉"
    elif a_turn and not v_turn:
        turn = "A 轉"
    elif v_turn and a_turn:
        turn = "V 轉" if rebound >= pullback else "A 轉"
    elif span < 0.5:
        turn = "盤整"
    else:
        turn = "趨勢上" if pct[-1] > 0 else "趨勢下"

    return Profile(turn=turn, close_pct=float(pct[-1]), high_pct=float(pct[hi]), low_pct=float(pct[lo]),
                   high_at=str(times[hi]), low_at=str(times[lo]),
                   rebound=float(rebound), pullback=float(pullback))


def find_similar(market: str, date: pd.Timestamp | None = None, top_n: int = 8,
                 metric: str = "euclid", shape_weight: float = DEFAULT_SHAPE_WEIGHT,
                 group_weights: dict[str, float] | None = None,
                 start: str = "2016-10-18") -> tuple[pd.Timestamp, list[Match], pd.DataFrame]:
    panel = sim.build_panel(market, bar=SHAPE_BAR, start=start)
    shape_days = pd.DatetimeIndex(panel.days)
    group_weights = group_weights or DEFAULT_GROUP_WEIGHTS
    groups = {name: cols for name, cols in DATA_GROUPS.items() if group_weights.get(name, 0) > 0}
    columns = [c for cols in groups.values() for c in cols]

    frame = feat.build_features(market, start=start)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"沒有這些特徵：{', '.join(missing)}")
    stats = frame.dropna(subset=columns) if columns else frame

    days = shape_days.intersection(stats.index)
    query_date = pd.Timestamp(date) if date is not None else days[-1]
    if query_date not in days:
        raise SystemExit(f"{query_date.date()} 沒有完整資料（可能是休市或分 K 不全）")
    pos = days.get_loc(query_date)
    if pos < MIN_HISTORY:
        raise SystemExit(f"{query_date.date()} 之前只有 {pos} 天，不足 {MIN_HISTORY} 天")

    shape_pct = pd.DataFrame(panel.pct(), index=shape_days).loc[days].to_numpy()
    dist, contrib = _distance(metric, shape_weight, shape_pct, stats.loc[days],
                              groups, group_weights, pos)

    daily = store.read(market, "daily").set_index("date").sort_index()
    daily.index = pd.to_datetime(daily.index)
    minutes = store.read(market, "1min")
    minutes["day"] = minutes["datetime"].dt.normalize()
    by_day = {day: g.sort_values("datetime") for day, g in minutes.groupby("day")}

    picked: list[int] = []
    for cand in np.argsort(dist, kind="stable"):
        if cand >= pos or any(abs(cand - kept) <= DEDUPE_GAP for kept in picked):
            continue
        if _next_trading_day(daily, days[cand]) not in by_day:
            continue
        picked.append(int(cand))
        if len(picked) == top_n:
            break

    matches = [_describe(daily, by_day, days[i], float(dist[i])) for i in picked]
    explain = contrib.iloc[picked].set_index(pd.Index([days[i].date() for i in picked]))
    return query_date, matches, explain


def _distance(metric: str, shape_weight: float, shape_pct: np.ndarray, stats: pd.DataFrame,
              groups: dict[str, list[str]], group_weights: dict[str, float], pos: int):
    """距離 = 形狀 × shape_weight + 各數據組 × 分到的權重。

    每一項都先用「查詢日之前的日子」的分布標準化，否則尺度不同、權重形同虛設。
    回傳 (距離, 各項占比表)。
    """
    parts: dict[str, np.ndarray] = {}
    if shape_weight > 0:
        query, history = shape_pct[pos], shape_pct
        if metric == "dtw":
            parts["形狀"] = sim.dtw(query, history, window=3)
        elif metric == "euclid":
            parts["形狀"] = sim.euclid(query, history)
        else:
            raise SystemExit(f"未知的 metric {metric!r}")

    for name, cols in groups.items():
        values = stats[cols].to_numpy(float)
        history, query = values[:pos], values[pos]
        mean = history.mean(axis=0)
        std = np.where(history.std(axis=0) > 1e-9, history.std(axis=0), 1.0)
        # 組內等權：標準化後取平均平方差
        parts[name] = np.sqrt((((values - mean) / std - (query - mean) / std) ** 2).mean(axis=1))

    data_total = sum(group_weights.get(name, 0.0) for name in groups)
    if shape_weight <= 0 and data_total <= 0:
        raise SystemExit("形狀權重與數據權重不能同時為 0")

    weights = {"形狀": shape_weight}
    for name in groups:
        weights[name] = (1 - shape_weight) * group_weights.get(name, 0.0) / data_total if data_total else 0.0

    scaled = {}
    for name, values in parts.items():
        baseline = values[:pos]
        z = (values - baseline.mean()) / max(baseline.std(), 1e-9)
        scaled[name] = (z - z.min()) * weights[name]     # 平移到 ≥0，方便看占比

    table = pd.DataFrame(scaled, index=stats.index)
    contrib = table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return table.sum(axis=1).to_numpy(), contrib


def _next_trading_day(daily: pd.DataFrame, day: pd.Timestamp) -> pd.Timestamp | None:
    later = daily.index[daily.index > day]
    return later[0] if len(later) else None


def _describe(daily: pd.DataFrame, by_day: dict, date_t: pd.Timestamp, score: float) -> Match:
    date_t1 = _next_trading_day(daily, date_t)
    close_t, close_prev = float(daily.loc[date_t, "close"]), float(daily["close"].shift().loc[date_t])
    open_t1 = float(daily.loc[date_t1, "open"])
    return Match(
        date_t=date_t, date_t1=date_t1, score=score,
        profile_t=profile_day(by_day[date_t]),
        profile_t1=profile_day(by_day[date_t1]),
        gap_t1=(open_t1 / close_t - 1) * 100,
        ret_t1=(float(daily.loc[date_t1, "close"]) / close_t - 1) * 100,
        ret_t=(close_t / close_prev - 1) * 100,
        turnover_ratio_t=float(daily.loc[date_t, "turnover"] / daily["turnover"].shift().rolling(20).mean().loc[date_t]),
    )


def summarise(matches: list[Match]) -> dict:
    ret = np.array([m.ret_t1 for m in matches])
    gap = np.array([m.gap_t1 for m in matches])
    oc = np.array([m.profile_t1.close_pct for m in matches])
    turns = pd.Series([m.profile_t1.turn for m in matches]).value_counts().to_dict()
    return {"up": int((ret > 0).sum()), "down": int((ret < 0).sum()),
            "mean": float(ret.mean()), "median": float(np.median(ret)),
            "gap_mean": float(gap.mean()), "oc_mean": float(oc.mean()),
            "oc_up": int((oc > 0).sum()), "turns": turns}


def print_report(market: str, query_date: pd.Timestamp, matches: list[Match], today: Profile) -> None:
    print(f"{market} 查詢日 {query_date.date()}　今天的江波：{today.describe()}\n")
    print(f"{'相似日(T)':11s} {'距離':>6s}  {'T 日江波':38s} | {'隔日(T+1)':11s} {'跳空':>7s} {'T+1 江波'}")
    print("-" * 132)
    for m in matches:
        print(f"{m.date_t.date()!s:11s} {m.score:6.3f}  {m.profile_t.describe():38s} | "
              f"{m.date_t1.date()!s:11s} {m.gap_t1:+6.2f}% {m.profile_t1.describe()}")

    s = summarise(matches)
    print(f"\n隔日收盤（vs T 收盤）：{s['up']} 漲 / {s['down']} 跌，平均 {s['mean']:+.2f}%、中位 {s['median']:+.2f}%")
    print(f"隔日開盤後到收盤：平均 {s['oc_mean']:+.2f}%（{s['oc_up']}/{len(matches)} 天為正）；跳空平均 {s['gap_mean']:+.2f}%")
    print("隔日江波型態：" + "、".join(f"{k} {v} 天" for k, v in s["turns"].items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", default="otc", choices=list(store.MARKETS))
    parser.add_argument("--date", help="查詢日（預設為最新一個交易日）")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--metric", default="euclid", choices=["euclid", "dtw"])
    parser.add_argument("--shape-weight", type=float, default=DEFAULT_SHAPE_WEIGHT,
                        help=f"當天江波形狀占的比重 0~1（預設 {DEFAULT_SHAPE_WEIGHT}），其餘分給數據組")
    parser.add_argument("--weights", nargs="+", metavar="組名=權重",
                        help=f"數據組之間怎麼分（0 = 不看）。組別：{'、'.join(DATA_GROUPS)}")
    parser.add_argument("--explain", action="store_true", help="印出形狀與各數據組實際占多少距離")
    parser.add_argument("--start", default="2016-10-18")
    parser.add_argument("--chart", nargs="?", const="auto", help="輸出江波走勢圖")
    args = parser.parse_args()

    group_weights = dict(DEFAULT_GROUP_WEIGHTS)
    for item in args.weights or []:
        name, value = _parse_weight(item)
        if name not in DATA_GROUPS:
            raise SystemExit(f"沒有這個組別 {name!r}，可用：{'、'.join(DATA_GROUPS)}")
        group_weights[name] = value
    query_date, matches, explain = find_similar(
        args.market, args.date, args.top, args.metric, args.shape_weight,
        group_weights, args.start)

    minutes = store.read(args.market, "1min")
    today_bars = minutes[minutes["datetime"].dt.normalize() == query_date].sort_values("datetime")
    print_report(args.market, query_date, matches, profile_day(today_bars))

    if args.explain and explain is not None:
        print("\n各項實際占距離的比例（%，越小＝那一項和今天越接近）：")
        print(explain.round(1).to_string())
        print("平均：" + "、".join(f"{k} {v:.1f}%" for k, v in explain.mean().items()))

    if args.chart:
        from similar_days_chart import plot_matches
        path = None if args.chart == "auto" else args.chart
        print("\n圖表：" + plot_matches(args.market, query_date, matches, path))
    return 0


def _parse_weight(text: str) -> tuple[str, float]:
    name, _, value = text.partition("=")
    if not value:
        raise SystemExit(f"權重格式應該是 名稱=數字，收到 {text!r}")
    return name, float(value)


if __name__ == "__main__":
    raise SystemExit(main())
