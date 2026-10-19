r"""相似度比對的 walk-forward 回測：它預測「截止後到收盤」的方向，比基準好嗎？

    ..\.venv\Scripts\python.exe backtest_similarity.py                      預設網格
    ..\.venv\Scripts\python.exe backtest_similarity.py --cuts 10:30 12:00
    ..\.venv\Scripts\python.exe backtest_similarity.py --csv out.csv

每一天都只用「那天以前」的歷史找相似日，取相似日「截止後到收盤」漲跌幅的平均當預測。
對照的基準刻意設得務實：用過去所有日子同一時段尾盤漲跌的平均（也就是大盤的自然偏多/偏空），
贏不過這個基準就等於沒有資訊。命中率附 95% 信賴區間，區間重疊就不能說有差。
"""
from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

import similarity as sim

FLAT = 0.05  # 實際漲跌小於這個 % 的日子不計入方向命中率


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def _ci(hits: np.ndarray) -> tuple[float, float]:
    """命中率與 95% 信賴區間半寬（百分點）。"""
    if len(hits) == 0:
        return float("nan"), float("nan")
    p = float(hits.mean())
    return p * 100, 1.96 * np.sqrt(p * (1 - p) / len(hits)) * 100


def evaluate(panel: sim.Panel, cut_bars: int, config: sim.MatchConfig) -> dict:
    closes, pct = panel.closes, panel.pct()
    actual_all = (closes[:, -1] / closes[:, cut_bars - 1] - 1) * 100.0

    rows = []
    for pos in range(config.min_history, len(panel.days)):
        idx, _ = sim.neighbors(panel, pos, cut_bars, config)
        if len(idx) == 0:
            continue
        rem = sim.remainder_pct(panel, idx, cut_bars)
        rows.append({
            "day": panel.days[pos],
            "pred": rem.mean(),
            "actual": actual_all[pos],
            "baseline": actual_all[:pos].mean(),   # 只用過去的大盤偏向
            "up_votes": int((rem > 0).sum()),
            "n_votes": len(rem),
            "morning": pct[pos, cut_bars - 1],
        })
    df = pd.DataFrame(rows)

    graded = df[df["actual"].abs() >= FLAT]
    hit, hit_ci = _ci((np.sign(graded["pred"]) == np.sign(graded["actual"])).to_numpy())
    base, base_ci = _ci((np.sign(graded["baseline"]) == np.sign(graded["actual"])).to_numpy())

    # 共識比例 ≥80%（原本寫「少於 1 票反對」，相似日取 50 天時幾乎不可能成立）
    share = graded["up_votes"] / graded["n_votes"]
    strong = graded[(share >= 0.8) | (share <= 0.2)]
    strong_hit, strong_ci = _ci((np.sign(strong["pred"]) == np.sign(strong["actual"])).to_numpy())

    pear = float(np.corrcoef(df["pred"], df["actual"])[0, 1])
    spear = float(np.corrcoef(_rank(df["pred"].to_numpy()), _rank(df["actual"].to_numpy()))[0, 1])
    # 和「只看上午漲跌」比：相似日有沒有提供上午走勢以外的資訊
    morning_corr = float(np.corrcoef(df["morning"], df["actual"])[0, 1])

    half = len(df) // 2
    halves = []
    for part in (df.iloc[:half], df.iloc[half:]):
        g = part[part["actual"].abs() >= FLAT]
        halves.append(_ci((np.sign(g["pred"]) == np.sign(g["actual"])).to_numpy())[0])

    return {
        "測試天數": len(graded),
        "命中率": f"{hit:.1f}±{hit_ci:.1f}%",
        "基準": f"{base:.1f}±{base_ci:.1f}%",
        "前半/後半": f"{halves[0]:.1f}/{halves[1]:.1f}%",
        "強共識天數": len(strong),
        "強共識命中": f"{strong_hit:.1f}±{strong_ci:.1f}%",
        "相關(pearson)": round(pear, 3),
        "相關(spearman)": round(spear, 3),
        "上午漲跌相關": round(morning_corr, 3),
        "_detail": df,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", default="otc", choices=list(sim.store.MARKETS))
    parser.add_argument("--bar", default="15min")
    parser.add_argument("--start", default="2016-10-18", help="研究範圍起始日（預設近 10 年）")
    parser.add_argument("--min-history", type=int, default=250, help="歷史不足這麼多天就不比")
    parser.add_argument("--cuts", nargs="+", default=["10:30", "12:00"])
    parser.add_argument("--metrics", nargs="+", default=["euclid", "dtw"])
    parser.add_argument("--normalize", nargs="+", default=["level", "shape"])
    parser.add_argument("--top-n", nargs="+", type=int, default=[8, 20])
    parser.add_argument("--dedupe", nargs="+", type=int, default=[5])
    parser.add_argument("--csv")
    args = parser.parse_args()

    panel = sim.build_panel(args.market, bar=args.bar, start=args.start)
    print(f"{args.market}: {len(panel.days)} 個交易日 {panel.days[0].date()} ~ {panel.days[-1].date()}，"
          f"每天 {panel.n_bars} 根 {args.bar} bar", flush=True)

    results = []
    combos = list(itertools.product(args.cuts, args.metrics, args.normalize, args.top_n, args.dedupe))
    for cut, metric, norm, top_n, dedupe in combos:
        config = sim.MatchConfig(top_n=top_n, metric=metric, normalize=norm, dedupe_gap=dedupe,
                                 min_history=args.min_history)
        stats = evaluate(panel, panel.bars_done_at(cut), config)
        detail = stats.pop("_detail")
        results.append({"截止": cut, "距離": metric, "標準化": norm, "相似日": top_n, "去重": dedupe, **stats})
        print(f"  跑完 {cut} {metric}/{norm} top{top_n} dedupe{dedupe}", flush=True)
        if args.csv and len(combos) == 1:
            detail.to_csv(args.csv, index=False)

    table = pd.DataFrame(results)
    print("\n" + table.to_string(index=False))
    if args.csv and len(combos) > 1:
        table.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
