r"""把相似日畫成「T 日接 T+1 日」的連貫走勢圖。

每一格：T 日 1 分鐘走勢（灰）接 T+1 日（彩），中間藍虛線分界、標示跳空；
Y 軸基準是 T 日開盤（0%）。最下面一列是各相似日 T+1 的 K 棒（相對 T+1 開盤，含上下引線）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import store  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BG, PANEL, GRID, TEXT = "#ffffff", "#f7f7f7", "#cccccc", "#111111"
UP, DOWN, SEP, ZERO, GREY = "#cc2222", "#008844", "#2255bb", "#999999", "#aaaaaa"
COLORS = ["#cc2222", "#cc8800", "#228822", "#1155cc", "#882288", "#008899", "#886600", "#cc4400"]
BARS_PER_DAY = 271


def plot_matches(market: str, query_date: pd.Timestamp, matches: list, path: str | None = None) -> str:
    minutes = store.read(market, "1min")
    minutes["day"] = minutes["datetime"].dt.normalize()
    by_day = {day: g.sort_values("datetime") for day, g in minutes.groupby("day")}
    daily = store.read(market, "daily").set_index("date").sort_index()
    daily.index = pd.to_datetime(daily.index)

    n = len(matches)
    fig = plt.figure(figsize=(13, 2.1 * n + 3.0), facecolor=BG)
    grid = gridspec.GridSpec(n + 1, 1, figure=fig, height_ratios=[1.0] * n + [1.3],
                             hspace=0.55, left=0.06, right=0.98, top=0.955, bottom=0.05)

    today = daily.loc[query_date]
    prev_close = daily["close"].shift().loc[query_date]
    fig.suptitle(
        f"{market.upper()} 相似日的隔天走勢　查詢日 {query_date.date()}（收盤 {today['close']:.2f}，"
        f"{(today['close'] / prev_close - 1) * 100:+.2f}%）\n"
        f"灰線=相似日當天、彩線=它的隔天；Y 軸基準＝相似日開盤（0%），藍虛線落差＝跳空",
        color=TEXT, fontsize=12, fontweight="bold", y=0.985,
    )

    for i, match in enumerate(matches):
        _panel(fig.add_subplot(grid[i, 0]), by_day, match, COLORS[i % len(COLORS)])
    _summary_panel(fig.add_subplot(grid[n, 0]), matches)

    if path is None:
        out = Path(__file__).resolve().parent / "results"
        out.mkdir(exist_ok=True)
        path = str(out / f"{market}_similar_{query_date:%Y%m%d}.jpg")
    fig.savefig(path, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.6)


def _panel(ax, by_day: dict, match, color: str) -> None:
    _style(ax)
    day_t, day_t1 = by_day.get(match.date_t), by_day.get(match.date_t1)
    if day_t is None or day_t1 is None:
        ax.text(0.5, 0.5, f"{match.date_t.date()} 缺分 K", transform=ax.transAxes,
                ha="center", va="center", color=GREY, fontsize=9)
        return

    base = float(day_t["open"].iloc[0])
    pct_t = (day_t["close"].to_numpy() - base) / base * 100
    pct_t1 = (day_t1["close"].to_numpy() - base) / base * 100
    sep = len(pct_t)

    ax.plot(np.arange(sep), pct_t, color=GREY, linewidth=0.9, zorder=2)
    ax.plot(np.arange(sep, sep + len(pct_t1)), pct_t1, color=color, linewidth=1.1, zorder=3)
    ax.axvline(sep - 0.5, color=SEP, linewidth=1.2, linestyle="--", alpha=0.85, zorder=4)
    ax.axhline(0, color=ZERO, linewidth=0.7, alpha=0.5, zorder=1)

    gap_color = UP if match.gap_t1 >= 0 else DOWN
    ax.annotate(f"跳空{match.gap_t1:+.2f}%", xy=(sep - 0.5, (pct_t[-1] + pct_t1[0]) / 2),
                xytext=(sep + 14, (pct_t[-1] + pct_t1[0]) / 2), color=gap_color, fontsize=7,
                va="center", ha="left", zorder=5,
                arrowprops=dict(arrowstyle="->", color=gap_color, lw=0.8),
                bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec=gap_color, lw=0.5, alpha=0.9))

    ax.text(0.995, 0.96, f"T+1 {match.date_t1.date()}　{match.profile_t1.turn}　收 {match.ret_t1:+.2f}%",
            transform=ax.transAxes, color=UP if match.ret_t1 >= 0 else DOWN,
            fontsize=8, ha="right", va="top", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="none", alpha=0.85))
    prof = match.profile_t
    ax.set_title(
        f"T:{match.date_t.date()}　{prof.turn}　收 {prof.close_pct:+.2f}%　"
        f"高 {prof.high_pct:+.2f}%({prof.high_at}) 低 {prof.low_pct:+.2f}%({prof.low_at})　"
        f"反彈 {prof.rebound:.2f}% 回落 {prof.pullback:.2f}%　量比 {match.turnover_ratio_t:.2f}x　距離 {match.score:.3f}",
        color=TEXT, fontsize=8, pad=3, loc="left")
    ax.set_ylabel("%", color=TEXT, fontsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}"))

    ticks, labels = [], []
    for offset, label in ((0, "09:00"), (120, "11:00"), (240, "13:00")):
        ticks += [offset, sep + offset]
        labels += [f"T\n{label}", f"T+1\n{label}"]
    ticks.append(sep + len(pct_t1) - 1)
    labels.append("T+1\n13:30")
    order = np.argsort(ticks)
    ax.set_xticks(np.array(ticks)[order])
    ax.set_xticklabels(np.array(labels)[order], fontsize=6, color=TEXT)
    ax.set_xlim(-3, sep + len(pct_t1) + 3)


def _summary_panel(ax, matches: list) -> None:
    _style(ax)
    closes = np.array([m.profile_t1.close_pct for m in matches])
    for i, match in enumerate(matches):
        prof = match.profile_t1
        color = UP if prof.close_pct >= 0 else DOWN
        bottom, top = sorted((0.0, prof.close_pct))
        height = max(top - bottom, 0.02)
        ax.bar(i, height, bottom=bottom, width=0.42, color=color, edgecolor=color, zorder=3)
        ax.plot([i, i], [max(top, prof.high_pct), min(bottom, prof.low_pct)],
                color=color, linewidth=1.1, zorder=2)

    ax.axhline(0, color=ZERO, linewidth=0.8, zorder=1)
    ax.axhline(closes.mean(), color="#d06010", linewidth=1.4, linestyle="--", zorder=4,
               label=f"開→收平均 {closes.mean():+.2f}%")
    ret = np.array([m.ret_t1 for m in matches])
    ax.set_title(
        f"隔日 K 棒（相對 T+1 開盤，含上下引線）　"
        f"收盤 vs 前一日：{int((ret > 0).sum())} 漲 / {int((ret < 0).sum())} 跌，"
        f"平均 {ret.mean():+.2f}%、中位 {np.median(ret):+.2f}%",
        color=TEXT, fontsize=9, pad=4)
    ax.set_xticks(range(len(matches)))
    ax.set_xticklabels([f"{m.date_t1.date()}\n收{m.ret_t1:+.2f}%" for m in matches], fontsize=7, color=TEXT)
    ax.set_xlim(-0.8, len(matches) - 0.2)
    ax.set_ylabel("% (vs T+1 開盤)", color=TEXT, fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}"))
    ax.legend(loc="upper right", fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
