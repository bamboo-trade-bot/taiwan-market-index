r"""收盤後預測隔日：用今天的資料算出明天的預估。

    ..\.venv\Scripts\python.exe predict_daily.py --market otc
    ..\.venv\Scripts\python.exe predict_daily.py --market otc --target y_cc --top 8

模型是脊迴歸（全部特徵、expanding window），與 backtest_daily.py 完全同一套程式碼，
所以「這裡印出來的預測」和「回測評估的那個模型」是同一個東西。

回測結論（櫃買、2019-01 ~ 2026-10、1,878 天）：
    y_oc（隔日開盤進、收盤出）：平均每次 +0.098%、t=3.37，8 年裡 1 年虧損；
                                 方向命中率只有 52~53%，報酬來自「大波動時方向對」。
    y_cc（收盤抱到隔日收盤）  ：贏不過「單純做多」（超越基準 t=0.73，不顯著）。
成本很關鍵：來回成本 0.05% 時 t 值降到 1.4，0.1% 就不賺了。詳見 FINDINGS.md。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import daily_features as feat
from backtest_daily import FLAT, MIN_HISTORY, RIDGE_ALPHA


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    mean = x.mean(axis=0)
    std = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    z = (x - mean) / std
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta, mean, std


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", default="otc")
    parser.add_argument("--target", default="y_oc", choices=feat.TARGETS)
    parser.add_argument("--start", default="2016-10-18")
    parser.add_argument("--top", type=int, default=6, help="列出貢獻最大的幾個特徵")
    args = parser.parse_args()

    frame = feat.build_features(args.market, start=args.start)
    columns = feat.feature_names(frame)

    today = frame.dropna(subset=columns).index[-1]
    train = frame.dropna(subset=columns + [args.target])
    train = train[train.index < today]
    if len(train) < MIN_HISTORY:
        raise SystemExit(f"歷史只有 {len(train)} 天，不足 {MIN_HISTORY} 天")

    x_train = train[columns].to_numpy(float)
    y_train = train[args.target].to_numpy(float)
    beta, mean, std = fit_ridge(x_train, y_train, RIDGE_ALPHA)

    x_today = frame.loc[today, columns].to_numpy(float)
    z_today = (x_today - mean) / std
    contrib = pd.Series(z_today * beta[1:], index=columns)
    prediction = float(beta[0] + contrib.sum())

    target_label = {"y_oc": "隔日開盤進、收盤出", "y_cc": "隔日收盤 vs 今日收盤", "y_gap": "隔日開盤跳空"}[args.target]
    print(f"{args.market}  資料截止 {today.date()}（用 {len(train)} 天歷史訓練）")
    print(f"預測目標：{target_label}")
    print(f"預估：{prediction:+.3f}%（方向：{'偏多' if prediction > 0 else '偏空'}）")

    hist = np.sign(np.array([
        beta[0] + ((train[columns].to_numpy(float)[i] - mean) / std) @ beta[1:]
        for i in range(max(0, len(train) - 250), len(train))
    ]))
    print(f"（同一組係數套在最近 250 天：看多 {int((hist > 0).sum())} 天、看空 {int((hist < 0).sum())} 天）")

    print(f"\n貢獻最大的 {args.top} 個特徵：")
    for name, value in contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(args.top).items():
        print(f"  {name:20s} 今日值 {frame.loc[today, name]:+9.3f}  貢獻 {value:+.3f}%")

    print("\n提醒：回測命中率只有 52~53%，單日預估不可當成確定方向；"
          f"實際漲跌小於 {FLAT}% 的日子在回測裡本來就不計分。來回成本超過 0.1% 就不划算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
