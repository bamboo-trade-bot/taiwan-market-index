r"""收盤後預測隔日：walk-forward 回測。

    ..\.venv\Scripts\python.exe backtest_daily.py --market otc
    ..\.venv\Scripts\python.exe backtest_daily.py --market otc --target y_oc --csv results/otc_t1.csv

每一天都只用「那天以前」的資料擬合模型（含標準化用的平均與標準差），再預測隔日，
所以沒有任何前視偏誤。對照基準刻意設得務實：

    base   過去所有日子的平均漲跌（也就是大盤的自然偏多/偏空），贏不過它等於沒有資訊
    ret1   只看今天漲跌幅的單變數線性（動能 / 反轉）
    knn    特徵標準化後取最相似的 K 天，用它們隔日的平均當預測（原型的做法，但尺度修好）
    ridge  全部特徵的脊迴歸

命中率只計實際漲跌 ≥0.1% 的日子，後面的 ± 是 95% 信賴區間；
「每次報酬」是照預測方向做一天的平均報酬（未扣成本）。
"""
from __future__ import annotations

import argparse
from math import erfc, sqrt

import numpy as np
import pandas as pd

import daily_features as feat

FLAT = 0.1          # 實際漲跌小於這個 % 不計入方向命中率
MIN_HISTORY = 500   # 至少兩年才開始預測
RIDGE_ALPHA = 10.0
KNN_K = 50


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray, alpha: float) -> float:
    """標準化後的脊迴歸（截距不罰）。"""
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    z_train, z_new = (x_train - mean) / std, (x_new - mean) / std

    design = np.column_stack([np.ones(len(z_train)), z_train])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return float(beta[0] + z_new @ beta[1:])


def knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_new: np.ndarray, k: int) -> float:
    """特徵先用歷史的平均與標準差標準化，再取歐氏距離最近的 k 天。"""
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    dist = np.sqrt((((x_train - mean) / std - (x_new - mean) / std) ** 2).mean(axis=1))
    idx = np.argsort(dist, kind="stable")[:k]
    return float(y_train[idx].mean())


def walk_forward(frame: pd.DataFrame, target: str, columns: list[str]) -> pd.DataFrame:
    data = frame.dropna(subset=columns + [target])
    x_all = data[columns].to_numpy(float)
    y_all = data[target].to_numpy(float)
    ret1_col = columns.index("ret1")

    rows = []
    for pos in range(MIN_HISTORY, len(data)):
        x_train, y_train = x_all[:pos], y_all[:pos]
        rows.append({
            "date": data.index[pos],
            "actual": y_all[pos],
            "base": y_train.mean(),
            "ret1": ridge_predict(x_train[:, [ret1_col]], y_train, x_all[pos, [ret1_col]], alpha=1.0),
            "knn": knn_predict(x_train, y_train, x_all[pos], k=KNN_K),
            "ridge": ridge_predict(x_train, y_train, x_all[pos], alpha=RIDGE_ALPHA),
        })
    return pd.DataFrame(rows).set_index("date")


def summarise(preds: pd.DataFrame, models: list[str], costs: tuple[float, ...] = (0.0, 0.02, 0.05)) -> pd.DataFrame:
    graded = preds[preds["actual"].abs() >= FLAT]
    base_ok = (np.sign(graded["base"]) == np.sign(graded["actual"])).to_numpy()

    rows = []
    for name in models:
        ok = (np.sign(graded[name]) == np.sign(graded["actual"])).to_numpy()
        hit = ok.mean()
        ret = np.sign(preds[name]) * preds["actual"]
        half = len(preds) // 2
        halves = [
            (np.sign(part[name]) == np.sign(part["actual"]))[part["actual"].abs() >= FLAT].mean() * 100
            for part in (preds.iloc[:half], preds.iloc[half:])
        ]
        b = int((ok & ~base_ok).sum())
        c = int((~ok & base_ok).sum())
        z = (abs(b - c) - 1) / sqrt(b + c) if b + c else 0.0
        base_ret = np.sign(preds["base"]) * preds["actual"]
        diff = ret - base_ret
        diff_t = float(diff.mean() / (diff.std() / sqrt(len(diff)))) if diff.std() > 0 else 0.0
        row = {
            "模型": name,
            "做空比例": f"{(preds[name] < 0).mean() * 100:.0f}%",
            "命中率": f"{hit * 100:.1f}±{1.96 * sqrt(hit * (1 - hit) / len(ok)) * 100:.1f}%",
            "前半/後半": f"{halves[0]:.1f}/{halves[1]:.1f}%",
            "相關": round(float(np.corrcoef(preds[name], preds["actual"])[0, 1]), 3),
            "MAE": round(float((preds[name] - preds["actual"]).abs().mean()), 3),
            "vs base p": "—" if name == "base" else f"{erfc(z / sqrt(2)):.3f}",
            "t值": round(float(ret.mean() / (ret.std() / sqrt(len(ret)))), 2),
            "超越base": f"{diff.mean():+.4f}%",
            "超越base t值": round(diff_t, 2),
        }
        for cost in costs:
            net = ret - cost
            row[f"每次淨報酬({cost:g}%成本)"] = f"{net.mean():+.4f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", default="otc")
    parser.add_argument("--target", default="y_cc", choices=feat.TARGETS)
    parser.add_argument("--start", default="2016-10-18")
    parser.add_argument("--csv")
    args = parser.parse_args()

    frame = feat.build_features(args.market, start=args.start)
    columns = feat.feature_names(frame)
    preds = walk_forward(frame, args.target, columns)

    print(f"{args.market} 目標 {args.target}：特徵 {len(columns)} 個、"
          f"測試 {len(preds)} 天（{preds.index[0].date()} ~ {preds.index[-1].date()}）")
    print(f"特徵：{', '.join(columns)}")
    graded = preds[preds['actual'].abs() >= FLAT]
    print(f"實際漲跌 ≥{FLAT}% 的天數 {len(graded)}；上漲比例 {(graded['actual'] > 0).mean() * 100:.1f}%；"
          f"平均絕對漲跌 {preds['actual'].abs().mean():.3f}%\n")
    print(summarise(preds, ["base", "ret1", "knn", "ridge"]).to_string(index=False))

    if args.csv:
        preds.to_csv(args.csv)
        print(f"\n逐日預測已存到 {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
