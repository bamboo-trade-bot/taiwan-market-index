r"""用富邦 Neo API 抓加權（TSE）與櫃買（OTC）指數的日 K 與 1 分 K。

    ..\.venv\Scripts\python.exe fetch_fubon.py                 兩個市場、兩種週期，增量更新
    ..\.venv\Scripts\python.exe fetch_fubon.py --market otc --timeframe 1min
    ..\.venv\Scripts\python.exe fetch_fubon.py --full          忽略本地檔案從頭抓

富邦能給的最早資料（2026-09 實測）：
    日 K：加權 2004-10-15、櫃買 2014-01-06
    分 K：兩者皆 2023-05-23
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store  # noqa: E402
from core.datasource.fubon import FubonSource  # noqa: E402

logger = logging.getLogger("fetch_fubon")

# 早於富邦資料起點的區段 API 回 404，candles() 會當成空資料，所以起點抓寬一點無妨
EARLIEST = {"daily": "2004-01-01", "1min": "2023-05-01"}
API_TIMEFRAME = {"daily": "D", "1min": "1"}
DAILY_FIELDS = "open,high,low,close,volume,turnover,change"


def to_frame(raw: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()
    if timeframe == "daily":
        df["date"] = pd.to_datetime(df["date"])
    else:
        # 指數分 K 的 volume 實際上是成交金額（與當日成交金額加總相符），改名避免誤用
        df = df.rename(columns={"date": "datetime", "volume": "turnover"})
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
        # 富邦分 K 由交易所 5 秒快照合成（與 TPEx miIndex 逐根比對一致）
        df["sample_sec"] = 5
    return df


def fetch(source: FubonSource, market: str, timeframe: str, full: bool) -> None:
    symbol = store.MARKETS[market]
    last = None if full else store.last_time(market, timeframe)
    # 從最後一天重抓：若上次是盤中抓的，把那天補完整
    start = last.strftime("%Y-%m-%d") if last is not None else EARLIEST[timeframe]
    end = date.today().strftime("%Y-%m-%d")

    fields = DAILY_FIELDS if timeframe == "daily" else None
    raw = source.candles(symbol, start, end, timeframe=API_TIMEFRAME[timeframe], fields=fields)
    df = to_frame(raw, timeframe)
    if df.empty:
        logger.info("%s %s（%s）%s~%s 沒有新資料", market, timeframe, symbol, start, end)
        return

    key = store.LAYOUTS[timeframe][1]
    total = store.write(market, timeframe, df)
    logger.info(
        "%s %s（%s）抓到 %d 筆 %s ~ %s，檔案共 %d 筆",
        market, timeframe, symbol, len(df), df[key].min(), df[key].max(), total,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", choices=[*store.MARKETS, "all"], default="all")
    parser.add_argument("--timeframe", choices=[*store.LAYOUTS, "all"], default="all")
    parser.add_argument("--full", action="store_true", help="忽略本地資料從頭抓")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    markets = list(store.MARKETS) if args.market == "all" else [args.market]
    timeframes = list(store.LAYOUTS) if args.timeframe == "all" else [args.timeframe]

    with FubonSource() as source:
        for market in markets:
            for timeframe in timeframes:
                fetch(source, market, timeframe, args.full)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
