r"""用櫃買中心公開資料補櫃買指數（otc）富邦抓不到的歷史。

    ..\.venv\Scripts\python.exe backfill_tpex.py                    日 K + 1 分 K 都補
    ..\.venv\Scripts\python.exe backfill_tpex.py --timeframe daily
    ..\.venv\Scripts\python.exe backfill_tpex.py --timeframe 1min --start 2020-01-01 --limit 5

只補本地檔案裡「沒有的日子」，不覆蓋富邦資料；中斷後重跑會接著補。
    日 K ：1999-09-01 起，只有開高低收與漲跌（volume / turnover 留空）
    1 分 K：2007-07-02 起，交易日曆取自日 K 檔，所以要先補日 K
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import store  # noqa: E402
from core.datasource.tpex import TpexIndexClient  # noqa: E402
from core.datasource.twse import TwseIndexClient  # noqa: E402

logger = logging.getLogger("backfill_official")

CLIENTS = {"otc": TpexIndexClient, "tse": TwseIndexClient}
EARLIEST = {
    "otc": {"daily": "1999-09-01", "1min": "2007-07-02", "margin": "2016-10-18"},
    "tse": {"1min": "2004-10-15", "margin": "2016-10-18"},
}
DAILY_BATCH_MONTHS = 12
INTRADAY_BATCH_DAYS = 20
MARGIN_BATCH_DAYS = 40
BARS_PER_DAY = 271


def backfill_daily(client: TpexIndexClient, market: str, start: pd.Timestamp) -> None:
    """從本地最早一天所在月份往回補到 start，每 12 個月寫一次檔。"""
    existing = store.read(market, "daily")
    have = set(pd.to_datetime(existing["date"]))
    first = min(have) if have else pd.Timestamp.today().normalize()
    months = pd.date_range(start.replace(day=1), first.replace(day=1), freq="MS")[::-1]
    if len(months) == 0 or (have and first <= start):
        logger.info("日 K 已補到 %s，不需要再抓", first.date())
        return

    logger.info("日 K：%s ~ %s 共 %d 個月", months[-1].date(), months[0].date(), len(months))
    batch = []
    for i, month in enumerate(months, 1):
        df = client.daily_month(month)
        batch.append(df[~df["date"].isin(have) & (df["date"] >= start)])
        if i % DAILY_BATCH_MONTHS == 0 or i == len(months):
            new = pd.concat(batch, ignore_index=True)
            if not new.empty:
                total = store.write(market, "daily", new)
                logger.info("日 K 補到 %s，本批 %d 筆，檔案共 %d 筆", new["date"].min().date(), len(new), total)
            batch = []


def backfill_intraday(client, market: str, start: pd.Timestamp, end: pd.Timestamp, limit: int | None) -> int:
    todo = _missing_days(market, "1min", start, end, limit)
    if not todo:
        logger.info("%s 1 分 K：%s ~ %s 沒有缺的交易日", market, start.date(), end.date())
        return 0

    logger.info("%s 1 分 K：要補 %d 個交易日（%s ~ %s），約 %.1f 小時",
                market, len(todo), todo[0].date(), todo[-1].date(), len(todo) * (2.7 if market == "otc" else 4.3) / 3600)
    empty, odd = [], []
    batch = []

    def flush() -> None:
        if batch:
            total = store.write(market, "1min", pd.concat(batch, ignore_index=True))
            logger.info("  已寫入至 %s，檔案共 %d 根", batch[-1]["datetime"].iloc[-1].date(), total)
            batch.clear()

    def run(days: list[pd.Timestamp]) -> list[pd.Timestamp]:
        failed = []
        for i, day in enumerate(days, 1):
            try:
                bars = client.intraday_bars(day)
            except Exception as exc:
                logger.warning("%s 抓取失敗：%s", day.date(), exc)
                failed.append(day)
                continue
            if bars.empty:
                empty.append(day)
                continue
            if len(bars) != BARS_PER_DAY:
                odd.append((day, len(bars)))
            batch.append(bars)
            if i % INTRADAY_BATCH_DAYS == 0:
                flush()
        flush()
        return failed

    failed = run(todo)
    if failed:
        logger.info("第一輪有 %d 天失敗，最後再試一輪", len(failed))
        failed = run(failed)

    logger.info("1 分 K 完成：成功 %d 天、無盤中資料 %d 天、失敗 %d 天、根數異常 %d 天",
                len(todo) - len(failed) - len(empty), len(empty), len(failed), len(odd))
    for label, items in (("無盤中資料", empty), ("失敗（重跑會再試）", failed)):
        if items:
            logger.info("  %s：%s", label, ", ".join(str(d.date()) for d in items[:20]))
    if odd:
        logger.info("  根數異常：%s", ", ".join(f"{d.date()}={n}" for d, n in odd[:20]))
    return len(failed)


def backfill_margin(client, market: str, start: pd.Timestamp, end: pd.Timestamp, limit: int | None) -> int:
    """整體融資融券餘額，一天一列；交易日曆取自日 K 檔。"""
    todo = _missing_days(market, "margin", start, end, limit)
    if not todo:
        logger.info("%s 融資融券：%s ~ %s 沒有缺的交易日", market, start.date(), end.date())
        return 0

    per_day = 2.2 if market == "tse" else 2.7
    logger.info("%s 融資融券：要補 %d 個交易日（%s ~ %s），約 %.1f 小時",
                market, len(todo), todo[0].date(), todo[-1].date(), len(todo) * per_day / 3600)
    failed, empty, batch = [], [], []

    def flush() -> None:
        if batch:
            total = store.write(market, "margin", pd.concat(batch, ignore_index=True))
            logger.info("  已寫入至 %s，檔案共 %d 天", batch[-1]["date"].iloc[-1].date(), total)
            batch.clear()

    for i, day in enumerate(todo, 1):
        try:
            row = client.margin_totals(day)
        except Exception as exc:
            logger.warning("%s 抓取失敗：%s", day.date(), exc)
            failed.append(day)
            continue
        if row.empty:
            empty.append(day)
            continue
        batch.append(row)
        if i % MARGIN_BATCH_DAYS == 0:
            flush()
    flush()

    logger.info("%s 融資融券完成：成功 %d 天、無資料 %d 天、失敗 %d 天",
                market, len(todo) - len(failed) - len(empty), len(empty), len(failed))
    for label, items in (("無資料", empty), ("失敗（重跑會再試）", failed)):
        if items:
            logger.info("  %s：%s", label, ", ".join(str(d.date()) for d in items[:20]))
    return len(failed)


def _missing_days(market: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
                  limit: int | None) -> list[pd.Timestamp]:
    """日 K 當交易日曆，扣掉本地已有的日子。"""
    daily = store.read(market, "daily")
    calendar = pd.to_datetime(daily["date"])
    calendar = calendar[(calendar >= start) & (calendar <= end)]

    existing = store.read(market, timeframe)
    key = store.LAYOUTS[timeframe][1]
    have = set(pd.to_datetime(existing[key]).dt.normalize()) if not existing.empty else set()
    todo = sorted(set(calendar) - have)
    return todo[:limit] if limit else todo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market", choices=list(CLIENTS), default="otc")
    parser.add_argument("--timeframe", choices=[*store.LAYOUTS, "all"], default="all")
    parser.add_argument("--start", help="起始日（預設為各週期可取得的最早日）")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--limit", type=int, help="最多補幾天（測試用）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    earliest = EARLIEST[args.market]
    timeframes = [t for t in (list(store.LAYOUTS) if args.timeframe == "all" else [args.timeframe]) if t in earliest]
    if not timeframes:
        parser.error(f"{args.market} 不支援 {args.timeframe}（可用：{', '.join(earliest)}）")
    end = pd.Timestamp(args.end)

    failures = 0
    with CLIENTS[args.market]() as client:
        if "daily" in timeframes:
            backfill_daily(client, args.market, pd.Timestamp(args.start or earliest["daily"]))
        if "1min" in timeframes:
            failures += backfill_intraday(client, args.market, pd.Timestamp(args.start or earliest["1min"]), end, args.limit)
        if "margin" in timeframes:
            failures += backfill_margin(client, args.market, pd.Timestamp(args.start or earliest["margin"]), end, args.limit)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
