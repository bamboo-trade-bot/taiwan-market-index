"""大盤資料的檔案版面與讀寫。本專案只有這裡拼資料路徑。

    data/
      tse/daily.parquet   加權指數 IX0001 日 K   date open high low close volume turnover change
      tse/1min.parquet    加權指數 IX0001 1 分 K datetime open high low close turnover sample_sec
      tse/margin.parquet  上市整體信用交易   date margin_*（融資）short_*（融券）
      otc/...             櫃買指數 IX0043，版面同上

融資融券：margin_* / short_* 的數量單位是張，帶 _value 的是金額（元）；
margin_change = 今日餘額 - 前日餘額。證交所沒有提供融券金額。

單位：volume = 成交股數，turnover = 成交金額（元）。分 K 只有成交金額。
分 K 的 datetime 是台北時間、該分鐘的起點（09:30 那根涵蓋 09:30:00~09:30:55）。
sample_sec 是合成分 K 的原始快照間隔（5/10/15/60 秒）；60 表示每分鐘只有一筆，開高低收相同。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# market → 富邦指數代號
MARKETS = {"tse": "IX0001", "otc": "IX0043"}

# timeframe → (檔名, 時間欄位, 欄位)
LAYOUTS = {
    "daily": ("daily.parquet", "date",
              ["date", "open", "high", "low", "close", "volume", "turnover", "change"]),
    "margin": ("margin.parquet", "date",
               ["date",
                "margin_prev", "margin_buy", "margin_sell", "margin_redeem", "margin_balance", "margin_change",
                "margin_prev_value", "margin_buy_value", "margin_sell_value", "margin_redeem_value",
                "margin_balance_value", "margin_change_value",
                "short_prev", "short_sell", "short_buy", "short_redeem", "short_balance", "short_change"]),
    "1min": ("1min.parquet", "datetime",
             ["datetime", "open", "high", "low", "close", "turnover", "sample_sec"]),
}


def path(market: str, timeframe: str) -> Path:
    if market not in MARKETS:
        raise ValueError(f"未知的市場 {market!r}，可用：{', '.join(MARKETS)}")
    return DATA_DIR / market / LAYOUTS[timeframe][0]


def read(market: str, timeframe: str) -> pd.DataFrame:
    p = path(market, timeframe)
    if not p.exists():
        return pd.DataFrame(columns=LAYOUTS[timeframe][2])
    return pd.read_parquet(p)


def last_time(market: str, timeframe: str) -> pd.Timestamp | None:
    p = path(market, timeframe)
    if not p.exists():
        return None
    key = LAYOUTS[timeframe][1]
    col = pd.read_parquet(p, columns=[key])[key]
    return None if col.empty else pd.Timestamp(col.max())


def write(market: str, timeframe: str, df: pd.DataFrame) -> int:
    """與既有檔案以時間欄位合併，新資料覆蓋舊資料（盤中抓到的半天會被收盤後重抓蓋掉）。"""
    _, key, columns = LAYOUTS[timeframe]
    p = path(market, timeframe)
    p.parent.mkdir(parents=True, exist_ok=True)

    df = df.reindex(columns=columns)
    if p.exists():
        df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        df = df.drop_duplicates(subset=key, keep="last")
    df = df.sort_values(key).reset_index(drop=True)
    df.to_parquet(p, index=False, compression="zstd")
    return len(df)
