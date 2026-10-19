"""證交所／櫃買中心公開資料來源（免費、免登入）。

用途：在拿到富邦憑證前先把 pipeline 跑通，或用來交叉驗證富邦回來的數字。

限制（刻意不隱藏）：
  * 證交所 STOCK_DAY 一次只給「一個月、一檔股票」，抓多年多檔會很慢。
  * 對方有流量管制，這裡設 3 秒一次；別調太快，會被擋。
  * 只實作上市（TSE）日線。上櫃歷史日線請走富邦。
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from core.datasource.base import DataSource

logger = logging.getLogger(__name__)

_STOCK_DAY = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
_TSE_TICKERS = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
_OTC_TICKERS = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
_MIN_INTERVAL = 3.0
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; personal backtesting)"}


class TwseSource(DataSource):
    name = "twse"

    def __init__(self, timeout: int = 20) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._timeout = timeout
        self._last_call = 0.0

    def close(self) -> None:
        self._session.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def list_tickers(self) -> pd.DataFrame:
        rows: list[dict] = []
        for url, market, code_key, name_key in (
            (_TSE_TICKERS, "TSE", "公司代號", "公司簡稱"),
            (_OTC_TICKERS, "OTC", "SecuritiesCompanyCode", "CompanyAbbreviation"),
        ):
            try:
                self._throttle()
                payload = self._session.get(url, timeout=self._timeout).json()
            except Exception:
                logger.warning("取得 %s 股票清單失敗，略過", market, exc_info=True)
                continue
            for item in payload:
                code = str(item.get(code_key, "")).strip()
                if code:
                    rows.append(
                        {
                            "symbol": code,
                            "name": str(item.get(name_key, "")).strip(),
                            "market": market,
                            "type": "EQUITY",
                        }
                    )
        return pd.DataFrame(rows).drop_duplicates(subset="symbol").reset_index(drop=True)

    def daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        months = pd.date_range(
            pd.Timestamp(start).normalize().replace(day=1),
            pd.Timestamp(end),
            freq="MS",
        )
        frames = []
        for month in months:
            df = self._fetch_month(symbol, month)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df[mask].sort_values("date").reset_index(drop=True)

    def _fetch_month(self, symbol: str, month: pd.Timestamp) -> pd.DataFrame | None:
        self._throttle()
        params = {"date": month.strftime("%Y%m01"), "stockNo": symbol, "response": "json"}
        try:
            payload = self._session.get(_STOCK_DAY, params=params, timeout=self._timeout).json()
        except Exception:
            logger.warning("取 %s %s 失敗", symbol, month.strftime("%Y-%m"), exc_info=True)
            return None

        if payload.get("stat") != "OK" or not payload.get("data"):
            return None

        raw = pd.DataFrame(payload["data"], columns=payload["fields"])
        return pd.DataFrame(
            {
                "date": raw["日期"].map(_roc_to_date),
                "open": raw["開盤價"].map(_to_float),
                "high": raw["最高價"].map(_to_float),
                "low": raw["最低價"].map(_to_float),
                "close": raw["收盤價"].map(_to_float),
                # 證交所給的是「股」，除以 1000 換成張，與富邦口徑對齊
                "volume": raw["成交股數"].map(_to_float) / 1000.0,
                "turnover": raw["成交金額"].map(_to_float),
            }
        ).dropna(subset=["date"])


def _roc_to_date(value: str) -> pd.Timestamp | None:
    """民國日期 '113/01/02' → Timestamp('2024-01-02')。"""
    try:
        year, month, day = str(value).split("/")
        return pd.Timestamp(int(year) + 1911, int(month), int(day))
    except (ValueError, TypeError):
        return None


def _to_float(value) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


# ====================================================================== 指數盤中
"""以下是證交所的加權指數盤中快照，與上面的個股日線無關，所以另開一個 client。
規則與 core/datasource/tpex.py 相同（已與富邦 IX0001 分 K 逐根比對）：
    * 開盤前那筆（09:00:00）值等於昨收，不計入。
    * 依分鐘向下取整分組；13:30 那筆已含收盤撮合，直接當 13:30 那根。
    * 成交金額由另一支 API 的累計值相減而得，單位是百萬元（比櫃買的萬元粗）。
"""

_MI_5MINS_INDEX = "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_INDEX"
_MI_5MINS_STATS = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_5MINS"
_INDEX_COLUMN = "發行量加權股價指數"
_TURNOVER_COLUMN = "累積成交金額"
_MI_MARGN = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
_INDEX_MIN_INTERVAL = 2.0
_OPEN_TIME = pd.Timedelta(hours=9)
_CLOSE_TIME = pd.Timedelta(hours=13, minutes=30)


class TwseIndexClient:
    """加權指數的盤中 1 分 K。與 TpexIndexClient 介面相同。"""

    def __init__(self, timeout: int = 60) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._timeout = timeout
        self._last_call = 0.0

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _INDEX_MIN_INTERVAL:
            time.sleep(_INDEX_MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def _payload(self, url: str, day: pd.Timestamp, extra: dict | None = None) -> dict | None:
        params = {"date": day.strftime("%Y%m%d"), "response": "json", **(extra or {})}
        for wait in (10, 30, 60, None):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                if wait is None:
                    raise RuntimeError(f"TWSE {url} {params['date']} 連續失敗") from exc
                logger.debug("TWSE %s %s 失敗（%s），%ss 後重試", url, params["date"], type(exc).__name__, wait)
                time.sleep(wait)
                continue
            has_data = payload.get("data") or any(t.get("data") for t in payload.get("tables", []))
            return payload if payload.get("stat") == "OK" and has_data else None
        return None

    def margin_totals(self, day: pd.Timestamp) -> pd.DataFrame:
        """上市整體的信用交易統計（一天一列）。張數是交易單位（張），金額換成元。"""
        day = pd.Timestamp(day).normalize()
        payload = self._payload(_MI_MARGN, day, extra={"selectType": "MS"})
        if payload is None:
            return pd.DataFrame()
        rows = payload.get("data") or payload["tables"][0].get("data")
        return pd.DataFrame([_margin_row(day, rows)]) if rows else pd.DataFrame()

    def intraday_bars(self, day: pd.Timestamp) -> pd.DataFrame:
        """某交易日的加權指數 1 分 K：datetime open high low close turnover sample_sec。

        成交金額那支 API 失敗或沒資料時，turnover 留 NaN，不影響指數。
        """
        day = pd.Timestamp(day).normalize()
        index_payload = self._payload(_MI_5MINS_INDEX, day)
        if index_payload is None:
            return pd.DataFrame()

        snaps = _index_snapshots(day, index_payload, _INDEX_COLUMN)
        if snaps.empty:
            return pd.DataFrame()

        bars = _snapshots_to_bars(snaps)
        stats_payload = self._payload(_MI_5MINS_STATS, day)
        bars["turnover"] = float("nan")
        if stats_payload is not None:
            cum = _index_snapshots(day, stats_payload, _TURNOVER_COLUMN)
            if not cum.empty:
                per_minute = cum.groupby(cum["ts"].dt.floor("min"))["value"].last() * 1_000_000
                turnover = per_minute.diff()
                turnover.iloc[0] = per_minute.iloc[0]
                bars["turnover"] = bars["datetime"].map(turnover)
        return bars[["datetime", "open", "high", "low", "close", "turnover", "sample_sec"]]


def _index_snapshots(day: pd.Timestamp, payload: dict, column: str) -> pd.DataFrame:
    raw = pd.DataFrame(payload["data"], columns=[f.strip() for f in payload["fields"]])
    if column not in raw.columns:
        logger.warning("TWSE %s 缺少「%s」欄位", day.date(), column)
        return pd.DataFrame()
    df = pd.DataFrame({
        "ts": day + pd.to_timedelta(raw["時間"].str.strip()),
        "value": raw[column].map(_to_float),
    }).sort_values("ts")
    return df[(df["ts"] > day + _OPEN_TIME) & (df["ts"] <= day + _CLOSE_TIME)]


def _snapshots_to_bars(snaps: pd.DataFrame) -> pd.DataFrame:
    step = int(snaps["ts"].diff().dt.total_seconds().median())
    minute = snaps["ts"].dt.floor("min")
    grouped = snaps.groupby(minute)["value"]
    bars = pd.DataFrame({
        "open": grouped.first(),
        "high": grouped.max(),
        "low": grouped.min(),
        "close": grouped.last(),
    })
    bars["sample_sec"] = step
    bars.index.name = "datetime"
    return bars.reset_index()


def _margin_row(day: pd.Timestamp, rows: list[list[str]]) -> dict:
    """「信用交易統計」三列（融資張數 / 融券張數 / 融資金額仟元）轉成一列。

    表頭順序：項目, 買進, 賣出, 現金(券)償還, 前日餘額, 今日餘額。
    融券那列的「買進」是券買回補、「賣出」是新增空單；證交所這支沒有融券金額。
    """
    table = {str(r[0]).strip(): [_to_float(x) for x in r[1:6]] for r in rows}
    out: dict = {"date": day}
    if (units := table.get("融資(交易單位)")):
        buy, sell, redeem, prev, balance = units
        out |= {"margin_buy": buy, "margin_sell": sell, "margin_redeem": redeem,
                "margin_prev": prev, "margin_balance": balance, "margin_change": balance - prev}
    if (value := table.get("融資金額(仟元)")):
        buy, sell, redeem, prev, balance = [v * 1_000 for v in value]   # 仟元 → 元
        out |= {"margin_buy_value": buy, "margin_sell_value": sell, "margin_redeem_value": redeem,
                "margin_prev_value": prev, "margin_balance_value": balance,
                "margin_change_value": balance - prev}
    if (short := table.get("融券(交易單位)")):
        buy, sell, redeem, prev, balance = short
        out |= {"short_buy": buy, "short_sell": sell, "short_redeem": redeem,
                "short_prev": prev, "short_balance": balance, "short_change": balance - prev}
    return out
