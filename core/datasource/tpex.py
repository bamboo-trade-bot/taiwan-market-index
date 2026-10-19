"""櫃買中心指數公開資料（免費、免登入）：櫃買指數日 K 與盤中快照。

不是 DataSource 子類別：DataSource 的契約是「個股清單 + 個股日 K」，這裡只有指數，
硬塞進去只會多出兩個丟 NotImplementedError 的方法，所以也不註冊到 SOURCES。

資料期間（2026-09 實測）：
    日 K        inx          1999-09-01 起，一次一個月，只有開高低收與漲跌
    盤中快照    miIndexHis   2007-07-02 ~ 2014-01-03（1 分鐘，2011-01-17 起 15 秒）
                miIndex      2014-01-06 起（15 秒，2014-02-24 起 10 秒，2014-12-29 起 5 秒）

盤中快照轉 1 分 K 的規則，已用 2023-05 之後與富邦 IX0043 分 K 逐根比對（開高低收完全一致，
成交金額差距在萬元進位內）：
    * 開盤前那筆（09:00:00，或 1 分鐘時期的 9:00）值等於昨收，不計入。
    * 秒級資料：依分鐘向下取整分組，09:30 那根 = 09:30:00~09:30:55 的快照。
    * 1 分鐘資料：hh:mm 的快照是「前一分鐘結束時」的狀態，歸到前一根。
    * 13:30 那根 = 最後的 99:99:99 收盤彙總列。
    * 成交金額由累計值相減而得。

連線注意：
    * 對方憑證沒附中繼憑證，certifi 驗不過，改用系統憑證庫（truststore）驗證。
    * 大筆回應偶爾傳到一半斷線、或後端暫時回 520，一律重試。
"""
from __future__ import annotations

import logging
import ssl
import time

import pandas as pd
import requests
import truststore
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.tpex.org.tw/www/zh-tw"
# 融資融券合計列的欄位順序（櫃買中心只給個股明細，整體數字塞在 summary）
_MARGIN_UNIT_KEYS = ["margin_prev", "margin_buy", "margin_sell", "margin_redeem", "margin_balance",
                     "short_prev", "short_sell", "short_buy", "short_redeem", "short_balance"]
_MARGIN_VALUE_KEYS = ["margin_prev_value", "margin_buy_value", "margin_sell_value",
                      "margin_redeem_value", "margin_balance_value"]
_MIN_INTERVAL = 2.5
# 對方 Cloudflare 後端偶爾會連續幾分鐘回 520，等久一點才撐得過去
_RETRY_WAITS = (10, 30, 60, 120, 300)
_MIINDEX_START = pd.Timestamp("2014-01-06")
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; personal backtesting)"}

MARKET_OPEN = pd.Timedelta(hours=9)
MARKET_CLOSE = pd.Timedelta(hours=13, minutes=30)


class _SystemTrustAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return super().init_poolmanager(*args, **kwargs)


class TpexIndexClient:
    def __init__(self, timeout: int = 60) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._session.mount("https://", _SystemTrustAdapter())
        self._timeout = timeout
        self._last_call = 0.0

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ 連線
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def _table(self, endpoint: str, day: pd.Timestamp, group: str = "indexInfo") -> dict:
        """POST 查詢並回傳第一張表；斷線、截斷、非 JSON 都重試，全部失敗才拋例外。"""
        form = {"date": day.strftime("%Y/%m/%d"), "response": "json"}
        for attempt, wait in enumerate((*_RETRY_WAITS, None)):
            self._throttle()
            try:
                resp = self._session.post(f"{_BASE_URL}/{group}/{endpoint}", data=form, timeout=self._timeout)
                resp.raise_for_status()
                return resp.json()["tables"][0]
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                if wait is None:
                    raise RuntimeError(f"TPEx {endpoint} {form['date']} 連續失敗") from exc
                logger.debug("TPEx %s %s 第 %d 次失敗（%s），%ss 後重試",
                             endpoint, form["date"], attempt + 1, type(exc).__name__, wait)
                time.sleep(wait)

    # ------------------------------------------------------------------ 資料
    def daily_month(self, month: pd.Timestamp) -> pd.DataFrame:
        """櫃買指數某個月的日 K。欄位：date open high low close change。"""
        table = self._table("inx", pd.Timestamp(month).replace(day=1))
        if not table.get("data"):
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "change"])
        raw = pd.DataFrame(table["data"], columns=["date", "open", "high", "low", "close", "change"])
        df = raw.assign(date=pd.to_datetime(raw["date"]))
        for col in ("open", "high", "low", "close", "change"):
            df[col] = _to_float(df[col])
        return df

    def margin_totals(self, day: pd.Timestamp) -> pd.DataFrame:
        """上櫃整體的融資融券餘額（一天一列）。張數是張，金額換成元。

        櫃買中心只提供個股明細，整體數字在表格的 summary 裡：
        「合計(張)」那列是張數，「融資金(仟元)」那列是融資金額。
        """
        day = pd.Timestamp(day).normalize()
        table = self._table("balance", day, group="margin")
        out: dict = {"date": day}
        for row in table.get("summary") or []:
            label = next((str(v).strip() for v in row if str(v).strip().startswith(("合計", "融資金"))), "")
            numbers = [n for n in (_to_number(v) for v in row) if n is not None]
            if label.startswith("合計") and len(numbers) >= len(_MARGIN_UNIT_KEYS):
                out |= dict(zip(_MARGIN_UNIT_KEYS, numbers))
            elif label.startswith("融資金") and len(numbers) >= len(_MARGIN_VALUE_KEYS):
                out |= {k: v * 1_000 for k, v in zip(_MARGIN_VALUE_KEYS, numbers)}  # 仟元 → 元
        if "margin_balance" not in out:
            logger.warning("TPEx %s 融資合計解析失敗", day.date())
            return pd.DataFrame()

        out["margin_change"] = out["margin_balance"] - out["margin_prev"]
        out["short_change"] = out["short_balance"] - out["short_prev"]
        if "margin_balance_value" in out:
            out["margin_change_value"] = out["margin_balance_value"] - out["margin_prev_value"]
        return pd.DataFrame([out])

    def intraday_bars(self, day: pd.Timestamp) -> pd.DataFrame:
        """某交易日的櫃買指數 1 分 K（規則見模組說明）。

        欄位：datetime open high low close turnover sample_sec
        sample_sec 為原始快照間隔（5/10/15/60）；60 的那幾年每分鐘只有一筆，開高低收相同。
        沒有資料時回傳空 DataFrame。
        """
        day = pd.Timestamp(day).normalize()
        endpoint = "miIndex" if day >= _MIINDEX_START else "miIndexHis"
        table = self._table(endpoint, day)
        if not table.get("data"):
            return pd.DataFrame()
        return snapshots_to_bars(day, table["fields"], table["data"])


def snapshots_to_bars(day: pd.Timestamp, fields: list[str], rows: list[list[str]]) -> pd.DataFrame:
    names = [f.replace(" ", "") for f in fields]
    raw = pd.DataFrame(rows, columns=names)
    df = pd.DataFrame({
        "time": raw["時間"].str.strip(),
        "index": _to_float(raw["櫃買指數"]),
        "cum_turnover": _to_float(raw["成交金額(萬元)"]) * 10_000,
    })

    summary = df[df["time"].str.startswith("99")]
    df = df[~df["time"].str.startswith("99")].copy()
    if df.empty or summary.empty:
        return pd.DataFrame()

    df["ts"] = day + pd.to_timedelta(df["time"].map(_normalize_clock))
    df = df.sort_values("ts")
    step = int(df["ts"].diff().dt.total_seconds().median())

    df = df[df["ts"] > day + MARKET_OPEN]  # 開盤前快照值 = 昨收
    if step >= 60:
        # 1 分鐘快照記錄的是前一分鐘結束時的狀態
        df["minute"] = df["ts"] - pd.Timedelta(minutes=1)
    else:
        df["minute"] = df["ts"].dt.floor("min")
    df = df[df["minute"] < day + MARKET_CLOSE]  # 13:30 由收盤彙總列取代

    grouped = df.groupby("minute")
    bars = pd.DataFrame({
        "open": grouped["index"].first(),
        "high": grouped["index"].max(),
        "low": grouped["index"].min(),
        "close": grouped["index"].last(),
        "cum": grouped["cum_turnover"].last(),
    })
    final = summary.iloc[-1]
    bars.loc[day + MARKET_CLOSE] = [final["index"]] * 4 + [final["cum_turnover"]]

    bars["turnover"] = bars["cum"].diff().fillna(bars["cum"].iloc[0])
    bars["sample_sec"] = step
    bars.index.name = "datetime"
    return bars.drop(columns="cum").reset_index()


def _normalize_clock(value: str) -> str:
    """'9:01' → '09:01:00'，'09:00:15' 原樣。"""
    parts = value.split(":")
    if len(parts) == 2:
        parts.append("00")
    return ":".join(p.zfill(2) for p in parts)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _to_number(value) -> float | None:
    """單一數字字串 → float；表頭文字、空白、'-' 都回 None。"""
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
