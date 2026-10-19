r"""富邦 Neo SDK 資料來源。

安裝（SDK 不在 PyPI，需從富邦證券開發者網站下載對應 Python 版本的 wheel）：

    .venv\Scripts\pip install fubon_neo-<version>-cp312-cp312-win_amd64.whl

登入資訊放 .env（見 .env.example），本檔案不碰任何硬編碼的帳密。

注意：Neo SDK 的方法簽章在改版時動過幾次。若登入或取資料時噴 AttributeError，
先用 `python -c "import fubon_neo; help(fubon_neo.sdk.FubonSDK)"` 對一下你手上
那版的介面，再調整 _login / daily_bars 這兩處即可，其他程式碼不受影響。
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from core.config import FubonCredentials
from core.datasource.base import DataSource

logger = logging.getLogger(__name__)

# 富邦行情 historical API 上限每分鐘 60 次，超過回 429；留一點餘裕
_MIN_INTERVAL = 1.05
# 被 429 擋下時，官方建議等 1 分鐘再試
_RATE_LIMIT_WAIT = 61
# 歷史 K 線單次查詢的區間上限，超過就分段抓
_CHUNK_DAYS = 365
_CANDLE_FIELDS = "open,high,low,close,volume,turnover"


class FubonSource(DataSource):
    name = "fubon"

    def __init__(self, credentials: FubonCredentials | None = None) -> None:
        self._credentials = credentials or FubonCredentials.from_env()
        self._sdk = None
        self._rest = None
        self._last_call = 0.0

    # ------------------------------------------------------------------ 連線
    def _ensure_login(self):
        if self._rest is not None:
            return self._rest

        try:
            from fubon_neo.sdk import FubonSDK
        except ImportError as exc:  # pragma: no cover - 取決於本機是否裝了 SDK
            raise ImportError(
                "找不到 fubon_neo。請先從富邦開發者網站下載 wheel 並安裝，"
                "或改用 --source twse 走免費開放資料。"
            ) from exc

        cred = self._credentials
        sdk = FubonSDK()
        result = sdk.login(
            cred.personal_id, cred.password, cred.cert_path, cred.cert_password
        )
        # 不同版本回傳型別不一致：有的是 Result 物件，有的直接回帳戶清單
        if getattr(result, "is_success", True) is False:
            raise RuntimeError(f"富邦登入失敗：{getattr(result, 'message', result)}")

        sdk.init_realtime()  # 行情服務要先初始化才能拿到 rest_client
        self._sdk = sdk
        self._rest = sdk.marketdata.rest_client.stock
        logger.info("富邦 Neo SDK 登入成功（%s）", cred)
        return self._rest

    def close(self) -> None:
        if self._sdk is not None:
            try:
                self._sdk.logout()
            except Exception:  # pragma: no cover - 登出失敗不該中斷流程
                logger.warning("富邦登出時發生例外，已忽略", exc_info=True)
            self._sdk = None
            self._rest = None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    # ------------------------------------------------------------------ 資料
    def list_tickers(self) -> pd.DataFrame:
        rest = self._ensure_login()
        rows: list[dict] = []
        for market in ("TSE", "OTC"):
            self._throttle()
            resp = rest.intraday.tickers(type="EQUITY", exchange="TWSE", market=market)
            for item in resp.get("data", []):
                rows.append(
                    {
                        "symbol": item.get("symbol"),
                        "name": item.get("name"),
                        "market": market,
                        "type": item.get("type", "EQUITY"),
                    }
                )
        return pd.DataFrame(rows).drop_duplicates(subset="symbol").reset_index(drop=True)

    def daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # 預設回傳的 candle 沒有成交金額，turnover 要明確指定才會給，
        # 而流動性過濾靠的就是 turnover
        return self.candles(symbol, start, end, timeframe="D", fields=_CANDLE_FIELDS)

    def candles(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "D",
        fields: str | None = None,
    ) -> pd.DataFrame:
        """歷史 K 線，timeframe 可為 1/3/5/10/15/30/60（分）或 D/W/M。

        欄位原樣回傳（date 在分 K 是帶 +08:00 的 ISO 字串）。注意：指數的分 K
        `volume` 實際上是成交金額，日 K 才是成交股數，由呼叫端自行命名。
        """
        rest = self._ensure_login()
        frames = []
        for chunk_start, chunk_end in _date_chunks(start, end, _CHUNK_DAYS):
            query = {
                "symbol": symbol, "from": chunk_start, "to": chunk_end,
                "timeframe": timeframe, "sort": "asc",
            }
            data = self._query_candles(rest, query, fields)
            if data:
                frames.append(pd.DataFrame(data))

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="date").sort_values("date")
        return df.reset_index(drop=True)

    def _query_candles(self, rest, query: dict, fields: str | None) -> list[dict]:
        label = f"{query['symbol']} {query['timeframe']} {query['from']}~{query['to']}"
        attempts = [{"fields": fields}, {}] if fields else [{}]
        for extra in attempts:
            for retry in range(2):
                self._throttle()
                try:
                    resp = rest.historical.candles(**query, **extra)
                except Exception as exc:
                    msg = str(exc)
                    if "Resource Not Found" in msg or "404" in msg:
                        # 區間整段在資料起始日前或全是休市日，官方回 404 而非空陣列
                        return []
                    if ("429" in msg or "Rate limit" in msg) and retry == 0:
                        logger.warning("%s 觸發流量限制，等 %ss 重試", label, _RATE_LIMIT_WAIT)
                        time.sleep(_RATE_LIMIT_WAIT)
                        continue
                    logger.debug("%s 查詢失敗（%s）", label, extra or "預設欄位", exc_info=True)
                    break
                return (resp.get("data") or []) if isinstance(resp, dict) else []
        logger.warning("取 %s 失敗，跳過此區間", label)
        return []


def _date_chunks(start: str, end: str, days: int) -> list[tuple[str, str]]:
    """把 [start, end] 切成不超過 days 天的區間，避免超過 API 單次查詢上限。"""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    cursor = start_ts
    while cursor <= end_ts:
        stop = min(cursor + pd.Timedelta(days=days - 1), end_ts)
        chunks.append((cursor.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        cursor = stop + pd.Timedelta(days=1)
    return chunks
