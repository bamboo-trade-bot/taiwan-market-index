"""資料來源的抽象介面。

策略與下載腳本只依賴這個介面，換來源（富邦 / TWSE / 未來別家）不用改上層程式碼。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """所有資料來源的共同契約。"""

    name: str = "base"

    @abstractmethod
    def list_tickers(self) -> pd.DataFrame:
        """回傳上市櫃股票清單。

        欄位至少要有：symbol, name, market（TSE/OTC）, type（EQUITY/ETF/...）
        """

    @abstractmethod
    def daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """回傳單檔日 K。

        欄位：date, open, high, low, close, volume, turnover
        沒有資料時回傳空 DataFrame（不要拋例外，讓呼叫端能跳過下市股票）。
        """

    def close(self) -> None:
        """釋放連線 / 登出。預設不做事。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
