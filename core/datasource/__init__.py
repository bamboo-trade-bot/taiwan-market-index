"""資料來源。透過 get_source() 取得實例，上層不直接 import 具體類別。"""
from __future__ import annotations

from core.datasource.base import DataSource
from core.datasource.fubon import FubonSource
from core.datasource.twse import TwseSource

SOURCES = {"fubon": FubonSource, "twse": TwseSource}


def get_source(name: str) -> DataSource:
    if name not in SOURCES:
        raise ValueError(f"未知的資料來源 {name!r}，可用：{', '.join(SOURCES)}")
    return SOURCES[name]()


__all__ = ["DataSource", "FubonSource", "TwseSource", "get_source", "SOURCES"]
