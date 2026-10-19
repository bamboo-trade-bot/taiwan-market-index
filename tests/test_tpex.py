import pandas as pd

from core.datasource.tpex import snapshots_to_bars

FIELDS = ["時 間", " 櫃買指數", " 成交金額(萬元)"]
DAY = pd.Timestamp("2024-02-21")


def _rows(pairs):
    return [[t, f"{v:.2f}", f"{amt:,}"] for t, v, amt in pairs]


def test_second_snapshots_skip_preopen_and_use_summary_for_close():
    rows = _rows([
        ("09:00:00", 100.00, 0),        # 開盤前，值 = 昨收
        ("09:00:05", 101.00, 10),
        ("09:00:10", 103.00, 25),
        ("09:00:55", 102.00, 30),
        ("09:01:00", 102.50, 40),
        ("13:29:55", 104.00, 900),
        ("13:30:00", 104.00, 900),      # 被收盤彙總取代
        ("99:99:99", 105.00, 1000),
    ])
    bars = snapshots_to_bars(DAY, FIELDS, rows).set_index("datetime")

    first = bars.loc[DAY + pd.Timedelta("9h")]
    assert (first.open, first.high, first.low, first.close) == (101.0, 103.0, 101.0, 102.0)
    assert first.turnover == 300_000

    close = bars.loc[DAY + pd.Timedelta("13h30min")]
    assert (close.open, close.close) == (105.0, 105.0)
    assert close.turnover == 1_000_000
    assert bars["turnover"].sum() == 10_000_000
    assert bars["sample_sec"].iloc[0] == 5


def test_minute_snapshots_belong_to_previous_minute():
    rows = _rows([
        ("9:00", 100.00, 5),            # 開盤前
        ("9:01", 101.00, 20),           # 09:00 那根的收盤
        ("9:02", 102.00, 30),
        ("13:30", 103.00, 80),          # 13:29 那根
        ("99:99", 103.10, 90),
    ])
    bars = snapshots_to_bars(DAY, FIELDS, rows).set_index("datetime")

    assert list(bars.index.strftime("%H:%M")) == ["09:00", "09:01", "13:29", "13:30"]
    assert bars.loc[DAY + pd.Timedelta("9h"), "open"] == 101.0
    assert bars.loc[DAY + pd.Timedelta("13h29min"), "close"] == 103.0
    assert bars.loc[DAY + pd.Timedelta("13h30min"), "close"] == 103.1
    assert bars["sample_sec"].iloc[0] == 60


def test_missing_summary_returns_empty():
    assert snapshots_to_bars(DAY, FIELDS, _rows([("09:00:05", 1.0, 1)])).empty
