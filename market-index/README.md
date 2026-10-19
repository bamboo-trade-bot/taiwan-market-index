# 大盤分析

加權指數（TSE, IX0001）與櫃買指數（OTC, IX0043）的歷史 K 線資料與分析。

**研究範圍：近 10 年（2016-10-18 起），兩個市場都是 2,438 個交易日。**
檔案裡保有更早的資料（櫃買分 K 到 2007、日 K 到 1999），刻意留著但不納入分析；
回測的 `--start` 預設就是 2016-10-18。最細只存到 1 分 K，不抓 5 秒 K。

## 抓資料

### 每日更新：富邦 Neo API

登入沿用上層專案的 `.env` 與 `core/datasource/fubon.py`。

```bash
cd market-index
..\.venv\Scripts\python.exe fetch_fubon.py                               # 全部增量更新
..\.venv\Scripts\python.exe fetch_fubon.py --market otc --timeframe 1min # 只抓一份
..\.venv\Scripts\python.exe fetch_fubon.py --full                        # 從頭重抓
```

重跑只會從本地最後一天開始補（最後一天會重抓，盤中抓到的半天收盤後會被蓋掉）。
從零開始全部抓完約 1 分鐘。

### 補歷史：交易所公開資料

富邦的歷史有限（櫃買日 K 只到 2014-01-06、兩個市場的分 K 都只到 2023-05-23），
更早的用 `core/datasource/tpex.py`（櫃買中心）與 `core/datasource/twse.py`（證交所）補。

```bash
..\.venv\Scripts\python.exe backfill_official.py --market otc                    # 日 K + 1 分 K
..\.venv\Scripts\python.exe backfill_official.py --market tse --start 2016-10-18 # 加權分 K
..\.venv\Scripts\python.exe backfill_official.py --market otc --timeframe 1min --limit 5
```

只補本地沒有的日子、不覆蓋富邦資料；中斷後重跑會接著補。交易日曆取自該市場的日 K 檔。
速度：櫃買約 2.7 秒/天，加權約 4.3 秒/天（指數與成交金額分兩支 API）。
對方後端偶爾回 520 或傳到一半斷線，會自動等待重試，最後再補抓一輪失敗的日子。

## 資料版面

```
data/
  tse/daily.parquet   2004-10-15 起   date open high low close volume turnover change
  tse/1min.parquet    2016-10-18 起   datetime open high low close turnover sample_sec
                                      （2023-05-22 以前來自證交所，成交金額只到百萬元進位）
  otc/daily.parquet   1999-09-01 起   （2014-01-03 以前來自櫃買中心，volume/turnover 為空）
  otc/1min.parquet    2007-07-02 起   （2023-05-22 以前來自櫃買中心）
  tse/margin.parquet  2016-10-18 起   整體信用交易（證交所）
  otc/margin.parquet  2016-10-18 起   整體信用交易（櫃買中心）
```

路徑一律透過 `store.py` 取得。

- `volume` = 成交股數、`turnover` = 成交金額（元）。
- **分 K 只有成交金額**：富邦指數分 K 的 `volume` 欄其實是成交金額，存檔時已改名為 `turnover`。
  全天分 K 加總約為日 K 成交金額的 97~99%。
- 分 K 的 `datetime` 是台北時間、該分鐘起點；每天 271 根（09:00~13:30），13:30 那根是收盤撮合。
- `sample_sec` 是合成分 K 的原始快照間隔：

  | 期間（櫃買） | sample_sec | 說明 |
  |---|---|---|
  | 2007-07-02 ~ 2011-01-13 | 60 | 每分鐘一筆快照，**開高低收相同** |
  | 2011-01-17 ~ 2014-02-21 | 15 | |
  | 2014-02-24 ~ 2014-12-26 | 10 | |
  | 2014-12-29 起 | 5 | 富邦分 K 也是這份資料合成 |

- 5/15/60 分 K、週 K、月 K 請由 1 分 K / 日 K 自行合成。
- **融資融券**（`margin.parquet`，一天一列，兩個市場欄位相同）：

  | 欄位 | 內容 |
  |---|---|
  | `margin_balance` / `margin_prev` / `margin_change` | 融資今日餘額、前日餘額、增減（張） |
  | `margin_buy` / `margin_sell` / `margin_redeem` | 資買、資賣、現金償還（張） |
  | `margin_*_value` | 以上對應金額（元） |
  | `short_balance` / `short_prev` / `short_change` | 融券餘額、前日餘額、增減（張） |
  | `short_sell` / `short_buy` / `short_redeem` | 券賣、券買回補、券償（張） |

  證交所沒有提供融券金額，所以沒有 `short_*_value`。

## 已知的資料狀況

- 日 K 已抽樣與證交所、櫃買中心官網比對，開高低收完全一致。
- 櫃買中心快照合成的分 K 與富邦 IX0043 分 K 逐根比對：開高低收完全一致，成交金額差距在萬元進位內
  （櫃買中心成交金額單位是萬元）。合成規則寫在 `core/datasource/tpex.py` 開頭。
- 分 K 合成的日 K 與官方日 K 一致，**例外**是櫃買 2024-02-20、2024-02-21：
  交易所盤中序列在收盤前就停止更新，分 K 最後收盤（245.47、245.44）與官方收盤（245.43、246.34）不同。
  需要收盤價時以日 K 為準。
- `sample_sec` 為 60 的年份，每分鐘只有一個點，分 K 的高低點會比真實值窄；
  早年收盤那筆常沒有另外的收盤撮合量，13:30 那根成交金額可能是 0。
- 證交所合成的加權分 K 也與富邦 IX0001 逐根比對過：開高低收完全一致，成交金額差距在百萬元進位內。
- 加權分 K 只補到 2016-10-18（研究範圍起點）；真要往前到 2004-10-15 可用同一支程式加 `--start`。
- 加權日 K 最早 2004-10-15（富邦上限），證交所的日 K 可回溯到 1999-01-05，還沒補。
- 富邦漏給 2024-07-23 09:00 那根分 K 的成交金額，已用證交所同一天的資料補上（其餘欄位兩邊一致）。
- **算融資增減請用 `margin_change`，不要用前後兩列的餘額相減。** 交易所公布的「前日餘額」
  和前一天的「今日餘額」偶爾對不上（加權 207 天、櫃買 129 天，加權最大一次差 52 萬張，
  例如 2024-12-11），這是交易所原始資料就有的重設，不是抓取問題。
  已驗證：`margin_buy - margin_sell - margin_redeem = margin_change`，加權 2,392/2,438 天完全相符
  （其餘最多差 1 張）、櫃買與兩市場的融券都是 100% 相符。
- 加權日 K 有 5 天沒有成交量與成交金額（2006-10-27、2010-06-24、2010-10-01、2011-03-23、2022-09-30）。

## 研究結果

回測結論都寫在 [FINDINGS.md](FINDINGS.md)。

**盤中相似走勢**（`similarity.py`、`backtest_similarity.py`）：原型的參數偏弱（相似日 8 天太少、
標準化基準錯），而且只有接近收盤才有預測力，早盤贏不過「只看上午漲跌」的一行規則。

**收盤後預測隔日**（`daily_features.py`、`backtest_daily.py`、`predict_daily.py`）：
抱過夜（y_cc）贏不過單純做多；「隔日開盤進、收盤出」（y_oc）則顯著有效
（+0.098%/次、t=3.37），最大的訊號是 20 日融資增減。成本超過 0.1% 就不划算。

```bash
..\.venv\Scripts\python.exe predict_daily.py --market otc    # 今天收盤後預測明天
```
