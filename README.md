# 台股大盤指數分析

加權指數（TSE, IX0001）與櫃買指數（OTC, IX0043）的 1 分 K、日 K 與信用交易資料，
以及「找出走勢相似的歷史日、看它們隔天怎麼走」的分析工具。

研究範圍是近 10 年（2016-10-18 起），兩個市場各 2,438 個交易日。

## 這個專案做了什麼

**資料**：交易所只公布盤中「快照」（每 5 秒一筆的指數值），不是 K 棒。
這裡把快照合成 1 分 K，合成規則已與富邦 Neo API 的分 K 逐根比對過：
開高低收完全一致，成交金額差距在交易所的進位單位內（櫃買萬元、證交所百萬元）。

| 資料 | 加權 | 櫃買 |
|---|---|---|
| 日 K | 2004-10-15 起（富邦） | 1999-09-01 起（櫃買中心＋富邦） |
| 1 分 K | 2016-10-18 起（證交所＋富邦） | 2007-07-02 起（櫃買中心＋富邦） |
| 融資融券 | 2016-10-18 起（證交所） | 2016-10-18 起（櫃買中心） |

**分析**：以「當天江波走勢形狀」為主、量能／融資／波動／位階為輔，
找出歷史上最相似的日子，列出它們隔天的走勢。

## 快速開始

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 1. 抓資料（只用交易所公開資料，不需要任何帳號；完整補完約 3 小時）
.venv\Scripts\python.exe market-index\backfill_official.py --market otc --start 2016-10-18
.venv\Scripts\python.exe market-index\backfill_official.py --market tse --start 2016-10-18

# 2. 找相似日，輸出走勢圖
.venv\Scripts\python.exe market-index\similar_days.py --market otc --chart --explain

# 3. 測試
.venv\Scripts\pytest
```

有富邦證券帳號的話，`market-index\fetch_fubon.py` 可以用 API 抓（快很多，每天更新約 10 秒），
把 `.env.example` 複製成 `.env` 填入登入資訊即可。沒有帳號不影響其他功能。

## 相似日比對

```bash
# 預設：形狀 60%，量能／融資／波動／位階各 10%
python market-index\similar_days.py --market otc --chart

# 調整比重（0 = 不看該組）
python market-index\similar_days.py --market otc --shape-weight 0.5 --weights 量能=2 融資=2

# 允許時間錯開（都是 V 轉但一個早一小時也算像）
python market-index\similar_days.py --market otc --metric dtw
```

比對時每一項都先用「查詢日之前」的分布標準化，否則尺度不同、權重形同虛設；
相鄰 5 個交易日只留最相似的一天，避免同一段行情重複入選。

## 回測結論

完整內容在 [market-index/FINDINGS.md](market-index/FINDINGS.md)，重點是**誠實地標出哪些做法沒有用**：

- **盤中相似走勢**：早盤（09:30~10:30）沒有預測力，贏不過「只看上午漲跌」的一行規則；
  越接近收盤越有效，13:00 之後櫃買方向命中率 65%、加權 57%。
- **收盤後預測隔日**：抱過夜贏不過單純做多（超越基準 t=0.73，不顯著）；
  「隔日開盤進、收盤出」則顯著（+0.098%/次、t=3.37），最大的訊號是 20 日融資增減。
- **成本很關鍵**：來回成本 0.1% 以上，上述優勢就消失了。
- 所有評估都是 walk-forward：每天只用該日之前的資料，測試裡也包含「模型不能偷看當天答案」的檢查。

## 目錄

```
core/datasource/     資料來源（API 呼叫只在這裡）
  fubon.py           富邦 Neo API（需帳號）
  twse.py            證交所公開資料
  tpex.py            櫃買中心公開資料
market-index/
  store.py           檔案版面與讀寫
  fetch_fubon.py     富邦抓日 K / 分 K
  backfill_official.py  用交易所公開資料補歷史與融資
  similarity.py      盤中走勢比對
  similar_days.py    找相似日 + 列出隔天走勢
  similar_days_chart.py  走勢圖
  daily_features.py  收盤後的特徵與目標
  backtest_*.py      walk-forward 回測
  predict_daily.py   收盤後預測隔日
tests/               含「不能用到未來資料」的檢查
```

## 資料來源

- 證交所 [MI_5MINS_INDEX](https://www.twse.com.tw/zh/trading/historical/mi-5mins-index.html)（每 5 秒指數）、MI_5MINS（成交統計）、MI_MARGN（信用交易）
- 櫃買中心 [miIndex](https://www.tpex.org.tw/zh-tw/indices/stock-index/industrial/sec.html)（每 5 秒指數）、inx（日 K）、margin/balance（融資融券）
- 富邦新一代 API（選配）

公開資料的使用請遵守各交易所的網站規範；本專案只保留抓取程式，不散布資料檔。

## 免責聲明

這是個人研究用的程式與統計結果，不是投資建議。回測是在無成本或低成本假設下的歷史統計，
不代表未來績效，也沒有模擬實際下單的滑價與成交狀況。
