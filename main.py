"""台股 AI 全市場選股與量化分析平台 - 無 Gradio API 版本

前端使用原生 HTML/CSS/JavaScript；後端使用 FastAPI。
API Key 僅存在伺服器環境變數，不暴露到瀏覽器。
"""
import os
import json
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from stock import StockAnalyzer, TaiwanMarketScanner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="台股 AI 全市場選股與量化分析平台",
    description="無 Gradio：原生 Web 前端 + FastAPI 後端",
    version="1.0.0",
)

# 若前後端同網域，實際上不需要 CORS；保留設定方便本機分離開發。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    ticker: str = Field(default="2330", min_length=1, max_length=20)
    period: str = Field(default="1y")


class WatchlistRequest(BaseModel):
    tickers: str = Field(default="2330,2317,2454,2308,2382,3231,6669,3017")


def df_records(df: pd.DataFrame):
    """Convert a DataFrame into strict JSON-safe Python values.

    Important: pandas/numpy NaN and +/-inf cannot be emitted by Starlette's
    JSONResponse. Convert them to JSON null (Python None) first.
    """
    if df is None or df.empty:
        return []

    clean = df.copy().replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(pd.notna(clean), None)
    records = clean.to_dict(orient="records")

    for row in records:
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                row[key] = float(value) if np.isfinite(value) else None
            elif isinstance(value, np.integer):
                row[key] = int(value)

    return records


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "finmind_configured": bool(os.environ.get("FINMIND_TOKEN")),
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "gradio": False,
    }


@app.get("/api/gainers")
def gainers(
    market: str = Query("上市＋上櫃"),
    top_n: int = Query(100, ge=1, le=100),
    min_trade_value: float = Query(0, ge=0),
):
    scanner = TaiwanMarketScanner()
    df, status = scanner.get_top_gainers(market, top_n, min_trade_value)
    return {"status": status, "data": df_records(df)}


@app.get("/api/scan")
def market_scan(
    market: str = Query("上市＋上櫃"),
    candidate_count: int = Query(60, ge=10, le=120),
    min_score: float = Query(60, ge=0, le=100),
):
    scanner = TaiwanMarketScanner()
    df, status = scanner.scan(market, candidate_count, min_score)
    return {"status": status, "data": df_records(df)}


和 Gemini 的對話
為什麼我的https://huggingface.co/spaces/twtshirt/Taiwan-stock-ai-dashboard

跑出來都是nan

在 Hugging Face Spaces 上運行台股儀表板（Taiwan-Stock Dashboard）出現全都是 NaN（或 None/空值）的情況，最常見的原因主要出在 資料獲取（yfinance/TWSE API 阻擋）、資料後處理（欄位未對齊）或指標計算 上。

以下為最常見的 5 大排查方向與修復方式：

1. yfinance 代碼後綴或資料抓取失敗（最常見）
問題：台股在 yfinance 需要加上市場後綴（上市為 .TW，上櫃為 .TWO）。例如台積電必須是 2330.TW，若只傳 2330，下載到的會是空 DataFrame，後續指標計算會全變成 NaN。

雲端 IP 被 Yahoo 阻擋 / 頻率限制：Hugging Face 伺服器所在的 AWS/GCP IP 常被 Yahoo Finance Rate Limit（429 Too Many Requests）。

檢查與修正：

Python
import yfinance as yf

ticker = "2330.TW"
df = yf.download(ticker, period="6mo", progress=False)

if df.empty:
    print("抓取失敗，DataFrame 為空！")
2. MultiIndex 欄位結構（yfinance 新版本變更）
問題：近期版本的 yfinance（0.2.x+）在 download() 時預設會返回 MultiIndex 欄位（如 ('Close', '2330.TW')）。如果程式碼直接取 df['Close']，可能抓取不到或結構錯位，導致所有後續計算結果全為 NaN。

修復方式：

Python
# 方式 A：只抓單一股票時推薦使用 Ticker().history()
stock = yf.Ticker("2330.TW")
df = stock.history(period="6mo")

# 方式 B：若使用 yf.download，強制壓平欄位
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
3. 技術指標計算長度不足（Rolling / EMA / RSI）
問題：若計算 60MA（季線）或布林通道時，如果抓取的歷史天數太少（例如只抓了 1 個月 / 30 筆資料），滾動窗口大於現有資料量，整串輸出就會全為 NaN。

修復方式：確保抓取的 period 至少為 1y 或 6mo，或設定 min_periods=1：

Python
df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
4. 時區與非交易日轉換（tz-aware vs tz-naive）
問題：yfinance 抓下來的資料索引帶有時區（如 UTC 或 Asia/Taipei）。若程式碼拿它與自定義的日期（如 pd.to_datetime('today')）進行 merge 或 reindex，因時區不匹配無法對齊，合併後的值會全變成 NaN。

修復方式：

Python
df.index = df.index.tz_localize(None)  # 移除時區資訊再做運算/合併
5. Hugging Face 建議檢查步驟
進入你的 Space 頁面，點擊右上角 Logs / Runtime logs。

查看終端是否有印出 HTTPError: 429 Too Many Requests、KeyError: 'Close' 或空資料警告。

若是 API 阻擋問題，可考慮加入請求標頭（Headers / Session）或改用台灣證券交易所開源 API（如 FinMind、twstock 或 TWSE OpenAPI）。

會跟用量有關係嗎

有非常大的關係。 用量（請求頻率、並發量、歷史抓取量）是導致雲端儀表板出現 NaN 最常見的隱形殺手。

1. Yahoo Finance 的 IP 級別限流（Rate Limit / 429）
共用 IP 災情：Hugging Face Spaces 免費版部署在 AWS 雲端伺服器上。Yahoo Finance 會針對整個資料中心 IP 網段進行嚴格限流。即使你的 Space 請求量不大，同伺服器其他人的爬蟲也可能觸發 Yahoo 阻擋。

短時間高頻請求：如果你在頁面載入時一口氣抓取數十檔台股，或者在短時間內反覆切換頁面刷新，Yahoo 就會回傳 429 Too Many Requests 或 Empty Response，此時 Python 拿不到報價，計算後就全部變成 NaN。

2. 本地（Local）與雲端（HF Space）的快取差異
本地跑正常、上傳就掛：本地電腦通常有本機快取（或住宅 IP 不易被擋），但在 Hugging Face 每次重啟容器時都沒有快取，全部從頭抓取，極易在開機或大量並發時直接撞到限流牆。

3. 如何驗證是否是用量/限流問題？
打開 Hugging Face Space 的 Runtime Logs，若看到以下關鍵字，就百分之百是用量與限流問題：

429 Client Error: Too Many Requests

YFPricesMissingError: ... No price data found

JSONDecodeError: Expecting value

4. 解決用量與限流的 3 種解法
解法 A：加入快取機制（Cache）

如果是使用 Streamlit 或 Gradio，一定要加上快取裝飾器，避免每次使用者點擊或重新整理就重複向 Yahoo 發送 API 請求：

Python
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=300)  # 快取 5 分鐘，避免短時間重複請求
def get_stock_data(ticker):
    return yf.download(ticker, period="6mo", progress=False)
解法 B：在大量抓取之間加入延遲（Sleep）

若一次要抓取多檔股票，避免同時並發，請加入隨機延遲：

Python
import time
# 在迴圈抓取時加入間隔
time.sleep(1.0)
解法 C：改用官方/專用 API

yfinance 屬於逆向工程爬蟲，極不穩定。若儀表板需要高頻率或多人存取，建議改用 FinMind（免費提供 API Token，額度高且穩定）或 證交所 OpenAPI。

想知道如何把 yfinance 換成更穩定的 FinMind API 嗎？

是
我目前是使用FinMind API Token

如果已經換成 FinMind API Token 依然全出 NaN，用量依然可能超標，但更常見的是「Secrets 環境變數遺失」或「FinMind 欄位名稱對應錯誤」。

1. Token 用量超標（Rate Limit: 600 次 / 小時）
限額機制：免費註冊驗證後的 FinMind Token 上限通常為 600 requests / hour（若未加 Token 只有 300/hr）。

儀表板常見陷阱：

若一次載入數十檔股票，且每檔分別發送「日K、三大法人、融資券」等多個 API 請求，點選 5~10 次頁面就會直接用光一小時的額度。

超額表現：FinMind 超過上限時會回傳 {'msg': 'Limit amount of request...'}，此時轉出的 DataFrame 為空，後續計算全數變成 NaN。

2. Hugging Face Spaces 讀不到 Token（最常見）
問題：如果在本機把 Token 寫在 .env 或變數中，上傳到 HF Space 後若沒有在後台設定，os.environ.get("FINMIND_TOKEN") 會拿到 None 或空字串。

解決方式：

進入你的 Hugging Face Space 頁面。

點擊 Settings -> 找到 Variables and secrets。

新增 Secret（例如 Key 名稱為 FINMIND_TOKEN，Value 填入你的 Token）。

程式碼中讀取方式：

Python
import os
token = os.environ.get("FINMIND_TOKEN", "")
3. 欄位名稱大小寫與格式不相符（Code Bug）
FinMind 回傳的 DataFrame 欄位名稱與一般的 yfinance / TA-Lib 常見名稱不同，若直接呼叫 df['Close'] 會找不到欄位，計算就會產生全 NaN：

資料項目	yfinance 欄位名	FinMind 欄位名
收盤價	Close	close（全小寫）
開盤價	Open	open（全小寫）
最高價	High	max（非 high）
最低價	Low	min（非 low）
成交量	Volume	Trading_Volume
日期	DatetimeIndex	date（是一般字串欄位，非 Index）
標準轉換寫法：

Python
# 取得 FinMind 資料後，建議先將欄位與索引對齊
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.rename(columns={'max': 'high', 'min': 'low'}, inplace=True)
4. 解決與優化方案
加入快取降低用量：

Python
import streamlit as st
from FinMind.data import DataLoader

@st.cache_data(ttl=3600)  # 快取 1 小時
def get_stock_data(stock_id, start_date):
    api = DataLoader()
    api.login_by_token(api_token=os.environ.get("FINMIND_TOKEN"))
    df = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
    return df
檢查 Log：至 Space 的 Logs 標籤查看是否有印出 Limit amount of request 或 KeyError，能立即鎖定是額度耗盡還是欄位名稱問題。

要我幫你檢查處理 FinMind 欄位與指標運算的程式碼嗎？

是
# 🏆 引入 XGBoost 梯度提升演算法進行股價漲跌預測

!pip install -q xgboost scikit-learn yfinance



import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

import yfinance as yf

import pandas as pd

import numpy as np



# 1. 下載並構建特徵

df = yf.download('2330.TW', period='3y', progress=False)

if isinstance(df.columns, pd.MultiIndex):

    df.columns = [c[0].lower() for c in df.columns]



df['ret_1d'] = df['close'].pct_change(1)

df['ret_5d'] = df['close'].pct_change(5)

df['ma20'] = df['close'].rolling(20).mean()

df['bias_20'] = (df['close'] - df['ma20']) / df['ma20']

df['vol_change'] = df['volume'].pct_change()



# 標籤：明日是否上漲

df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

df_clean = df.dropna()



features = ['ret_1d', 'ret_5d', 'bias_20', 'vol_change']

X = df_clean[features]

y = df_clean['Target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)



# 2. 訓練 XGBoost

model = xgb.XGBClassifier(

    n_estimators=150,

    max_depth=4,

    learning_rate=0.05,

    random_state=42

)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print(f"🎯 XGBoost 測試集準確度: {accuracy_score(y_test, y_pred):.2%}")

print("\n分類詳細報告:")

print(classification_report(y_test, y_pred))

這段程式碼的架構基本完整，但在實務與 Hugging Face 雲端部署環境中，有 3 個致命 Bug 與 1 個特徵工程缺陷 會導致抓不到資料、產生 NaN 或破壞預測邏輯：

1. yfinance 欄位壓平寫法錯誤（最直接的 NaN / KeyError 來源）
問題：yfinance 產生的 MultiIndex 格式通常為 ('Close', '2330.TW')。程式碼中使用 c[0].lower() 雖然取出了名稱，但若不是 MultiIndex（例如在特定 yfinance 版本下），c[0] 會直接抓取欄位字串的第一個字元（如 'c'），導致找不到 'close' 欄位而報錯或整張表變成 NaN。

修正方式：

Python
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(c).lower() for c in df.columns]
2. 資料洩漏（Data Leakage / Lookahead Bias）
問題：

Python
df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
最後一筆資料因為 shift(-1) 會變成 NaN，經過 dropna() 後，最新的今天資料會被直接刪掉。

影響：這導致模型「永遠無法預測明天的漲跌」，只能回測歷史資料。如果要用最新一天的特徵預測明天，必須將最新一列與歷史訓練集分開處理。

3. 未處理非交易日 / 零成交量的極端值
問題：台股有時會有停牌、連續假期或成交量變動極端的情況，df['volume'].pct_change() 可能產生 inf 或 -inf，dropna() 無法過濾 inf，後續丟入 XGBoost 會直接報錯（ValueError: Input contains NaN, infinity or a value too large）。

修正方式：

Python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
4. 完整優化修正後的程式碼
Python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
import pandas as pd
import numpy as np

# 1. 下載資料
ticker = '2330.TW'
df = yf.download(ticker, period='3y', progress=False)

if df.empty:
    raise ValueError(f"無法取得 {ticker} 的股價資料，請檢查代碼或網路連線。")

# 規範化欄位名稱為小寫
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.columns = [str(c).lower() for c in df.columns]

# 2. 構建特徵
df['ret_1d'] = df['close'].pct_change(1)
df['ret_5d'] = df['close'].pct_change(5)
df['ma20'] = df['close'].rolling(20).mean()
df['bias_20'] = (df['close'] - df['ma20']) / df['ma20']
df['vol_change'] = df['volume'].pct_change()

# 標籤：明日是否上漲 (1 = 漲, 0 = 跌/平)
df['target'] = (df['close'].shift(-1) > df['close']).astype(float)
# 確保最後一筆是 NaN (因為明天還沒發生)
df.loc[df.index[-1], 'target'] = np.nan

# 處理無窮大值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. 分離「最新一天待預測資料」與「歷史訓練資料」
features = ['ret_1d', 'ret_5d', 'bias_20', 'vol_change']

latest_row = df.iloc[[-1]][features]  # 最新一天的特徵 (用來預測明天)
train_df = df.iloc[:-1].dropna(subset=features + ['target'])  # 歷史有標籤的資料

X = train_df[features]
y = train_df['target'].astype(int)

# 時間序列切分 (嚴禁 shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. 訓練 XGBoost
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 5. 評估模型
y_pred = model.predict(X_test)
print(f"🎯 XGBoost 測試集準確度: {accuracy_score(y_test, y_pred):.2%}")
print("\n分類詳細報告:")
print(classification_report(y_test, y_pred))

# 6. 預測下一個交易日
next_day_pred = model.predict(latest_row)[0]
next_day_prob = model.predict_proba(latest_row)[0][1]

direction = "🟢 上漲" if next_day_pred == 1 else "🔴 下跌/平盤"
print(f"\n🔮 下一交易日預測結果: {direction} (上漲機率: {next_day_prob:.2%})")
想請問這個檔案為什麼無法讀取到上櫃股票

這個檔案無法讀取上櫃股票（OTC）的主要原因，是因為在擷取 Yahoo Finance (yfinance) 資料時，程式碼強制將所有股票代號都加上了上市的 .TW 後綴，導致上櫃股票專用的 .TWO 後綴被刪除並覆寫。

🔍 錯誤原因分析
根據提供的 stock.py，問題出在以下兩個連動的設計片段：

第一處（初始化時清除了上櫃後綴）：在 StockAnalyzer 類別的 __init__ 方法中，程式碼會把傳入的 .TWO 和 .TW 全部強制刪除，只保留純數字代號。  
PY

Python
self.ticker = ticker.replace('.TWO', '').replace('.TW', '') # 確保內部存儲不含 .TW
第二處（下載時寫死了上市後綴）：在 fetch_data 方法中呼叫 yfinance 下載 K 線資料時，字串格式化直接寫死了 .TW。這使得原本應該是上櫃的股票，也會被強制當作上市股票去抓取（例如上櫃股穩懋 3105 會變成 3105.TW，導致找不到資料回傳空值）。  
PY
+ 1

Python
self.df = yf.download(f'{self.ticker}.TW', period=period, progress=False)
🛠️ 建議的修改方式
要解決這個問題，必須將純數字代號（供 FinMind 使用）與帶有市場後綴的代號（供 Yahoo Finance 使用）分開處理。請針對 stock.py 進行以下兩處修改：

1. 修改 __init__ 方法，新增一個保留後綴的屬性：

Python
def __init__(self, ticker):
    # 判斷並保留 yfinance 專用的完整代號 (包含 .TW 或 .TWO)
    if not ticker.endswith(('.TW', '.TWO')):
        # 若使用者沒輸入後綴，這裡可以自訂預設值，或由外部強制要求輸入完整代號
        self.yahoo_ticker = f"{ticker}.TW" 
    else:
        self.yahoo_ticker = ticker
        
    # 原本的 self.ticker 保持純數字，供後續 FinMind API 使用
    self.ticker = ticker.replace('.TWO', '').replace('.TW', '') 
    # ... 以下保留原程式碼 ...
2. 修改 fetch_data 方法，使用正確的完整代號：

Python
def fetch_data(self, period='3y'):
    print(f"\n----- 開始為 {self.ticker} 下載數據 (期間: {period}) -----")
    
    # 將原本寫死的 f'{self.ticker}.TW' 改為 self.yahoo_ticker
    self.df = yf.download(self.yahoo_ticker, period=period, progress=False) 
    
    if self.df.empty:
        print(f"錯誤: 無法下載 {self.yahoo_ticker} 的 K 線數據。")
        return False
    # ... 以下保留原程式碼 ...
這樣修改後，程式就能順利透過 3105.TWO 抓取上櫃股票資料，同時不影響原本 FinMind API 需要純數字 3105 的查詢邏輯。

需要我順便幫你檢查 scan_watchlist 或是這份檔案中其他用到 API 的方法是否有遺漏的上櫃相容性問題嗎？

修改後此方法會造成錯誤無法執行, 請用其他方法修改

如果前一個方法（修改 __init__ 增加新屬性）會導致你的主程式報錯，通常是因為你的框架（例如 Streamlit 或其他模組）在實例化 StockAnalyzer 時有嚴格的屬性綁定，或者資料傳遞時不相容新的變數。

我們改用 「自動偵測與Fallback（降級嘗試）」 的無侵入式修改法。

你完全不需要修改 __init__，保持它原本過濾掉後綴的樣子。我們只針對 fetch_data 方法進行修改：讓程式先去 Yahoo Finance 嘗試抓取上市（.TW），如果發現資料是空的，就自動切換成上櫃（.TWO）再抓一次。  
PY

請將 stock.py 裡面的 fetch_data 替換成以下寫法：

Python
    def fetch_data(self, period='3y'):
        """下載股票數據、財報、法人籌碼、融資融券、期貨借券等數據。"""
        print(f"\n----- 開始為 {self.ticker} 下載數據 (期間: {period}) -----")
        
        # 1. 先嘗試當作「上市股票」抓取 (.TW)
        self.df = yf.download(f'{self.ticker}.TW', period=period, progress=False)
        
        # 2. 如果抓不到資料 (DataFrame為空)，自動切換為「上櫃股票」抓取 (.TWO)
        if self.df.empty:
            self.df = yf.download(f'{self.ticker}.TWO', period=period, progress=False)
            
        # 3. 如果還是空的，代表代號錯誤或網路問題
        if self.df.empty:
            print(f"錯誤: 無法下載 {self.ticker} 的 K 線數據 (上市與上櫃皆無資料)。")
            return False
            
        self.df = self._fix_col_names(self.df)
        self.df.index = pd.to_datetime(self.df.index)

        # 均線
        self.df['ma5'] = self.df['close'].rolling(window=5).mean()
        # ... 後續所有指標計算程式碼保持原樣不變 ...
為什麼這個寫法更安全？
不改變類別結構：你不需要新增 self.yahoo_ticker，self.ticker 依舊維持純數字（例如 3105），這確保了你後續呼叫 FinMind API 時不會出錯（因為 FinMind 只接受純數字）。  
PY

無痛相容：無論使用者在介面上輸入 3105、3105.TWO 還是 3105.TW，經過 __init__ 清洗後都會變成 3105，然後交由這個方法自動去向 Yahoo 試探真實的市場類別。  
PY

(另外建議：你的 get_bias_text 與 get_indicator_summary 方法中有寫死回傳字串為 {self.ticker}.TW，若要讓上櫃股票顯示得更精確，可以把字串裡的 .TW 直接刪除，改為 {self.ticker} 最新 20 日乖離率 即可)  
PY

請問這樣修改後，程式能順利印出上櫃股票的指標圖表了嗎？

目前無法讀取上櫃股票的問題還是沒解決, 是否是因為附件此檔案中有問題?

是的，你的懷疑完全正確，這份 main.py 檔案中確實存在攔截並破壞上櫃股票代號的邏輯。

你在 stock.py 做的修改之所以沒有生效，是因為在資料送達 stock.py 之前，main.py 就已經把 .TWO 的後綴給刪除了，而且還在回傳給前端時強制加上了 .TW。  
PY

🔍 找出 main.py 的問題點
在 main.py 的 /api/analyze 路由中，有兩個地方寫死了上市後綴：

破壞使用者輸入：

Python
ticker = req.ticker.strip().upper().replace(".TW", "").replace(".TWO", "")
這行程式碼會將前端傳來的 3105.TWO 強制閹割成 3105。  
PY

硬碟寫死回傳值：

Python
"ticker": f"{ticker}.TW",
不管後端實際抓到什麼資料，回傳給前端顯示的字串都被寫死為 {ticker}.TW，這會導致前端圖表與指標全部顯示上市的代號。  
PY

Python
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    allowed_periods = {"3mo", "6mo", "1y", "2y", "3y", "5y", "max"}
    if req.period not in allowed_periods:
        raise HTTPException(status_code=400, detail="不支援的資料區間")

    # 保留使用者的原始輸入 (例如: 3105, 3105.TW, 或 3105.TWO)
    original_ticker = req.ticker.strip().upper()
    
    # 提取純數字代號供 FinMind 使用
    pure_ticker = original_ticker.replace(".TW", "").replace(".TWO", "")
    
    if not pure_ticker:
        raise HTTPException(status_code=400, detail="請輸入股票代號")

    # 傳入純數字代號給 StockAnalyzer (搭配你在 stock.py 中實作的 TW/TWO 自動降級嘗試)
    analyzer = StockAnalyzer(pure_ticker)
    if not analyzer.fetch_data(period=req.period):
        raise HTTPException(status_code=404, detail=f"無法取得 {original_ticker} 的股票資料")

    analyzer.engineer_features()
    plot = analyzer.get_multi_panel_plot(analyzer.df)

    # 動態決定回傳的顯示代號 (若有抓到資料，可依據使用者輸入或預設返回)
    display_ticker = original_ticker if ".TWO" in original_ticker else f"{pure_ticker}.TW"

    return {
        "ticker": display_ticker,  # 移除寫死的 f"{ticker}.TW"
        "bias": analyzer.get_bias_text(analyzer.df),
        "indicator_summary": analyzer.get_indicator_summary(),
        "macd_analysis": analyzer.get_macd_diagnostics(),
        "leading_analysis": analyzer.get_leading_indicators_analysis(),
        "news": df_records(analyzer._fetch_news(days=5)),
        "news_analysis": analyzer.get_news_analysis(days=5),
        "ai_analysis": analyzer.get_ai_prediction_text(),
        "plot": json.loads(plot.to_json()),
    }

@app.post("/api/watchlist")
def watchlist(req: WatchlistRequest):
    analyzer = StockAnalyzer("2330")
    df = analyzer.scan_watchlist(req.tickers)
    return {
        "status": f"完成 {len(df)} 檔自選股量化排名。" if not df.empty else "沒有足夠歷史資料。",
        "data": df_records(df),
    }
