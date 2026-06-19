#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Market Analysis Telegram Bot
Major Pairs: EUR/USD, GBP/USD, USD/JPY, GBPAUD, AUD/USD
Data: TradingView (tvdatafeed) with yfinance fallback
"""

import os, sys, logging, time, urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import pytz
from datetime import datetime
from dotenv import load_dotenv

# tvdatafeed removed due to PyPI/Cloudflare issues. Using yfinance as primary source.

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("FOREX_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("FOREX_TELEGRAM_CHAT_ID", "")
BD_TZ  = pytz.timezone("Asia/Dhaka")
UTC_TZ = pytz.utc

PAIRS = {
    "EUR/USD": {"tv": "EURUSD", "yf": "EURUSD=X"},
    "GBP/USD": {"tv": "GBPUSD", "yf": "GBPUSD=X"},
    "USD/JPY": {"tv": "USDJPY", "yf": "JPY=X"},
    "GBPAUD":  {"tv": "GBPAUD", "yf": "GBPAUD=X"},
    "AUD/USD": {"tv": "AUDUSD", "yf": "AUDUSD=X"}
}

# ══════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════

def yf_fetch(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """Fallback: fetch from yfinance"""
    try:
        import yfinance as yf
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        df = yf.Ticker(symbol, session=session).history(period=period, interval=interval)
        if df is None or len(df) < 5:
            logger.warning(f"yf {symbol} {interval}: Data empty or less than 5 rows. Rows: {len(df) if df is not None else 0}")
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        return df
    except Exception as e:
        logger.warning(f"yf {symbol} {interval}: Exception {e}")
        return None

def get_ohlcv(pair: str, tf: str) -> pd.DataFrame | None:
    """Fetch OHLCV based on timeframe."""
    info = PAIRS[pair]
    df = None
    
    # Mapping
    if tf == "weekly":
        df = yf_fetch(info["yf"], "2y", "1wk")
    elif tf == "daily":
        df = yf_fetch(info["yf"], "90d", "1d")
    elif tf == "4h":
        # yfinance doesn't natively support robust 4h interval for all assets easily
        # "1h" used as substitute or fetch 1h and resample
        df = yf_fetch(info["yf"], "30d", "1h") 
        if df is not None:
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
    elif tf == "1h":
        df = yf_fetch(info["yf"], "7d", "1h")
    elif tf == "15m":
        df = yf_fetch(info["yf"], "3d", "15m")
        
    if df is not None:
        logger.info(f"{pair} {tf}: {len(df)} rows")
    else:
        logger.warning(f"{pair} {tf}: Failed to fetch")
    return df

def get_live_price(pair: str) -> float | None:
    """Get latest close price from 15m candle."""
    df = get_ohlcv(pair, "15m")
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1])
    return None

# ══════════════════════════════════════════════
# FUNDAMENTALS
# ══════════════════════════════════════════════

def fetch_forex_news() -> list[str]:
    """Fetch Currencies news from CNBC"""
    news = []
    try:
        url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        for item in root.findall('.//item')[:3]:
            title_elem = item.find('title')
            if title_elem is not None and title_elem.text:
                news.append(title_elem.text.strip())
    except Exception as e:
        logger.warning(f"News fetch error: {e}")
    return news

def analyze_forex_sentiment(news_list: list[str], pair: str) -> dict:
    if not news_list:
        return {"bias": "MIXED", "text": "⚪ নিউজের প্রভাব অস্পষ্ট।"}
    
    text = " ".join(news_list).lower()
    
    usd_weak = sum(1 for w in ["rate cut", "dovish", "weak dollar", "drop", "lower", "inflation drop", "slowdown"] if w in text)
    usd_strong = sum(1 for w in ["rate hike", "hawkish", "strong dollar", "rise", "inflation rise", "strong jobs", "growth"] if w in text)
    
    # Base vs Quote Logic
    if pair.endswith("USD"): # EUR/USD, GBP/USD, AUD/USD
        bull_score = usd_weak
        bear_score = usd_strong
    elif pair.startswith("USD"): # USD/JPY
        bull_score = usd_strong
        bear_score = usd_weak
    else: # GBPAUD
        return {"bias": "MIXED", "text": "⚪ ক্রস পেয়ারে ডিরেক্ট USD ইমপ্যাক্ট মিক্সড হতে পারে।"}
        
    if bull_score > bear_score:
        return {"bias": "BULLISH", "text": f"🟢 ফান্ডামেন্টাল নিউজ অনুযায়ী {pair} **উপরের দিকে (Bullish)** যেতে পারে।"}
    elif bear_score > bull_score:
        return {"bias": "BEARISH", "text": f"🔴 ফান্ডামেন্টাল নিউজ অনুযায়ী {pair} **নিচের দিকে (Bearish)** যেতে পারে।"}
    else:
        return {"bias": "MIXED", "text": "⚪ নিউজগুলো মিক্সড, মার্কেটে **ভোলাটিলিটি (উঠা-নামা)** দেখা যেতে পারে।"}

# ══════════════════════════════════════════════
# INDICATORS & LOGIC (Reused from Gold/BTC)
# ══════════════════════════════════════════════

def calc(df: pd.DataFrame, cur_price: float | None = None) -> dict | None:
    if df is None or len(df) < 14: return None
    close = df["close"].copy()
    high, low = df["high"], df["low"]
    if cur_price: close.iloc[-1] = cur_price
    try:
        n = len(close)
        rsi = RSIIndicator(close, 14).rsi().iloc[-1]
        mo = MACD(close)
        macd, msig, mhist = mo.macd().iloc[-1], mo.macd_signal().iloc[-1], mo.macd_diff().iloc[-1]
        e20 = EMAIndicator(close, min(20, n-1)).ema_indicator().iloc[-1]
        e50 = EMAIndicator(close, min(50, n-1)).ema_indicator().iloc[-1]
        s200 = SMAIndicator(close, min(200, n)).sma_indicator().iloc[-1]
        bb = BollingerBands(close, min(20, n-1), 2)
        bb_p = bb.bollinger_pband().iloc[-1]
        atr = AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        price = float(close.iloc[-1])
        pct = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        w = min(20, n)
        rh, rl = float(high.iloc[-w:].max()), float(low.iloc[-w:].min())
        pv = (rh + rl + price) / 3
        return {
            "price": price, "pct": pct, "rsi": rsi, "macd": macd, "msig": msig, "mhist": mhist,
            "e20": e20, "e50": e50, "s200": s200, "bb_p": bb_p, "atr": atr,
            "r2": pv+(rh-rl), "r1": 2*pv-rl, "s1": 2*pv-rh, "s2": pv-(rh-rl),
        }
    except:
        return None

def tf_bias(ind: dict | None) -> tuple[str, str]:
    if ind is None: return "❓", "No data"
    p, rsi, mhist, e20, e50, bb_p = ind["price"], ind["rsi"], ind["mhist"], ind["e20"], ind["e50"], ind["bb_p"]
    bull = sum([p > e20, p > e50, rsi > 50, mhist > 0, bb_p < 0.5])
    if bull >= 4: return "📈", "Bullish"
    elif bull == 3: return "🔼", "Slightly Bullish"
    elif bull <= 1: return "📉", "Bearish"
    else: return "🔽", "Slightly Bearish"

def get_signal(ind: dict) -> dict:
    p, rsi, mhist, e20, e50, s200, bb_p = ind["price"], ind["rsi"], ind["mhist"], ind["e20"], ind["e50"], ind["s200"], ind["bb_p"]
    b = s = 0
    if rsi < 35: b += 2
    elif rsi > 65: s += 2
    elif rsi > 50: b += 1
    else: s += 1
    b += 1 if mhist > 0 else 0; s += 0 if mhist > 0 else 1
    b += 1 if p > e20 else 0; s += 0 if p > e20 else 1
    b += 1 if p > e50 else 0; s += 0 if p > e50 else 1
    b += 1 if p > s200 else 0; s += 0 if p > s200 else 1
    if bb_p < 0.2: b += 2
    elif bb_p > 0.8: s += 2
    total = b + s
    if total == 0: return {"bias": "NEUTRAL", "str": "WEAK"}
    if b > s: return {"bias": "BUY", "str": "STRONG" if b/total >= 0.7 else "MODERATE"}
    if s > b: return {"bias": "SELL", "str": "STRONG" if s/total >= 0.7 else "MODERATE"}
    return {"bias": "NEUTRAL", "str": "MODERATE"}

# ══════════════════════════════════════════════
# MESSAGE FORMATTING
# ══════════════════════════════════════════════

def format_price(pair: str, price: float) -> str:
    return f"{price:,.3f}" if "JPY" in pair else f"{price:,.5f}"

def build_report(pair: str, is_daily: bool, global_news: list) -> str:
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    price = get_live_price(pair)
    
    if is_daily:
        w_df, d_df = get_ohlcv(pair, "weekly"), get_ohlcv(pair, "daily")
        w_ind, d_ind = calc(w_df), calc(d_df, price)
        if not price and d_ind: price = d_ind["price"]
        if not price: return f"💱 *{pair}*\n❌ ডেটা পাওয়া যায়নি।"
        
        w_icon, w_lbl = tf_bias(w_ind)
        d_icon, d_lbl = tf_bias(d_ind)
        primary = d_ind or w_ind
        sig = get_signal(primary) if primary else {"bias": "NEUTRAL"}
        bias_txt = {"BUY": "🟢 BULLISH", "SELL": "🔴 BEARISH", "NEUTRAL": "⚪ NEUTRAL"}[sig["bias"]]
        
        lev = primary or {}
        p_fmt = lambda x: format_price(pair, x)
        
        sentiment = analyze_forex_sentiment(global_news, pair)
        news_text = "\n📰 *ফান্ডামেন্টাল আপডেট:*\n" + "\n".join([f"🔹 {n}" for n in global_news]) + f"\n💡 *ফান্ডামেন্টাল বায়াস:* {sentiment['text']}\n" if global_news else ""

        return (
            f"💱 *{pair}*\n"
            f"🌅 *দৈনিক ফরেক্স বায়াস — {datetime.now(BD_TZ).strftime('%d %b %Y')}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 *মূল্য:* `{p_fmt(price)}`\n\n"
            f"📊 *হায়ার টাইমফ্রেম:*\n"
            f"  🗓 Weekly: {w_icon} {w_lbl}\n"
            f"  📅 Daily:  {d_icon} {d_lbl}\n\n"
            f"🔑 *আজকের লেভেল:*\n"
            f"  🔴 রেজিস্ট্যান্স: `{p_fmt(lev.get('r1',0))}` → `{p_fmt(lev.get('r2',0))}`\n"
            f"  🟢 সাপোর্ট: `{p_fmt(lev.get('s1',0))}` → `{p_fmt(lev.get('s2',0))}`\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *আজকের বায়াস: {bias_txt}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
            f"{news_text}"
        )
    else:
        h4_df, h1_df, m15_df = get_ohlcv(pair, "4h"), get_ohlcv(pair, "1h"), get_ohlcv(pair, "15m")
        h4_ind, h1_ind, m15_ind = calc(h4_df, price), calc(h1_df, price), calc(m15_df, price)
        primary = h4_ind or h1_ind or m15_ind
        if not primary: return f"💱 *{pair}*\n❌ ডেটা পাওয়া যায়নি।"
        
        cur = primary["price"]
        pct = primary.get("pct", 0)
        atr = primary["atr"]
        p_fmt = lambda x: format_price(pair, x)
        
        h4_icon, h4_lbl = tf_bias(h4_ind)
        h1_icon, h1_lbl = tf_bias(h1_ind)
        m15_icon, m15_lbl = tf_bias(m15_ind)
        
        sig = get_signal(primary)
        sl = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "NEUTRAL": "⚪ WAIT"}[sig["bias"]]
        
        if sig["bias"] == "BUY":
            sl_p, tp1, tp2 = cur-1.5*atr, cur+1.5*atr, cur+2.5*atr
            trade = f"🟢 *Buy* | Entry: `{p_fmt(cur)}`\n  SL: `{p_fmt(sl_p)}` | TP1: `{p_fmt(tp1)}` | TP2: `{p_fmt(tp2)}`"
        elif sig["bias"] == "SELL":
            sl_p, tp1, tp2 = cur+1.5*atr, cur-1.5*atr, cur-2.5*atr
            trade = f"🔴 *Sell* | Entry: `{p_fmt(cur)}`\n  SL: `{p_fmt(sl_p)}` | TP1: `{p_fmt(tp1)}` | TP2: `{p_fmt(tp2)}`"
        else:
            trade = f"⚪ *অপেক্ষা করুন* | S: `{p_fmt(primary['s1'])}` – R: `{p_fmt(primary['r1'])}`"

        sentiment = analyze_forex_sentiment(global_news, pair)
        news_text = "\n📰 *ফান্ডামেন্টাল আপডেট:*\n" + "\n".join([f"🔹 {n}" for n in global_news]) + f"\n💡 *ফান্ডামেন্টাল বায়াস:* {sentiment['text']}\n⚖️ *ওভারঅল ভিউ:* টেকনিক্যাল সিগন্যাল {sl}, এবং ফান্ডামেন্টাল অনুযায়ী {sentiment['bias']}।\n" if global_news else ""

        return (
            f"💱 *{pair}* — ফরেক্স ইন্ট্রাডে\n"
            f"🕐 {now_bd}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💰 `{p_fmt(cur)}` ({pct:+.2f}%)\n\n"
            f"📊 *টাইমফ্রেম বিশ্লেষণ:*\n"
            f"  ⏱ 4H:  {h4_icon} {h4_lbl}\n"
            f"  ⏰ 1H:  {h1_icon} {h1_lbl}\n"
            f"  ⚡ 15M: {m15_icon} {m15_lbl}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *সিগনাল: {sl}*\n"
            f"{trade}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
            f"{news_text}"
        )

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing Forex Telegram credentials!"); return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=30)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def main():
    # Weekend check
    utc_now = datetime.now(UTC_TZ)
    if utc_now.weekday() >= 5:
        logger.info("Weekend detected. Forex market is closed. Exiting.")
        return

    is_daily = (utc_now.hour == 0)
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    logger.info(f"Forex Bot Started. Mode: {'DAILY' if is_daily else 'INTRADAY'}")
    
    global_news = fetch_forex_news()
    
    send_telegram(f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                  f"🌎 *ফরেক্স মার্কেট আপডেট*\n"
                  f"🕐 {now_bd}\n"
                  f"━━━━━━━━━━━━━━━━━━━━━━━━")

    for pair in PAIRS.keys():
        logger.info(f"Processing {pair}...")
        msg = build_report(pair, is_daily, global_news)
        send_telegram(msg)
        time.sleep(5)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()
