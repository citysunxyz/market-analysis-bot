#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC & Gold Market Analysis Telegram Bot
- 06:00 AM BST (00:00 UTC): Daily Bias Report (Weekly + Daily)
- Other sessions: Intraday Report (4H + 1H + 15M)
Data: CoinGecko free API (no key needed)
"""

import os, sys, logging, requests, time, urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import pytz
from datetime import datetime
from dotenv import load_dotenv
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
BD_TZ  = pytz.timezone("Asia/Dhaka")
UTC_TZ = pytz.utc


# ══════════════════════════════════════════════
# SESSION DETECTION
# ══════════════════════════════════════════════

def is_daily_opening() -> bool:
    """True at 00:xx UTC = 06:00 AM BST (daily opening report)."""
    return datetime.now(UTC_TZ).hour == 0


# ══════════════════════════════════════════════
# DATA FETCHING — CoinGecko Only
# ══════════════════════════════════════════════

def cg_get(url: str, retries: int = 3) -> dict | None:
    """Generic CoinGecko GET with retry + rate-limit handling."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                logger.warning(f"Rate limit. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"CG attempt {attempt+1}: {e}")
            time.sleep(4)
    return None


def cg_ohlc(coin: str, days: int) -> pd.DataFrame | None:
    """CoinGecko OHLC endpoint -> DataFrame."""
    data = cg_get(f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc"
                  f"?vs_currency=usd&days={days}")
    if not data or len(data) < 5:
        return None
    df = pd.DataFrame(data, columns=["ts","open","high","low","close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").reset_index(drop=True)
    df["volume"] = 0
    logger.info(f"ohlc {coin} {days}d: {len(df)} rows")
    return df


def cg_chart_raw(coin: str, days: int) -> pd.DataFrame | None:
    """CoinGecko market_chart -> raw price series with UTC DatetimeIndex."""
    data = cg_get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
                  f"?vs_currency=usd&days={days}")
    if not data:
        return None
    prices = data.get("prices", [])
    if len(prices) < 10:
        return None
    df = pd.DataFrame(prices, columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    logger.info(f"chart {coin} {days}d: {len(df)} rows")
    return df


def resample_to_ohlcv(price_df: pd.DataFrame, freq: str) -> pd.DataFrame | None:
    """Resample close price series to OHLCV DataFrame."""
    if price_df is None or len(price_df) < 8:
        return None
    try:
        ohlc = price_df["close"].resample(freq).ohlc().dropna()
        ohlc["volume"] = 0
        result = ohlc.reset_index(drop=True)
        logger.info(f"resample {freq}: {len(result)} rows")
        return result if len(result) >= 8 else None
    except Exception as e:
        logger.warning(f"Resample {freq}: {e}")
        return None


def cg_price(coin: str) -> float | None:
    """Live price from CoinGecko simple/price."""
    data = cg_get(f"https://api.coingecko.com/api/v3/simple/price"
                  f"?ids={coin}&vs_currencies=usd")
    if data and coin in data:
        p = data[coin]["usd"]
        logger.info(f"price {coin}: ${p:,.2f}")
        return p
    return None


def yf_fallback(symbol: str, period: str, interval: str,
                mult: float = 1.0) -> pd.DataFrame | None:
    """yfinance fallback (only used if CoinGecko fails for 4H)."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df is None or len(df) < 5:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        if mult != 1.0:
            for c in ["open","high","low","close"]:
                df[c] = df[c] * mult
        return df
    except Exception as e:
        logger.warning(f"yf {symbol}: {e}")
        return None

def fetch_news(asset: str) -> list[str]:
    """Fetch top 3 fundamental news headlines using RSS."""
    news = []
    try:
        if asset == "GOLD":
            # CNBC Market News
            url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000115"
        else:
            # CoinTelegraph Bitcoin News
            url = "https://cointelegraph.com/rss"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
        
        root = ET.fromstring(xml_data)
        for item in root.findall('.//item')[:3]:
            title_elem = item.find('title')
            if title_elem is not None and title_elem.text:
                news.append(title_elem.text.strip())
                
    except Exception as e:
        logger.warning(f"News fetch error for {asset}: {e}")
    
    return news


def analyze_news_sentiment(news_list: list[str], asset: str) -> dict:
    """Analyze news headlines to determine fundamental bias."""
    if not news_list:
        return {"bias": "MIXED", "text": "⚪ নিউজের প্রভাব অস্পষ্ট।"}
    
    text = " ".join(news_list).lower()
    bull_score = 0
    bear_score = 0
    
    if asset == "GOLD":
        bull_words = ["cut", "weak", "dovish", "surge", "jump", "high", "war", "tension", "inflation", "stimulus", "gain", "up"]
        bear_words = ["hike", "strong", "hawkish", "drop", "fall", "low", "peak", "crash", "bear", "plunge", "steady", "down"]
    else:
        bull_words = ["adopt", "approve", "etf", "surge", "jump", "high", "bull", "rally", "integrate", "upgrade", "buy", "gain"]
        bear_words = ["ban", "hack", "crash", "drop", "fall", "crackdown", "reject", "sell", "hawkish", "scam", "regulation", "bear"]
        
    for w in bull_words:
        if w in text: bull_score += 1
    for w in bear_words:
        if w in text: bear_score += 1
        
    if bull_score > bear_score:
        return {"bias": "BULLISH", "text": "🟢 ফান্ডামেন্টাল নিউজ অনুযায়ী মার্কেট **উপরের দিকে (Bullish)** যেতে পারে।"}
    elif bear_score > bull_score:
        return {"bias": "BEARISH", "text": "🔴 ফান্ডামেন্টাল নিউজ অনুযায়ী মার্কেট **নিচের দিকে (Bearish)** যেতে পারে।"}
    else:
        return {"bias": "MIXED", "text": "⚪ নিউজগুলো মিক্সড, মার্কেটে **ভোলাটিলিটি (উঠা-নামা)** দেখা যেতে পারে।"}


# ── Fetch helpers ──────────────────────────

def fetch_daily_data(asset: str) -> dict:
    """Weekly + Daily data for morning bias report."""
    coin = "pax-gold" if asset == "GOLD" else "bitcoin"
    logger.info(f"{asset}: weekly ohlc...")
    weekly = cg_ohlc(coin, 365)
    time.sleep(4)
    logger.info(f"{asset}: daily ohlc...")
    daily = cg_ohlc(coin, 90)
    time.sleep(3)
    logger.info(f"{asset}: live price...")
    price = cg_price(coin)
    time.sleep(2)
    # yfinance fallbacks
    if asset == "GOLD":
        if weekly is None or len(weekly) < 10: weekly = yf_fallback("GLD","2y","1wk",10.0)
        if daily  is None or len(daily)  < 10: daily  = yf_fallback("GLD","90d","1d",10.0)
    else:
        if weekly is None or len(weekly) < 10: weekly = yf_fallback("BTC-USD","2y","1wk")
        if daily  is None or len(daily)  < 10: daily  = yf_fallback("BTC-USD","90d","1d")
    return {"weekly": weekly, "daily": daily, "price": price}


def fetch_intraday_data(asset: str) -> dict:
    """4H + 1H + 15M data. Fetches 7d chart ONCE -> resamples to 1H and 15M."""
    coin = "pax-gold" if asset == "GOLD" else "bitcoin"

    # 4H: CoinGecko OHLC
    logger.info(f"{asset}: 4H ohlc (30d)...")
    h4 = cg_ohlc(coin, 30)
    time.sleep(4)

    # ONE fetch for 1H + 15M
    logger.info(f"{asset}: 7d chart (for 1H & 15M)...")
    raw7 = cg_chart_raw(coin, 7)
    time.sleep(3)

    h1 = resample_to_ohlcv(raw7, "1h")

    # 15M from last 2 days of the 7d data
    raw2 = raw7.last("2D") if raw7 is not None else None
    m15  = resample_to_ohlcv(raw2, "15min")
    if m15 is None and raw7 is not None:
        m15 = resample_to_ohlcv(raw7, "30min")  # fallback: 30min

    # Live price
    logger.info(f"{asset}: live price...")
    price = cg_price(coin)
    time.sleep(2)

    # 4H fallback to yfinance
    if h4 is None or len(h4) < 10:
        sym  = "GLD"     if asset == "GOLD" else "BTC-USD"
        mult = 10.0      if asset == "GOLD" else 1.0
        h4   = yf_fallback(sym, "30d", "4h", mult)

    logger.info(f"{asset}: 4H={len(h4) if h4 is not None else 0}, "
                f"1H={len(h1) if h1 is not None else 0}, "
                f"15M={len(m15) if m15 is not None else 0}, price={price}")
    return {"h4": h4, "h1": h1, "m15": m15, "price": price}


# ══════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════

def calc(df: pd.DataFrame, cur_price: float | None = None) -> dict | None:
    if df is None or len(df) < 14:
        return None
    close = df["close"].copy()
    high, low = df["high"], df["low"]
    if cur_price:
        close.iloc[-1] = cur_price
    try:
        n   = len(close)
        rsi = RSIIndicator(close, 14).rsi().iloc[-1]
        mo  = MACD(close)
        macd, msig, mhist = mo.macd().iloc[-1], mo.macd_signal().iloc[-1], mo.macd_diff().iloc[-1]
        e20  = EMAIndicator(close, min(20, n-1)).ema_indicator().iloc[-1]
        e50  = EMAIndicator(close, min(50, n-1)).ema_indicator().iloc[-1]
        s200 = SMAIndicator(close, min(200, n)).sma_indicator().iloc[-1]
        bb   = BollingerBands(close, min(20, n-1), 2)
        bb_p = bb.bollinger_pband().iloc[-1]
        atr  = AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        price = float(close.iloc[-1])
        pct   = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        w  = min(20, n)
        rh = float(high.iloc[-w:].max())
        rl = float(low.iloc[-w:].min())
        pv = (rh + rl + price) / 3
        return {
            "price": price, "pct": pct, "rsi": rsi,
            "macd": macd, "msig": msig, "mhist": mhist,
            "e20": e20, "e50": e50, "s200": s200,
            "bb_p": bb_p, "atr": atr,
            "r2": pv+(rh-rl), "r1": 2*pv-rl,
            "s1": 2*pv-rh,    "s2": pv-(rh-rl),
        }
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return None


def tf_bias(ind: dict | None) -> tuple[str, str]:
    if ind is None:
        return "❓", "No data"
    p, rsi, mhist, e20, e50, bb_p = (
        ind["price"], ind["rsi"], ind["mhist"],
        ind["e20"], ind["e50"], ind["bb_p"])
    bull = sum([p > e20, p > e50, rsi > 50, mhist > 0, bb_p < 0.5])
    if   bull >= 4: return "📈", "Bullish"
    elif bull == 3: return "🔼", "Slightly Bullish"
    elif bull <= 1: return "📉", "Bearish"
    else:           return "🔽", "Slightly Bearish"


def get_signal(ind: dict) -> dict:
    p, rsi, mhist = ind["price"], ind["rsi"], ind["mhist"]
    e20, e50, s200, bb_p = ind["e20"], ind["e50"], ind["s200"], ind["bb_p"]
    b = s = 0
    if rsi < 35: b += 2
    elif rsi > 65: s += 2
    elif rsi > 50: b += 1
    else: s += 1
    b += 1 if mhist > 0 else 0;  s += 0 if mhist > 0 else 1
    b += 1 if p > e20  else 0;   s += 0 if p > e20  else 1
    b += 1 if p > e50  else 0;   s += 0 if p > e50  else 1
    b += 1 if p > s200 else 0;   s += 0 if p > s200 else 1
    if bb_p < 0.2: b += 2
    elif bb_p > 0.8: s += 2
    total = b + s
    if total == 0: return {"bias": "NEUTRAL", "str": "WEAK"}
    if b > s:  return {"bias": "BUY",  "str": "STRONG" if b/total >= 0.7 else "MODERATE"}
    if s > b:  return {"bias": "SELL", "str": "STRONG" if s/total >= 0.7 else "MODERATE"}
    return {"bias": "NEUTRAL", "str": "MODERATE"}


# ══════════════════════════════════════════════
# MESSAGE BUILDERS
# ══════════════════════════════════════════════

def build_daily_bias(name: str, emoji: str, data: dict) -> str:
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y")
    w_ind = calc(data.get("weekly"))
    d_ind = calc(data.get("daily"), data.get("price"))
    price = data.get("price") or (d_ind["price"] if d_ind else None)
    if price is None:
        return f"{emoji} *{name}*\n❌ ডেটা পাওয়া যায়নি।"

    w_icon, w_lbl = tf_bias(w_ind)
    d_icon, d_lbl = tf_bias(d_ind)
    primary = d_ind or w_ind
    sig  = get_signal(primary) if primary else {"bias": "NEUTRAL", "str": "WEAK"}
    bias = sig["bias"]
    bias_txt = {"BUY": "🟢 BULLISH", "SELL": "🔴 BEARISH", "NEUTRAL": "⚪ NEUTRAL"}
    advice   = {
        "BUY":     "→ ডিমান্ড জোনে বাই অগ্রাধিকার\n→ রেজিস্ট্যান্সে সতর্ক থাকুন",
        "SELL":    "→ পুলব্যাকে সেল অগ্রাধিকার\n→ সাপোর্টে বাই রিস্কি",
        "NEUTRAL": "→ ব্রেকআউটের অপেক্ষায় থাকুন\n→ দুই দিকেই সম্ভব"
    }
    lev = primary or {}
    r1 = lev.get("r1", 0); r2 = lev.get("r2", 0)
    s1 = lev.get("s1", 0); s2 = lev.get("s2", 0)
    atr = lev.get("atr", 0)
    w_rsi = f"`{w_ind['rsi']:.0f}`" if w_ind else "N/A"
    d_rsi = f"`{d_ind['rsi']:.0f}`" if d_ind else "N/A"

    asset_key = "GOLD" if "Gold" in name else "BTC"
    news_list = fetch_news(asset_key)
    news_text = ""
    if news_list:
        sentiment = analyze_news_sentiment(news_list, asset_key)
        news_text = "\n📰 *ফান্ডামেন্টাল আপডেট (Live):*\n"
        for n in news_list:
            news_text += f"🔹 {n}\n"
        news_text += f"\n💡 *ফান্ডামেন্টাল বায়াস:* {sentiment['text']}\n"
        news_text += f"⚖️ *ওভারঅল ভিউ:* টেকনিক্যাল অনুযায়ী {bias_txt[bias]}, এবং ফান্ডামেন্টাল অনুযায়ী {sentiment['bias']}।\n"

    return (
        f"{emoji} *{name}*\n"
        f"🌅 *দৈনিক বায়াস — {now_bd}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 *মূল্য:* `${price:,.2f}`\n\n"
        f"📊 *হায়ার টাইমফ্রেম বায়াস:*\n"
        f"  🗓 Weekly: {w_icon} {w_lbl} | RSI {w_rsi}\n"
        f"  📅 Daily:  {d_icon} {d_lbl} | RSI {d_rsi}\n\n"
        f"🔑 *আজকের কী লেভেল:*\n"
        f"  🔴 রেজিস্ট্যান্স: `${r1:,.2f}` → `${r2:,.2f}`\n"
        f"  ◀ বর্তমান: `${price:,.2f}`\n"
        f"  🟢 সাপোর্ট: `${s1:,.2f}` → `${s2:,.2f}`\n"
        f"  ⚡ ATR: `${atr:,.2f}`\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *আজকের বায়াস: {bias_txt[bias]}*\n"
        f"{advice[bias]}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
        f"{news_text}"
        f"\n⚠️ _শিক্ষামূলক বিশ্লেষণ। স্টপ লস ব্যবহার করুন।_"
    )


def build_intraday(name: str, emoji: str, data: dict) -> str:
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    price  = data.get("price")
    h4_ind  = calc(data.get("h4"),  price)
    h1_ind  = calc(data.get("h1"),  price)
    m15_ind = calc(data.get("m15"), price)
    primary = h4_ind or h1_ind or m15_ind
    if primary is None:
        return f"{emoji} *{name}*\n❌ ডেটা পাওয়া যায়নি।"
    if price: primary["price"] = price
    cur = primary["price"]
    pct = primary.get("pct", 0)
    atr = primary["atr"]
    chg_icon = "📈" if pct >= 0 else "📉"

    h4_icon,  h4_lbl  = tf_bias(h4_ind)
    h1_icon,  h1_lbl  = tf_bias(h1_ind)
    m15_icon, m15_lbl = tf_bias(m15_ind)

    def rsi_str(ind): return f"`{ind['rsi']:.0f}`" if ind else "N/A"
    def macd_str(ind): return ("✅" if ind["macd"] > ind["msig"] else "❌") if ind else "❓"

    sig  = get_signal(primary)
    bias = sig["bias"]
    si   = {"STRONG": "🔥", "MODERATE": "⚡", "WEAK": "💤"}.get(sig["str"], "")
    sl   = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "NEUTRAL": "⚪ WAIT"}[bias]

    if bias == "BUY":
        sl_p, tp1, tp2 = cur-1.5*atr, cur+1.5*atr, cur+2.5*atr
        trade = (f"🟢 *Buy* | Entry: `${cur:,.2f}`\n"
                 f"  SL: `${sl_p:,.2f}` | TP1: `${tp1:,.2f}` | TP2: `${tp2:,.2f}`")
    elif bias == "SELL":
        sl_p, tp1, tp2 = cur+1.5*atr, cur-1.5*atr, cur-2.5*atr
        trade = (f"🔴 *Sell* | Entry: `${cur:,.2f}`\n"
                 f"  SL: `${sl_p:,.2f}` | TP1: `${tp1:,.2f}` | TP2: `${tp2:,.2f}`")
    else:
        trade = f"⚪ *অপেক্ষা করুন* | S: `${primary['s1']:,.2f}` – R: `${primary['r1']:,.2f}`"

    asset_key = "GOLD" if "Gold" in name else "BTC"
    news_list = fetch_news(asset_key)
    news_text = ""
    if news_list:
        sentiment = analyze_news_sentiment(news_list, asset_key)
        news_text = "\n📰 *ফান্ডামেন্টাল আপডেট (Live):*\n"
        for n in news_list:
            news_text += f"🔹 {n}\n"
        news_text += f"\n💡 *ফান্ডামেন্টাল বায়াস:* {sentiment['text']}\n"
        news_text += f"⚖️ *ওভারঅল ভিউ:* টেকনিক্যাল সিগন্যাল {sl}, এবং ফান্ডামেন্টাল অনুযায়ী {sentiment['bias']}।\n"

    return (
        f"{emoji} *{name}* — ইন্ট্রাডে\n"
        f"🕐 {now_bd} (BST)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 `${cur:,.2f}` {chg_icon} `{pct:+.2f}%`\n\n"
        f"📊 *টাইমফ্রেম বিশ্লেষণ:*\n"
        f"  ⏱ 4H:  {h4_icon} {h4_lbl} | RSI {rsi_str(h4_ind)} | MACD {macd_str(h4_ind)}\n"
        f"  ⏰ 1H:  {h1_icon} {h1_lbl} | RSI {rsi_str(h1_ind)} | MACD {macd_str(h1_ind)}\n"
        f"  ⚡ 15M: {m15_icon} {m15_lbl} | RSI {rsi_str(m15_ind)} | MACD {macd_str(m15_ind)}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *সিগনাল: {sl}* {si}\n"
        f"{trade}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
        f"{news_text}"
        f"\n⚠️ _শিক্ষামূলক বিশ্লেষণ মাত্র।_"
    )


# ══════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════

def send_telegram(msg: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing Telegram credentials!"); return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "Markdown", "disable_web_page_preview": True},
            timeout=30)
        r.raise_for_status()
        logger.info("Telegram sent!")
        return True
    except Exception as e:
        logger.error(f"Telegram: {e}"); return False


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    mode   = "DAILY BIAS" if is_daily_opening() else "INTRADAY"
    logger.info(f"Mode: {mode} | {now_bd}")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing env vars!"); sys.exit(1)

    if is_daily_opening():
        send_telegram(
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌅 *দৈনিক বায়াস রিপোর্ট*\n"
            f"🕐 {now_bd} (BST)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━")

        gold_data = fetch_daily_data("GOLD")
        send_telegram(build_daily_bias("Gold (XAUUSD)", "🥇", gold_data))
        logger.info("Waiting 10s before BTC...")
        time.sleep(10)
        btc_data = fetch_daily_data("BTC")
        send_telegram(build_daily_bias("Bitcoin (BTC/USD)", "₿", btc_data))

    else:
        send_telegram(
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱ *ইন্ট্রাডে আপডেট*\n"
            f"🕐 {now_bd} (BST)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━")

        gold_data = fetch_intraday_data("GOLD")
        send_telegram(build_intraday("Gold (XAUUSD)", "🥇", gold_data))
        logger.info("Waiting 10s before BTC...")
        time.sleep(10)
        btc_data = fetch_intraday_data("BTC")
        send_telegram(build_intraday("Bitcoin (BTC/USD)", "₿", btc_data))

    logger.info("Done!")


if __name__ == "__main__":
    main()
