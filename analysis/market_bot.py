#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC & Gold Market Analysis Telegram Bot
- 06:00 AM BST (00:00 UTC): Daily Bias Report (Weekly + Daily)
- Other sessions: Intraday Report (4H + 1H + 15M)
"""

import os, sys, logging, requests, time
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
    """True if current UTC time is 00:xx — the daily opening (6 AM BST)."""
    utc_hour = datetime.now(UTC_TZ).hour
    return utc_hour == 0


# ══════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════

def coingecko_ohlc(coin_id: str, days: int, retries: int = 3) -> pd.DataFrame | None:
    """CoinGecko OHLC with retry + rate-limit handling."""
    for attempt in range(retries):
        try:
            url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                   f"/ohlc?vs_currency=usd&days={days}")
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                logger.warning(f"Rate limit on {coin_id}. Sleeping {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if not data or len(data) < 5:
                return None
            df = pd.DataFrame(data, columns=["ts","open","high","low","close"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.sort_values("ts").reset_index(drop=True)
            df["volume"] = 0
            logger.info(f"CoinGecko {coin_id} {days}d: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"CoinGecko {coin_id} attempt {attempt+1}: {e}")
            time.sleep(3)
    return None


def yf_ohlc(symbol: str, period: str, interval: str,
            mult: float = 1.0) -> pd.DataFrame | None:
    """yfinance OHLCV with optional price multiplier."""
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
        logger.info(f"yfinance {symbol} {interval}: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"yfinance {symbol}: {e}")
        return None


def live_price_cg(coin_id: str) -> float | None:
    """Live price from CoinGecko."""
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price"
                f"?ids={coin_id}&vs_currencies=usd", timeout=15)
            if r.status_code == 429:
                time.sleep(15)
                continue
            r.raise_for_status()
            return r.json()[coin_id]["usd"]
        except Exception as e:
            logger.warning(f"Price {coin_id} attempt {attempt+1}: {e}")
            time.sleep(3)
    return None


def fetch_daily_data(asset: str) -> dict:
    """Fetch Weekly + Daily data for the morning bias report."""
    if asset == "GOLD":
        logger.info("Gold: fetching weekly...")
        w = coingecko_ohlc("pax-gold", 365)
        time.sleep(3)
        logger.info("Gold: fetching daily...")
        d = coingecko_ohlc("pax-gold", 90)
        time.sleep(2)
        price = live_price_cg("pax-gold")
        time.sleep(2)
        # yfinance fallbacks
        if w is None or len(w) < 10:
            w = yf_ohlc("GLD", "2y", "1wk", 10.0)
        if d is None or len(d) < 10:
            d = yf_ohlc("GLD", "90d", "1d", 10.0)
        if price is None:
            try:
                import yfinance as yf
                h = yf.Ticker("GLD").history(period="1d", interval="30m")
                if h is not None and len(h) > 0:
                    price = float(h["Close"].iloc[-1]) * 10.0
            except:
                pass
        return {"weekly": w, "daily": d, "price": price}
    else:  # BTC
        logger.info("BTC: fetching weekly...")
        w = coingecko_ohlc("bitcoin", 365)
        time.sleep(3)
        logger.info("BTC: fetching daily...")
        d = coingecko_ohlc("bitcoin", 90)
        time.sleep(2)
        price = live_price_cg("bitcoin")
        time.sleep(2)
        if w is None or len(w) < 10: w = yf_ohlc("BTC-USD", "2y", "1wk")
        if d is None or len(d) < 10: d = yf_ohlc("BTC-USD", "90d", "1d")
        return {"weekly": w, "daily": d, "price": price}


def fetch_intraday_data(asset: str) -> dict:
    """Fetch 4H + 1H + 15M data for intraday reports."""
    if asset == "GOLD":
        logger.info("Gold: fetching 4H...")
        h4 = coingecko_ohlc("pax-gold", 30)
        time.sleep(2)
        price = live_price_cg("pax-gold")
        time.sleep(2)
        if h4 is None or len(h4) < 10:
            h4 = yf_ohlc("GLD", "30d", "4h", 10.0)
        # 1H and 15M via yfinance (CoinGecko doesn't provide < 4H for free)
        h1  = yf_ohlc("GLD", "7d",  "1h",  10.0)
        m15 = yf_ohlc("GLD", "5d",  "15m", 10.0)
        if price is None:
            try:
                import yfinance as yf
                h = yf.Ticker("GLD").history(period="1d", interval="30m")
                if h is not None and len(h) > 0:
                    price = float(h["Close"].iloc[-1]) * 10.0
            except:
                pass
        return {"h4": h4, "h1": h1, "m15": m15, "price": price}
    else:  # BTC
        logger.info("BTC: fetching 4H...")
        h4 = coingecko_ohlc("bitcoin", 30)
        time.sleep(2)
        price = live_price_cg("bitcoin")
        time.sleep(2)
        if h4 is None or len(h4) < 10: h4 = yf_ohlc("BTC-USD", "30d", "4h")
        h1  = yf_ohlc("BTC-USD", "7d",  "1h")
        m15 = yf_ohlc("BTC-USD", "5d",  "15m")
        return {"h4": h4, "h1": h1, "m15": m15, "price": price}


# ══════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════

def calc(df: pd.DataFrame, cur_price: float | None = None) -> dict | None:
    """Compute technical indicators from an OHLCV DataFrame."""
    if df is None or len(df) < 14:
        return None
    close = df["close"].copy()
    high, low = df["high"], df["low"]
    if cur_price:
        close.iloc[-1] = cur_price
    try:
        n = len(close)
        rsi   = RSIIndicator(close, 14).rsi().iloc[-1]
        mo    = MACD(close)
        macd  = mo.macd().iloc[-1]
        msig  = mo.macd_signal().iloc[-1]
        mhist = mo.macd_diff().iloc[-1]
        e20   = EMAIndicator(close, min(20, n-1)).ema_indicator().iloc[-1]
        e50   = EMAIndicator(close, min(50, n-1)).ema_indicator().iloc[-1]
        s200  = SMAIndicator(close, min(200, n)).sma_indicator().iloc[-1]
        bb    = BollingerBands(close, min(20, n-1), 2)
        bb_u  = bb.bollinger_hband().iloc[-1]
        bb_l  = bb.bollinger_lband().iloc[-1]
        bb_p  = bb.bollinger_pband().iloc[-1]
        atr   = AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        price = float(close.iloc[-1])
        pct   = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        w = min(20, n)
        rh = float(high.iloc[-w:].max())
        rl = float(low.iloc[-w:].min())
        piv = (rh + rl + price) / 3
        return {
            "price": price, "pct": pct, "rsi": rsi,
            "macd": macd, "msig": msig, "mhist": mhist,
            "e20": e20, "e50": e50, "s200": s200,
            "bb_u": bb_u, "bb_l": bb_l, "bb_p": bb_p, "atr": atr,
            "r2": piv+(rh-rl), "r1": 2*piv-rl,
            "s1": 2*piv-rh,    "s2": piv-(rh-rl),
        }
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return None


def tf_bias(ind: dict | None) -> tuple[str, str]:
    """Return (emoji, label) bias for a timeframe."""
    if ind is None:
        return "❓", "No data"
    price, rsi, mhist = ind["price"], ind["rsi"], ind["mhist"]
    e20, e50 = ind["e20"], ind["e50"]
    bull = sum([price > e20, price > e50, rsi > 50, mhist > 0, ind["bb_p"] < 0.5])
    if   bull >= 4: return "📈", "Bullish"
    elif bull == 3: return "🔼", "Slightly Bullish"
    elif bull <= 1: return "📉", "Bearish"
    else:           return "🔽", "Slightly Bearish"


def overall_signal(ind: dict) -> dict:
    """Generate BUY/SELL/NEUTRAL signal from indicators."""
    b = s = 0
    p, rsi, mhist = ind["price"], ind["rsi"], ind["mhist"]
    e20, e50, s200, bb_p = ind["e20"], ind["e50"], ind["s200"], ind["bb_p"]
    if rsi < 35:   b += 2
    elif rsi > 65: s += 2
    elif rsi > 50: b += 1
    else:          s += 1
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
    """🌅 Morning report: Weekly + Daily bias with key levels for the day."""
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y")

    w_ind = calc(data.get("weekly"))
    d_ind = calc(data.get("daily"), data.get("price"))
    price = data.get("price") or (d_ind["price"] if d_ind else None)

    if price is None:
        return f"{emoji} *{name}*\n❌ ডেটা পাওয়া যায়নি।"

    w_icon, w_lbl = tf_bias(w_ind)
    d_icon, d_lbl = tf_bias(d_ind)

    # Overall daily bias
    primary = d_ind or w_ind
    sig = overall_signal(primary) if primary else {"bias": "NEUTRAL", "str": "WEAK"}
    bias = sig["bias"]
    bias_labels = {"BUY": "🟢 BULLISH", "SELL": "🔴 BEARISH", "NEUTRAL": "⚪ NEUTRAL"}
    bias_advice = {
        "BUY":     "→ ডিমান্ড জোনে বাই অগ্রাধিকার\n→ রেজিস্ট্যান্সে সতর্ক থাকুন",
        "SELL":    "→ পুলব্যাকে সেল অগ্রাধিকার\n→ সাপোর্টে বাই রিস্কি",
        "NEUTRAL": "→ ব্রেকআউটের অপেক্ষায় থাকুন\n→ দুই দিকেই ট্রেড সম্ভব"
    }

    # Key levels
    lev = d_ind or w_ind
    r2, r1 = lev["r2"], lev["r1"]
    s1, s2 = lev["s1"], lev["s2"]
    atr = lev["atr"]

    # RSI info
    w_rsi = f"`{w_ind['rsi']:.0f}`" if w_ind else "N/A"
    d_rsi = f"`{d_ind['rsi']:.0f}`" if d_ind else "N/A"

    msg = (
        f"{emoji} *{name}*\n"
        f"🌅 *দৈনিক বায়াস রিপোর্ট — {now_bd}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"💰 *মূল্য:* `${price:,.2f}`\n"
        f"\n"
        f"📊 *হায়ার টাইমফ্রেম বায়াস:*\n"
        f"  🗓 Weekly: {w_icon} {w_lbl} | RSI {w_rsi}\n"
        f"  📅 Daily:  {d_icon} {d_lbl} | RSI {d_rsi}\n"
        f"\n"
        f"🔑 *আজকের কী লেভেল:*\n"
        f"  🔴 রেজিস্ট্যান্স: `${r1:,.2f}` → `${r2:,.2f}`\n"
        f"  ◀ বর্তমান: `${price:,.2f}`\n"
        f"  🟢 সাপোর্ট: `${s1:,.2f}` → `${s2:,.2f}`\n"
        f"  ⚡ ATR (দৈনিক রেঞ্জ): `${atr:,.2f}`\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *আজকের সামগ্রিক বায়াস:*\n"
        f"   *{bias_labels[bias]}*\n"
        f"{bias_advice[bias]}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _শিক্ষামূলক বিশ্লেষণ। স্টপ লস ব্যবহার করুন।_"
    )
    return msg


def build_intraday(name: str, emoji: str, data: dict) -> str:
    """⏱ Intraday report: 4H + 1H + 15M current market condition."""
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    price  = data.get("price")

    h4_ind  = calc(data.get("h4"),  price)
    h1_ind  = calc(data.get("h1"),  price)
    m15_ind = calc(data.get("m15"), price)

    primary = h4_ind or h1_ind or m15_ind
    if primary is None:
        return f"{emoji} *{name}*\n❌ ডেটা পাওয়া যায়নি।"

    if price:
        primary["price"] = price
    cur = primary["price"]
    pct = primary.get("pct", 0)
    chg_icon = "📈" if pct >= 0 else "📉"

    # TF biases
    h4_icon,  h4_lbl  = tf_bias(h4_ind)
    h1_icon,  h1_lbl  = tf_bias(h1_ind)
    m15_icon, m15_lbl = tf_bias(m15_ind)

    # RSI per TF
    h4_rsi  = f"`{h4_ind['rsi']:.0f}`"  if h4_ind  else "N/A"
    h1_rsi  = f"`{h1_ind['rsi']:.0f}`"  if h1_ind  else "N/A"
    m15_rsi = f"`{m15_ind['rsi']:.0f}`" if m15_ind else "N/A"

    # MACD per TF
    def macd_icon(ind):
        if ind is None: return "❓"
        return "✅" if ind["macd"] > ind["msig"] else "❌"

    # Signal from primary (4H preferred)
    sig  = overall_signal(primary)
    bias = sig["bias"]
    atr  = primary["atr"]
    str_icon = {"STRONG": "🔥", "MODERATE": "⚡", "WEAK": "💤"}.get(sig["str"], "")
    sig_labels = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "NEUTRAL": "⚪ WAIT"}

    # Trade setup
    if bias == "BUY":
        sl, tp1, tp2 = cur - 1.5*atr, cur + 1.5*atr, cur + 2.5*atr
        trade = (
            f"🟢 *Buy* | Entry: `${cur:,.2f}`\n"
            f"  SL: `${sl:,.2f}` | TP1: `${tp1:,.2f}` | TP2: `${tp2:,.2f}`"
        )
    elif bias == "SELL":
        sl, tp1, tp2 = cur + 1.5*atr, cur - 1.5*atr, cur - 2.5*atr
        trade = (
            f"🔴 *Sell* | Entry: `${cur:,.2f}`\n"
            f"  SL: `${sl:,.2f}` | TP1: `${tp1:,.2f}` | TP2: `${tp2:,.2f}`"
        )
    else:
        lev = primary
        trade = f"⚪ *অপেক্ষা করুন* | S: `${lev['s1']:,.2f}` – R: `${lev['r1']:,.2f}`"

    msg = (
        f"{emoji} *{name}* — ইন্ট্রাডে আপডেট\n"
        f"🕐 {now_bd} (BST)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"💰 `${cur:,.2f}` {chg_icon} `{pct:+.2f}%`\n"
        f"\n"
        f"📊 *টাইমফ্রেম বিশ্লেষণ:*\n"
        f"  ⏱ 4H:  {h4_icon} {h4_lbl} | RSI {h4_rsi} | MACD {macd_icon(h4_ind)}\n"
        f"  ⏰ 1H:  {h1_icon} {h1_lbl} | RSI {h1_rsi} | MACD {macd_icon(h1_ind)}\n"
        f"  ⚡ 15M: {m15_icon} {m15_lbl} | RSI {m15_rsi} | MACD {macd_icon(m15_ind)}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *সিগনাল: {sig_labels[bias]}* {str_icon}\n"
        f"{trade}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _শিক্ষামূলক বিশ্লেষণ মাত্র।_"
    )
    return msg


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
        logger.info("Telegram message sent!")
        return True
    except requests.HTTPError as e:
        logger.error(f"Telegram HTTP: {e.response.text}"); return False
    except Exception as e:
        logger.error(f"Telegram: {e}"); return False


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    mode   = "DAILY BIAS" if is_daily_opening() else "INTRADAY"
    logger.info(f"Starting — Mode: {mode} | Time: {now_bd}")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing env vars!"); sys.exit(1)

    if is_daily_opening():
        # ────────────────────────────────
        # 🌅 MORNING: Daily Bias Report
        # ────────────────────────────────
        send_telegram(
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌅 *দৈনিক বায়াস রিপোর্ট*\n"
            f"🕐 {now_bd} (BST)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        # Gold
        logger.info("Fetching Gold daily data...")
        gold_data = fetch_daily_data("GOLD")
        gold_msg  = build_daily_bias("Gold (XAUUSD)", "🥇", gold_data)
        send_telegram(gold_msg)

        logger.info("Waiting 8s before BTC...")
        time.sleep(8)

        # BTC
        logger.info("Fetching BTC daily data...")
        btc_data = fetch_daily_data("BTC")
        btc_msg  = build_daily_bias("Bitcoin (BTC/USD)", "₿", btc_data)
        send_telegram(btc_msg)

    else:
        # ────────────────────────────────
        # ⏱ INTRADAY: 4H + 1H + 15M
        # ────────────────────────────────
        send_telegram(
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏱ *ইন্ট্রাডে মার্কেট আপডেট*\n"
            f"🕐 {now_bd} (BST)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        # Gold
        logger.info("Fetching Gold intraday data...")
        gold_data = fetch_intraday_data("GOLD")
        gold_msg  = build_intraday("Gold (XAUUSD)", "🥇", gold_data)
        send_telegram(gold_msg)

        logger.info("Waiting 8s before BTC...")
        time.sleep(8)

        # BTC
        logger.info("Fetching BTC intraday data...")
        btc_data = fetch_intraday_data("BTC")
        btc_msg  = build_intraday("Bitcoin (BTC/USD)", "₿", btc_data)
        send_telegram(btc_msg)

    logger.info("Done!")


if __name__ == "__main__":
    main()
