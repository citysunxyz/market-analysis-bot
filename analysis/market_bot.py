#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC & Gold Market Analysis Telegram Bot
Multi-timeframe analysis with key levels and trade setups.
"""

import os, sys, logging, requests, time
import numpy as np
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
BD_TZ = pytz.timezone("Asia/Dhaka")

# ══════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════

def coingecko_ohlc(coin_id: str, days: int, retries: int = 3) -> pd.DataFrame | None:
    """Fetch OHLC from CoinGecko free API with retry."""
    for attempt in range(retries):
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
            r = requests.get(url, timeout=30)
            if r.status_code == 429:  # Rate limit
                wait = 10 * (attempt + 1)
                logger.warning(f"CoinGecko rate limit. Waiting {wait}s...")
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
            logger.warning(f"CoinGecko {coin_id} {days}d attempt {attempt+1}: {e}")
            time.sleep(3)
    return None

def yf_ohlc(symbol: str, period: str, interval: str, multiplier: float = 1.0) -> pd.DataFrame | None:
    """Fetch OHLC from yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df is None or len(df) < 5:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        if multiplier != 1.0:
            for c in ["open","high","low","close"]:
                df[c] = df[c] * multiplier
        return df
    except Exception as e:
        logger.warning(f"yfinance {symbol} error: {e}")
        return None

def live_price(coin_id: str) -> float | None:
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd",
                timeout=15)
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            price = r.json()[coin_id]["usd"]
            logger.info(f"Live price {coin_id}: ${price:,.2f}")
            return price
        except Exception as e:
            logger.warning(f"Live price {coin_id} attempt {attempt+1}: {e}")
            time.sleep(3)
    return None

def fetch_all_gold():
    """Fetch gold data for Weekly, Daily, 4H."""
    logger.info("Fetching Gold weekly data...")
    weekly = coingecko_ohlc("pax-gold", 365)
    time.sleep(2)
    logger.info("Fetching Gold daily data...")
    daily  = coingecko_ohlc("pax-gold", 90)
    time.sleep(2)
    logger.info("Fetching Gold 4H data...")
    h4     = coingecko_ohlc("pax-gold", 30)
    time.sleep(2)

    # Fallback: yfinance GLD ETF × 10 (≈ XAUUSD)
    if weekly is None or len(weekly) < 10:
        weekly = yf_ohlc("GLD", "2y", "1wk", 10.0)
    if daily is None or len(daily) < 10:
        daily = yf_ohlc("GLD", "90d", "1d", 10.0)
    if h4 is None or len(h4) < 10:
        h4 = yf_ohlc("GLD", "30d", "4h", 10.0) or yf_ohlc("GC=F", "30d", "4h")

    price = live_price("pax-gold")
    time.sleep(2)
    if price is None:
        try:
            import yfinance as yf
            hist = yf.Ticker("GLD").history(period="1d", interval="1h")
            if hist is not None and len(hist) > 0:
                price = float(hist["Close"].iloc[-1]) * 10.0
        except:
            pass

    logger.info(f"Gold data: weekly={len(weekly) if weekly is not None else 0}, daily={len(daily) if daily is not None else 0}, 4H={len(h4) if h4 is not None else 0}, price={price}")
    return weekly, daily, h4, price

def fetch_all_btc():
    """Fetch BTC data for Weekly, Daily, 4H."""
    logger.info("Fetching BTC weekly data...")
    weekly = coingecko_ohlc("bitcoin", 365)
    time.sleep(3)  # Extra sleep to avoid rate limits after Gold
    logger.info("Fetching BTC daily data...")
    daily  = coingecko_ohlc("bitcoin", 90)
    time.sleep(2)
    logger.info("Fetching BTC 4H data...")
    h4     = coingecko_ohlc("bitcoin", 30)
    time.sleep(2)

    if weekly is None or len(weekly) < 10: weekly = yf_ohlc("BTC-USD", "2y", "1wk")
    if daily  is None or len(daily)  < 10: daily  = yf_ohlc("BTC-USD", "90d", "1d")
    if h4     is None or len(h4)     < 10: h4     = yf_ohlc("BTC-USD", "30d", "4h")

    price = live_price("bitcoin")
    time.sleep(2)
    logger.info(f"BTC data: weekly={len(weekly) if weekly is not None else 0}, daily={len(daily) if daily is not None else 0}, 4H={len(h4) if h4 is not None else 0}, price={price}")
    return weekly, daily, h4, price

# ══════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════

def indicators(df: pd.DataFrame, cur_price: float | None = None) -> dict | None:
    if df is None or len(df) < 14:
        return None
    close = df["close"].copy()
    high  = df["high"]
    low   = df["low"]
    if cur_price:
        close.iloc[-1] = cur_price
    try:
        n = len(close)
        rsi  = RSIIndicator(close, 14).rsi().iloc[-1]
        macd_obj = MACD(close)
        macd = macd_obj.macd().iloc[-1]
        msig = macd_obj.macd_signal().iloc[-1]
        mhist= macd_obj.macd_diff().iloc[-1]
        e20  = EMAIndicator(close, min(20, n-1)).ema_indicator().iloc[-1]
        e50  = EMAIndicator(close, min(50, n-1)).ema_indicator().iloc[-1]
        s200 = SMAIndicator(close, min(200, n)).sma_indicator().iloc[-1]
        bb   = BollingerBands(close, min(20, n-1), 2)
        bb_u = bb.bollinger_hband().iloc[-1]
        bb_m = bb.bollinger_mavg().iloc[-1]
        bb_l = bb.bollinger_lband().iloc[-1]
        bb_p = bb.bollinger_pband().iloc[-1]
        atr  = AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
        price = float(close.iloc[-1])
        pct   = (price - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        rh = float(high.iloc[-20:].max()) if n >= 20 else float(high.max())
        rl = float(low.iloc[-20:].min())  if n >= 20 else float(low.min())
        piv = (rh + rl + price) / 3
        return {
            "price": price, "pct": pct, "rsi": rsi,
            "macd": macd, "msig": msig, "mhist": mhist,
            "e20": e20, "e50": e50, "s200": s200,
            "bb_u": bb_u, "bb_m": bb_m, "bb_l": bb_l, "bb_p": bb_p,
            "atr": atr,
            "r2": piv + (rh - rl), "r1": 2*piv - rl,
            "s1": 2*piv - rh,      "s2": piv - (rh - rl),
        }
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return None

def tf_bias(ind: dict | None) -> tuple[str, str]:
    """Return (emoji_bias, short label) for a timeframe."""
    if ind is None:
        return "❓", "No data"
    price, rsi, mhist = ind["price"], ind["rsi"], ind["mhist"]
    e20, e50, s200    = ind["e20"], ind["e50"], ind["s200"]
    bull = sum([price > e20, price > e50, price > s200, rsi > 50, mhist > 0])
    bear = 5 - bull
    if bull >= 4:   return "📈", "Bullish"
    elif bull == 3: return "🔼", "Slightly Bullish"
    elif bear >= 4: return "📉", "Bearish"
    elif bear == 3: return "🔽", "Slightly Bearish"
    else:           return "↔️", "Neutral"

def signal(ind: dict) -> dict:
    price, rsi, mhist = ind["price"], ind["rsi"], ind["mhist"]
    e20, e50, s200, bb_p = ind["e20"], ind["e50"], ind["s200"], ind["bb_p"]
    b = s = 0
    if rsi < 35:    b += 2
    elif rsi > 65:  s += 2
    elif rsi > 50:  b += 1
    else:           s += 1
    b += (1 if mhist > 0 else 0);  s += (0 if mhist > 0 else 1)
    b += (1 if price > e20  else 0); s += (0 if price > e20  else 1)
    b += (1 if price > e50  else 0); s += (0 if price > e50  else 1)
    b += (1 if price > s200 else 0); s += (0 if price > s200 else 1)
    if bb_p < 0.2: b += 2
    elif bb_p > 0.8: s += 2
    total = b + s
    if total == 0: return {"bias": "NEUTRAL", "str": "WEAK"}
    if b > s:   return {"bias": "BUY",  "str": "STRONG" if b/total >= 0.7 else "MODERATE"}
    if s > b:   return {"bias": "SELL", "str": "STRONG" if s/total >= 0.7 else "MODERATE"}
    return {"bias": "NEUTRAL", "str": "MODERATE"}

# ══════════════════════════════════════════════
# MESSAGE BUILDER
# ══════════════════════════════════════════════

def build_report(name: str, emoji: str,
                 w_ind, d_ind, h4_ind,
                 price: float | None) -> str:

    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")

    # Use 4H as primary for signals
    primary = h4_ind or d_ind
    if primary is None:
        return f"{emoji} *{name}*\n❌ ডেটা পাওয়া যায়নি।"

    if price:
        primary["price"] = price

    sig = signal(primary)
    cur = primary["price"]
    pct = primary.get("pct", 0)
    atr = primary["atr"]
    chg_icon = "📈" if pct >= 0 else "📉"

    # TF bias
    w_icon,  w_lbl  = tf_bias(w_ind)
    d_icon,  d_lbl  = tf_bias(d_ind)
    h4_icon, h4_lbl = tf_bias(h4_ind)

    # Key levels (use daily if available, else 4H)
    lev = d_ind or h4_ind
    r2, r1 = lev["r2"], lev["r1"]
    s1, s2 = lev["s1"], lev["s2"]

    # Trade setup
    bias = sig["bias"]
    str_icon = {"STRONG": "🔥", "MODERATE": "⚡", "WEAK": "💤"}.get(sig["str"], "")
    if bias == "BUY":
        sig_label = "🟢 BUY"
        sl  = cur - 1.5 * atr
        tp1 = cur + 1.5 * atr
        tp2 = cur + 2.5 * atr
        tp3 = cur + 4.0 * atr
        trade = (
            f"🟢 *Buy Setup*\n"
            f"  Entry: `${cur:,.2f}` | SL: `${sl:,.2f}`\n"
            f"  TP1: `${tp1:,.2f}` | TP2: `${tp2:,.2f}` | TP3: `${tp3:,.2f}`\n"
            f"  R:R → 1:1.5 / 1:2.5 / 1:4"
        )
    elif bias == "SELL":
        sig_label = "🔴 SELL"
        sl  = cur + 1.5 * atr
        tp1 = cur - 1.5 * atr
        tp2 = cur - 2.5 * atr
        tp3 = cur - 4.0 * atr
        trade = (
            f"🔴 *Sell Setup*\n"
            f"  Entry: `${cur:,.2f}` | SL: `${sl:,.2f}`\n"
            f"  TP1: `${tp1:,.2f}` | TP2: `${tp2:,.2f}` | TP3: `${tp3:,.2f}`\n"
            f"  R:R → 1:1.5 / 1:2.5 / 1:4"
        )
    else:
        sig_label = "⚪ WAIT"
        trade = f"⚪ *সিগনাল নেই — অপেক্ষা করুন*\n  Range: `${s1:,.2f}` – `${r1:,.2f}`"

    # RSI info
    rsi = primary["rsi"]
    rsi_lbl = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Normal"

    msg = (
        f"{emoji} *{name}*\n"
        f"🕐 {now_bd} (BST)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"💰 *মূল্য:* `${cur:,.2f}` {chg_icon} `{pct:+.2f}%`\n"
        f"\n"
        f"📊 *মাল্টি-টাইমফ্রেম বায়াস:*\n"
        f"  🗓 Weekly:  {w_icon} {w_lbl}\n"
        f"  📅 Daily:   {d_icon} {d_lbl}\n"
        f"  ⏱ 4H:      {h4_icon} {h4_lbl}\n"
        f"\n"
        f"📐 *ইন্ডিকেটর (4H):*\n"
        f"  RSI: `{rsi:.0f}` {rsi_lbl}\n"
        f"  EMA20: `{primary['e20']:,.0f}` | EMA50: `{primary['e50']:,.0f}`\n"
        f"\n"
        f"🔑 *কী লেভেল:*\n"
        f"  🔴 R2: `${r2:,.2f}` | R1: `${r1:,.2f}`\n"
        f"  ◀ Now: `${cur:,.2f}`\n"
        f"  🟢 S1: `${s1:,.2f}` | S2: `${s2:,.2f}`\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *সিগনাল: {sig_label}* {str_icon}\n"
        f"\n"
        f"{trade}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _শিক্ষামূলক বিশ্লেষণ। ট্রেডের আগে নিজস্ব যাচাই করুন।_"
    )
    return msg

# ══════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════

def send_telegram(msg: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing Telegram credentials!")
        return False
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg,
            "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=data, timeout=30)
        r.raise_for_status()
        logger.info("Telegram sent!")
        return True
    except requests.HTTPError as e:
        logger.error(f"Telegram HTTP: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Telegram: {e}")
        return False

# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    logger.info("Market Analysis Bot starting...")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing env vars!"); sys.exit(1)

    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    send_telegram(
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Automated Market Report*\n"
        f"🕐 {now_bd} (BST)\n"
        f"Gold & BTC বিশ্লেষণ নিচে 👇\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # ── GOLD ──
    gw, gd, gh4, gprice = fetch_all_gold()
    gw_ind  = indicators(gw)
    gd_ind  = indicators(gd)
    gh4_ind = indicators(gh4, gprice)
    gold_msg = build_report("Gold (XAUUSD)", "🥇", gw_ind, gd_ind, gh4_ind, gprice)
    gold_ok = send_telegram(gold_msg)

    # Wait before BTC to respect rate limits
    logger.info("Waiting 5s before BTC fetch...")
    time.sleep(5)

    # ── BTC ──
    bw, bd, bh4, bprice = fetch_all_btc()
    bw_ind  = indicators(bw)
    bd_ind  = indicators(bd)
    bh4_ind = indicators(bh4, bprice)
    btc_msg = build_report("Bitcoin (BTC/USD)", "₿", bw_ind, bd_ind, bh4_ind, bprice)
    btc_ok = send_telegram(btc_msg)

    if gold_ok and btc_ok:
        logger.info("All reports sent!")
    else:
        logger.warning("Some reports failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
