#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 BTC & Gold Market Analysis Bot
====================================
প্রতি ৪ ঘণ্টায় BTC ও Gold-এর টেকনিক্যাল এনালাইসিস করে
Telegram-এ রিপোর্ট পাঠায়।

তৈরিকারক: GitHub Actions Automated Bot
"""

import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from dotenv import load_dotenv

# ta লাইব্রেরি থেকে ইন্ডিকেটর
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# yfinance
try:
    import yfinance as yf
except ImportError:
    yf = None

# ------------------------------------------------------------------
# লগিং সেটআপ
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# .env ফাইল লোড (লোকাল টেস্টিং-এর জন্য)
load_dotenv()

# ------------------------------------------------------------------
# কনফিগারেশন
# ------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

SYMBOLS = {
    "GOLD": {"yf_symbol": "GC=F",  "name": "Gold (XAUUSD)", "emoji": "🥇"},
    "BTC":  {"yf_symbol": "BTC-USD","name": "Bitcoin (BTC/USD)", "emoji": "₿"},
}

BD_TZ = pytz.timezone("Asia/Dhaka")


# ==================================================================
# ডেটা ফেচ ফাংশন
# ==================================================================
def fetch_ohlcv(yf_symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame | None:
    """Yahoo Finance থেকে OHLCV ডেটা আনুন।"""
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        if df is None or df.empty:
            logger.warning(f"No data for {yf_symbol}")
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        logger.info(f"✅ Fetched {len(df)} rows for {yf_symbol}")
        return df
    except Exception as e:
        logger.error(f"❌ Error fetching {yf_symbol}: {e}")
        return None


def fetch_daily(yf_symbol: str) -> pd.DataFrame | None:
    """Daily ডেটা আনুন (সাপোর্ট/রেজিস্ট্যান্স ও ট্রেন্ডের জন্য)।"""
    return fetch_ohlcv(yf_symbol, period="180d", interval="1d")


# ==================================================================
# টেকনিক্যাল ইন্ডিকেটর
# ==================================================================
def compute_indicators(df: pd.DataFrame) -> dict:
    """সব টেকনিক্যাল ইন্ডিকেটর হিসাব করুন।"""
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ---- RSI ----
    rsi_val = RSIIndicator(close=close, window=14).rsi().iloc[-1]

    # ---- MACD ----
    macd_obj  = MACD(close=close)
    macd_val  = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    macd_diff = macd_obj.macd_diff().iloc[-1]

    # ---- Moving Averages ----
    ema20  = EMAIndicator(close=close, window=20).ema_indicator().iloc[-1]
    ema50  = EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]
    sma200 = SMAIndicator(close=close, window=min(200, len(close))).sma_indicator().iloc[-1]

    # ---- Bollinger Bands ----
    bb     = BollingerBands(close=close, window=20, window_dev=2)
    bb_up  = bb.bollinger_hband().iloc[-1]
    bb_mid = bb.bollinger_mavg().iloc[-1]
    bb_low = bb.bollinger_lband().iloc[-1]
    bb_pct = bb.bollinger_pband().iloc[-1]  # 0–1: 0=at lower, 1=at upper

    # ---- ATR (ভোলাটিলিটি) ----
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]

    # ---- সাম্প্রতিক ক্যান্ডেল ডেটা ----
    current_price = close.iloc[-1]
    prev_close    = close.iloc[-2]
    pct_change    = (current_price - prev_close) / prev_close * 100

    # ---- সাপোর্ট ও রেজিস্ট্যান্স (শেষ ২০ ক্যান্ডেলের লো/হাই) ----
    recent_high  = high.iloc[-20:].max()
    recent_low   = low.iloc[-20:].min()
    pivot        = (recent_high + recent_low + current_price) / 3
    resistance1  = 2 * pivot - recent_low
    support1     = 2 * pivot - recent_high
    resistance2  = pivot + (recent_high - recent_low)
    support2     = pivot - (recent_high - recent_low)

    return {
        "price":      current_price,
        "pct_change": pct_change,
        "rsi":        rsi_val,
        "macd":       macd_val,
        "macd_sig":   macd_sig,
        "macd_hist":  macd_diff,
        "ema20":      ema20,
        "ema50":      ema50,
        "sma200":     sma200,
        "bb_upper":   bb_up,
        "bb_mid":     bb_mid,
        "bb_lower":   bb_low,
        "bb_pct":     bb_pct,
        "atr":        atr,
        "support1":   support1,
        "support2":   support2,
        "resistance1":resistance1,
        "resistance2":resistance2,
    }


# ==================================================================
# সিগনাল জেনারেটর
# ==================================================================
def generate_signal(ind: dict, daily_df: pd.DataFrame | None) -> dict:
    """
    সকল ইন্ডিকেটর থেকে একটি সামগ্রিক সিগনাল তৈরি করুন।
    Returns: {"bias": "BUY/SELL/NEUTRAL", "strength": "STRONG/MODERATE/WEAK", "reasons": [...]}
    """
    buy_points  = 0
    sell_points = 0
    reasons     = []

    price   = ind["price"]
    rsi     = ind["rsi"]
    macd_h  = ind["macd_hist"]
    ema20   = ind["ema20"]
    ema50   = ind["ema50"]
    sma200  = ind["sma200"]
    bb_pct  = ind["bb_pct"]

    # RSI বিশ্লেষণ
    if rsi < 35:
        buy_points += 2
        reasons.append(f"📉 RSI অতিবিক্রীত ({rsi:.1f}) — বাই সুযোগ")
    elif rsi > 65:
        sell_points += 2
        reasons.append(f"📈 RSI অতিক্রীত ({rsi:.1f}) — সেল চাপ সম্ভব")
    elif rsi > 50:
        buy_points += 1
        reasons.append(f"RSI বুলিশ জোনে ({rsi:.1f})")
    else:
        sell_points += 1
        reasons.append(f"RSI বিয়ারিশ জোনে ({rsi:.1f})")

    # MACD হিস্টোগ্রাম
    if macd_h > 0:
        buy_points += 1
        reasons.append("MACD হিস্টোগ্রাম পজিটিভ (বুলিশ মোমেন্টাম)")
    else:
        sell_points += 1
        reasons.append("MACD হিস্টোগ্রাম নেগেটিভ (বিয়ারিশ মোমেন্টাম)")

    # মূল্য vs EMA20
    if price > ema20:
        buy_points += 1
        reasons.append(f"মূল্য EMA20 (${ema20:,.2f}) উপরে ✅")
    else:
        sell_points += 1
        reasons.append(f"মূল্য EMA20 (${ema20:,.2f}) নিচে ❌")

    # মূল্য vs EMA50
    if price > ema50:
        buy_points += 1
        reasons.append(f"মূল্য EMA50 (${ema50:,.2f}) উপরে ✅")
    else:
        sell_points += 1
        reasons.append(f"মূল্য EMA50 (${ema50:,.2f}) নিচে ❌")

    # মূল্য vs SMA200
    if price > sma200:
        buy_points += 1
        reasons.append(f"মূল্য SMA200 (${sma200:,.2f}) উপরে — দীর্ঘমেয়াদী বুলিশ ✅")
    else:
        sell_points += 1
        reasons.append(f"মূল্য SMA200 (${sma200:,.2f}) নিচে — দীর্ঘমেয়াদী বিয়ারিশ ❌")

    # Bollinger Band অবস্থান
    if bb_pct < 0.2:
        buy_points += 2
        reasons.append(f"BB লোয়ার ব্যান্ডের কাছে — ওভারসোল্ড রিবাউন্ড সম্ভব")
    elif bb_pct > 0.8:
        sell_points += 2
        reasons.append(f"BB আপার ব্যান্ডের কাছে — ওভারবট, রিজেকশন সম্ভব")

    # সামগ্রিক বায়াস নির্ধারণ
    total = buy_points + sell_points
    if total == 0:
        bias = "NEUTRAL"
        strength = "WEAK"
    elif buy_points > sell_points:
        bias = "BUY"
        ratio = buy_points / total
        strength = "STRONG" if ratio >= 0.7 else "MODERATE"
    elif sell_points > buy_points:
        bias = "SELL"
        ratio = sell_points / total
        strength = "STRONG" if ratio >= 0.7 else "MODERATE"
    else:
        bias = "NEUTRAL"
        strength = "MODERATE"

    return {"bias": bias, "strength": strength, "reasons": reasons}


# ==================================================================
# ট্রেড সেটআপ
# ==================================================================
def build_trade_setup(ind: dict, signal: dict, name: str) -> str:
    """সিগনালের ভিত্তিতে ট্রেড সেটআপ তৈরি করুন।"""
    price  = ind["price"]
    atr    = ind["atr"]
    bias   = signal["bias"]

    if bias == "BUY":
        sl  = price - 1.5 * atr
        tp1 = price + 1.5 * atr
        tp2 = price + 2.5 * atr
        tp3 = price + 4.0 * atr
        rr1 = 1.5
        setup = (
            f"🟢 *লং (Buy) সেটআপ*\n"
            f"  ├ এন্ট্রি: `${price:,.2f}`\n"
            f"  ├ স্টপ লস: `${sl:,.2f}` (-{1.5*atr/price*100:.1f}%)\n"
            f"  ├ TP1: `${tp1:,.2f}` (+{1.5*atr/price*100:.1f}%) R:R=1:1.5\n"
            f"  ├ TP2: `${tp2:,.2f}` (+{2.5*atr/price*100:.1f}%) R:R=1:2.5\n"
            f"  └ TP3: `${tp3:,.2f}` (+{4.0*atr/price*100:.1f}%) R:R=1:4\n"
        )
    elif bias == "SELL":
        sl  = price + 1.5 * atr
        tp1 = price - 1.5 * atr
        tp2 = price - 2.5 * atr
        tp3 = price - 4.0 * atr
        setup = (
            f"🔴 *শর্ট (Sell) সেটআপ*\n"
            f"  ├ এন্ট্রি: `${price:,.2f}`\n"
            f"  ├ স্টপ লস: `${sl:,.2f}` (+{1.5*atr/price*100:.1f}%)\n"
            f"  ├ TP1: `${tp1:,.2f}` (-{1.5*atr/price*100:.1f}%) R:R=1:1.5\n"
            f"  ├ TP2: `${tp2:,.2f}` (-{2.5*atr/price*100:.1f}%) R:R=1:2.5\n"
            f"  └ TP3: `${tp3:,.2f}` (-{4.0*atr/price*100:.1f}%) R:R=1:4\n"
        )
    else:
        setup = (
            f"⚪ *নিউট্রাল — অপেক্ষা করুন*\n"
            f"  ├ সাপোর্ট: `${ind['support1']:,.2f}`\n"
            f"  └ রেজিস্ট্যান্স: `${ind['resistance1']:,.2f}`\n"
        )

    return setup


# ==================================================================
# Telegram মেসেজ বিল্ডার
# ==================================================================
def build_message(symbol: str, meta: dict, ind: dict, signal: dict, df_1h: pd.DataFrame) -> str:
    """Telegram মার্কডাউন মেসেজ তৈরি করুন।"""
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")

    # সিগনাল ইমোজি
    bias_map = {
        "BUY":     ("🟢 BUY",   "📈 বুলিশ"),
        "SELL":    ("🔴 SELL",  "📉 বিয়ারিশ"),
        "NEUTRAL": ("⚪ WAIT",  "↔️ নিউট্রাল"),
    }
    sig_label, bias_label = bias_map[signal["bias"]]
    strength_label = {"STRONG": "🔥 শক্তিশালী", "MODERATE": "⚡ মাঝারি", "WEAK": "💤 দুর্বল"}.get(signal["strength"], "")

    # প্রাইস পরিবর্তন ইমোজি
    chg = ind["pct_change"]
    chg_icon = "📈" if chg >= 0 else "📉"

    # RSI ব্যাখ্যা
    rsi = ind["rsi"]
    rsi_label = "অতিক্রীত (Overbought)" if rsi > 70 else "অতিবিক্রীত (Oversold)" if rsi < 30 else "নরমাল"

    # MACD ব্যাখ্যা
    macd_label = "✅ বুলিশ ক্রসওভার" if ind["macd"] > ind["macd_sig"] else "❌ বিয়ারিশ ক্রসওভার"

    # ট্রেড সেটআপ
    trade_setup = build_trade_setup(ind, signal, meta["name"])

    # কারণগুলো
    reasons_text = "\n".join([f"  • {r}" for r in signal["reasons"][:5]])

    msg = f"""
{meta['emoji']} *{meta['name']} — মার্কেট এনালাইসিস*
🕐 *সময়:* {now_bd} (BST)
━━━━━━━━━━━━━━━━━━━━━━━━━

💰 *বর্তমান মূল্য:* `${ind['price']:,.2f}`
{chg_icon} *পরিবর্তন:* `{chg:+.2f}%` (গত ক্যান্ডেল থেকে)

📊 *টেকনিক্যাল ইন্ডিকেটর (1H)*
  ├ RSI (14): `{rsi:.1f}` — {rsi_label}
  ├ MACD: {macd_label}
  ├ EMA20: `${ind['ema20']:,.2f}`
  ├ EMA50: `${ind['ema50']:,.2f}`
  └ SMA200: `${ind['sma200']:,.2f}`

📐 *Bollinger Bands*
  ├ 🔴 আপার: `${ind['bb_upper']:,.2f}`
  ├ ⚪ মিড:   `${ind['bb_mid']:,.2f}`
  └ 🟢 লোয়ার: `${ind['bb_lower']:,.2f}`

🔑 *কী লেভেল (Pivot Points)*
  ├ 🔴 R2: `${ind['resistance2']:,.2f}`
  ├ 🔴 R1: `${ind['resistance1']:,.2f}`
  ├ ⚪ বর্তমান: `${ind['price']:,.2f}` ◀
  ├ 🟢 S1: `${ind['support1']:,.2f}`
  └ 🟢 S2: `${ind['support2']:,.2f}`

⚡ *ATR (ভোলাটিলিটি):* `${ind['atr']:,.2f}`

━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 *সামগ্রিক বায়াস: {sig_label}* | {strength_label}

📋 *বিশ্লেষণের কারণ:*
{reasons_text}

━━━━━━━━━━━━━━━━━━━━━━━━━
{trade_setup}
━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ _এটি শুধুমাত্র শিক্ষামূলক বিশ্লেষণ। ট্রেডের আগে নিজস্ব গবেষণা করুন এবং রিস্ক ম্যানেজমেন্ট মেনে চলুন।_
"""
    return msg.strip()


# ==================================================================
# Telegram পাঠানো
# ==================================================================
def send_telegram(message: str) -> bool:
    """Telegram Bot API-তে মেসেজ পাঠান।"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("❌ TELEGRAM_BOT_TOKEN বা TELEGRAM_CHAT_ID সেট নেই!")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        logger.info("✅ Telegram-এ মেসেজ পাঠানো সফল!")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Telegram HTTP Error: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"❌ Telegram Error: {e}")
        return False


def send_divider():
    """দুটি সিম্বলের মাঝে একটি বিভাজক মেসেজ পাঠান।"""
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    msg = f"━━━━━━━━━━━━━━━━━━━━━━━━━\n📊 *অটোমেটেড মার্কেট রিপোর্ট*\n🕐 {now_bd} (BST)\nদুটি সিম্বলের এনালাইসিস নিচে দেওয়া হলো 👇"
    send_telegram(msg)


# ==================================================================
# মেইন ফাংশন
# ==================================================================
def analyze_symbol(symbol: str) -> bool:
    """একটি সিম্বল এনালাইজ করে Telegram পাঠান।"""
    meta = SYMBOLS[symbol]
    yf_sym = meta["yf_symbol"]

    logger.info(f"🔍 Analyzing {symbol} ({yf_sym})...")

    # ডেটা আনুন
    df_1h = fetch_ohlcv(yf_sym, period="60d", interval="1h")
    if df_1h is None or len(df_1h) < 50:
        logger.error(f"❌ {symbol}: পর্যাপ্ত ডেটা নেই")
        send_telegram(f"❌ {meta['name']}: মার্কেট ডেটা পাওয়া যায়নি। পরবর্তী সেশনে আবার চেষ্টা করা হবে।")
        return False

    df_daily = fetch_daily(yf_sym)

    # ইন্ডিকেটর ও সিগনাল
    ind    = compute_indicators(df_1h)
    signal = generate_signal(ind, df_daily)

    # মেসেজ তৈরি ও পাঠানো
    message = build_message(symbol, meta, ind, signal, df_1h)
    return send_telegram(message)


def main():
    logger.info("🚀 Market Analysis Bot শুরু হচ্ছে...")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("❌ পরিবেশ চলকগুলো সেট করা নেই। GitHub Secrets চেক করুন।")
        sys.exit(1)

    # শুরুর মেসেজ
    send_divider()

    # Gold এনালাইসিস
    gold_ok = analyze_symbol("GOLD")

    # BTC এনালাইসিস
    btc_ok = analyze_symbol("BTC")

    if gold_ok and btc_ok:
        logger.info("✅ সব এনালাইসিস সফলভাবে পাঠানো হয়েছে!")
    else:
        logger.warning("⚠️ কিছু এনালাইসিস ব্যর্থ হয়েছে।")
        sys.exit(1)


if __name__ == "__main__":
    main()
