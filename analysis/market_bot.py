#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC & Gold Market Analysis Telegram Bot
- Gold: CoinGecko free API (XAUUSD via metals-api fallback to yfinance GLD)
- BTC:  CoinGecko free API
"""

import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
BD_TZ = pytz.timezone("Asia/Dhaka")

# ──────────────────────────────────────────────
# DATA FETCHING — CoinGecko Free API
# ──────────────────────────────────────────────

def fetch_btc_ohlc(days: int = 30) -> pd.DataFrame | None:
    """BTC OHLC data from CoinGecko (free, no API key needed)."""
    try:
        # CoinGecko returns OHLC in 4h candles for 30 days
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["volume"] = 0  # CoinGecko OHLC doesn't include volume
        logger.info(f"BTC: fetched {len(df)} candles from CoinGecko")
        return df
    except Exception as e:
        logger.error(f"BTC CoinGecko error: {e}")
        return None


def fetch_btc_price() -> float | None:
    """Get current BTC price."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()["bitcoin"]["usd"]
    except Exception as e:
        logger.error(f"BTC price error: {e}")
        return None


def fetch_gold_ohlc() -> pd.DataFrame | None:
    """
    Gold OHLC — tries multiple sources:
    1. yfinance GLD (SPDR Gold ETF, very reliable)
    2. yfinance XAUUSD=X
    3. yfinance GC=F (Gold Futures)
    """
    symbols_to_try = ["GLD", "XAUUSD=X", "GC=F", "IAU"]
    multipliers    = {"GLD": 10.0, "GC=F": 1.0, "XAUUSD=X": 1.0, "IAU": 19.5}

    try:
        import yfinance as yf
        for sym in symbols_to_try:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(period="60d", interval="4h")
                if df is not None and len(df) >= 30:
                    df.columns = [c.lower() for c in df.columns]
                    df = df[["open", "high", "low", "close", "volume"]].dropna()
                    mult = multipliers.get(sym, 1.0)
                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] * mult
                    logger.info(f"Gold: fetched {len(df)} candles via {sym} (x{mult})")
                    return df
            except Exception as e2:
                logger.warning(f"Gold {sym} failed: {e2}")
                continue
    except ImportError:
        pass

    # Fallback: CoinGecko XAU via Paxos Gold (PAXG)
    try:
        url = "https://api.coingecko.com/api/v3/coins/pax-gold/ohlc?vs_currency=usd&days=30"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data and len(data) >= 20:
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["volume"] = 0
            logger.info(f"Gold: fetched {len(df)} candles via PAXG (CoinGecko)")
            return df
    except Exception as e3:
        logger.error(f"Gold PAXG error: {e3}")

    return None


def fetch_gold_price() -> float | None:
    """Get current Gold price (PAXG ≈ XAUUSD)."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=pax-gold&vs_currencies=usd"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()["pax-gold"]["usd"]
    except Exception as e:
        logger.error(f"Gold price error: {e}")
        # Fallback: yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker("GLD")
            hist = ticker.history(period="1d", interval="1h")
            if hist is not None and len(hist) > 0:
                return float(hist["Close"].iloc[-1]) * 10.0
        except:
            pass
        return None


# ──────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, current_price: float | None = None) -> dict | None:
    """Calculate all technical indicators from OHLCV dataframe."""
    if df is None or len(df) < 20:
        return None

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    if current_price:
        # Use live price as the last close
        close = close.copy()
        close.iloc[-1] = current_price

    try:
        rsi_val  = RSIIndicator(close=close, window=14).rsi().iloc[-1]
        macd_obj = MACD(close=close)
        macd_val = macd_obj.macd().iloc[-1]
        macd_sig = macd_obj.macd_signal().iloc[-1]
        macd_diff= macd_obj.macd_diff().iloc[-1]
        ema20    = EMAIndicator(close=close, window=min(20, len(close)-1)).ema_indicator().iloc[-1]
        ema50    = EMAIndicator(close=close, window=min(50, len(close)-1)).ema_indicator().iloc[-1]
        sma200   = SMAIndicator(close=close, window=min(200, len(close))).sma_indicator().iloc[-1]
        bb       = BollingerBands(close=close, window=min(20, len(close)-1), window_dev=2)
        bb_up    = bb.bollinger_hband().iloc[-1]
        bb_mid   = bb.bollinger_mavg().iloc[-1]
        bb_low   = bb.bollinger_lband().iloc[-1]
        bb_pct   = bb.bollinger_pband().iloc[-1]
        atr      = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]

        price      = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        pct_change = (price - prev_close) / prev_close * 100

        recent_high  = float(high.iloc[-20:].max())
        recent_low   = float(low.iloc[-20:].min())
        pivot        = (recent_high + recent_low + price) / 3
        resistance1  = 2 * pivot - recent_low
        support1     = 2 * pivot - recent_high
        resistance2  = pivot + (recent_high - recent_low)
        support2     = pivot - (recent_high - recent_low)

        return {
            "price": price, "pct_change": pct_change,
            "rsi": rsi_val, "macd": macd_val, "macd_sig": macd_sig, "macd_hist": macd_diff,
            "ema20": ema20, "ema50": ema50, "sma200": sma200,
            "bb_upper": bb_up, "bb_mid": bb_mid, "bb_lower": bb_low, "bb_pct": bb_pct,
            "atr": atr,
            "support1": support1, "support2": support2,
            "resistance1": resistance1, "resistance2": resistance2,
        }
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return None


# ──────────────────────────────────────────────
# SIGNAL GENERATOR
# ──────────────────────────────────────────────

def generate_signal(ind: dict) -> dict:
    buy_points = sell_points = 0
    reasons = []
    price, rsi, macd_h = ind["price"], ind["rsi"], ind["macd_hist"]
    ema20, ema50, sma200, bb_pct = ind["ema20"], ind["ema50"], ind["sma200"], ind["bb_pct"]

    if rsi < 35:
        buy_points += 2; reasons.append(f"RSI oversold ({rsi:.1f}) — buy opportunity")
    elif rsi > 65:
        sell_points += 2; reasons.append(f"RSI overbought ({rsi:.1f}) — sell pressure")
    elif rsi > 50:
        buy_points += 1; reasons.append(f"RSI bullish zone ({rsi:.1f})")
    else:
        sell_points += 1; reasons.append(f"RSI bearish zone ({rsi:.1f})")

    if macd_h > 0:
        buy_points += 1; reasons.append("MACD histogram positive (bullish momentum)")
    else:
        sell_points += 1; reasons.append("MACD histogram negative (bearish momentum)")

    if price > ema20:
        buy_points += 1; reasons.append(f"Price above EMA20 (${ema20:,.2f})")
    else:
        sell_points += 1; reasons.append(f"Price below EMA20 (${ema20:,.2f})")

    if price > ema50:
        buy_points += 1; reasons.append(f"Price above EMA50 (${ema50:,.2f})")
    else:
        sell_points += 1; reasons.append(f"Price below EMA50 (${ema50:,.2f})")

    if price > sma200:
        buy_points += 1; reasons.append(f"Price above SMA200 — long-term bullish")
    else:
        sell_points += 1; reasons.append(f"Price below SMA200 — long-term bearish")

    if bb_pct < 0.2:
        buy_points += 2; reasons.append("Near BB lower band — oversold bounce possible")
    elif bb_pct > 0.8:
        sell_points += 2; reasons.append("Near BB upper band — overbought, rejection possible")

    total = buy_points + sell_points
    if total == 0:
        bias, strength = "NEUTRAL", "WEAK"
    elif buy_points > sell_points:
        bias = "BUY"; ratio = buy_points / total
        strength = "STRONG" if ratio >= 0.7 else "MODERATE"
    elif sell_points > buy_points:
        bias = "SELL"; ratio = sell_points / total
        strength = "STRONG" if ratio >= 0.7 else "MODERATE"
    else:
        bias, strength = "NEUTRAL", "MODERATE"

    return {"bias": bias, "strength": strength, "reasons": reasons}


# ──────────────────────────────────────────────
# MESSAGE BUILDER
# ──────────────────────────────────────────────

def build_message(name: str, emoji: str, ind: dict, signal: dict) -> str:
    """Build a concise summary message for Telegram."""
    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")

    bias_map = {
        "BUY":     "🟢 BUY",
        "SELL":    "🔴 SELL",
        "NEUTRAL": "⚪ WAIT",
    }
    sig_label = bias_map[signal["bias"]]
    str_map   = {"STRONG": "🔥", "MODERATE": "⚡", "WEAK": "💤"}
    str_icon  = str_map.get(signal["strength"], "")

    chg       = ind["pct_change"]
    chg_icon  = "📈" if chg >= 0 else "📉"
    rsi       = ind["rsi"]
    rsi_icon  = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
    macd_icon = "✅" if ind["macd"] > ind["macd_sig"] else "❌"
    price     = ind["price"]
    atr       = ind["atr"]
    bias      = signal["bias"]

    # Compact trade setup
    if bias == "BUY":
        sl  = price - 1.5 * atr
        tp1 = price + 1.5 * atr
        tp2 = price + 2.5 * atr
        trade = (
            f"🟢 *Buy* | SL: `${sl:,.0f}` | TP1: `${tp1:,.0f}` | TP2: `${tp2:,.0f}`"
        )
    elif bias == "SELL":
        sl  = price + 1.5 * atr
        tp1 = price - 1.5 * atr
        tp2 = price - 2.5 * atr
        trade = (
            f"🔴 *Sell* | SL: `${sl:,.0f}` | TP1: `${tp1:,.0f}` | TP2: `${tp2:,.0f}`"
        )
    else:
        trade = f"⚪ *Wait* | S: `${ind['support1']:,.0f}` | R: `${ind['resistance1']:,.0f}`"

    msg = (
        f"{emoji} *{name}*\n"
        f"🕐 {now_bd} (BST)\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"💰 `${price:,.2f}` {chg_icon} `{chg:+.2f}%`\n\n"
        f"📊 RSI: {rsi_icon}`{rsi:.0f}` | MACD: {macd_icon} | ATR: `{atr:,.0f}`\n"
        f"📐 EMA20: `{ind['ema20']:,.0f}` | EMA50: `{ind['ema50']:,.0f}`\n\n"
        f"🔑 R1:`${ind['resistance1']:,.0f}` | S1:`${ind['support1']:,.0f}`\n"
        f"   R2:`${ind['resistance2']:,.0f}` | S2:`${ind['support2']:,.0f}`\n\n"
        f"🎯 *{sig_label}* {str_icon}\n"
        f"{trade}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"⚠️ _শিক্ষামূলক বিশ্লেষণ মাত্র_"
    )
    return msg


# ──────────────────────────────────────────────
# TELEGRAM
# ──────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
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
        logger.info("Telegram message sent!")
        return True
    except requests.exceptions.HTTPError as e:
        logger.error(f"Telegram HTTP error: {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    logger.info("Market Analysis Bot starting...")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing environment variables!")
        sys.exit(1)

    now_bd = datetime.now(BD_TZ).strftime("%d %b %Y | %I:%M %p")
    send_telegram(
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Automated Market Report*\n"
        f"🕐 {now_bd} (BST)\n"
        f"BTC & Gold analysis below 👇\n"
        f"━━━━━━━━━━━━━━━━━━━━━━"
    )

    results = []

    # ── GOLD ──
    logger.info("Fetching Gold data...")
    gold_df    = fetch_gold_ohlc()
    gold_price = fetch_gold_price()
    gold_ind   = compute_indicators(gold_df, gold_price)

    if gold_ind:
        gold_signal = generate_signal(gold_ind)
        gold_msg    = build_message("Gold (XAUUSD)", "🥇", gold_ind, gold_signal)
        results.append(send_telegram(gold_msg))
        logger.info(f"Gold: ${gold_ind['price']:,.2f} | {gold_signal['bias']} {gold_signal['strength']}")
    else:
        send_telegram("❌ Gold: Could not fetch market data. Will retry next session.")
        results.append(False)

    # ── BTC ──
    logger.info("Fetching BTC data...")
    btc_df    = fetch_btc_ohlc(days=30)
    btc_price = fetch_btc_price()
    btc_ind   = compute_indicators(btc_df, btc_price)

    if btc_ind:
        btc_signal = generate_signal(btc_ind)
        btc_msg    = build_message("Bitcoin (BTC/USD)", "₿", btc_ind, btc_signal)
        results.append(send_telegram(btc_msg))
        logger.info(f"BTC: ${btc_ind['price']:,.2f} | {btc_signal['bias']} {btc_signal['strength']}")
    else:
        send_telegram("❌ BTC: Could not fetch market data. Will retry next session.")
        results.append(False)

    if all(results):
        logger.info("All reports sent successfully!")
    else:
        logger.warning("Some reports failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
