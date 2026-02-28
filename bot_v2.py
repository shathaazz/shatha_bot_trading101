import asyncio
import os
import logging
import requests
import random
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas as pd
from telegram import Bot
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ConversationHandler, CallbackQueryHandler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== حفظ وتحميل البيانات =====
BOT_DATA_FILE = "bot_data.json"
import json

# ===== Google Sheets Integration =====
from sheets_db import (
    setup_sheets, journal_add, journal_set_status,
    journal_set_result, journal_load,
    account_save, account_load,
    weights_save, weights_load, stats_add_week
)

def save_data():
    """يحفظ كل البيانات في Google Sheets + ملف JSON backup"""
    try:
        account_save(ACCOUNT, DAILY_RISK)
        data = {
            "account": ACCOUNT,
            "daily_risk": DAILY_RISK,
            "journal": {k: {
                kk: vv for kk, vv in v.items() if kk != "analysis"
            } for k, v in JOURNAL.items()},
            "weights": WEIGHTS_MEMORY,
        }
        with open(BOT_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"خطأ حفظ البيانات: {e}")

def load_data():
    """يحمل البيانات من Google Sheets عند التشغيل"""
    # تحميل Account
    saved_acc = account_load()
    for k, v in saved_acc.items():
        if k in ACCOUNT:
            ACCOUNT[k] = v

    # تحميل Journal من Sheets
    sheets_journal = journal_load()
    if sheets_journal:
        for tid, t in sheets_journal.items():
            t["yf_sym"] = SYMBOLS.get(t.get("symbol", ""), "")
        JOURNAL.update(sheets_journal)
    else:
        # fallback: ملف محلي
        try:
            if not os.path.exists(BOT_DATA_FILE):
                return
            with open(BOT_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_risk = data.get("daily_risk", {})
            for k, v in saved_risk.items():
                if k in DAILY_RISK:
                    DAILY_RISK[k] = v
            JOURNAL.update(data.get("journal", {}))
        except Exception as e:
            logger.error(f"خطأ تحميل البيانات: {e}")

    # تحميل Weights
    w = weights_load()
    if w:
        for k in DEFAULT_WEIGHTS:
            if k not in w:
                w[k] = DEFAULT_WEIGHTS[k]
        WEIGHTS_MEMORY.update(w)

    logger.info(f"✅ تم تحميل البيانات: {len(JOURNAL)} صفقة في الجورنال")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")
RIYADH_TZ = pytz.timezone("Asia/Riyadh")

# ===== حساب البروب فيرم =====
PHASE_TARGETS = {
    "challenge":     {"target": 8.0,  "max_dd": 10.0, "daily_dd": 5.0},
    "verification":  {"target": 4.0,  "max_dd": 10.0, "daily_dd": 5.0},
    "funded":        {"target": None, "max_dd": 10.0, "daily_dd": 5.0},
}

ACCOUNT = {
    "balance": float(os.environ.get("ACCOUNT_BALANCE", "10000")),
    "current_balance": float(os.environ.get("ACCOUNT_BALANCE", "10000")),
    "max_drawdown": float(os.environ.get("MAX_DRAWDOWN", "10.0")),
    "daily_drawdown": float(os.environ.get("DAILY_DRAWDOWN", "5.0")),
    "drawdown_used": 0.0,
    "daily_used": 0.0,
    "trades_week": 0,
    "trades_today": 0,
    "pnl_percent": 0.0,
    "firm_name": os.environ.get("FIRM_NAME", "Prop Firm"),
    "phase": os.environ.get("ACCOUNT_PHASE", "challenge"),
    "profit_split": float(os.environ.get("PROFIT_SPLIT", "20")),
}

SYMBOLS = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "BTCUSD": "BTC-USD",
    "USDCHF": "USDCHF=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
}

TRADINGVIEW_LINKS = {
    "XAUUSD": "https://www.tradingview.com/chart/?symbol=OANDA%3AXAUUSD",
    "XAGUSD": "https://www.tradingview.com/chart/?symbol=OANDA%3AXAGUSD",
    "EURUSD": "https://www.tradingview.com/chart/?symbol=OANDA%3AEURUSD",
    "GBPUSD": "https://www.tradingview.com/chart/?symbol=OANDA%3AGBPUSD",
    "BTCUSD": "https://www.tradingview.com/chart/?symbol=BITSTAMP%3ABTCUSD",
    "USDCHF": "https://www.tradingview.com/chart/?symbol=OANDA%3AUSDCHF",
    "USDJPY": "https://www.tradingview.com/chart/?symbol=OANDA%3AUSDJPY",
    "AUDUSD": "https://www.tradingview.com/chart/?symbol=OANDA%3AAUDUSD",
}

HIGH_IMPACT_KEYWORDS = [
    "Fed", "Federal Reserve", "FOMC", "Interest Rate",
    "CPI", "NFP", "Non-Farm", "GDP", "Powell", "ECB", "BOE", "BOJ"
]

# ===== رسايل البوت =====
WAITING_MSGS = [
    "عيني على الشارت، لحظة ⏳",
    "أفحص الأزواج واحد واحد 🔍",
    "ثانية وأخبرك وش شايف 👀",
]

NO_SETUP_MSGS = [
    "ما في سيتاب يستاهل الحين يا شذا 🤷‍♀️\nروحي اتقهوي وأنا أراقب ☕",
    "السوق هادي، ما في فرصة بشروطنا 😌\nالصبر ذهب 💛",
    "فحصت كل شي، ما لقيت شي صح 🙅‍♀️\nأحسن من صفقة غلط صح؟",
]

DAILY_TIPS = [
    "ما في صفقة تستاهل تكسرين عشانها خطتك 💡",
    "الصفقة الصح تجيك، ما تروحين إليها ⏳",
    "المهم إدارة المخاطرة مو الربح السريع 🛡️",
    "أي ضغط داخل الصفقة؟ اطلعي منها 🧠",
    "الانضباط يفرق بين المحترف والمبتدئ 🏆",
    "كل صفقة في الجورنال، اللي ما يوثق ما يتعلم 📝",
    "الحساب أهم من أي صفقة، خذي استراحة لو تعبتِ 🌿",
]

# ===== حالات المحادثة للتحديث =====
(S_BALANCE, S_PNL, S_DD, S_DAILY, S_TRADES_W, S_TRADES_D) = range(6)

# ===== نظام الأوزان المتعلمة =====
WEIGHTS_FILE = "weights.json"

DEFAULT_WEIGHTS = {
    "idm_wick_high": 1.0,
    "idm_wick_medium": 1.0,
    "h4_of_high": 1.0,
    "h4_of_medium": 1.0,
    "h1_of_high": 1.0,
    "has_liquidity": 1.0,
    "daily_match": 1.0,
    "weekly_match": 1.0,
    "ob_body_high": 1.0,
}

WEIGHTS_MEMORY = DEFAULT_WEIGHTS.copy()

def load_weights():
    global WEIGHTS_MEMORY
    # أول شي: Google Sheets
    w = weights_load()
    if w:
        for k in DEFAULT_WEIGHTS:
            if k not in w:
                w[k] = DEFAULT_WEIGHTS[k]
        WEIGHTS_MEMORY = w.copy()
        return w.copy()
    # fallback: ملف محلي
    try:
        with open(WEIGHTS_FILE, "r") as f:
            w = json.load(f)
            for k in DEFAULT_WEIGHTS:
                if k not in w:
                    w[k] = DEFAULT_WEIGHTS[k]
            WEIGHTS_MEMORY = w.copy()
            return w
    except:
        return WEIGHTS_MEMORY.copy()

def save_weights(weights):
    global WEIGHTS_MEMORY
    WEIGHTS_MEMORY = weights.copy()
    weights_save(weights)  # Google Sheets
    try:                   # backup محلي
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception as e:
        logger.error(f"خطأ حفظ الأوزان: {e}")

def update_weights_entered(analysis, reached_tp=False):
    weights = load_weights()
    boost = 0.35 if reached_tp else 0.15
    if analysis.get("idm_wick", 0) > 0.5:
        weights["idm_wick_high"] = min(3.0, weights["idm_wick_high"] + boost)
    elif analysis.get("idm_wick", 0) > 0.35:
        weights["idm_wick_medium"] = min(3.0, weights["idm_wick_medium"] + boost)
    h4_of = analysis.get("h4_of", 0)
    if h4_of >= 0.7:
        weights["h4_of_high"] = min(3.0, weights["h4_of_high"] + boost)
    elif h4_of >= 0.5:
        weights["h4_of_medium"] = min(3.0, weights["h4_of_medium"] + boost)
    if analysis.get("h1_of", 0) >= 0.7:
        weights["h1_of_high"] = min(3.0, weights["h1_of_high"] + boost)
    if analysis.get("has_liquidity"):
        weights["has_liquidity"] = min(3.0, weights["has_liquidity"] + boost)
    if analysis.get("daily_match"):
        weights["daily_match"] = min(3.0, weights["daily_match"] + boost)
    if analysis.get("weekly_match"):
        weights["weekly_match"] = min(3.0, weights["weekly_match"] + boost)
    if analysis.get("ob", {}).get("body_ratio", 0) > 0.6:
        weights["ob_body_high"] = min(3.0, weights["ob_body_high"] + boost)
    save_weights(weights)

def update_weights_skipped(analysis):
    weights = load_weights()
    penalty = 0.10
    if analysis.get("idm_wick", 0) > 0.5:
        weights["idm_wick_high"] = max(0.1, weights["idm_wick_high"] - penalty)
    elif analysis.get("idm_wick", 0) > 0.35:
        weights["idm_wick_medium"] = max(0.1, weights["idm_wick_medium"] - penalty)
    h4_of = analysis.get("h4_of", 0)
    if h4_of >= 0.7:
        weights["h4_of_high"] = max(0.1, weights["h4_of_high"] - penalty)
    elif h4_of >= 0.5:
        weights["h4_of_medium"] = max(0.1, weights["h4_of_medium"] - penalty)
    if analysis.get("has_liquidity"):
        weights["has_liquidity"] = max(0.1, weights["has_liquidity"] - penalty)
    save_weights(weights)

def calc_quality_weighted(dbos, idm, ob, h4_of, h1_of, has_liquidity,
                           daily_match, weekly_match, in_ob, has_news,
                           idm_wick=0, ob_body=0):
    weights = WEIGHTS_MEMORY
    score = 0
    if dbos: score += 20
    if idm: score += 15
    if ob: score += 15
    if idm_wick > 0.5: score += 8 * weights["idm_wick_high"]
    elif idm_wick > 0.35: score += 5 * weights["idm_wick_medium"]
    if h4_of >= 0.7: score += 8 * weights["h4_of_high"]
    elif h4_of >= 0.5: score += 5 * weights["h4_of_medium"]
    if h1_of >= 0.7: score += 6 * weights["h1_of_high"]
    if has_liquidity: score += 8 * weights["has_liquidity"]
    if daily_match: score += 6 * weights["daily_match"]
    if weekly_match: score += 4 * weights["weekly_match"]
    if ob_body > 0.6: score += 5 * weights["ob_body_high"]
    if in_ob: score += 5
    if has_news: score -= 20
    return max(0, min(100, round(score)))


# ===== جورنال الصفقات =====
JOURNAL = {}
TRADE_COUNTER = [0]
SENT_SETUPS = {}
PENDING_SETUPS = {}

# ===== Daily Risk Breaker =====
DAILY_RISK = {
    "trading_stopped": False,
    "consecutive_losses": 0,
    "daily_loss_pct": 0.0,
    "stop_reason": "",
}


# ===== الأخبار =====
def check_news():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        if r.status_code != 200:
            return {"has_news": False, "events": []}
        now = datetime.utcnow()
        upcoming = []
        for ev in r.json():
            try:
                if ev.get("impact") != "High":
                    continue
                t = datetime.fromisoformat(ev.get("date", "").replace("Z", ""))
                diff = t - now
                if timedelta(hours=-1) <= diff <= timedelta(hours=24):
                    title = ev.get("title", "")
                    if any(k.lower() in title.lower() for k in HIGH_IMPACT_KEYWORDS):
                        upcoming.append({
                            "title": title,
                            "hours": round(diff.total_seconds() / 3600, 1)
                        })
            except:
                continue
        return {"has_news": len(upcoming) > 0, "events": upcoming[:3]}
    except:
        return {"has_news": False, "events": []}


# ===== تحليل السوق =====
def get_candles(yf_sym, tf, limit=100):
    try:
        period = {"1h": "7d", "4h": "60d", "1d": "180d", "1wk": "2y"}.get(tf, "60d")
        df = yf.Ticker(yf_sym).history(period=period, interval=tf)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        return df.tail(limit)
    except:
        return pd.DataFrame()


def detect_trend_structure(df, lookback=30):
    if len(df) < lookback:
        return "neutral"
    recent = df.tail(lookback)
    highs = []
    lows = []
    for i in range(2, len(recent) - 2):
        if (recent["high"].iloc[i] > recent["high"].iloc[i-1] and
            recent["high"].iloc[i] > recent["high"].iloc[i-2] and
            recent["high"].iloc[i] > recent["high"].iloc[i+1] and
            recent["high"].iloc[i] > recent["high"].iloc[i+2]):
            highs.append(recent["high"].iloc[i])
        if (recent["low"].iloc[i] < recent["low"].iloc[i-1] and
            recent["low"].iloc[i] < recent["low"].iloc[i-2] and
            recent["low"].iloc[i] < recent["low"].iloc[i+1] and
            recent["low"].iloc[i] < recent["low"].iloc[i+2]):
            lows.append(recent["low"].iloc[i])
    if len(highs) < 2 or len(lows) < 2:
        return "neutral"
    hh = highs[-1] > highs[-2]
    hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]
    ll = lows[-1] < lows[-2]
    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    return "neutral"


def detect_trend(df):
    return detect_trend_structure(df)


def find_swings(df, lb=5):
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def detect_order_flow(df, direction, lookback=10):
    if len(df) < lookback + 1:
        return 0.0
    recent = df.tail(lookback)
    score = 0
    total = lookback - 1
    for i in range(1, len(recent)):
        curr_high = recent["high"].iloc[i]
        prev_high = recent["high"].iloc[i-1]
        curr_low = recent["low"].iloc[i]
        prev_low = recent["low"].iloc[i-1]
        if direction == "bullish":
            if curr_high > prev_high: score += 0.5
            if curr_low > prev_low: score += 0.5
        else:
            if curr_high < prev_high: score += 0.5
            if curr_low < prev_low: score += 0.5
    return round(score / total, 2)


def detect_dbos(df, direction=None, highs=None, lows=None):
    lb = 5
    h_list, l_list = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i-lb:i+lb+1].max():
            h_list.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == df["low"].iloc[i-lb:i+lb+1].min():
            l_list.append((i, df["low"].iloc[i]))

    if direction == "bullish" and len(h_list) >= 2:
        for i in range(len(h_list)-1, 0, -1):
            h2_idx, h2_val = h_list[i]
            h1_idx, h1_val = h_list[i-1]
            if h2_val <= h1_val: continue
            segment = df.iloc[h1_idx:h2_idx+1]
            if len(segment) < 2 or len(segment) > 60: continue
            move_size = h2_val - segment["low"].min()
            if move_size <= 0: continue
            max_pb = max([segment["high"].iloc[k-1] - segment["low"].iloc[k] for k in range(1, len(segment))], default=0)
            if max_pb / move_size > 0.40: continue
            for j in range(h2_idx, min(h2_idx+8, len(df))):
                candle = df.iloc[j]
                body_top = max(candle["open"], candle["close"])
                if body_top > h2_val:
                    return {"index": j, "price": h2_val, "impulse_start": h1_idx, "sweep_level": segment["low"].min(), "broke_two": True}

    elif direction == "bearish" and len(l_list) >= 2:
        for i in range(len(l_list)-1, 0, -1):
            l2_idx, l2_val = l_list[i]
            l1_idx, l1_val = l_list[i-1]
            if l2_val >= l1_val: continue
            segment = df.iloc[l1_idx:l2_idx+1]
            if len(segment) < 2 or len(segment) > 60: continue
            move_size = segment["high"].max() - l2_val
            if move_size <= 0: continue
            max_pb = max([segment["high"].iloc[k] - segment["low"].iloc[k-1] for k in range(1, len(segment))], default=0)
            if max_pb / move_size > 0.40: continue
            for j in range(l2_idx, min(l2_idx+8, len(df))):
                candle = df.iloc[j]
                body_bottom = min(candle["open"], candle["close"])
                if body_bottom < l2_val:
                    return {"index": j, "price": l2_val, "impulse_start": l1_idx, "sweep_level": segment["high"].max(), "broke_two": True}
    return None

def find_idm(df, dbos_idx, direction):
    search_end = min(dbos_idx + 25, len(df))
    for i in range(dbos_idx + 1, search_end):
        c = df.iloc[i]
        candle_range = c["high"] - c["low"]
        if candle_range == 0:
            continue
        body = abs(c["close"] - c["open"])
        body_ratio = body / candle_range
        if direction == "bullish":
            lower_wick = min(c["open"], c["close"]) - c["low"]
            wick_ratio = lower_wick / candle_range
            is_pin_bar = wick_ratio > 0.35 and body_ratio < 0.5
            is_bearish_engulf = c["close"] < c["open"] and body_ratio > 0.6
            if (is_pin_bar or is_bearish_engulf):
                if c["low"] < df["low"].iloc[max(0,i-3):i].min():
                    return {"index": i, "price": c["low"], "wick_ratio": round(wick_ratio, 2), "type": "pin_bar" if is_pin_bar else "engulf"}
        else:
            upper_wick = c["high"] - max(c["open"], c["close"])
            wick_ratio = upper_wick / candle_range
            is_pin_bar = wick_ratio > 0.35 and body_ratio < 0.5
            is_bullish_engulf = c["close"] > c["open"] and body_ratio > 0.6
            if (is_pin_bar or is_bullish_engulf):
                if c["high"] > df["high"].iloc[max(0,i-3):i].max():
                    return {"index": i, "price": c["high"], "wick_ratio": round(wick_ratio, 2), "type": "pin_bar" if is_pin_bar else "engulf"}
    return None

def find_ob(df, idm_idx, direction):
    if idm_idx is None or idm_idx < 2:
        return None
    for i in range(idm_idx - 1, max(idm_idx - 7, 0), -1):
        c = df.iloc[i]
        candle_range = c["high"] - c["low"]
        if candle_range == 0: continue
        body = abs(c["close"] - c["open"])
        if body / candle_range < 0.45: continue
        if direction == "bullish" and c["close"] < c["open"]:
            if i + 1 < len(df) and df["close"].iloc[i+1] > df["open"].iloc[i+1]:
                return {"top": c["open"], "bottom": c["close"], "index": i, "body_ratio": round(body/candle_range, 2)}
        elif direction == "bearish" and c["close"] > c["open"]:
            if i + 1 < len(df) and df["close"].iloc[i+1] < df["open"].iloc[i+1]:
                return {"top": c["close"], "bottom": c["open"], "index": i, "body_ratio": round(body/candle_range, 2)}
    for i in range(idm_idx - 1, max(idm_idx - 13, 0), -1):
        c = df.iloc[i]
        candle_range = c["high"] - c["low"]
        if candle_range == 0: continue
        body = abs(c["close"] - c["open"])
        if body / candle_range < 0.40: continue
        if direction == "bullish" and c["close"] < c["open"]:
            return {"top": c["open"], "bottom": c["close"], "index": i, "body_ratio": 0}
        elif direction == "bearish" and c["close"] > c["open"]:
            return {"top": c["close"], "bottom": c["open"], "index": i, "body_ratio": 0}
    return None

def ob_sweeps_liquidity(df, ob, direction, highs, lows):
    if not ob:
        return False
    ob_idx = ob.get("index", 0)
    prev_highs = [h[1] for h in highs if h[0] < ob_idx]
    prev_lows = [l[1] for l in lows if l[0] < ob_idx]
    if direction == "bullish" and prev_lows:
        nearest_low = max(prev_lows)
        return ob["bottom"] <= nearest_low <= ob["top"]
    elif direction == "bearish" and prev_highs:
        nearest_high = min(prev_highs)
        return ob["bottom"] <= nearest_high <= ob["top"]
    return False


def check_liquidity_sweep(df, direction):
    if len(df) < 20:
        return False
    recent = df.tail(20)
    prev_high = recent["high"].iloc[:-3].max()
    prev_low = recent["low"].iloc[:-3].min()
    last2 = df.iloc[-3:-1]
    last_close = df["close"].iloc[-1]
    if direction == "bullish":
        swept = last2["low"].min() < prev_low
        recovered = last_close > prev_low
        return swept and recovered
    else:
        swept = last2["high"].max() > prev_high
        recovered = last_close < prev_high
        return swept and recovered


def is_price_in_ob(current, ob, buffer=0.2):
    ob_range = ob["top"] - ob["bottom"]
    extended_top = ob["top"] + ob_range * buffer
    extended_bottom = ob["bottom"] - ob_range * buffer
    return extended_bottom <= current <= extended_top


def check_liquidity_above(df, direction, lookback=40):
    if len(df) < lookback:
        return False, 0
    recent = df.tail(lookback)
    current = df["close"].iloc[-1]
    if direction == "bullish":
        bsl_level = recent["high"].max()
        distance_pct = (bsl_level - current) / current * 100
        has_liquidity = 0.3 < distance_pct < 4.0
        return has_liquidity, round(bsl_level, 4)
    else:
        ssl_level = recent["low"].min()
        distance_pct = (current - ssl_level) / current * 100
        has_liquidity = 0.3 < distance_pct < 4.0
        return has_liquidity, round(ssl_level, 4)


def calc_quality(dbos, idm, ob, sweep, weekly_match, daily_match, in_ob, ob_sweep, has_news, h4_of=0, h1_of=0, has_liquidity=False):
    score = 0
    if dbos: score += 20
    if idm: score += 20
    if ob: score += 20
    if h4_of >= 0.7: score += 12
    elif h4_of >= 0.5: score += 6
    if h1_of >= 0.7: score += 8
    elif h1_of >= 0.5: score += 4
    if has_liquidity: score += 10
    if daily_match: score += 8
    if weekly_match: score += 5
    if in_ob: score += 5
    if idm and idm.get("wick_ratio", 0) > 0.45: score += 5
    if has_news: score -= 20
    return max(0, min(100, score))


def calc_entry_sl_tp(ob, direction, tf="1h"):
    ob_range = ob["top"] - ob["bottom"]
    sl_buffer = ob_range * 0.15
    swing_tf = tf in ["1h", "4h"]
    min_risk = ob_range * 1.5 if swing_tf else ob_range * 1.1
    if direction == "bullish":
        entry = round(ob["top"], 5)
        sl_raw = ob["bottom"] - sl_buffer
        if swing_tf and (entry - sl_raw) < min_risk:
            sl_raw = entry - min_risk
        sl = round(sl_raw, 5)
        risk = entry - sl
        tp1 = round(entry + risk * 2.0, 5)
        tp2 = round(entry + risk * 4.0, 5)
    else:
        entry = round(ob["bottom"], 5)
        sl_raw = ob["top"] + sl_buffer
        if swing_tf and (sl_raw - entry) < min_risk:
            sl_raw = entry + min_risk
        sl = round(sl_raw, 5)
        risk = sl - entry
        tp1 = round(entry - risk * 2.0, 5)
        tp2 = round(entry - risk * 4.0, 5)
    return entry, sl, tp1, tp2, 2.0, 4.0


def get_risk_advice(quality):
    dd_used = ACCOUNT["drawdown_used"]
    daily_used = ACCOUNT["daily_used"]
    max_dd = ACCOUNT["max_drawdown"]
    daily_dd = ACCOUNT["daily_drawdown"]
    remaining_max = max_dd - dd_used
    remaining_daily = daily_dd - daily_used
    phase = ACCOUNT["phase"]
    if remaining_max <= 1.5:
        return 0, "🚨 الدروداون حرج، لا تدخلين أي صفقة!"
    if remaining_daily <= 0.5:
        return 0, "⛔ وصلتِ الحد اليومي، استريحي اليوم"
    if phase == "challenge":
        max_risk = min(remaining_daily * 0.3, 1.0)
    elif phase == "verification":
        max_risk = min(remaining_daily * 0.35, 1.5)
    else:
        max_risk = min(remaining_daily * 0.4, 2.0)
    if quality >= 90:
        risk = min(max_risk, 1.5 if phase != "challenge" else 1.0)
        label = "ممتازة 🔥 تستاهل المخاطرة"
    elif quality >= 80:
        risk = min(max_risk, 1.0)
        label = "قوية 💪 مخاطرة عادية"
    elif quality >= 70:
        risk = min(max_risk, 0.75)
        label = "كويسة 👍 خففي الحجم شوي"
    elif quality >= 60:
        risk = min(max_risk, 0.5)
        label = "مقبولة، خففي المخاطرة 🤏"
    else:
        return 0, "ضعيفة، ما ندخل ❌"
    if remaining_max < 4:
        label += f"\n⚠️ باقي {remaining_max:.1f}% دروداون، اضغطي على الكوالتي"
    return round(risk, 2), label


def analyze(sym_name, yf_sym, tf, news, debug=False):
    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 50:
        if debug: return f"{sym_name} {tf}: ❌ بيانات فاضية"
        return None
    trend = detect_trend_structure(df)
    if trend == "neutral":
        if debug: return f"{sym_name} {tf}: ❌ ترند محايد"
        return None
    df_h4 = get_candles(yf_sym, "4h", 30)
    h4_trend = detect_trend_structure(df_h4) if not df_h4.empty else "neutral"
    h4_of = detect_order_flow(df_h4, trend) if not df_h4.empty else 0.0
    if h4_trend != "neutral" and h4_trend != trend:
        if debug: return f"{sym_name} {tf}: ❌ H4 عكس الاتجاه"
        return None
    if h4_of < 0.6:
        if debug: return f"{sym_name} {tf}: ❌ H4 order flow ضعيف ({h4_of})"
        return None
    df_h1 = get_candles(yf_sym, "1h", 20)
    h1_of = detect_order_flow(df_h1, trend) if not df_h1.empty else 0.0
    dbos = detect_dbos(df, trend)
    if not dbos:
        if debug: return f"{sym_name} {tf}: ❌ ما في DBOS"
        return None
    idm = find_idm(df, dbos["index"], trend)
    if not idm:
        if debug: return f"{sym_name} {tf}: ❌ ما في IDM (DBOS عند {dbos['index']})"
        return None
    ob = find_ob(df, idm["index"], trend)
    if not ob:
        if debug: return f"{sym_name} {tf}: ❌ ما في OB"
        return None
    current = df["close"].iloc[-1]
    direction = trend
    ob_range = ob["top"] - ob["bottom"]
    if direction == "bullish":
        if current < ob["bottom"] - ob_range:
            if debug: return f"{sym_name} {tf}: ❌ فات الـ OB (السعر تحته)"
            return None
        if current > ob["top"] + ob_range * 2:
            if debug: return f"{sym_name} {tf}: ⏳ السعر بعيد عن OB - انتظار"
            return None
    else:
        if current > ob["top"] + ob_range:
            if debug: return f"{sym_name} {tf}: ❌ فات الـ OB (السعر فوقه)"
            return None
        if current < ob["bottom"] - ob_range * 2:
            if debug: return f"{sym_name} {tf}: ⏳ السعر بعيد عن OB - انتظار"
            return None
    ob_size = ob["top"] - ob["bottom"]
    min_ob_size = 0.0
    if "BTC" in sym_name or "ETH" in sym_name:
        min_ob_size = 300.0
    elif "XAU" in sym_name:
        min_ob_size = 3.0
    elif "XAG" in sym_name:
        min_ob_size = 0.05
    else:
        min_ob_size = 0.0020
    if ob_size < min_ob_size:
        if debug: return f"{sym_name} {tf}: ❌ OB صغير جداً ({round(ob_size, 4)} < {min_ob_size})"
        return None
    in_ob = ob["bottom"] <= current <= ob["top"]
    has_liquidity, liq_level = check_liquidity_above(df, trend)
    sweep = False
    ob_sweep = False
    df_d = get_candles(yf_sym, "1d", 50)
    daily_trend = detect_trend_structure(df_d) if not df_d.empty else "neutral"
    daily_match = daily_trend == trend
    df_w = get_candles(yf_sym, "1wk", 20)
    weekly_trend = detect_trend_structure(df_w) if not df_w.empty else "neutral"
    weekly_match = weekly_trend == trend
    quality = calc_quality_weighted(
        dbos, idm, ob, h4_of, h1_of, has_liquidity,
        daily_match, weekly_match, in_ob, news["has_news"],
        idm_wick=idm.get("wick_ratio", 0) if idm else 0,
        ob_body=ob.get("body_ratio", 0) if ob else 0
    )
    if quality < 65:
        if debug: return f"{sym_name} {tf}: ❌ جودة منخفضة {quality}%"
        return None
    ob_age = len(df) - ob.get("index", 0)
    if ob_age > 60:
        if debug: return f"{sym_name} {tf}: ❌ OB قديم ({ob_age} شمعة)"
        return None
    idm_age = len(df) - idm["index"]
    if idm_age > 40:
        if debug: return f"{sym_name} {tf}: ❌ IDM قديم ({idm_age} شمعة)"
        return None
    entry, sl, tp1, tp2, rr1, rr2 = calc_entry_sl_tp(ob, trend, tf)
    return {
        "symbol": sym_name, "tf": tf, "trend": trend,
        "current": current, "ob": ob, "in_ob": in_ob,
        "sweep": sweep, "ob_sweep": ob_sweep,
        "h4_of": h4_of, "h1_of": h1_of,
        "has_liquidity": has_liquidity, "liq_level": liq_level,
        "daily_match": daily_match, "daily_trend": daily_trend,
        "weekly_match": weekly_match, "weekly_trend": weekly_trend,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": rr1, "rr2": rr2, "quality": quality, "news": news,
        "idm_type": idm.get("type", ""), "idm_wick": idm.get("wick_ratio", 0),
    }

def setup_msg(a):
    direction = "شراء 📈" if a["trend"] == "bullish" else "بيع 📉"
    arrow = "🟢" if a["trend"] == "bullish" else "🔴"
    risk, label = get_risk_advice(a["quality"])
    d_icon = "✅" if a["daily_match"] else "❌"
    w_icon = "✅" if a["weekly_match"] else "⚠️"
    d_txt = {"bullish": "صاعد", "bearish": "هابط"}.get(a["daily_trend"], "محايد")
    w_txt = {"bullish": "صاعد", "bearish": "هابط"}.get(a["weekly_trend"], "محايد")
    quality_bar = "█" * (a["quality"] // 20) + "░" * (5 - a["quality"] // 20)
    extras = []
    if a.get("ob_sweep"):
        extras.append("⚡ OB يسحب سيولة = قوي جداً")
    if a["sweep"]:
        extras.append("✅ سحب سيولة")
    if a["daily_match"] and a["weekly_match"]:
        extras.append("✅ توافق كامل")
    elif a["daily_match"]:
        extras.append("✅ اليومي يدعم")
    news_txt = ""
    if a["news"]["has_news"]:
        news_txt = "⚠️ أخبار مهمة قريبة!\n"
        for ev in a["news"]["events"]:
            news_txt += f"  • {ev['title']} بعد {ev['hours']}س\n"
    risk_txt = f"❌ ما ندخل - {label}" if risk == 0 else f"💰 مخاطرة: {risk}% - {label}"
    tv = TRADINGVIEW_LINKS.get(a["symbol"], "https://www.tradingview.com")
    if a["in_ob"]:
        action_header = f"⚡ وصل الـ OB - ادخلي الحين!"
        order_type = "دخول فوري (Market)"
    else:
        action_header = f"⏳ ما وصل بعد - حطي ليمت أوردر"
        order_type = f"ليمت أوردر عند: {a['entry']}"
    msg = f"🔵 {arrow} {direction} | {a['symbol']} | {a['tf']}\n"
    msg += "─────────────────\n"
    msg += f"{w_icon} أسبوعي: {w_txt}  {d_icon} يومي: {d_txt}\n"
    if extras:
        msg += "  ".join(extras) + "\n"
    msg += news_txt
    msg += "─────────────────\n"
    msg += f"{action_header}\n"
    msg += f"📌 {order_type}\n"
    msg += f"🛑 ستوب:   {a['sl']}  (تحت الـ OB)\n"
    msg += f"✅ هدف 1:  {a['tp1']}  (1:2)\n"
    msg += f"🚀 هدف 2:  {a['tp2']}  (1:4)\n"
    msg += f"السعر الحالي: {round(a['current'], 4)}\n"
    msg += f"منطقة OB: {round(a['ob']['bottom'],4)} - {round(a['ob']['top'],4)}\n"
    msg += "─────────────────\n"
    msg += f"جودة: {a['quality']}/100  {quality_bar}\n"
    msg += f"{risk_txt}\n"
    msg += f"📈 {tv}\n"
    msg += "القرار إلك يا شذا 💪"
    return msg


def challenge_progress_msg():
    phase = ACCOUNT["phase"]
    pnl = ACCOUNT["pnl_percent"]
    target = PHASE_TARGETS.get(phase, {}).get("target", 0)
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    if target:
        progress = max(0, min(100, round(pnl / target * 100)))
        bar = "█" * (progress // 20) + "░" * (5 - progress // 20)
        target_txt = f"الهدف: {target}% | وصلت: {pnl}%\n{bar} {progress}%"
    else:
        target_txt = f"حساب ممول | ربح: {pnl}%"
    phase_label = {"challenge": "Challenge", "verification": "Verification", "funded": "Funded"}.get(phase, "")
    msg = f"📊 {phase_label} Progress\n"
    msg += "─────────────────\n"
    msg += f"{target_txt}\n"
    msg += f"دروداون باقي: {remaining_max:.1f}%\n"
    if target and pnl >= target:
        msg += "✅ حققتِ الهدف! انتقلي للمرحلة التالية"
    elif remaining_max < 3:
        msg += "⚠️ دروداون ضيق، تعاملي بحذر"
    else:
        msg += "واصلي يا شذا 💪"
    return msg


def daily_advice_msg():
    dd = ACCOUNT["drawdown_used"]
    remaining_max = ACCOUNT["max_drawdown"] - dd
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    pnl = ACCOUNT["pnl_percent"]
    trades = ACCOUNT["trades_week"]
    phase_txt = {"challenge": "🔴 چالنج", "verification": "🟡 تحقق", "funded": "🟢 ممول"}.get(ACCOUNT["phase"], "")
    if pnl > 3:
        pnl_txt = f"رابح {pnl}%، واصلي 🌟"
    elif pnl > 0:
        pnl_txt = f"رابح {pnl}%، شغل كويس 👍"
    elif pnl == 0:
        pnl_txt = "عند نقطة البداية 🎯"
    elif pnl >= -3:
        pnl_txt = f"خسارة {abs(pnl)}%، خففي الحجم ⚠️"
    else:
        pnl_txt = f"خسارة {abs(pnl)}%، حمي الحساب ❗"
    if remaining_max >= 7:
        dd_txt = f"باقي {remaining_max:.1f}% الحمدلله ✅"
    elif remaining_max >= 4:
        dd_txt = f"باقي {remaining_max:.1f}% - تعاملي بحذر 🟡"
    else:
        dd_txt = f"باقي {remaining_max:.1f}% فقط! 🔴"
    if remaining_daily >= 3:
        daily_txt = f"باقي {remaining_daily:.1f}% يومي ✅"
    elif remaining_daily >= 1:
        daily_txt = f"باقي {remaining_daily:.1f}% يومي ⚠️"
    else:
        daily_txt = "وصلتِ الحد اليومي 🛑"
    trades_txt = (
        "ما دخلتِ صفقات، الصبر ذهب 💎" if trades == 0
        else f"{trades} صفقة، ممتاز 👏" if trades <= 2
        else f"{trades} صفقات، شوي كثير 🤔"
    )
    msg = f"صباح الخير يا شذا ☀️\n"
    msg += f"─────────────────\n"
    msg += f"{ACCOUNT['firm_name']} | {phase_txt}\n"
    msg += f"💰 الحساب: ${ACCOUNT['current_balance']:,.0f}\n"
    msg += f"─────────────────\n"
    msg += f"الحساب: {pnl_txt}\n"
    msg += f"دروداون كلي: {dd_txt}\n"
    msg += f"دروداون يومي: {daily_txt}\n"
    msg += f"الصفقات: {trades_txt}\n"
    msg += f"─────────────────\n"
    msg += f"{random.choice(DAILY_TIPS)}\n"
    msg += "وفقك الله 🤍"
    return msg


def status_msg():
    now = datetime.now(RIYADH_TZ)
    pnl = ACCOUNT["pnl_percent"]
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    icon = "🟢" if pnl >= 0 and remaining_max > 5 else "🟡" if remaining_max > 2 else "🔴"
    msg = f"{icon} حالة الحساب | {now.strftime('%H:%M')} الرياض\n"
    msg += f"─────────────────\n"
    msg += f"الحساب: {'+' if pnl >= 0 else ''}{pnl}%\n"
    msg += f"دروداون كلي: {ACCOUNT['drawdown_used']}% (باقي {remaining_max:.1f}%)\n"
    msg += f"دروداون يومي: {ACCOUNT['daily_used']}% (باقي {remaining_daily:.1f}%)\n"
    msg += f"صفقات اليوم: {ACCOUNT['trades_today']} | الأسبوع: {ACCOUNT['trades_week']}"
    return msg


# ===== التحديث التفاعلي =====
async def update_start(update, context):
    await update.message.reply_text(
        "يلا نحدث حسابك يا شذا 📋\n\n"
        "كم الرصيد الحالي بالدولار؟\n"
        "مثال: 10000\n"
        "(أو /skip)"
    )
    return S_BALANCE

async def got_balance(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text.replace(",", "").replace("$", ""))
            ACCOUNT["current_balance"] = val
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_BALANCE
    await update.message.reply_text(
        "كم نسبة الربح أو الخسارة الكلية؟\n"
        "مثال: +3.5 أو -2.0\n"
        "(أو /skip)"
    )
    return S_PNL

async def got_pnl(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text.replace("+", "").replace("%", ""))
            ACCOUNT["pnl_percent"] = val
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_PNL
    await update.message.reply_text(
        "كم الدروداون الكلي المستخدم حتى الحين؟\n"
        "مثال: 2.5\n"
        "(أو /skip)"
    )
    return S_DD

async def got_dd(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text.replace("%", ""))
            ACCOUNT["drawdown_used"] = val
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_DD
    await update.message.reply_text(
        "كم الدروداون اليومي المستخدم اليوم؟\n"
        "مثال: 1.0\n"
        "(أو /skip)"
    )
    return S_DAILY

async def got_daily(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text.replace("%", ""))
            ACCOUNT["daily_used"] = val
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_DAILY
    await update.message.reply_text(
        "كم صفقة دخلتِ هاالأسبوع؟\n"
        "مثال: 2\n"
        "(أو /skip)"
    )
    return S_TRADES_W

async def got_trades_w(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = int(text)
            ACCOUNT["trades_week"] = val
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_TRADES_W
    await update.message.reply_text(
        "كم صفقة اليوم؟\n"
        "مثال: 1\n"
        "(أو /skip)"
    )
    return S_TRADES_D

async def got_trades_d(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = int(text)
            ACCOUNT["trades_today"] = val
        except:
            pass
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    msg = "✅ تم التحديث!\n"
    msg += f"─────────────────\n"
    msg += f"💰 الرصيد: ${ACCOUNT['current_balance']:,.0f}\n"
    msg += f"📊 PnL: {'+' if ACCOUNT['pnl_percent'] >= 0 else ''}{ACCOUNT['pnl_percent']}%\n"
    msg += f"📉 دروداون كلي: {ACCOUNT['drawdown_used']}% (باقي {remaining_max:.1f}%)\n"
    msg += f"📅 دروداون يومي: {ACCOUNT['daily_used']}% (باقي {remaining_daily:.1f}%)\n"
    msg += f"🔢 صفقات الأسبوع: {ACCOUNT['trades_week']}\n"
    msg += f"📌 صفقات اليوم: {ACCOUNT['trades_today']}\n"
    msg += "\nبوتك يحلل بناء على بياناتك الجديدة 💪"
    await update.message.reply_text(msg)
    return ConversationHandler.END

async def cancel_update(update, context):
    await update.message.reply_text("إلغاء التحديث ❌")
    return ConversationHandler.END


# ===== جورنال - إرسال سيتاب مع أزرار =====
async def send_setup_with_buttons(bot, a, custom_msg=None):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    global TRADE_COUNTER
    TRADE_COUNTER[0] += 1
    trade_id = str(TRADE_COUNTER[0])

    JOURNAL[trade_id] = {
        "symbol": a["symbol"],
        "tf": a["tf"],
        "direction": a["trend"],
        "entry": a["entry"],
        "sl": a["sl"],
        "tp1": a["tp1"],
        "tp2": a["tp2"],
        "yf_sym": SYMBOLS.get(a["symbol"], ""),
        "risk": 0,
        "status": "pending",
        "result_r": None,
        "timestamp": datetime.now(RIYADH_TZ).strftime("%Y-%m-%d %H:%M"),
        "quality": a.get("quality", 0),
        "h4_of": a.get("h4_of", 0),
        "daily_match": a.get("daily_match", False),
        "weekly_match": a.get("weekly_match", False),
    }
    # حفظ في Google Sheets
    journal_add(trade_id, JOURNAL[trade_id])

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ دخلت", callback_data=f"entered_{trade_id}"),
            InlineKeyboardButton("❌ ما دخلت", callback_data=f"skipped_{trade_id}"),
        ]
    ])
    msg_text = custom_msg if custom_msg else setup_msg(a)
    await bot.send_message(chat_id=CHAT_ID, text=msg_text, reply_markup=keyboard)


async def handle_callback(update, context):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("entered_"):
        trade_id = data.split("_")[1]
        if trade_id not in JOURNAL:
            await query.edit_message_reply_markup(reply_markup=None)
            return
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("0.5%", callback_data=f"risk_{trade_id}_0.5"),
                InlineKeyboardButton("1%",   callback_data=f"risk_{trade_id}_1.0"),
                InlineKeyboardButton("1.5%", callback_data=f"risk_{trade_id}_1.5"),
            ]
        ])
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await context.bot.send_message(chat_id=CHAT_ID, text="✅ دخلتِ الصفقة! كم المخاطرة؟")

    elif data.startswith("risk_"):
        parts = data.split("_")
        trade_id = parts[1]
        risk = float(parts[2])
        if trade_id in JOURNAL:
            JOURNAL[trade_id]["risk"] = risk
            JOURNAL[trade_id]["status"] = "active"
            ACCOUNT["trades_week"] += 1
            ACCOUNT["trades_today"] += 1
            journal_set_status(trade_id, "active", risk)  # Google Sheets
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"📌 مسجلة! {JOURNAL[trade_id]['symbol']} | مخاطرة: {risk}%\n"
                    "أراقبها وأخبرك لما تصل الهدف أو الستوب 👀"
                )
            )

    elif data.startswith("skipped_"):
        trade_id = data.split("_")[1]
        if trade_id in JOURNAL:
            JOURNAL[trade_id]["status"] = "skipped"
            journal_set_status(trade_id, "skipped")  # Google Sheets
        await query.edit_message_reply_markup(reply_markup=None)

    elif data.startswith("result_"):
        parts = data.split("_")
        trade_id = parts[1]
        result = parts[2]
        if trade_id in JOURNAL:
            t = JOURNAL[trade_id]
            if result == "tp1":
                t["result_r"] = 2.0
                t["status"] = "closed"
                DAILY_RISK["consecutive_losses"] = 0
                update_weights_entered(t.get("analysis", {}), reached_tp=True)
                journal_set_result(trade_id, 2.0)  # Google Sheets
                msg = f"✅ هدف 1 وصل! +2R على {t['symbol']} 🎯"
            elif result == "tp2":
                t["result_r"] = 4.0
                t["status"] = "closed"
                DAILY_RISK["consecutive_losses"] = 0
                update_weights_entered(t.get("analysis", {}), reached_tp=True)
                journal_set_result(trade_id, 4.0)  # Google Sheets
                msg = f"🚀 هدف 2 وصل! +4R على {t['symbol']} 🔥"
            else:
                t["result_r"] = -1.0
                t["status"] = "closed"
                risk_used = t.get("risk", 1.0)
                DAILY_RISK["daily_loss_pct"] += risk_used
                DAILY_RISK["consecutive_losses"] += 1
                journal_set_result(trade_id, -1.0)  # Google Sheets
                if DAILY_RISK["consecutive_losses"] >= 2:
                    DAILY_RISK["trading_stopped"] = True
                    DAILY_RISK["stop_reason"] = "ستوبين متتاليين"
                elif DAILY_RISK["daily_loss_pct"] >= 2.0:
                    DAILY_RISK["trading_stopped"] = True
                    DAILY_RISK["stop_reason"] = f"خسارة {DAILY_RISK['daily_loss_pct']:.1f}% اليوم"
                if DAILY_RISK["trading_stopped"]:
                    stop_msg = f"🛑 Daily Risk Breaker! السبب: {DAILY_RISK['stop_reason']}\nما في إشارات لباقي اليوم 💪\nبكرة تعود تلقائياً"
                    await context.bot.send_message(chat_id=CHAT_ID, text=stop_msg)
                msg = f"🔴 ستوب على {t['symbol']} | -1R - كل صفقة خاسرة درس، واصلي 💪"
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(chat_id=CHAT_ID, text=msg)


# ===== مراقبة الصفقات النشطة =====
async def monitor_trades(bot):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    active = {k: v for k, v in JOURNAL.items() if v["status"] == "active"}
    for trade_id, t in active.items():
        try:
            yf_sym = t["yf_sym"]
            if not yf_sym:
                continue
            df = get_candles(yf_sym, "1h", 5)
            if df.empty:
                continue
            current = df["close"].iloc[-1]
            direction = t["direction"]
            hit_tp2 = (direction == "bullish" and current >= t["tp2"]) or (direction == "bearish" and current <= t["tp2"])
            hit_tp1 = (direction == "bullish" and current >= t["tp1"]) or (direction == "bearish" and current <= t["tp1"])
            hit_sl  = (direction == "bullish" and current <= t["sl"])  or (direction == "bearish" and current >= t["sl"])
            if hit_tp2:
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("✅ أكدي TP2", callback_data=f"result_{trade_id}_tp2")]])
                await bot.send_message(chat_id=CHAT_ID, text=f"🚀 يبدو وصل هدف 2 على {t['symbol']}! أكدي:", reply_markup=keyboard)
            elif hit_tp1:
                keyboard = InlineKeyboardMarkup([[
                    InlineKeyboardButton("✅ TP1", callback_data=f"result_{trade_id}_tp1"),
                    InlineKeyboardButton("🚀 TP2", callback_data=f"result_{trade_id}_tp2"),
                ]])
                await bot.send_message(chat_id=CHAT_ID, text=f"✅ يبدو وصل هدف 1 على {t['symbol']}! وين أغلقتِ؟", reply_markup=keyboard)
            elif hit_sl:
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔴 أكدي الستوب", callback_data=f"result_{trade_id}_sl")]])
                await bot.send_message(chat_id=CHAT_ID, text=f"⚠️ يبدو لمس الستوب على {t['symbol']}! أكدي:", reply_markup=keyboard)
        except Exception as e:
            logger.error(f"خطأ مراقبة صفقة {trade_id}: {e}")


# ===== تقرير الأسبوع =====
def weekly_report_msg():
    closed = [t for t in JOURNAL.values() if t["status"] == "closed"]
    skipped = [t for t in JOURNAL.values() if t["status"] == "skipped"]
    active = [t for t in JOURNAL.values() if t["status"] == "active"]
    if not closed and not active:
        return "ما في صفقات مسجلة هالأسبوع يا شذا 📋\nبداية الأسبوع الجاي إن شاء الله 💪"
    wins = [t for t in closed if t["result_r"] and t["result_r"] > 0]
    losses = [t for t in closed if t["result_r"] and t["result_r"] < 0]
    win_rate = round(len(wins) / len(closed) * 100) if closed else 0
    total_r_clean = round(sum(t["result_r"] for t in closed if t["result_r"]), 1)
    msg = "📊 تقرير الأسبوع يا شذا\n"
    msg += "─────────────────\n"
    msg += f"إجمالي الصفقات: {len(closed)}\n"
    msg += f"✅ رابحة: {len(wins)} | 🔴 خاسرة: {len(losses)}\n"
    msg += f"📈 نسبة الفوز: {win_rate}%\n"
    msg += f"💰 مجموع الـ R: {'+' if total_r_clean >= 0 else ''}{total_r_clean}R\n"
    if skipped:
        msg += f"⏭ تجاهلتِ: {len(skipped)} صفقة\n"
    if active:
        msg += f"⏳ لا تزال مفتوحة: {len(active)}\n"
    msg += "─────────────────\n"
    if closed:
        msg += "تفاصيل:\n"
        for t in closed:
            icon = "✅" if t["result_r"] and t["result_r"] > 0 else "🔴"
            r_txt = f"+{t['result_r']}R" if t["result_r"] and t["result_r"] > 0 else f"{t['result_r']}R"
            msg += f"{icon} {t['symbol']} {t['tf']} → {r_txt}\n"
    msg += "─────────────────\n"
    if total_r_clean >= 4:
        msg += "أسبوع ممتاز، واصلي بنفس المنهج 🌟"
    elif total_r_clean >= 0:
        msg += "أسبوع كويس، استمري 💪"
    else:
        msg += "أسبوع صعب، راجعي الجورنال وشوفي وين الخلل 🧠"
    stats_add_week(JOURNAL)  # Google Sheets
    JOURNAL.clear()
    return msg


def is_dd_safe():
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    return remaining_max > 1.5 and remaining_daily > 0.5


# ============================================================
# ===== استراتيجية Morning Star OB =====
# ============================================================

def detect_morning_star(df, direction="bullish"):
    if len(df) < 10:
        return None
    search_start = max(3, len(df) - 20)
    for i in range(search_start, len(df) - 2):
        c1 = df.iloc[i]
        c2 = df.iloc[i+1]
        c3 = df.iloc[i+2]
        r1 = c1["high"] - c1["low"]
        r3 = c3["high"] - c3["low"]
        if r1 == 0 or r3 == 0:
            continue
        body1 = abs(c1["close"] - c1["open"])
        body2 = abs(c2["close"] - c2["open"])
        body3 = abs(c3["close"] - c3["open"])
        if direction == "bullish":
            if c1["close"] >= c1["open"]: continue
            if body1 / r1 < 0.50: continue
            prev_low = df["low"].iloc[max(0,i-10):i].min()
            if c1["low"] >= prev_low: continue
            if body2 > body1 * 0.30: continue
            if c3["close"] <= c3["open"]: continue
            midpoint_c1 = (c1["open"] + c1["close"]) / 2
            if c3["close"] <= midpoint_c1: continue
            bsl = df["high"].iloc[max(0,i-20):i+3].max()
            current = df["close"].iloc[-1]
            distance_to_bsl = (bsl - current) / current * 100
            has_bsl = 0.3 < distance_to_bsl < 5.0
            ob = {"top": c1["open"], "bottom": c1["close"], "index": i}
            return {
                "pattern_idx": i, "c1_idx": i, "c2_idx": i+1, "c3_idx": i+2,
                "ob": ob, "entry": round(c3["close"], 5),
                "sl": round(c1["low"] - (c1["high"] - c1["low"]) * 0.05, 5),
                "bsl": round(bsl, 5), "has_bsl": has_bsl,
                "liq_pool": round(prev_low, 5),
            }
        else:
            if c1["close"] <= c1["open"]: continue
            if body1 / r1 < 0.50: continue
            prev_high = df["high"].iloc[max(0,i-10):i].max()
            if c1["high"] <= prev_high: continue
            if body2 > body1 * 0.30: continue
            if c3["close"] >= c3["open"]: continue
            midpoint_c1 = (c1["open"] + c1["close"]) / 2
            if c3["close"] >= midpoint_c1: continue
            ssl = df["low"].iloc[max(0,i-20):i+3].min()
            current = df["close"].iloc[-1]
            distance_to_ssl = (current - ssl) / current * 100
            has_ssl = 0.3 < distance_to_ssl < 5.0
            ob = {"top": c1["close"], "bottom": c1["open"], "index": i}
            return {
                "pattern_idx": i, "c1_idx": i, "c2_idx": i+1, "c3_idx": i+2,
                "ob": ob, "entry": round(c3["close"], 5),
                "sl": round(c1["high"] + (c1["high"] - c1["low"]) * 0.05, 5),
                "bsl": round(ssl, 5), "has_bsl": has_ssl,
                "liq_pool": round(prev_high, 5),
            }
    return None


def analyze_morning_star(sym_name, yf_sym, tf, news, debug=False):
    if tf not in ["1h", "4h"]:
        return None
    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 30:
        if debug: return f"🌟 {sym_name} {tf}: بيانات فاضية"
        return None
    df_h4 = get_candles(yf_sym, "4h", 30)
    h4_trend = detect_trend_structure(df_h4) if not df_h4.empty else "neutral"
    if h4_trend == "neutral":
        if debug: return f"🌟 {sym_name} {tf}: H4 محايد"
        return None
    pattern = detect_morning_star(df, h4_trend)
    if not pattern:
        if debug: return f"🌟 {sym_name} {tf}: ما في Morning Star"
        return None
    if pattern["c3_idx"] < len(df) - 3:
        if debug: return f"🌟 {sym_name} {tf}: Pattern قديم"
        return None
    current = df["close"].iloc[-1]
    entry = pattern["entry"]
    sl = pattern["sl"]
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    tp1 = round(entry + risk * 2.0, 5) if h4_trend == "bullish" else round(entry - risk * 2.0, 5)
    tp2 = round(entry + risk * 4.0, 5) if h4_trend == "bullish" else round(entry - risk * 4.0, 5)
    quality = 70
    if pattern["has_bsl"]: quality += 15
    if news["has_news"]: quality -= 20
    quality = max(0, min(100, quality))
    if quality < 65:
        if debug: return f"🌟 {sym_name} {tf}: جودة منخفضة {quality}%"
        return None
    return {
        "strategy": "morning_star",
        "symbol": sym_name, "tf": tf, "trend": h4_trend,
        "current": current, "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2, "rr1": 2.0, "rr2": 4.0,
        "quality": quality, "ob": pattern["ob"], "in_ob": True,
        "has_bsl": pattern["has_bsl"], "bsl": pattern["bsl"],
        "liq_pool": pattern["liq_pool"], "news": news,
        "daily_match": True, "weekly_match": False,
        "daily_trend": h4_trend, "weekly_trend": "neutral",
        "sweep": False, "ob_sweep": False,
        "h4_of": 0, "h1_of": 0,
        "has_liquidity": pattern["has_bsl"], "liq_level": pattern["bsl"],
        "idm_type": "", "idm_wick": 0,
    }


def morning_star_msg(a):
    direction = "شراء 📈" if a["trend"] == "bullish" else "بيع 📉"
    risk, label = get_risk_advice(a["quality"])
    risk_txt = f"❌ ما ندخل - {label}" if risk == 0 else f"💰 مخاطرة: {risk}% - {label}"
    tv = TRADINGVIEW_LINKS.get(a["symbol"], "https://www.tradingview.com")
    quality_bar = "█" * (a["quality"] // 20) + "░" * (5 - a["quality"] // 20)
    sep = "─" * 17
    lines = [
        f"🌟 Morning Star OB | {direction}", sep,
        f"📊 {a['symbol']} | {a['tf']}", sep,
        f"⚡ دخول فوري عند: {a['entry']}",
        f"🛑 ستوب: {a['sl']}  (تحت ذيل الشمعة الأولى)",
        f"✅ هدف 1: {a['tp1']}  (1:2)",
        f"🚀 هدف 2: {a['tp2']}  (1:4)",
        f"السعر الحالي: {round(a['current'], 4)}",
    ]
    if a.get("has_bsl"):
        lines.append(f"💧 BSL فوق عند: {a['bsl']}")
    lines.append(f"🔻 Liq Pool عند: {a['liq_pool']}")
    lines += [sep, f"جودة: {a['quality']}/100  {quality_bar}", risk_txt, f"📈 {tv}", "القرار إلك يا شذا 💪"]
    return "\n".join(lines)


async def scan_markets(bot):
    if not is_dd_safe():
        remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
        remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
        if remaining_max <= 1.5:
            await bot.send_message(chat_id=CHAT_ID, text=f"🛑 Max DD critical! Only {remaining_max:.1f}% left\nNo trades until you review your account\n/update to refresh")
        elif remaining_daily <= 0.5:
            await bot.send_message(chat_id=CHAT_ID, text=f"⛔ Daily DD limit reached! {remaining_daily:.1f}% left today\nRest for today, back tomorrow 💪")
        return False
    if DAILY_RISK["trading_stopped"]:
        return False
    news = check_news()
    found = []
    now_ts = datetime.now()
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            key = f"{name}_{tf}"
            last_sent = SENT_SETUPS.get(key)
            if last_sent:
                hours_ago = (now_ts - last_sent).total_seconds() / 3600
                if hours_ago < 4:
                    continue
            try:
                r = analyze(name, yf_sym, tf, news)
                if r:
                    found.append(r)
            except Exception as e:
                logger.error(f"خطأ {name} {tf}: {e}")
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            key = f"ms_{name}_{tf}"
            last_sent = SENT_SETUPS.get(key)
            if last_sent:
                hours_ago = (now_ts - last_sent).total_seconds() / 3600
                if hours_ago < 4:
                    continue
            try:
                ms = analyze_morning_star(name, yf_sym, tf, news)
                if ms and isinstance(ms, dict):
                    ms["_sent_key"] = key
                    found.append(ms)
                    logger.info(f"Morning Star found: {name} {tf} quality={ms.get('quality')}")
                else:
                    logger.info(f"Morning Star none: {name} {tf}")
            except Exception as e:
                logger.error(f"Morning Star خطأ {name} {tf}: {e}", exc_info=True)
    if found:
        found.sort(key=lambda x: x["quality"], reverse=True)
        for s in found:
            if s.get("strategy") == "morning_star":
                msg_text = morning_star_msg(s)
                await send_setup_with_buttons(bot, s, custom_msg=msg_text)
                SENT_SETUPS[s.get("_sent_key", f"ms_{s['symbol']}_{s['tf']}")] = datetime.now()
            else:
                await send_setup_with_buttons(bot, s)
                SENT_SETUPS[f"{s['symbol']}_{s['tf']}"] = datetime.now()
            await asyncio.sleep(2)
        return True
    return False


# ===== الحلقة الرئيسية =====
async def trading_loop(bot):
    load_data()
    phase_txt = {"challenge": "🔴 چالنج", "verification": "🟡 تحقق", "funded": "🟢 ممول"}.get(ACCOUNT["phase"], "")
    await bot.send_message(
        chat_id=CHAT_ID,
        text=(
            f"بوتك اشتغل يا شذا ✅\n"
            f"─────────────────\n"
            f"{ACCOUNT['firm_name']} | {phase_txt}\n"
            f"💰 ${ACCOUNT['balance']:,.0f} | دروداون: {ACCOUNT['max_drawdown']}% / {ACCOUNT['daily_drawdown']}% يومي\n"
            f"─────────────────\n"
            f"/scan فحص فوري\n"
            f"/advice نصايح اليوم\n"
            f"/status حالة الحساب\n"
            f"/update تحديث الحساب\n"
        )
    )
    last_advice_day = None
    last_scan_hour = -1
    while True:
        try:
            now = datetime.now(RIYADH_TZ)
            today = now.date()
            if now.hour == 8 and now.minute < 5 and last_advice_day != today:
                await bot.send_message(chat_id=CHAT_ID, text=daily_advice_msg())
                await bot.send_message(chat_id=CHAT_ID, text=challenge_progress_msg())
                ACCOUNT["daily_used"] = 0.0
                ACCOUNT["trades_today"] = 0
                if DAILY_RISK["trading_stopped"]:
                    DAILY_RISK["trading_stopped"] = False
                    DAILY_RISK["consecutive_losses"] = 0
                    DAILY_RISK["daily_loss_pct"] = 0.0
                    DAILY_RISK["stop_reason"] = ""
                    await bot.send_message(chat_id=CHAT_ID, text="✅ يوم جديد! الإشارات عادت - تداولي بحكمة 💪")
                save_data()
                last_advice_day = today
            if now.weekday() == 4 and now.hour == 20 and now.minute < 5:
                if not hasattr(trading_loop, 'last_report') or trading_loop.last_report != today:
                    await bot.send_message(chat_id=CHAT_ID, text=weekly_report_msg())
                    trading_loop.last_report = today
            if now.hour % 4 == 0 and now.hour != last_scan_hour and now.minute < 5:
                found = await scan_markets(bot)
                if not found:
                    await bot.send_message(chat_id=CHAT_ID, text=random.choice(NO_SETUP_MSGS))
                last_scan_hour = now.hour
            else:
                await scan_markets(bot)
            await monitor_trades(bot)
            await asyncio.sleep(1800)
        except Exception as e:
            logger.error(f"خطأ: {e}")
            await asyncio.sleep(60)


# ===== الأوامر =====
async def start_cmd(update, context):
    await update.message.reply_text(
        "يا هلا يا شذا! 🌟\n"
        "أنا بوتك، أراقب الأسواق 24/7\n\n"
        "/scan فحص فوري\n"
        "/advice نصايح اليوم\n"
        "/status حالة الحساب\n"
        "/progress Challenge progress\n"
        "/update تحديث الحساب\n"
        "/journal تقرير الجورنال\n"
    )

async def scan_cmd(update, context):
    await update.message.reply_text(random.choice(WAITING_MSGS))
    found = await scan_markets(context.bot)
    if not found:
        await update.message.reply_text(random.choice(NO_SETUP_MSGS))

async def debug_cmd(update, context):
    news = check_news()
    msg = "🔍 تشخيص كامل:\n─────────────────\n"
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                result = analyze(name, yf_sym, tf, news, debug=True)
                if isinstance(result, str):
                    msg += f"{result}\n"
                elif isinstance(result, dict):
                    msg += f"{name} {tf}: ✅ DBOS جودة {result['quality']}%\n"
                else:
                    msg += f"{name} {tf}: ❌ ما في سيتاب\n"
            except Exception as e:
                logger.error(f"debug error {name} {tf}: {e}")
                msg += f"{name} {tf}: ⚠️ {str(e)[:40]}\n"
    msg += "\n🌟 Morning Star:\n─────────────────\n"
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                result = analyze_morning_star(name, yf_sym, tf, news, debug=True)
                if isinstance(result, str):
                    msg += f"{result}\n"
                elif isinstance(result, dict):
                    msg += f"🌟 {name} {tf}: ✅ Morning Star جودة {result['quality']}%\n"
                else:
                    msg += f"🌟 {name} {tf}: ❌ ما في\n"
            except Exception as e:
                msg += f"🌟 {name} {tf}: ⚠️ {str(e)[:40]}\n"
    await update.message.reply_text(msg)

async def advice_cmd(update, context):
    await update.message.reply_text(daily_advice_msg())

async def status_cmd(update, context):
    await update.message.reply_text(status_msg())

async def progress_cmd(update, context):
    await update.message.reply_text(challenge_progress_msg())

async def journal_cmd(update, context):
    await update.message.reply_text(weekly_report_msg())


# ===== التشغيل =====
async def main():
    setup_sheets()  # إعداد Google Sheets عند أول تشغيل
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    update_conv = ConversationHandler(
        entry_points=[CommandHandler("update", update_start)],
        states={
            S_BALANCE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, got_balance)],
            S_PNL:      [MessageHandler(filters.TEXT & ~filters.COMMAND, got_pnl)],
            S_DD:       [MessageHandler(filters.TEXT & ~filters.COMMAND, got_dd)],
            S_DAILY:    [MessageHandler(filters.TEXT & ~filters.COMMAND, got_daily)],
            S_TRADES_W: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_trades_w)],
            S_TRADES_D: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_trades_d)],
        },
        fallbacks=[CommandHandler("cancel", cancel_update)],
    )
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("advice", advice_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("debug", debug_cmd))
    app.add_handler(CommandHandler("progress", progress_cmd))
    app.add_handler(CommandHandler("journal", journal_cmd))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(update_conv)
    bot = Bot(token=TELEGRAM_TOKEN)
    async with app:
        await app.start()
        await app.updater.start_polling()
        await trading_loop(bot)


if __name__ == "__main__":
    asyncio.run(main())
