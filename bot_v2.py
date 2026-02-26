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
    "phase": os.environ.get("ACCOUNT_PHASE", "challenge"),  # challenge / verification / funded
    "profit_split": float(os.environ.get("PROFIT_SPLIT", "20")),  # % شركة تاخذه
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

# ===== جورنال الصفقات =====
JOURNAL = {}  # { trade_id: {symbol, tf, entry, sl, tp1, tp2, direction, risk, status, result_r, timestamp} }
TRADE_COUNTER = [0]  # قائمة عشان نقدر نعدلها داخل الدوال

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


def detect_trend(df):
    if len(df) < 20:
        return "neutral"
    r = df.tail(20)
    if r["high"].iloc[-1] > r["high"].iloc[0] and r["low"].iloc[-1] > r["low"].iloc[0]:
        return "bullish"
    if r["high"].iloc[-1] < r["high"].iloc[0] and r["low"].iloc[-1] < r["low"].iloc[0]:
        return "bearish"
    return "neutral"


def find_swings(df, lb=5):
    """إيجاد قمم وقيعان واضحة - lb=5 عشان يكون أدق"""
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def detect_dbos(df, highs, lows, direction):
    """
    DBOS: ضلع واحد قوي يكسر مستويين مهمين
    الضلع الواحد = حركة قوية بدون تراجع > 30%
    يكسر قمتين (bullish) أو قاعين (bearish) متتاليتين
    """
    if direction == "bullish" and len(highs) >= 2:
        for i in range(len(highs) - 1, 0, -1):
            h2 = highs[i]    # القمة الأحدث
            h1 = highs[i-1]  # القمة الأقدم
            if h2[1] <= h1[1]:
                continue
            # الضلع الواحد: من h1 لـ h2 بدون تراجع كبير
            seg = df.iloc[h1[0]:h2[0]+1]
            if len(seg) < 2 or len(seg) > 50:
                continue
            move = h2[1] - df["low"].iloc[h1[0]:h2[0]+1].min()
            max_pullback = 0
            for k in range(1, len(seg)):
                pb = seg["high"].iloc[k-1] - seg["low"].iloc[k]
                if pb > max_pullback:
                    max_pullback = pb
            # تراجع لا يتجاوز 35% = ضلع واحد
            if move > 0 and max_pullback / move > 0.50:
                continue
            # تأكد الكسر واضح
            for j in range(h2[0], min(h2[0]+10, len(df))):
                if df["close"].iloc[j] > h1[1]:
                    return {"index": j, "price": h1[1], "sweep_level": df["low"].iloc[h1[0]:h2[0]+1].min()}
    elif direction == "bearish" and len(lows) >= 2:
        for i in range(len(lows) - 1, 0, -1):
            l2 = lows[i]
            l1 = lows[i-1]
            if l2[1] >= l1[1]:
                continue
            seg = df.iloc[l1[0]:l2[0]+1]
            if len(seg) < 2 or len(seg) > 50:
                continue
            move = df["high"].iloc[l1[0]:l2[0]+1].max() - l2[1]
            max_pullback = 0
            for k in range(1, len(seg)):
                pb = seg["high"].iloc[k] - seg["low"].iloc[k-1]
                if pb > max_pullback:
                    max_pullback = pb
            if move > 0 and max_pullback / move > 0.50:
                continue
            for j in range(l2[0], min(l2[0]+10, len(df))):
                if df["close"].iloc[j] < l1[1]:
                    return {"index": j, "price": l1[1], "sweep_level": df["high"].iloc[l1[0]:l2[0]+1].max()}
    return None


def find_idm(df, dbos_idx, direction):
    """
    IDM: أول بول باك بعد الضلع القوي
    لازم يكون شمعة سيولة واضحة = ذيل طويل أو شمعة سيولة كاملة
    - Bullish: ذيل سفلي طويل (> 40% من الشمعة) أو شمعة هابطة بجسم كبير
    - Bearish: ذيل علوي طويل (> 40% من الشمعة) أو شمعة صاعدة بجسم كبير
    مو مجرد شمعة صغيرة عشوائية
    """
    search_end = min(dbos_idx + 20, len(df))
    for i in range(dbos_idx + 1, search_end):
        c = df.iloc[i]
        candle_range = c["high"] - c["low"]
        if candle_range == 0:
            continue

        if direction == "bullish":
            lower_wick = min(c["open"], c["close"]) - c["low"]
            wick_ratio = lower_wick / candle_range
            body = abs(c["close"] - c["open"])
            body_ratio = body / candle_range

            # شمعة سيولة = ذيل سفلي طويل (> 40%) أو شمعة هابطة بجسم واضح (> 50%)
            is_liquidity_candle = wick_ratio > 0.4
            is_strong_bearish = c["close"] < c["open"] and body_ratio > 0.5

            if (is_liquidity_candle or is_strong_bearish) and c["low"] < df["low"].iloc[i - 1]:
                return {"index": i, "price": c["low"], "wick_ratio": round(wick_ratio, 2)}

        else:
            upper_wick = c["high"] - max(c["open"], c["close"])
            wick_ratio = upper_wick / candle_range
            body = abs(c["close"] - c["open"])
            body_ratio = body / candle_range

            is_liquidity_candle = wick_ratio > 0.4
            is_strong_bullish = c["close"] > c["open"] and body_ratio > 0.5

            if (is_liquidity_candle or is_strong_bullish) and c["high"] > df["high"].iloc[i - 1]:
                return {"index": i, "price": c["high"], "wick_ratio": round(wick_ratio, 2)}

    return None


def find_ob(df, idm_idx, direction):
    """
    OB: آخر شمعة عكسية أدت للحركة القوية مباشرة
    - نبحث أقرب شمعة عكسية للـ IDM والشمعة بعدها في نفس اتجاه الحركة
    - جسم واضح فوق 50%
    - لو ما لقينا، نوسع البحث بدون شرط الشمعة التالية
    """
    if idm_idx is None or idm_idx < 2:
        return None

    # بحث ضيق أولاً: 5 شمعات قبل IDM مع شرط الشمعة التالية
    for i in range(idm_idx - 1, max(idm_idx - 6, 0), -1):
        c = df.iloc[i]
        body = abs(c["close"] - c["open"])
        candle_range = c["high"] - c["low"]
        if candle_range == 0:
            continue
        if body / candle_range < 0.5:
            continue
        next_c = df.iloc[i + 1] if i + 1 < len(df) else None
        if direction == "bullish" and c["close"] < c["open"]:
            if next_c is not None and next_c["close"] > next_c["open"]:
                return {"top": c["open"], "bottom": c["close"], "index": i}
        elif direction == "bearish" and c["close"] > c["open"]:
            if next_c is not None and next_c["close"] < next_c["open"]:
                return {"top": c["close"], "bottom": c["open"], "index": i}

    # بحث موسع: 10 شمعات بدون شرط الشمعة التالية
    for i in range(idm_idx - 1, max(idm_idx - 11, 0), -1):
        c = df.iloc[i]
        body = abs(c["close"] - c["open"])
        candle_range = c["high"] - c["low"]
        if candle_range == 0:
            continue
        if body / candle_range < 0.4:
            continue
        if direction == "bullish" and c["close"] < c["open"]:
            return {"top": c["open"], "bottom": c["close"], "index": i}
        elif direction == "bearish" and c["close"] > c["open"]:
            return {"top": c["close"], "bottom": c["open"], "index": i}
    return None


def ob_sweeps_liquidity(df, ob, direction, highs, lows):
    """
    هل الـ OB فوق/تحت مستوى سيولة مهم؟ = OB أقوى
    """
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
    """
    سحب السيولة: السعر يخترق قمة/قاع سابقة ثم يرجع
    هذا يؤكد الاتجاه ويعطي قوة للسيتاب
    """
    if len(df) < 20:
        return False
    recent = df.tail(20)
    prev_high = recent["high"].iloc[:-3].max()
    prev_low = recent["low"].iloc[:-3].min()
    last2 = df.iloc[-3:-1]
    last_close = df["close"].iloc[-1]

    if direction == "bullish":
        # اخترق القاع ثم رجع فوقه
        swept = last2["low"].min() < prev_low
        recovered = last_close > prev_low
        return swept and recovered
    else:
        # اخترق القمة ثم رجع تحتها
        swept = last2["high"].max() > prev_high
        recovered = last_close < prev_high
        return swept and recovered


def is_price_in_ob(current, ob, buffer=0.2):
    """هل السعر داخل أو قريب من الـ OB؟"""
    ob_range = ob["top"] - ob["bottom"]
    extended_top = ob["top"] + ob_range * buffer
    extended_bottom = ob["bottom"] - ob_range * buffer
    return extended_bottom <= current <= extended_top


def calc_quality(dbos, idm, ob, sweep, weekly_match, daily_match, in_ob, ob_sweep, has_news):
    score = 0
    if dbos: score += 20         # كسر هيكل مزدوج - أساسي
    if idm: score += 20          # بول باك - أساسي
    if ob: score += 20           # أوردر بلوك - أساسي
    if ob_sweep: score += 15     # OB يسحب سيولة = أقوى ⚡
    if sweep: score += 10        # سحب سيولة عام
    if daily_match: score += 10  # توافق يومي
    if weekly_match: score += 5  # توافق أسبوعي
    if in_ob: score += 5         # السعر في المنطقة الحين
    # بونص IDM ذيل سيولة واضح
    if idm and idm.get("wick_ratio", 0) > 0.4: score += 5
    if has_news: score -= 20     # أخبار = خطر
    return max(0, min(100, score))


def calc_entry_sl_tp(ob, direction):
    """
    الدخول: أعلى الـ OB تماماً (bullish) أو أسفله تماماً (bearish)
    الستوب: تحت أسفل الـ OB بهامش 10% (bullish) أو فوق أعلاه (bearish)
    الأهداف: RR 1:2 و 1:4
    """
    ob_range = ob["top"] - ob["bottom"]
    sl_buffer = ob_range * 0.1  # 10% تحت/فوق الـ OB

    if direction == "bullish":
        entry = round(ob["top"], 5)               # دخول عند أعلى الـ OB
        sl = round(ob["bottom"] - sl_buffer, 5)   # ستوب تحت أسفل الـ OB
        risk = entry - sl
        tp1 = round(entry + risk * 2.0, 5)
        tp2 = round(entry + risk * 4.0, 5)
    else:
        entry = round(ob["bottom"], 5)             # دخول عند أسفل الـ OB
        sl = round(ob["top"] + sl_buffer, 5)       # ستوب فوق أعلى الـ OB
        risk = sl - entry
        tp1 = round(entry - risk * 2.0, 5)
        tp2 = round(entry - risk * 4.0, 5)

    return entry, sl, tp1, tp2, 2.0, 4.0


def get_risk_advice(quality):
    """نصيحة المخاطرة بناء على حالة الحساب والجودة"""
    dd_used = ACCOUNT["drawdown_used"]
    daily_used = ACCOUNT["daily_used"]
    max_dd = ACCOUNT["max_drawdown"]
    daily_dd = ACCOUNT["daily_drawdown"]
    remaining_max = max_dd - dd_used
    remaining_daily = daily_dd - daily_used
    phase = ACCOUNT["phase"]

    # فحص حدود الدروداون أولاً
    if remaining_max <= 1.5:
        return 0, "🚨 الدروداون حرج، لا تدخلين أي صفقة!"
    if remaining_daily <= 0.5:
        return 0, "⛔ وصلتِ الحد اليومي، استريحي اليوم"

    # حد المخاطرة حسب المرحلة
    if phase == "challenge":
        max_risk = min(remaining_daily * 0.3, 1.0)
    elif phase == "verification":
        max_risk = min(remaining_daily * 0.35, 1.5)
    else:
        max_risk = min(remaining_daily * 0.4, 2.0)

    # مخاطرة حسب الجودة
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

    # تحذير لو الحساب تحت ضغط
    if remaining_max < 4:
        label += f"\n⚠️ باقي {remaining_max:.1f}% دروداون، اضغطي على الكوالتي"

    return round(risk, 2), label


def analyze(sym_name, yf_sym, tf, news, debug=False):
    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 40:
        if debug: return f"{sym_name} {tf}: ❌ بيانات فاضية"
        return None

    trend = detect_trend(df)
    if trend == "neutral":
        if debug: return f"{sym_name} {tf}: ❌ ترند محايد"
        return None

    highs, lows = find_swings(df, lb=5)
    dbos = detect_dbos(df, highs, lows, trend)
    if not dbos:
        if debug: return f"{sym_name} {tf}: ❌ ما في DBOS"
        return None

    idm = find_idm(df, dbos["index"], trend)
    if not idm:
        if debug: return f"{sym_name} {tf}: ❌ ما في IDM (ترند: {trend}، DBOS عند شمعة {dbos['index']})"
        return None

    ob = find_ob(df, idm["index"], trend)
    if not ob:
        if debug: return f"{sym_name} {tf}: ❌ ما في OB (IDM عند {round(idm['price'],4)})"
        return None

    current = df["close"].iloc[-1]
    direction = trend  # alias عشان ما في لبس

    # الشرط الأساسي: السعر لازم يكون فوق الـ OB وقادم له (bullish)
    # أو تحت الـ OB وقادم له (bearish)
    # مو بعيد عنه بأكثر من 3x حجم الـ OB
    ob_range = ob["top"] - ob["bottom"]
    max_distance = ob_range * 15  # أقصى مسافة مقبولة

    if direction == "bullish":
        if current < ob["bottom"] - ob_range:
            if debug: return f"{sym_name} {tf}: ❌ السعر تحت OB (فات الفرصة)"
            return None
        if current > ob["top"] + max_distance:
            if debug: return f"{sym_name} {tf}: ❌ السعر بعيد جداً عن OB"
            return None
    else:
        if current > ob["top"] + ob_range:
            if debug: return f"{sym_name} {tf}: ❌ السعر فوق OB (فات الفرصة)"
            return None
        if current < ob["bottom"] - max_distance:
            if debug: return f"{sym_name} {tf}: ❌ السعر بعيد جداً عن OB"
            return None

    in_ob = is_price_in_ob(current, ob)
    sweep = check_liquidity_sweep(df, trend)
    ob_sweep = ob_sweeps_liquidity(df, ob, trend, highs, lows)

    # توافق الفريمات العليا
    df_d = get_candles(yf_sym, "1d", 50)
    daily_trend = detect_trend(df_d) if not df_d.empty else "neutral"
    daily_match = daily_trend == trend

    df_w = get_candles(yf_sym, "1wk", 20)
    weekly_trend = detect_trend(df_w) if not df_w.empty else "neutral"
    weekly_match = weekly_trend == trend

    quality = calc_quality(dbos, idm, ob, sweep, weekly_match, daily_match, in_ob, ob_sweep, news["has_news"])
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
    


    entry, sl, tp1, tp2, rr1, rr2 = calc_entry_sl_tp(ob, trend)

    return {
        "symbol": sym_name,
        "tf": tf,
        "trend": trend,
        "current": current,
        "ob": ob,
        "in_ob": in_ob,
        "sweep": sweep,
        "ob_sweep": ob_sweep,
        "daily_match": daily_match,
        "daily_trend": daily_trend,
        "weekly_match": weekly_match,
        "weekly_trend": weekly_trend,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": rr1,
        "rr2": rr2,
        "quality": quality,
        "news": news,
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
        # السعر وصل الـ OB - دخول فوري
        action_header = f"⚡ وصل الـ OB - ادخلي الحين!"
        order_type = "دخول فوري (Market)"
    else:
        # ما وصل بعد - ليمت أوردر
        action_header = f"⏳ ما وصل بعد - حطي ليمت أوردر"
        order_type = f"ليمت أوردر عند: {a['entry']}"

    msg = f"{arrow} {direction} | {a['symbol']} | {a['tf']}\n"
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


# ===== التحديث التفاعلي - محادثة خطوة خطوة =====

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
async def send_setup_with_buttons(bot, a):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    global TRADE_COUNTER
    TRADE_COUNTER[0] += 1
    trade_id = str(TRADE_COUNTER[0])

    # حفظ الصفقة في الجورنال بحالة "انتظار"
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
        "status": "pending",   # pending / active / closed
        "result_r": None,
        "timestamp": datetime.now(RIYADH_TZ).strftime("%Y-%m-%d %H:%M"),
    }

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ دخلت", callback_data=f"entered_{trade_id}"),
            InlineKeyboardButton("❌ ما دخلت", callback_data=f"skipped_{trade_id}"),
        ]
    ])
    await bot.send_message(chat_id=CHAT_ID, text=setup_msg(a), reply_markup=keyboard)


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
        # اسألها كم المخاطرة
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
        await query.edit_message_reply_markup(reply_markup=None)

    elif data.startswith("result_"):
        parts = data.split("_")
        trade_id = parts[1]
        result = parts[2]  # tp1 / tp2 / sl
        if trade_id in JOURNAL:
            t = JOURNAL[trade_id]
            if result == "tp1":
                t["result_r"] = 2.0
                t["status"] = "closed"
                DAILY_RISK["consecutive_losses"] = 0
                msg = f"✅ هدف 1 وصل! +2R على {t['symbol']} 🎯"
            elif result == "tp2":
                t["result_r"] = 4.0
                t["status"] = "closed"
                DAILY_RISK["consecutive_losses"] = 0
                msg = f"🚀 هدف 2 وصل! +4R على {t['symbol']} 🔥"
            else:
                t["result_r"] = -1.0
                t["status"] = "closed"
                risk_used = t.get("risk", 1.0)
                DAILY_RISK["daily_loss_pct"] += risk_used
                DAILY_RISK["consecutive_losses"] += 1
                # Daily Risk Breaker
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
    """يفحص كل ساعة وين وصلت الصفقات النشطة"""
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

            # فحص وصول الأهداف أو الستوب
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
    total_r = sum(t["result_r"] * t["risk"] / 1.0 for t in closed if t["result_r"])

    win_rate = round(len(wins) / len(closed) * 100) if closed else 0
    total_r_clean = round(sum(t["result_r"] for t in closed if t["result_r"]), 1)

    msg = "📊 تقرير الأسبوع يا شذا"
    msg += "─────────────────"
    msg += f"إجمالي الصفقات: {len(closed)}"
    msg += f"✅ رابحة: {len(wins)} | 🔴 خاسرة: {len(losses)}"
    msg += f"📈 نسبة الفوز: {win_rate}%"
    msg += f"💰 مجموع الـ R: {'+' if total_r_clean >= 0 else ''}{total_r_clean}R"
    if skipped:
        msg += f"⏭ تجاهلتِ: {len(skipped)} صفقة"
    if active:
        msg += f"⏳ لا تزال مفتوحة: {len(active)}"
    msg += "─────────────────"

    if closed:
        msg += "تفاصيل:"
        for t in closed:
            icon = "✅" if t["result_r"] and t["result_r"] > 0 else "🔴"
            r_txt = f"+{t['result_r']}R" if t["result_r"] and t["result_r"] > 0 else f"{t['result_r']}R"
            msg += f"{icon} {t['symbol']} {t['tf']} → {r_txt}"

    msg += "─────────────────"
    if total_r_clean >= 4:
        msg += "أسبوع ممتاز، واصلي بنفس المنهج 🌟"
    elif total_r_clean >= 0:
        msg += "أسبوع كويس، استمري 💪"
    else:
        msg += "أسبوع صعب، راجعي الجورنال وشوفي وين الخلل 🧠"

    # تصفير الجورنال للأسبوع الجديد
    JOURNAL.clear()
    return msg


# ===== الفحص =====
def is_dd_safe():
    """هل الحساب آمن للتداول؟"""
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    return remaining_max > 1.5 and remaining_daily > 0.5


async def scan_markets(bot):
    # حماية DD - لو اقتربنا من الحد نوقف
    if not is_dd_safe():
        remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
        remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
        if remaining_max <= 1.5:
            await bot.send_message(
                chat_id=CHAT_ID,
                text=f"🛑 Max DD critical! Only {remaining_max:.1f}% left\nNo trades until you review your account\n/update to refresh"
            )
        elif remaining_daily <= 0.5:
            await bot.send_message(
                chat_id=CHAT_ID,
                text=f"⛔ Daily DD limit reached! {remaining_daily:.1f}% left today\nRest for today, back tomorrow 💪"
            )
        return False

    # Daily Risk Breaker
    if DAILY_RISK["trading_stopped"]:
        return False

    news = check_news()
    found = []
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                r = analyze(name, yf_sym, tf, news)
                if r:
                    found.append(r)
                else:
                    # تشخيص مؤقت
                    dbg = analyze(name, yf_sym, tf, news, debug=True)
                    if isinstance(dbg, str) and "سيتاب" not in dbg:
                        logger.info(f"SCAN REJECT: {dbg}")
            except Exception as e:
                logger.error(f"خطأ {name} {tf}: {e}")
    if found:
        found.sort(key=lambda x: x["quality"], reverse=True)
        for s in found:
            await send_setup_with_buttons(bot, s)
            await asyncio.sleep(2)
        return True
    return False


# ===== الحلقة الرئيسية =====
async def trading_loop(bot):
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
                # Reset Daily Risk Breaker
                if DAILY_RISK["trading_stopped"]:
                    DAILY_RISK["trading_stopped"] = False
                    DAILY_RISK["consecutive_losses"] = 0
                    DAILY_RISK["daily_loss_pct"] = 0.0
                    DAILY_RISK["stop_reason"] = ""
                    await bot.send_message(chat_id=CHAT_ID, text="✅ يوم جديد! الإشارات عادت - تداولي بحكمة 💪")
                last_advice_day = today

            # تقرير الجمعة
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

            # مراقبة الصفقات النشطة
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
    """تشخيص - يرسل وش يصير مع كل زوج"""
    news = check_news()
    msg = "🔍 تشخيص كامل:\n─────────────────\n"
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                result = analyze(name, yf_sym, tf, news, debug=True)
                if isinstance(result, str):
                    msg += f"{result}\n"
                elif isinstance(result, dict):
                    msg += f"{name} {tf}: ✅ سيتاب جودة {result['quality']}%\n"
                else:
                    msg += f"{name} {tf}: ❌ ما في سيتاب\n"
            except Exception as e:
                logger.error(f"debug error {name} {tf}: {e}")
                msg += f"{name} {tf}: ⚠️ {str(e)[:40]}\n"
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
