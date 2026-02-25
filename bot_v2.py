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
    filters, ConversationHandler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")
RIYADH_TZ = pytz.timezone("Asia/Riyadh")

# ===== حساب البروب فيرم =====
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

HIGH_IMPACT_NEWS = [
    "Fed", "Federal Reserve", "FOMC", "Interest Rate",
    "CPI", "NFP", "Non-Farm", "GDP", "Powell", "ECB", "BOE", "BOJ",
    "Inflation", "Unemployment", "Retail Sales", "PPI"
]

# ===== رسايل البوت =====
WAITING_MSGS = [
    "عيني على الشارت، لحظة ⏳",
    "أفحص الأزواج واحد واحد 🔍",
    "ثانية وأخبرك وش شايف 👀",
]

NO_SETUP_MSGS = [
    "ما في dBOS واضح الحين يا شذا 🤷‍♀️\nروحي اتقهوي وأنا أراقب ☕",
    "السوق ما عطانا سيتاب بشروطنا 😌\nالصبر ذهب 💛",
    "فحصت كل شي، ما في ضلع واحد قوي الحين 🙅‍♀️\nأحسن من صفقة غلط",
]

DAILY_TIPS = [
    "ما في صفقة تستاهل تكسرين عشانها خطتك 💡",
    "الصفقة الصح تجيك، ما تروحين إليها ⏳",
    "المهم إدارة المخاطرة مو الربح السريع 🛡️",
    "أي ضغط داخل الصفقة؟ اطلعي منها 🧠",
    "الانضباط يفرق بين المحترف والمبتدئ 🏆",
    "كل صفقة في الجورنال، اللي ما يوثق ما يتعلم 📝",
    "الحساب أهم من أي صفقة 🌿",
    "dBOS نادر = لما يجي يستاهل 🎯",
]

(S_BALANCE, S_PNL, S_DD, S_DAILY, S_TRADES_W, S_TRADES_D) = range(6)


# ===== الأخبار =====
def check_news():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        if r.status_code != 200:
            return {"has_news": False, "events": [], "imminent": False}
        now = datetime.utcnow()
        upcoming = []
        imminent = False  # أخبار خلال 4 ساعات
        for ev in r.json():
            try:
                if ev.get("impact") != "High":
                    continue
                t = datetime.fromisoformat(ev.get("date", "").replace("Z", ""))
                diff = t - now
                if timedelta(hours=-1) <= diff <= timedelta(hours=24):
                    title = ev.get("title", "")
                    if any(k.lower() in title.lower() for k in HIGH_IMPACT_NEWS):
                        hours = round(diff.total_seconds() / 3600, 1)
                        upcoming.append({"title": title, "hours": hours})
                        if hours <= 4:
                            imminent = True
            except:
                continue
        return {"has_news": len(upcoming) > 0, "events": upcoming[:3], "imminent": imminent}
    except:
        return {"has_news": False, "events": [], "imminent": False}


# ===== البيانات =====
def get_candles(yf_sym, tf, limit=150):
    try:
        period = {"1h": "7d", "4h": "60d", "1d": "180d", "1wk": "2y"}.get(tf, "60d")
        df = yf.Ticker(yf_sym).history(period=period, interval=tf)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        return df.tail(limit)
    except:
        return pd.DataFrame()


def find_swing_points(df, lb=5):
    """إيجاد قمم وقيعان واضحة"""
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


# ===== الشرط 1: سحب السيولة بذيل شمعة =====
def detect_liquidity_sweep(df, highs, lows, direction, lookback=30):
    """
    سحب السيولة: ذيل شمعة يخترق قمة/قاع سابقة ثم السعر يرجع
    - Bullish: ذيل تحت يخترق قاع سابق ثم يرجع فوقه (سحب سيولة تحتية)
    - Bearish: ذيل فوق يخترق قمة سابقة ثم يرجع تحتها (سحب سيولة علوية)
    يرجع: index الشمعة اللي سحبت السيولة ومستوى السيولة
    """
    if len(df) < lookback:
        return None

    search_start = max(0, len(df) - lookback)

    for i in range(len(df) - 2, search_start, -1):
        candle = df.iloc[i]
        next_close = df["close"].iloc[i + 1] if i + 1 < len(df) else candle["close"]

        if direction == "bullish":
            # نبحث عن قاع سابق واضح
            prev_lows = [l[1] for l in lows if l[0] < i - 3]
            if not prev_lows:
                continue
            nearest_low = max(prev_lows)  # أقرب قاع سابق

            # الذيل السفلي يخترق القاع
            lower_wick = candle["open"] - candle["low"] if candle["close"] > candle["open"] else candle["close"] - candle["low"]
            wick_ratio = lower_wick / (candle["high"] - candle["low"]) if (candle["high"] - candle["low"]) > 0 else 0

            swept = candle["low"] < nearest_low  # اخترق القاع
            recovered = candle["close"] > nearest_low  # أغلق فوقه
            has_wick = wick_ratio > 0.3  # ذيل واضح

            if swept and recovered and has_wick:
                return {"index": i, "level": nearest_low, "type": "bullish_sweep"}

        else:  # bearish
            prev_highs = [h[1] for h in highs if h[0] < i - 3]
            if not prev_highs:
                continue
            nearest_high = min(prev_highs)

            # الذيل العلوي يخترق القمة
            upper_wick = candle["high"] - candle["close"] if candle["close"] > candle["open"] else candle["high"] - candle["open"]
            wick_ratio = upper_wick / (candle["high"] - candle["low"]) if (candle["high"] - candle["low"]) > 0 else 0

            swept = candle["high"] > nearest_high
            recovered = candle["close"] < nearest_high
            has_wick = wick_ratio > 0.3

            if swept and recovered and has_wick:
                return {"index": i, "level": nearest_high, "type": "bearish_sweep"}

    return None


# ===== الشرط 2: الضلع الواحد القوي + dBOS =====
def detect_single_leg_dbos(df, highs, lows, sweep, direction):
    """
    بعد سحب السيولة، يجب أن يكسر السعر قمتين/قاعين بـ ضلع واحد قوي.
    الضلع الواحد القوي = 3-7 شمعات متتالية في نفس الاتجاه بدون تراجع كبير
    بعده يكسر قمتين (bullish) أو قاعين (bearish)
    """
    if not sweep:
        return None

    sweep_idx = sweep["index"]
    search_start = sweep_idx + 1
    search_end = min(sweep_idx + 40, len(df))

    if direction == "bullish":
        # نبحث عن قمتين بعد السحب
        post_sweep_highs = [h for h in highs if search_start <= h[0] < search_end]
        if len(post_sweep_highs) < 2:
            return None

        # فرز تصاعدي
        post_sweep_highs.sort(key=lambda x: x[0])

        for i in range(len(post_sweep_highs) - 1):
            h1 = post_sweep_highs[i]
            h2 = post_sweep_highs[i + 1]

            # h2 أعلى من h1 = صاعد
            if h2[1] <= h1[1]:
                continue

            # الضلع الواحد: السعر من h1 لـ h2 في 3-8 شمعات بدون تراجع > 50%
            seg = df.iloc[h1[0]:h2[0] + 1]
            if len(seg) < 2 or len(seg) > 10:
                continue

            move = h2[1] - h1[1]
            max_pullback = 0
            for j in range(1, len(seg)):
                pullback = seg["high"].iloc[j - 1] - seg["low"].iloc[j]
                if pullback > max_pullback:
                    max_pullback = pullback

            # التراجع لا يتجاوز 40% من الحركة = ضلع واحد
            if move > 0 and max_pullback / move > 0.4:
                continue

            # تأكيد الكسر: إغلاق فوق h1
            broke = False
            for j in range(h1[0], min(h2[0] + 5, len(df))):
                if df["close"].iloc[j] > h1[1]:
                    broke = True
                    break

            if broke:
                return {
                    "high1": h1,
                    "high2": h2,
                    "break_idx": h2[0],
                    "sweep_level": sweep["level"]
                }

    else:  # bearish
        post_sweep_lows = [l for l in lows if search_start <= l[0] < search_end]
        if len(post_sweep_lows) < 2:
            return None

        post_sweep_lows.sort(key=lambda x: x[0])

        for i in range(len(post_sweep_lows) - 1):
            l1 = post_sweep_lows[i]
            l2 = post_sweep_lows[i + 1]

            if l2[1] >= l1[1]:
                continue

            seg = df.iloc[l1[0]:l2[0] + 1]
            if len(seg) < 2 or len(seg) > 10:
                continue

            move = l1[1] - l2[1]
            max_pullback = 0
            for j in range(1, len(seg)):
                pullback = seg["high"].iloc[j] - seg["low"].iloc[j - 1]
                if pullback > max_pullback:
                    max_pullback = pullback

            if move > 0 and max_pullback / move > 0.4:
                continue

            broke = False
            for j in range(l1[0], min(l2[0] + 5, len(df))):
                if df["close"].iloc[j] < l1[1]:
                    broke = True
                    break

            if broke:
                return {
                    "low1": l1,
                    "low2": l2,
                    "break_idx": l2[0],
                    "sweep_level": sweep["level"]
                }

    return None


# ===== الشرط 3: IDM =====
def detect_idm(df, dbos, direction):
    """
    أول تراجع بعد الـ dBOS = IDM
    - Bullish: أول قاع يتشكل بعد الكسر
    - Bearish: أول قمة تتشكل بعد الكسر
    """
    if not dbos:
        return None

    start = dbos["break_idx"] + 1
    end = min(start + 25, len(df))

    for i in range(start, end):
        if direction == "bullish":
            # قاع محلي = شمعة هابطة بعد صاعدة
            if (df["close"].iloc[i] < df["open"].iloc[i] and
                    df["low"].iloc[i] < df["low"].iloc[i - 1]):
                return {"index": i, "price": df["low"].iloc[i]}
        else:
            # قمة محلية = شمعة صاعدة بعد هابطة
            if (df["close"].iloc[i] > df["open"].iloc[i] and
                    df["high"].iloc[i] > df["high"].iloc[i - 1]):
                return {"index": i, "price": df["high"].iloc[i]}

    return None


# ===== الشرط 4: OB غير ملموس تحت/فوق IDM =====
def detect_unmitigated_ob(df, idm, direction):
    """
    OB مباشرة تحت IDM (bullish) أو فوقه (bearish)
    يجب أن يكون:
    1. آخر شمعة عكسية قبل الحركة القوية
    2. غير ملموس (السعر ما رجع إليه بعد)
    3. جسم واضح > 50%
    """
    if not idm:
        return None

    idm_idx = idm["index"]
    search_start = max(0, idm_idx - 8)

    # نبحث من IDM للخلف
    for i in range(idm_idx - 1, search_start, -1):
        c = df.iloc[i]
        body = abs(c["close"] - c["open"])
        candle_range = c["high"] - c["low"]
        if candle_range == 0:
            continue
        if body / candle_range < 0.5:
            continue

        if direction == "bullish" and c["close"] < c["open"]:
            ob_top = c["open"]
            ob_bottom = c["close"]

            # تحقق إنه غير ملموس: السعر ما نزل لداخل الـ OB بعد تشكله
            mitigated = False
            for j in range(i + 1, len(df)):
                if df["low"].iloc[j] < ob_top and df["close"].iloc[j] < ob_top:
                    mitigated = True
                    break

            if not mitigated:
                return {"top": ob_top, "bottom": ob_bottom, "index": i, "unmitigated": True}

        elif direction == "bearish" and c["close"] > c["open"]:
            ob_top = c["close"]
            ob_bottom = c["open"]

            mitigated = False
            for j in range(i + 1, len(df)):
                if df["high"].iloc[j] > ob_bottom and df["close"].iloc[j] > ob_bottom:
                    mitigated = True
                    break

            if not mitigated:
                return {"top": ob_top, "bottom": ob_bottom, "index": i, "unmitigated": True}

    return None


# ===== هل السعر وصل الـ OB؟ =====
def price_at_ob(current, ob, direction):
    """
    السعر وصل الـ OB أو لا؟
    وصل = داخل المنطقة أو لمسها
    """
    ob_range = ob["top"] - ob["bottom"]
    buffer = ob_range * 0.15  # هامش 15%

    if direction == "bullish":
        # نبحث شراء = السعر نزل للـ OB
        return (ob["bottom"] - buffer) <= current <= (ob["top"] + buffer)
    else:
        # نبحث بيع = السعر صعد للـ OB
        return (ob["bottom"] - buffer) <= current <= (ob["top"] + buffer)


# ===== حساب الدخول والستوب والهدف =====
def calc_trade_levels(ob, sweep, direction):
    """
    دخول: عند ملامسة الـ OB
    ستوب: أسفل الـ OB أو أسفل ذيل سحب السيولة (أيهما أبعد)
    هدف: 4R
    """
    ob_range = ob["top"] - ob["bottom"]
    sl_buffer = ob_range * 0.1

    if direction == "bullish":
        entry = round(ob["top"], 5)
        sl_ob = round(ob["bottom"] - sl_buffer, 5)
        sl_sweep = round(sweep["level"] * 0.999, 5)  # أسفل ذيل السيولة بقليل
        sl = min(sl_ob, sl_sweep)  # أبعد الاثنين
        risk = entry - sl
        tp = round(entry + risk * 4.0, 5)  # 4R
    else:
        entry = round(ob["bottom"], 5)
        sl_ob = round(ob["top"] + sl_buffer, 5)
        sl_sweep = round(sweep["level"] * 1.001, 5)
        sl = max(sl_ob, sl_sweep)
        risk = sl - entry
        tp = round(entry - risk * 4.0, 5)  # 4R

    return entry, sl, tp


# ===== نصيحة المخاطرة =====
def get_risk_advice(account):
    dd_used = account["drawdown_used"]
    daily_used = account["daily_used"]
    remaining_max = account["max_drawdown"] - dd_used
    remaining_daily = account["daily_drawdown"] - daily_used
    phase = account["phase"]

    if remaining_max <= 1.5:
        return 0, "🚨 الدروداون حرج، لا تدخلين!"
    if remaining_daily <= 0.5:
        return 0, "⛔ وصلتِ الحد اليومي"

    if phase == "challenge":
        max_risk = min(remaining_daily * 0.3, 1.0)
    elif phase == "verification":
        max_risk = min(remaining_daily * 0.35, 1.5)
    else:
        max_risk = min(remaining_daily * 0.4, 2.0)

    # dBOS = جودة عالية دايماً، نخاطر بالحد الأقصى المسموح
    risk = round(max_risk, 2)
    label = "dBOS عالي الجودة 🔥"

    if remaining_max < 4:
        label += f"\n⚠️ باقي {remaining_max:.1f}% دروداون"

    return risk, label


# ===== الترند العام =====
def detect_trend(df):
    if len(df) < 20:
        return "neutral"
    r = df.tail(20)
    if r["high"].iloc[-1] > r["high"].iloc[0] and r["low"].iloc[-1] > r["low"].iloc[0]:
        return "bullish"
    if r["high"].iloc[-1] < r["high"].iloc[0] and r["low"].iloc[-1] < r["low"].iloc[0]:
        return "bearish"
    return "neutral"


# ===== التحليل الكامل (5 شروط) =====
def analyze(sym_name, yf_sym, tf, news):
    # تجاهل الصفقات وقت الأخبار المهمة القريبة
    if news["imminent"]:
        return None

    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 50:
        return None

    highs, lows = find_swing_points(df, lb=5)

    # نجرب الشراء والبيع
    for direction in ["bullish", "bearish"]:

        # الشرط 1: سحب السيولة بذيل شمعة
        sweep = detect_liquidity_sweep(df, highs, lows, direction)
        if not sweep:
            continue

        # الشرط 2: ضلع واحد قوي + dBOS
        dbos = detect_single_leg_dbos(df, highs, lows, sweep, direction)
        if not dbos:
            continue

        # الشرط 3: IDM بعد الكسر
        idm = detect_idm(df, dbos, direction)
        if not idm:
            continue

        # الشرط 4: OB غير ملموس تحت/فوق IDM
        ob = detect_unmitigated_ob(df, idm, direction)
        if not ob:
            continue

        current = df["close"].iloc[-1]

        # الشرط 5: السعر وصل الـ OB أو قريب
        at_ob = price_at_ob(current, ob, direction)

        # توافق الفريمات العليا
        df_d = get_candles(yf_sym, "1d", 50)
        daily_trend = detect_trend(df_d) if not df_d.empty else "neutral"
        daily_match = daily_trend == direction

        df_w = get_candles(yf_sym, "1wk", 20)
        weekly_trend = detect_trend(df_w) if not df_w.empty else "neutral"
        weekly_match = weekly_trend == direction

        entry, sl, tp = calc_trade_levels(ob, sweep, direction)
        risk, label = get_risk_advice(ACCOUNT)

        return {
            "symbol": sym_name,
            "tf": tf,
            "direction": direction,
            "current": current,
            "ob": ob,
            "at_ob": at_ob,
            "sweep": sweep,
            "dbos": dbos,
            "idm": idm,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "daily_match": daily_match,
            "daily_trend": daily_trend,
            "weekly_match": weekly_match,
            "weekly_trend": weekly_trend,
            "risk": risk,
            "risk_label": label,
            "news": news,
        }

    return None


# ===== رسالة السيتاب =====
def setup_msg(a):
    direction = "شراء 📈" if a["direction"] == "bullish" else "بيع 📉"
    arrow = "🟢" if a["direction"] == "bullish" else "🔴"

    d_icon = "✅" if a["daily_match"] else "❌"
    w_icon = "✅" if a["weekly_match"] else "⚠️"
    d_txt = {"bullish": "صاعد", "bearish": "هابط"}.get(a["daily_trend"], "محايد")
    w_txt = {"bullish": "صاعد", "bearish": "هابط"}.get(a["weekly_trend"], "محايد")

    news_txt = ""
    if a["news"]["has_news"]:
        news_txt = "⚠️ أخبار قريبة:\n"
        for ev in a["news"]["events"]:
            news_txt += f"  • {ev['title']} بعد {ev['hours']}س\n"

    if a["at_ob"]:
        action = "⚡ وصل الـ OB - ادخلي الحين!\n📌 دخول فوري (Market)"
    else:
        action = f"⏳ ما وصل بعد - حطي ليمت أوردر\n📌 ليمت عند: {a['entry']}"

    risk_txt = (
        f"❌ ما ندخل - {a['risk_label']}" if a["risk"] == 0
        else f"💰 مخاطرة: {a['risk']}% | {a['risk_label']}"
    )

    tv = TRADINGVIEW_LINKS.get(a["symbol"], "https://www.tradingview.com")

    msg = f"{arrow} dBOS {direction} | {a['symbol']} | {a['tf']}\n"
    msg += "─────────────────\n"
    msg += "✅ الشروط الـ5 تحققت:\n"
    msg += f"  1️⃣ سحب سيولة عند {round(a['sweep']['level'], 4)}\n"
    msg += f"  2️⃣ ضلع واحد قوي + dBOS\n"
    msg += f"  3️⃣ IDM عند {round(a['idm']['price'], 4)}\n"
    msg += f"  4️⃣ OB غير ملموس: {round(a['ob']['bottom'], 4)} - {round(a['ob']['top'], 4)}\n"
    msg += f"  5️⃣ {'السعر في الـ OB ✅' if a['at_ob'] else 'انتظار الـ OB ⏳'}\n"
    msg += "─────────────────\n"
    msg += f"{w_icon} أسبوعي: {w_txt}  {d_icon} يومي: {d_txt}\n"
    msg += news_txt
    msg += "─────────────────\n"
    msg += f"{action}\n"
    msg += f"🛑 ستوب: {a['sl']}\n"
    msg += f"🚀 هدف:  {a['tp']}  (4R)\n"
    msg += f"السعر: {round(a['current'], 4)}\n"
    msg += "─────────────────\n"
    msg += f"{risk_txt}\n"
    msg += f"📈 {tv}\n"
    msg += "القرار إلك يا شذا 💪"
    return msg


# ===== رسايل الحساب =====
def daily_advice_msg():
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    pnl = ACCOUNT["pnl_percent"]
    trades = ACCOUNT["trades_week"]
    phase_txt = {"challenge": "🔴 چالنج", "verification": "🟡 تحقق", "funded": "🟢 ممول"}.get(ACCOUNT["phase"], "")

    pnl_txt = (
        f"رابح {pnl}%، واصلي 🌟" if pnl > 3 else
        f"رابح {pnl}%، شغل كويس 👍" if pnl > 0 else
        "عند نقطة البداية 🎯" if pnl == 0 else
        f"خسارة {abs(pnl)}%، خففي الحجم ⚠️" if pnl >= -3 else
        f"خسارة {abs(pnl)}%، حمي الحساب ❗"
    )

    dd_txt = (
        f"باقي {remaining_max:.1f}% الحمدلله ✅" if remaining_max >= 7 else
        f"باقي {remaining_max:.1f}% - تعاملي بحذر 🟡" if remaining_max >= 4 else
        f"باقي {remaining_max:.1f}% فقط! 🔴"
    )

    daily_txt = (
        f"باقي {remaining_daily:.1f}% يومي ✅" if remaining_daily >= 3 else
        f"باقي {remaining_daily:.1f}% يومي ⚠️" if remaining_daily >= 1 else
        "وصلتِ الحد اليومي 🛑"
    )

    trades_txt = (
        "ما دخلتِ صفقات، الصبر ذهب 💎" if trades == 0 else
        f"{trades} صفقة، ممتاز 👏" if trades <= 2 else
        f"{trades} صفقات، شوي كثير 🤔"
    )

    msg = f"صباح الخير يا شذا ☀️\n"
    msg += f"─────────────────\n"
    msg += f"{ACCOUNT['firm_name']} | {phase_txt}\n"
    msg += f"💰 ${ACCOUNT['current_balance']:,.0f}\n"
    msg += f"─────────────────\n"
    msg += f"الحساب: {pnl_txt}\n"
    msg += f"دروداون: {dd_txt}\n"
    msg += f"اليومي: {daily_txt}\n"
    msg += f"الصفقات: {trades_txt}\n"
    msg += f"─────────────────\n"
    msg += f"{random.choice(DAILY_TIPS)}\n"
    msg += "وفقك الله 🤍"
    return msg


def status_msg():
    now = datetime.now(RIYADH_TZ)
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    pnl = ACCOUNT["pnl_percent"]
    icon = "🟢" if pnl >= 0 and remaining_max > 5 else "🟡" if remaining_max > 2 else "🔴"

    msg = f"{icon} الحساب | {now.strftime('%H:%M')} الرياض\n"
    msg += f"─────────────────\n"
    msg += f"PnL: {'+' if pnl >= 0 else ''}{pnl}%\n"
    msg += f"دروداون: {ACCOUNT['drawdown_used']}% (باقي {remaining_max:.1f}%)\n"
    msg += f"اليومي: {ACCOUNT['daily_used']}% (باقي {remaining_daily:.1f}%)\n"
    msg += f"صفقات اليوم: {ACCOUNT['trades_today']} | الأسبوع: {ACCOUNT['trades_week']}"
    return msg


# ===== فحص السوق =====
async def scan_markets(bot):
    news = check_news()

    # تحذير لو في أخبار وشيكة
    if news["imminent"]:
        await bot.send_message(
            chat_id=CHAT_ID,
            text="⚠️ في أخبار مهمة خلال 4 ساعات، ما أرسل صفقات حتى تمر:\n" +
                 "\n".join([f"• {e['title']} بعد {e['hours']}س" for e in news["events"]])
        )
        return False

    found = []
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                r = analyze(name, yf_sym, tf, news)
                if r:
                    found.append(r)
            except Exception as e:
                logger.error(f"خطأ {name} {tf}: {e}")

    if found:
        for s in found:
            await bot.send_message(chat_id=CHAT_ID, text=setup_msg(s))
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
            f"استراتيجية: dBOS (5 شروط صارمة)\n"
            f"{ACCOUNT['firm_name']} | {phase_txt}\n"
            f"💰 ${ACCOUNT['balance']:,.0f} | {ACCOUNT['max_drawdown']}% / {ACCOUNT['daily_drawdown']}% يومي\n"
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
                ACCOUNT["daily_used"] = 0.0
                ACCOUNT["trades_today"] = 0
                last_advice_day = today

            if now.hour % 4 == 0 and now.hour != last_scan_hour and now.minute < 5:
                found = await scan_markets(bot)
                if not found:
                    await bot.send_message(chat_id=CHAT_ID, text=random.choice(NO_SETUP_MSGS))
                last_scan_hour = now.hour
            else:
                await scan_markets(bot)

            await asyncio.sleep(3600)

        except Exception as e:
            logger.error(f"خطأ: {e}")
            await asyncio.sleep(60)


# ===== التحديث التفاعلي =====
async def update_start(update, context):
    await update.message.reply_text(
        "يلا نحدث حسابك 📋\n\nكم الرصيد الحالي؟\nمثال: 10000\n(أو /skip)"
    )
    return S_BALANCE


async def got_balance(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["current_balance"] = float(text.replace(",", "").replace("$", ""))
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_BALANCE
    await update.message.reply_text("كم نسبة الربح/الخسارة الكلية؟\nمثال: +3.5 أو -2.0\n(أو /skip)")
    return S_PNL


async def got_pnl(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["pnl_percent"] = float(text.replace("+", "").replace("%", ""))
        except:
            await update.message.reply_text("رقم غلط، جربي /skip")
            return S_PNL
    await update.message.reply_text("كم الدروداون الكلي المستخدم؟\nمثال: 2.5\n(أو /skip)")
    return S_DD


async def got_dd(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["drawdown_used"] = float(text.replace("%", ""))
        except:
            await update.message.reply_text("رقم غلط، جربي /skip")
            return S_DD
    await update.message.reply_text("كم الدروداون اليومي المستخدم اليوم؟\nمثال: 1.0\n(أو /skip)")
    return S_DAILY


async def got_daily(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["daily_used"] = float(text.replace("%", ""))
        except:
            await update.message.reply_text("رقم غلط، جربي /skip")
            return S_DAILY
    await update.message.reply_text("كم صفقة هاالأسبوع؟\nمثال: 2\n(أو /skip)")
    return S_TRADES_W


async def got_trades_w(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["trades_week"] = int(text)
        except:
            await update.message.reply_text("رقم غلط، جربي /skip")
            return S_TRADES_W
    await update.message.reply_text("كم صفقة اليوم؟\nمثال: 1\n(أو /skip)")
    return S_TRADES_D


async def got_trades_d(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["trades_today"] = int(text)
        except:
            pass

    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]

    msg = "✅ تم التحديث!\n"
    msg += f"─────────────────\n"
    msg += f"💰 ${ACCOUNT['current_balance']:,.0f}\n"
    msg += f"PnL: {'+' if ACCOUNT['pnl_percent'] >= 0 else ''}{ACCOUNT['pnl_percent']}%\n"
    msg += f"دروداون: {ACCOUNT['drawdown_used']}% (باقي {remaining_max:.1f}%)\n"
    msg += f"يومي: {ACCOUNT['daily_used']}% (باقي {remaining_daily:.1f}%)\n"
    msg += f"صفقات الأسبوع: {ACCOUNT['trades_week']} | اليوم: {ACCOUNT['trades_today']}\n"
    msg += "جاهز أراقب بناء على بياناتك 💪"
    await update.message.reply_text(msg)
    return ConversationHandler.END


async def cancel_update(update, context):
    await update.message.reply_text("إلغاء ❌")
    return ConversationHandler.END


# ===== الأوامر =====
async def start_cmd(update, context):
    await update.message.reply_text(
        "يا هلا يا شذا! 🌟\n"
        "بوتك يبحث عن dBOS فقط - نادر وعالي الجودة\n\n"
        "/scan فحص فوري\n"
        "/advice نصايح اليوم\n"
        "/status حالة الحساب\n"
        "/update تحديث الحساب\n"
    )


async def scan_cmd(update, context):
    await update.message.reply_text(random.choice(WAITING_MSGS))
    found = await scan_markets(context.bot)
    if not found:
        await update.message.reply_text(random.choice(NO_SETUP_MSGS))


async def advice_cmd(update, context):
    await update.message.reply_text(daily_advice_msg())


async def status_cmd(update, context):
    await update.message.reply_text(status_msg())


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
    app.add_handler(update_conv)

    bot = Bot(token=TELEGRAM_TOKEN)
    async with app:
        await app.start()
        await app.updater.start_polling()
        await trading_loop(bot)


if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
