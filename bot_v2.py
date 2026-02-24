import asyncio
import os
import logging
import requests
import random
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas as pd
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ConversationHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")
RIYADH_TZ = pytz.timezone("Asia/Riyadh")

# ===== ACCOUNT (Prop Firm) =====
ACCOUNT = {
    "balance": float(os.environ.get("ACCOUNT_BALANCE", "10000")),
    "current_balance": float(os.environ.get("ACCOUNT_BALANCE", "10000")),
    "max_drawdown": float(os.environ.get("MAX_DRAWDOWN", "10.0")),       # % من الشركة
    "daily_drawdown": float(os.environ.get("DAILY_DRAWDOWN", "5.0")),    # % يومي
    "drawdown_used": 0.0,
    "daily_used": 0.0,
    "trades_week": 0,
    "trades_today": 0,
    "pnl_percent": 0.0,
    "pnl_today": 0.0,
    "firm_name": os.environ.get("FIRM_NAME", "Prop Firm"),
    "phase": os.environ.get("ACCOUNT_PHASE", "challenge"),  # challenge / verification / funded
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

# ===== PERSONALITY MESSAGES =====
WAITING_MSGS = [
    "🔍 جالس أفحص الأسواق لك يا قمر، لحظة صبر",
    "👀 عيني على الشارت، ما يفوتني شي",
    "⚡ البحث مستمر، السوق مو دايم يعطي فرص بس أنا صاحي دايم",
    "🧐 أفحص كل زوج بعين، خليني أشوف وش عندنا",
]

NO_SETUP_MSGS = [
    "😅 فحصت كل شي يا حبيبتي، السوق مو عاطينا سيتاب يستاهل الحين. روحي اتقهوي وأنا هنا أراقب لك 🫖",
    "🙅 ما لقيت فرصة بشروطنا الحين. الصبر ذهب يا شذا، والفرص دايم تجي للصابرين 💪",
    "😌 السوق هادي، ما في حركة تستاهل. استغلي الوقت تراجعين الجورنال أو تستريحين ☕",
    "🤷 ما في سيتاب صح الحين. أحسن من صفقة خاسرة بسبب عجلة، صح؟ 😉",
]

STATUS_MSGS = [
    "💪 جالس أبحث لك، عيني ما تفارق الشارت",
    "🔥 أفحص الأزواج واحد واحد، لو في شي أنبهك فوراً",
    "😎 صاحي ومراقب، لا تقلقين أبد",
    "🚀 شغال بكامل طاقتي، ما شي يفوتني إن شاء الله",
]

DAILY_TIPS = [
    "💡 ما في صفقة تستاهل تكسرين عشانها خطتك. الخطة هي الملك",
    "⏳ السوينق يحتاج صبر. الصفقة الصح تجيك، ما تروحين إليها",
    "🛡️ الخسارة جزء من التداول، المهم إدارة المخاطرة مو الربح السريع",
    "🧠 أي ضغط داخل الصفقة؟ هذا إشارة توقفين مو تكملين",
    "🏆 الفرق بين المحترف والمبتدئ مو في الصفقات، في الانضباط",
    "📝 اكتبي كل صفقة في الجورنال. اللي ما يوثق ما يتعلم",
    "🌿 لو حسيتِ بالثقل من السوق، خذي استراحة. الحساب أهم من أي صفقة",
    "🎯 ركزي على الجودة مو الكمية، صفقة واحدة صح أفضل من عشر مشكوك فيها",
]

GREET_MSGS = [
    "يا هلا والله يا شذا! 🌟",
    "صباح الخير يا نجمة! ☀️",
    "أهلاً بالبطلة! 💪",
    "يا هلا يا قمر! 🌙",
]

# ===== CONVERSATION STATES =====
(ASK_BALANCE, ASK_PNL, ASK_DD_USED, ASK_DAILY_USED, ASK_TRADES, ASK_TRADES_TODAY) = range(6)


# ===== NEWS =====
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
                            "currency": ev.get("country", ""),
                            "hours": round(diff.total_seconds() / 3600, 1)
                        })
            except:
                continue
        return {"has_news": len(upcoming) > 0, "events": upcoming[:3]}
    except:
        return {"has_news": False, "events": []}


# ===== MARKET ANALYSIS =====
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


def find_swings(df, lb=3):
    highs, lows = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def detect_dbos(df, highs, lows, direction):
    if direction == "bullish" and len(highs) >= 2:
        for i in range(len(highs) - 1, 0, -1):
            if highs[i][1] > highs[i - 1][1]:
                for j in range(highs[i - 1][0], len(df)):
                    if df["close"].iloc[j] > highs[i - 1][1]:
                        return {"index": j, "price": highs[i - 1][1]}
    elif direction == "bearish" and len(lows) >= 2:
        for i in range(len(lows) - 1, 0, -1):
            if lows[i][1] < lows[i - 1][1]:
                for j in range(lows[i - 1][0], len(df)):
                    if df["close"].iloc[j] < lows[i - 1][1]:
                        return {"index": j, "price": lows[i - 1][1]}
    return None


def find_idm(df, idx, direction):
    for i in range(idx + 1, min(idx + 25, len(df))):
        if direction == "bullish":
            if df["close"].iloc[i] < df["open"].iloc[i] and df["low"].iloc[i] < df["low"].iloc[i - 1]:
                return {"index": i, "price": df["low"].iloc[i]}
        else:
            if df["close"].iloc[i] > df["open"].iloc[i] and df["high"].iloc[i] > df["high"].iloc[i - 1]:
                return {"index": i, "price": df["high"].iloc[i]}
    return None


def find_ob(df, idx, direction):
    if not idx or idx < 2:
        return None
    for i in range(idx, max(idx - 15, 0), -1):
        c = df.iloc[i]
        if direction == "bullish" and c["close"] < c["open"]:
            return {"top": c["open"], "bottom": c["close"]}
        elif direction == "bearish" and c["close"] > c["open"]:
            return {"top": c["close"], "bottom": c["open"]}
    return None


def check_sweep(df, direction):
    if len(df) < 15:
        return False
    rh = df["high"].tail(15).iloc[:-2].max()
    rl = df["low"].tail(15).iloc[:-2].min()
    last = df.iloc[-2]
    if direction == "bullish":
        return last["low"] < rl and df["close"].iloc[-1] > rl
    return last["high"] > rh and df["close"].iloc[-1] < rh


def calc_quality(dbos, idm, ob, sweep, weekly_match, daily_match, has_news):
    score = 0
    if dbos:
        score += 20
    if idm:
        score += 20
    if ob:
        score += 20
    if sweep:
        score += 15
    if daily_match:
        score += 15
    if weekly_match:
        score += 10
    if has_news:
        score -= 15
    return max(0, min(100, score))


def calc_entry_sl_tp(current, ob, direction, symbol):
    """حساب الدخول والستوب والهدف"""
    ob_mid = (ob["top"] + ob["bottom"]) / 2
    ob_range = ob["top"] - ob["bottom"]

    # نسب مختلفة حسب الزوج
    pip_multiplier = 1.0
    if symbol in ["XAUUSD", "XAGUSD"]:
        pip_multiplier = 1.0
    elif symbol == "BTCUSD":
        pip_multiplier = 1.0
    elif symbol in ["USDJPY"]:
        pip_multiplier = 0.01
    else:
        pip_multiplier = 0.0001

    if direction == "bullish":
        entry = round(ob["top"] * 0.98 + ob["bottom"] * 0.02, 5)  # قرب أعلى OB
        sl = round(ob["bottom"] - ob_range * 0.3, 5)               # تحت OB بشوي
        tp1 = round(entry + (entry - sl) * 1.5, 5)                 # RR 1.5
        tp2 = round(entry + (entry - sl) * 2.5, 5)                 # RR 2.5
    else:
        entry = round(ob["bottom"] * 0.98 + ob["top"] * 0.02, 5)  # قرب أسفل OB
        sl = round(ob["top"] + ob_range * 0.3, 5)                  # فوق OB بشوي
        tp1 = round(entry - (sl - entry) * 1.5, 5)                 # RR 1.5
        tp2 = round(entry - (sl - entry) * 2.5, 5)                 # RR 2.5

    sl_pips = abs(entry - sl)
    rr1 = round(abs(tp1 - entry) / sl_pips, 1) if sl_pips > 0 else 0
    rr2 = round(abs(tp2 - entry) / sl_pips, 1) if sl_pips > 0 else 0

    return entry, sl, tp1, tp2, rr1, rr2


def get_risk_advice(quality, account):
    """نصيحة المخاطرة بناء على حالة الحساب والجودة"""
    dd_used = account["drawdown_used"]
    daily_used = account["daily_used"]
    max_dd = account["max_drawdown"]
    daily_dd = account["daily_drawdown"]
    remaining_max = max_dd - dd_used
    remaining_daily = daily_dd - daily_used
    phase = account["phase"]

    # فحص الدروداون
    if remaining_max <= 1.5:
        return 0, "🚨 الدروداون حرج جداً! لا تدخلين أي صفقة الآن"
    if remaining_daily <= 0.5:
        return 0, "⛔ وصلتِ للحد اليومي، استريحي لهاليوم"

    # الحد الأقصى للمخاطرة حسب الحالة
    max_risk_per_trade = min(remaining_daily * 0.4, remaining_max * 0.2)

    if phase == "challenge":
        max_risk_per_trade = min(max_risk_per_trade, 1.0)  # أكثر حذراً في الچالنج
    elif phase == "verification":
        max_risk_per_trade = min(max_risk_per_trade, 1.5)
    else:  # funded
        max_risk_per_trade = min(max_risk_per_trade, 2.0)

    # مخاطرة بناء على الجودة
    if quality >= 90:
        risk = min(max_risk_per_trade, 1.5)
        label = "ممتازة 🔥"
    elif quality >= 80:
        risk = min(max_risk_per_trade, 1.0)
        label = "قوية 💪"
    elif quality >= 70:
        risk = min(max_risk_per_trade, 0.75)
        label = "كويسة 👍"
    elif quality >= 60:
        risk = min(max_risk_per_trade, 0.5)
        label = "مقبولة، خففي المخاطرة 🤏"
    else:
        return 0, "ضعيفة، ما ندخل ❌"

    # تحذير إضافي لو الحساب تحت ضغط
    warning = ""
    if remaining_max < 4:
        warning = f"\n⚠️ باقي {remaining_max:.1f}% دروداون فقط، تعاملي بحذر شديد"
    elif remaining_daily < 2:
        warning = f"\n⚠️ باقي {remaining_daily:.1f}% يومي فقط"

    return round(risk, 2), label + warning


def analyze(sym_name, yf_sym, tf, news):
    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 30:
        return None

    trend = detect_trend(df)
    if trend == "neutral":
        return None

    highs, lows = find_swings(df)
    dbos = detect_dbos(df, highs, lows, trend)
    if not dbos:
        return None

    idm = find_idm(df, dbos["index"], trend)
    if not idm:
        return None

    ob = find_ob(df, idm["index"], trend)
    if not ob:
        return None

    current = df["close"].iloc[-1]
    ob_range = ob["top"] - ob["bottom"]
    in_ob = (ob["bottom"] - ob_range * 0.3) <= current <= (ob["top"] + ob_range * 0.3)
    sweep = check_sweep(df, trend)

    # Daily & Weekly trend
    df_d = get_candles(yf_sym, "1d", 30)
    daily_trend = detect_trend(df_d) if not df_d.empty else "neutral"
    daily_match = daily_trend == trend

    df_w = get_candles(yf_sym, "1wk", 20)
    weekly_trend = detect_trend(df_w) if not df_w.empty else "neutral"
    weekly_match = weekly_trend == trend

    quality = calc_quality(dbos, idm, ob, sweep, weekly_match, daily_match, news["has_news"])
    if quality < 60:
        return None

    entry, sl, tp1, tp2, rr1, rr2 = calc_entry_sl_tp(current, ob, trend, sym_name)

    return {
        "symbol": sym_name,
        "tf": tf,
        "trend": trend,
        "current": current,
        "ob_top": ob["top"],
        "ob_bottom": ob["bottom"],
        "in_ob": in_ob,
        "sweep": sweep,
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
    risk, label = get_risk_advice(a["quality"], ACCOUNT)

    # توافق الفريمات
    daily_icon = "✅" if a["daily_match"] else "❌"
    weekly_icon = "✅" if a["weekly_match"] else "⚠️"
    daily_txt = a["daily_trend"].replace("bullish", "صاعد").replace("bearish", "هابط").replace("neutral", "محايد")
    weekly_txt = a["weekly_trend"].replace("bullish", "صاعد").replace("bearish", "هابط").replace("neutral", "محايد")

    # أخبار
    news_txt = ""
    if a["news"]["has_news"]:
        news_txt = "\n⚠️ تنبيه أخبار مهمة!\n"
        for ev in a["news"]["events"]:
            news_txt += f"  • {ev['title']} بعد {ev['hours']} ساعة\n"
        news_txt += "خذي بالك وخففي الحجم 🙏\n"

    # extras
    extras = []
    if a["sweep"]:
        extras.append("✅ سحب سيولة قبل الحركة")
    if a["daily_match"] and a["weekly_match"]:
        extras.append("✅ توافق كامل: اليومي والأسبوعي يدعمان")
    elif a["daily_match"]:
        extras.append("✅ اليومي يدعم، الأسبوعي محايد")

    # شريط الجودة
    filled = a["quality"] // 20
    quality_bar = "█" * filled + "░" * (5 - filled)
    quality_label = "ممتاز 🔥" if a["quality"] >= 90 else "قوي 💪" if a["quality"] >= 80 else "كويس 👍" if a["quality"] >= 70 else "مقبول"

    # zone
    if a["in_ob"]:
        zone_txt = "⚡ السعر داخل الـ OB الحين! فرصة الدخول قائمة"
    else:
        zone_txt = f"⏳ انتظري السعر يوصل للمنطقة"

    # risk
    if risk == 0:
        risk_txt = f"❌ ما ندخل - {label}"
        lot_txt = ""
    else:
        risk_amount = round(ACCOUNT["current_balance"] * risk / 100, 2)
        risk_txt = f"💰 مخاطرة: {risk}% (≈ ${risk_amount}) - {label}"
        lot_txt = ""

    tv_link = TRADINGVIEW_LINKS.get(a["symbol"], "https://www.tradingview.com")

    msg = f"{arrow} سيتاب {direction} | {a['symbol']}\n"
    msg += f"⏱ فريم: {a['tf']}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "📊 تحليل الفريمات:\n"
    msg += f"  {weekly_icon} أسبوعي: {weekly_txt}\n"
    msg += f"  {daily_icon} يومي: {daily_txt}\n"
    msg += f"  {arrow} {a['tf']}: إشارة الدخول\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "🔬 السيتاب:\n"
    msg += "  • DBOS ✅ كسر هيكل مزدوج\n"
    msg += "  • IDM ✅ أول بول باك\n"
    msg += "  • OB ✅ أوردر بلوك جاهز\n"
    if extras:
        for e in extras:
            msg += f"  • {e}\n"
    msg += news_txt
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"💵 السعر الحالي: {round(a['current'], 5)}\n"
    msg += f"🎯 دخول:  {a['entry']}\n"
    msg += f"🛑 ستوب:  {a['sl']}\n"
    msg += f"✅ هدف 1: {a['tp1']}  (RR {a['rr1']}:1)\n"
    msg += f"🚀 هدف 2: {a['tp2']}  (RR {a['rr2']}:1)\n"
    msg += f"{zone_txt}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"⭐ الجودة: {a['quality']}/100 {quality_label}\n"
    msg += f"  {quality_bar}\n"
    msg += f"{risk_txt}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"📈 شوفي الشارت: {tv_link}\n"
    msg += "القرار النهائي إلك شذا 💪"
    return msg


def daily_advice_msg():
    dd = ACCOUNT["drawdown_used"]
    trades = ACCOUNT["trades_week"]
    pnl = ACCOUNT["pnl_percent"]
    pnl_today = ACCOUNT["pnl_today"]
    remaining_max = ACCOUNT["max_drawdown"] - dd
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    phase_txt = {
        "challenge": "🔴 چالنج",
        "verification": "🟡 تحقق",
        "funded": "🟢 ممول"
    }.get(ACCOUNT["phase"], "")

    # PnL
    if pnl > 3:
        pnl_txt = f"رابح {pnl}%، واصلي بنفس المنهج يا نجمة 🌟"
    elif pnl > 0:
        pnl_txt = f"رابح {pnl}%، شغل كويس، ثبتي على الخطة 👍"
    elif pnl == 0:
        pnl_txt = "عند نقطة البداية، ركزي على الجودة 🎯"
    elif pnl >= -3:
        pnl_txt = f"خسارة {abs(pnl)}%، خففي الحجم وانتبهي ⚠️"
    else:
        pnl_txt = f"خسارة {abs(pnl)}%، الأولوية الآن حماية الحساب ❗"

    # Drawdown
    if dd == 0:
        dd_txt = "الحساب طازج 100%، الحمدلله ✨"
    elif remaining_max >= 7:
        dd_txt = f"استخدمتِ {dd}%، باقي {remaining_max:.1f}% الحمدلله 👌"
    elif remaining_max >= 4:
        dd_txt = f"باقي {remaining_max:.1f}% دروداون، تعاملي بحذر 🟡"
    else:
        dd_txt = f"باقي {remaining_max:.1f}% فقط! الحساب يحتاج عناية قصوى 🔴"

    # Daily
    if remaining_daily >= 3:
        daily_txt = f"اليوم استخدمتِ {ACCOUNT['daily_used']:.1f}%، باقي {remaining_daily:.1f}% يومي ✅"
    elif remaining_daily >= 1:
        daily_txt = f"تقربين من الحد اليومي! باقي {remaining_daily:.1f}% فقط ⚠️"
    else:
        daily_txt = "وصلتِ للحد اليومي، لا مزيد من الصفقات اليوم 🛑"

    # Trades
    if trades == 0:
        trades_txt = "ما دخلتِ صفقات، الصبر ذهب انتظري السيتاب الصح 💎"
    elif trades <= 2:
        trades_txt = f"دخلتِ {trades} صفقة، ممتاز 👏"
    elif trades <= 4:
        trades_txt = f"{trades} صفقات الأسبوع، شوي كثير للسوينق 🤔"
    else:
        trades_txt = f"{trades} صفقات! أكثر من اللازم، ركزي على الجودة 🛑"

    msg = f"☀️ {random.choice(GREET_MSGS)}\n"
    msg += f"نصايح اليوم من بوتك\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"🏢 {ACCOUNT['firm_name']} | {phase_txt}\n"
    msg += f"💰 الحساب: ${ACCOUNT['current_balance']:,.0f}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"1️⃣ وضع الحساب:\n   {pnl_txt}\n\n"
    msg += f"2️⃣ الدروداون الكلي:\n   {dd_txt}\n\n"
    msg += f"3️⃣ الدروداون اليومي:\n   {daily_txt}\n\n"
    msg += f"4️⃣ الصفقات:\n   {trades_txt}\n\n"
    msg += f"5️⃣ نصيحة اليوم:\n   {random.choice(DAILY_TIPS)}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "وفقك الله يا شذا 🤍"
    return msg


def status_msg():
    now = datetime.now(RIYADH_TZ)
    pnl = ACCOUNT["pnl_percent"]
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]

    status_icon = "🟢" if pnl >= 0 and remaining_max > 5 else "🟡" if remaining_max > 2 else "🔴"

    msg = f"{random.choice(STATUS_MSGS)}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"🕐 الوقت: {now.strftime('%H:%M')} الرياض\n"
    msg += f"{status_icon} الحساب: {'+' if pnl >= 0 else ''}{pnl}%\n"
    msg += f"📊 دروداون مستخدم: {ACCOUNT['drawdown_used']}%\n"
    msg += f"📅 دروداون يومي: {ACCOUNT['daily_used']}%\n"
    msg += f"🔢 صفقات اليوم: {ACCOUNT['trades_today']}\n"
    msg += f"📈 صفقات الأسبوع: {ACCOUNT['trades_week']}\n"
    msg += f"💰 باقي (كلي): {remaining_max:.1f}% | يومي: {remaining_daily:.1f}%"
    return msg


# ===== INTERACTIVE UPDATE (Conversation) =====
async def update_start(update, context):
    await update.message.reply_text(
        "💬 خلنا نحدث بيانات حسابك!\n\nكم رصيد حسابك الحالي بالدولار؟\n(مثال: 10000 أو اكتبي /skip لتخطي)"
    )
    return ASK_BALANCE


async def ask_pnl(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text.replace(",", ""))
            ACCOUNT["current_balance"] = val
        except:
            await update.message.reply_text("⚠️ رقم غلط، جربي مرة ثانية أو /skip")
            return ASK_BALANCE
    await update.message.reply_text(
        "📊 كم نسبة الربح/الخسارة الكلية للحساب؟\n(مثال: +3.5 أو -2.0 أو /skip)"
    )
    return ASK_PNL


async def ask_dd_used(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text.replace("+", ""))
            ACCOUNT["pnl_percent"] = val
        except:
            await update.message.reply_text("⚠️ رقم غلط، جربي /skip")
            return ASK_PNL
    await update.message.reply_text(
        "📉 كم الدروداون الكلي المستخدم حتى الآن؟\n(مثال: 2.5 أو /skip)"
    )
    return ASK_DD_USED


async def ask_daily_used(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text)
            ACCOUNT["drawdown_used"] = val
        except:
            await update.message.reply_text("⚠️ رقم غلط، جربي /skip")
            return ASK_DD_USED
    await update.message.reply_text(
        "📅 كم الدروداون اليومي المستخدم اليوم؟\n(مثال: 1.0 أو /skip)"
    )
    return ASK_DAILY_USED


async def ask_trades_week(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = float(text)
            ACCOUNT["daily_used"] = val
        except:
            await update.message.reply_text("⚠️ رقم غلط، جربي /skip")
            return ASK_DAILY_USED
    await update.message.reply_text(
        "🔢 كم عدد صفقاتك هاالأسبوع؟\n(مثال: 3 أو /skip)"
    )
    return ASK_TRADES


async def ask_trades_today(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = int(text)
            ACCOUNT["trades_week"] = val
        except:
            await update.message.reply_text("⚠️ رقم غلط، جربي /skip")
            return ASK_TRADES
    await update.message.reply_text(
        "📌 كم عدد صفقاتك اليوم؟\n(مثال: 1 أو /skip)"
    )
    return ASK_TRADES_TODAY


async def finish_update(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            val = int(text)
            ACCOUNT["trades_today"] = val
        except:
            pass

    msg = "✅ تم تحديث الحساب!\n\n"
    msg += f"💰 الرصيد: ${ACCOUNT['current_balance']:,.0f}\n"
    msg += f"📊 PnL: {'+' if ACCOUNT['pnl_percent'] >= 0 else ''}{ACCOUNT['pnl_percent']}%\n"
    msg += f"📉 دروداون كلي: {ACCOUNT['drawdown_used']}%\n"
    msg += f"📅 دروداون يومي: {ACCOUNT['daily_used']}%\n"
    msg += f"🔢 صفقات الأسبوع: {ACCOUNT['trades_week']}\n"
    msg += f"📌 صفقات اليوم: {ACCOUNT['trades_today']}\n"
    msg += "\nبوتك جاهز يراقب ويحلل بناء على بياناتك الجديدة 💪"
    await update.message.reply_text(msg)
    return ConversationHandler.END


async def cancel_update(update, context):
    await update.message.reply_text("❌ إلغاء التحديث. البيانات القديمة ما تغيرت.")
    return ConversationHandler.END


# ===== SCAN =====
async def scan_markets(bot):
    news = check_news()
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
        found.sort(key=lambda x: x["quality"], reverse=True)
        for s in found:
            await bot.send_message(chat_id=CHAT_ID, text=setup_msg(s))
            await asyncio.sleep(2)
        return True
    return False


# ===== TRADING LOOP =====
async def trading_loop(bot):
    await bot.send_message(
        chat_id=CHAT_ID,
        text=(
            f"🤖 أهلاً يا شذا! بوتك اشتغل ✅\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🏢 {ACCOUNT['firm_name']}\n"
            f"💰 رصيد: ${ACCOUNT['balance']:,.0f}\n"
            f"📉 حد دروداون: {ACCOUNT['max_drawdown']}% | يومي: {ACCOUNT['daily_drawdown']}%\n"
            f"━━━━━━━━━━━━━━━\n"
            f"أراقب الأسواق كل ساعة وأنبهك بأي سيتاب 👀\n\n"
            f"/scan فحص فوري\n"
            f"/advice نصايح اليوم\n"
            f"/status حالة الحساب\n"
            f"/update تحديث بيانات الحساب\n"
        )
    )
    last_advice_day = None
    last_status_hour = -1

    while True:
        try:
            now = datetime.now(RIYADH_TZ)
            today = now.date()

            # نصايح الصباح
            if now.hour == 8 and now.minute < 5 and last_advice_day != today:
                await bot.send_message(chat_id=CHAT_ID, text=daily_advice_msg())
                ACCOUNT["daily_used"] = 0.0
                ACCOUNT["trades_today"] = 0
                last_advice_day = today

            # فحص كل 4 ساعات مع رسالة
            if now.hour % 4 == 0 and now.hour != last_status_hour and now.minute < 5:
                found = await scan_markets(bot)
                if not found:
                    await bot.send_message(chat_id=CHAT_ID, text=random.choice(NO_SETUP_MSGS))
                last_status_hour = now.hour
            else:
                await scan_markets(bot)

            await asyncio.sleep(3600)

        except Exception as e:
            logger.error(f"خطأ: {e}")
            await asyncio.sleep(60)


# ===== COMMANDS =====
async def start_cmd(update, context):
    await update.message.reply_text(
        f"🌟 {random.choice(GREET_MSGS)}\n"
        "أنا بوتك للتداول، أراقب الأسواق 24/7 وما يفوتني شي!\n\n"
        "📌 الأوامر:\n"
        "/scan فحص فوري للأسواق\n"
        "/advice نصايح اليوم والحساب\n"
        "/status حالة الحساب الآن\n"
        "/update تحديث بيانات الحساب\n"
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


# ===== MAIN =====
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Conversation handler للتحديث التفاعلي
    update_conv = ConversationHandler(
        entry_points=[CommandHandler("update", update_start)],
        states={
            ASK_BALANCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_pnl)],
            ASK_PNL: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_dd_used)],
            ASK_DD_USED: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_daily_used)],
            ASK_DAILY_USED: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_trades_week)],
            ASK_TRADES: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_trades_today)],
            ASK_TRADES_TODAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, finish_update)],
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
