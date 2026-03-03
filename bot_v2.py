import asyncio
import os
import logging
import requests
import random
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas as pd
import json
from telegram import Bot
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ConversationHandler, CallbackQueryHandler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Google Sheets =====
from sheets_db import (
    setup_sheets, journal_add, journal_set_status,
    journal_set_result, journal_load,
    account_save, account_load,
    weights_save, weights_load, stats_add_week
)

# ===== ثوابت =====
BOT_DATA_FILE  = "bot_data.json"
WEIGHTS_FILE   = "weights.json"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
RIYADH_TZ      = pytz.timezone("Asia/Riyadh")

# ===== حالات المحادثة =====
(S_FIRM, S_PHASE, S_BALANCE, S_MAX_DD, S_DAILY_DD,
 S_PNL, S_DD_USED, S_DAILY_USED, S_TRADES_W, S_TRADES_D) = range(10)

# ===== بيانات الحساب =====
ACCOUNT = {
    "balance":         float(os.environ.get("ACCOUNT_BALANCE", "10000")),
    "current_balance": float(os.environ.get("ACCOUNT_BALANCE", "10000")),
    "max_drawdown":    float(os.environ.get("MAX_DRAWDOWN", "10.0")),
    "daily_drawdown":  float(os.environ.get("DAILY_DRAWDOWN", "5.0")),
    "drawdown_used":   0.0,
    "daily_used":      0.0,
    "trades_week":     0,
    "trades_today":    0,
    "pnl_percent":     0.0,
    "firm_name":       os.environ.get("FIRM_NAME", ""),
    "phase":           os.environ.get("ACCOUNT_PHASE", ""),
    "profit_split":    float(os.environ.get("PROFIT_SPLIT", "20")),
    "setup_done":      False,
}

PHASE_TARGETS = {
    "challenge":    {"target": 8.0,  "max_dd": 10.0, "daily_dd": 5.0},
    "verification": {"target": 4.0,  "max_dd": 10.0, "daily_dd": 5.0},
    "funded":       {"target": None, "max_dd": 10.0, "daily_dd": 5.0},
}

# ===== الرموز =====
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

HARD_NEWS_BLOCK = ["CPI", "NFP", "Non-Farm", "FOMC", "Federal Reserve", "Fed Rate"]
HIGH_IMPACT_KEYWORDS = [
    "Fed", "Federal Reserve", "FOMC", "Interest Rate",
    "CPI", "NFP", "Non-Farm", "GDP", "Powell", "ECB", "BOE", "BOJ"
]

WAITING_MSGS = [
    "عيني على الشارت، لحظة ⏳",
    "أفحص الأزواج واحد واحد 🔍",
    "ثانية وأخبرك وش شايف 👀",
]

NO_SETUP_MSGS = [
    "ما في سيتاب يستاهل الحين 🤷‍♀️\nروحي اتقهوي وأنا أراقب ☕",
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

JOURNAL        = {}
TRADE_COUNTER  = [0]
SENT_SETUPS    = {}
CHAT_ID = int(os.environ.get("CHAT_ID", "0"))

# قائمة انتظار الـ OB — أزواج وصلت DBOS وننتظر السعر يقترب
OB_WATCHLIST = {}  # key -> {symbol, tf, ob, trend, entry, sl, tp1, tp2, quality, added_at}

DAILY_RISK = {
    "trading_stopped":    False,
    "stopped_until_date": None,
    "loss_today":         False,
    "trades_entered_today": 0,
    "last_warning_hour":  -1,
}

DEFAULT_WEIGHTS = {
    "liq_sweep":    1.0,
    "h4_of":        1.0,
    "daily_of":     1.0,
    "bsl_ssl":      1.0,
    "pdh_pdl":      1.0,
    "lwh_lwl":      1.0,
    "idm_wick":     1.0,
    "ob_body":      1.0,
}
WEIGHTS_MEMORY = DEFAULT_WEIGHTS.copy()


# ============================================================
# ===== حفظ وتحميل =====
# ============================================================

def save_data():
    try:
        account_save(ACCOUNT, DAILY_RISK)
        data = {
            "account": ACCOUNT,
            "daily_risk": DAILY_RISK,
            "journal": {k: {kk: vv for kk, vv in v.items() if kk != "analysis"}
                        for k, v in JOURNAL.items()},
        }
        with open(BOT_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.error(f"خطأ حفظ البيانات: {e}")


def load_data():
    saved_acc = account_load()
    float_fields = ["balance", "current_balance", "max_drawdown", "daily_drawdown",
                    "drawdown_used", "daily_used", "pnl_percent", "profit_split"]
    int_fields   = ["trades_week", "trades_today"]
    bool_fields  = ["setup_done"]
    for k, v in saved_acc.items():
        if k in ACCOUNT:
            try:
                if k in float_fields:
                    ACCOUNT[k] = float(v) if v not in (None, "") else ACCOUNT[k]
                elif k in int_fields:
                    ACCOUNT[k] = int(float(v)) if v not in (None, "") else ACCOUNT[k]
                elif k in bool_fields:
                    ACCOUNT[k] = str(v).strip().lower() in ("true", "1", "yes")
                else:
                    ACCOUNT[k] = v
            except:
                pass

    sheets_journal = journal_load()
    if sheets_journal:
        for tid, t in sheets_journal.items():
            t["yf_sym"] = SYMBOLS.get(t.get("symbol", ""), "")
        JOURNAL.update(sheets_journal)
    else:
        try:
            if os.path.exists(BOT_DATA_FILE):
                with open(BOT_DATA_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saved_risk = data.get("daily_risk", {})
                for k, v in saved_risk.items():
                    if k in DAILY_RISK:
                        DAILY_RISK[k] = v
                JOURNAL.update(data.get("journal", {}))
        except Exception as e:
            logger.error(f"خطأ تحميل البيانات: {e}")

    w = weights_load()
    if w:
        for k in DEFAULT_WEIGHTS:
            if k not in w:
                w[k] = DEFAULT_WEIGHTS[k]
        WEIGHTS_MEMORY.update(w)

    logger.info(f"✅ تم تحميل البيانات: {len(JOURNAL)} صفقة")


def load_weights():
    global WEIGHTS_MEMORY
    w = weights_load()
    if w:
        for k in DEFAULT_WEIGHTS:
            if k not in w:
                w[k] = DEFAULT_WEIGHTS[k]
        WEIGHTS_MEMORY = w.copy()
        return w.copy()
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
    weights_save(weights)
    try:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)
    except Exception as e:
        logger.error(f"خطأ حفظ الأوزان: {e}")


# ============================================================
# ===== إعداد الحساب =====
# ============================================================

async def setup_start(update, context):
    await update.message.reply_text(
        "أهلاً! 👋\nخليني أعرف حسابك عشان أخدمك صح\n\n"
        "1️⃣ اسم شركة التمويل؟\n"
        "مثال: Funding Pips"
    )
    return S_FIRM

async def got_firm(update, context):
    ACCOUNT["firm_name"] = update.message.text.strip()
    await update.message.reply_text(
        "2️⃣ المرحلة؟\n"
        "اكتبي: challenge أو verification أو funded"
    )
    return S_PHASE

async def got_phase(update, context):
    phase = update.message.text.strip().lower()
    if phase not in ["challenge", "verification", "funded"]:
        await update.message.reply_text("اكتبي: challenge أو verification أو funded")
        return S_PHASE
    ACCOUNT["phase"] = phase
    await update.message.reply_text(
        "3️⃣ حجم الحساب بالدولار؟\n"
        "مثال: 5000"
    )
    return S_BALANCE

async def got_balance_setup(update, context):
    try:
        val = float(update.message.text.strip().replace(",", "").replace("$", ""))
        ACCOUNT["balance"] = val
        ACCOUNT["current_balance"] = val
    except:
        await update.message.reply_text("رقم غلط، جربي مرة ثانية")
        return S_BALANCE
    await update.message.reply_text(
        "4️⃣ حد الدروداون الكلي؟\n"
        "مثال: 10"
    )
    return S_MAX_DD

async def got_max_dd(update, context):
    try:
        val = float(update.message.text.strip().replace("%", ""))
        ACCOUNT["max_drawdown"] = val
    except:
        await update.message.reply_text("رقم غلط، جربي مرة ثانية")
        return S_MAX_DD
    await update.message.reply_text(
        "5️⃣ حد الدروداون اليومي؟\n"
        "مثال: 5"
    )
    return S_DAILY_DD

async def got_daily_dd(update, context):
    try:
        val = float(update.message.text.strip().replace("%", ""))
        ACCOUNT["daily_drawdown"] = val
    except:
        await update.message.reply_text("رقم غلط، جربي مرة ثانية")
        return S_DAILY_DD
    ACCOUNT["setup_done"] = True
    save_data()
    phase_label = {"challenge": "Challenge", "verification": "Verification", "funded": "Funded"}.get(ACCOUNT["phase"], "")
    await update.message.reply_text(
        f"✅ تم الإعداد!\n"
        f"─────────────────\n"
        f"🏢 {ACCOUNT['firm_name']} | {phase_label}\n"
        f"💰 ${ACCOUNT['balance']:,.0f}\n"
        f"📉 دروداون كلي: {ACCOUNT['max_drawdown']}%\n"
        f"📅 دروداون يومي: {ACCOUNT['daily_drawdown']}%\n"
        f"─────────────────\n"
        f"البوت يراقب السوق الآن 24/7 👀\n\n"
        f"/scan فحص فوري\n"
        f"/advice نصايح اليوم\n"
        f"/status حالة الحساب\n"
        f"/update تحديث الحساب"
    )
    return ConversationHandler.END

async def update_start(update, context):
    await update.message.reply_text(
        "يلا نحدث حسابك 📋\n\n"
        "كم الرصيد الحالي؟\n(أو /skip)"
    )
    return S_BALANCE

async def got_balance(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["current_balance"] = float(text.replace(",", "").replace("$", ""))
            calc_auto_drawdown()
        except:
            await update.message.reply_text("رقم غلط، جربي مرة ثانية أو /skip")
            return S_BALANCE
    await update.message.reply_text("كم نسبة الربح أو الخسارة الكلية؟\nمثال: +3.5 أو -2.0\n(أو /skip)")
    return S_PNL

async def got_pnl(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["pnl_percent"] = float(text.replace("+", "").replace("%", ""))
        except:
            await update.message.reply_text("رقم غلط أو /skip")
            return S_PNL
    await update.message.reply_text("كم الدروداون الكلي المستخدم؟\nمثال: 2.5\n(أو /skip)")
    return S_DD_USED

async def got_dd_used(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["drawdown_used"] = float(text.replace("%", ""))
        except:
            await update.message.reply_text("رقم غلط أو /skip")
            return S_DD_USED
    await update.message.reply_text("كم الدروداون اليومي المستخدم اليوم؟\nمثال: 1.0\n(أو /skip)")
    return S_DAILY_USED

async def got_daily_used(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["daily_used"] = float(text.replace("%", ""))
        except:
            await update.message.reply_text("رقم غلط أو /skip")
            return S_DAILY_USED
    await update.message.reply_text("كم صفقة الأسبوع؟\n(أو /skip)")
    return S_TRADES_W

async def got_trades_w(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["trades_week"] = int(text)
        except:
            await update.message.reply_text("رقم غلط أو /skip")
            return S_TRADES_W
    await update.message.reply_text("كم صفقة اليوم؟\n(أو /skip)")
    return S_TRADES_D

async def got_trades_d(update, context):
    text = update.message.text.strip()
    if text.lower() != "/skip":
        try:
            ACCOUNT["trades_today"] = int(text)
        except:
            pass
    remaining_max   = ACCOUNT["max_drawdown"]   - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    save_data()
    await update.message.reply_text(
        f"✅ تم التحديث!\n"
        f"─────────────────\n"
        f"💰 الرصيد: ${ACCOUNT['current_balance']:,.0f}\n"
        f"📊 PnL: {'+' if ACCOUNT['pnl_percent']>=0 else ''}{ACCOUNT['pnl_percent']}%\n"
        f"📉 دروداون كلي: {ACCOUNT['drawdown_used']}% (باقي {remaining_max:.1f}%)\n"
        f"📅 دروداون يومي: {ACCOUNT['daily_used']}% (باقي {remaining_daily:.1f}%)\n"
        f"🔢 صفقات الأسبوع: {ACCOUNT['trades_week']}\n"
        f"📌 صفقات اليوم: {ACCOUNT['trades_today']}\n"
        f"\nبوتك يحلل بناء على بياناتك الجديدة 💪"
    )
    return ConversationHandler.END

async def cancel_conv(update, context):
    await update.message.reply_text("إلغاء ❌")
    return ConversationHandler.END


# ============================================================
# ===== الأخبار =====
# ============================================================

def check_news():
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        if r.status_code != 200:
            return {"has_news": False, "hard_block": False, "events": []}
        now = datetime.utcnow()
        upcoming = []
        hard_block = False
        for ev in r.json():
            try:
                if ev.get("impact") != "High":
                    continue
                t = datetime.fromisoformat(ev.get("date", "").replace("Z", ""))
                diff = t - now
                if timedelta(hours=-1) <= diff <= timedelta(hours=24):
                    title = ev.get("title", "")
                    if any(k.lower() in title.lower() for k in HIGH_IMPACT_KEYWORDS):
                        upcoming.append({"title": title, "hours": round(diff.total_seconds()/3600, 1)})
                    if any(k.lower() in title.lower() for k in HARD_NEWS_BLOCK):
                        hard_block = True
            except:
                continue
        return {"has_news": len(upcoming) > 0, "hard_block": hard_block, "events": upcoming[:3]}
    except:
        return {"has_news": False, "hard_block": False, "events": []}


def news_comedy_msg(events):
    """رسالة كوميدية سعودية لأيام الأخبار الكبيرة"""
    import random

    # نحدد نوع الخبر
    all_titles = " ".join(e["title"] for e in events).upper()
    if "FOMC" in all_titles or "FEDERAL" in all_titles or "FED RATE" in all_titles:
        news_type = "FOMC"
    elif "NFP" in all_titles or "NON-FARM" in all_titles:
        news_type = "NFP"
    elif "CPI" in all_titles:
        news_type = "CPI"
    else:
        news_type = "خبر"

    hours = events[0]["hours"] if events else 0
    title = events[0]["title"] if events else ""

    timing = (
        "بعد شوي!" if hours < 1
        else f"بعد {hours:.0f} ساعة" if hours < 6
        else f"اليوم الساعة {hours:.0f}"
    )

    fomc_msgs = [
        f"⚠️ اليوم يوم FOMC يا شذا!\n─────────────────\n"
        f"يعني البنك الفيدرالي راح يفتح بزه ويتكلم عن الفايدة 🎙️\n"
        f"السوق كله واقف على رجل ونص ينتظر كلامه 😬\n"
        f"احنا؟ جالسين بره الملعب نشرب قهوة ☕\n"
        f"─────────────────\n"
        f"⏰ {timing} | البوت موقوف حتى تنتهي العاصفة 🛑",

        f"🚨 FOMC اليوم — بيان الفيدرالي {timing}!\n"
        f"─────────────────\n"
        f"باول راح يقف ويتكلم وكل كلمة تحرك السوق مليار دولار 💸\n"
        f"لو قال 'hawkish' السوق ينزل\n"
        f"لو قال 'dovish' السوق يطلع\n"
        f"لو تعطس — الله يستر 😅\n"
        f"─────────────────\n"
        f"قرارنا: نلعب بعيد اليوم 🏃‍♀️ | البوت مو شايل أمانة",
    ]

    nfp_msgs = [
        f"📢 يوم NFP يا شذا — أرقام الوظائف الأمريكية {timing}!\n"
        f"─────────────────\n"
        f"هذا الرقم يخلي المتداولين يتعرقون من غير ما يتحركون 😰\n"
        f"لو الوظائف زادت — الدولار يعلّق\n"
        f"لو نقصت — الذهب يطير\n"
        f"لو جاء 'مخيب' — الكل يصرخ وإحنا نتفرج 🍿\n"
        f"─────────────────\n"
        f"اليوم الخبز يمشي لحاله | ما نلحق على أحد 🛑",

        f"🔥 NFP اليوم! أخطر رقم في الشهر {timing}\n"
        f"─────────────────\n"
        f"يعني بكره الصبح أمريكا تقول: كم واحد اشتغل هالشهر؟\n"
        f"السوق يخمن، ونحن نخمن، وكلنا غلط في الآخر 😂\n"
        f"─────────────────\n"
        f"قرار البوت: إجازة مدفوعة الأجر اليوم ☕🛑",
    ]

    cpi_msgs = [
        f"📊 CPI اليوم — أرقام التضخم الأمريكي {timing}!\n"
        f"─────────────────\n"
        f"يعني بيقيسون غلاء المعيشة هناك 🧺\n"
        f"لو الأسعار طالعة — الفيدرالي يرفع الفايدة\n"
        f"لو نازلة — يمكن يريحنا 😅\n"
        f"كلام كثير والسوق يهتز على طول\n"
        f"─────────────────\n"
        f"إحنا اليوم في بيتنا — السوق يلعب لوحده 🛑",

        f"⚡ CPI اليوم يا شذا {timing}\n"
        f"─────────────────\n"
        f"رقم التضخم — اللي لو طلع عالي\n"
        f"الناس تصيح، الذهب ينط، الدولار يتعالى 📈\n"
        f"ولو نزل؟ نفس الصياح بس معكوس 😂\n"
        f"─────────────────\n"
        f"البوت قرر يتأمل اليوم 🧘‍♀️ | ما في صفقات 🛑",
    ]

    msgs = {
        "FOMC": fomc_msgs,
        "NFP":  nfp_msgs,
        "CPI":  cpi_msgs,
        "خبر":  [
            f"⚠️ خبر مهم اليوم {timing}!\n"
            f"─────────────────\n"
            f"السوق متوتر وما يدري وين يروح 😬\n"
            f"إحنا؟ جالسين نشاهد من بعيد 🍿\n"
            f"─────────────────\n"
            f"البوت موقوف حتى يهدى الوضع 🛑"
        ],
    }

    msg = random.choice(msgs.get(news_type, msgs["خبر"]))
    if len(events) > 1:
        msg += f"\n\n📋 أخبار اليوم:\n"
        for e in events[:3]:
            msg += f"• {e['title']}\n"
    return msg


# ============================================================
# ===== تحليل السوق =====
# ============================================================

def get_candles(yf_sym, tf, limit=150):
    try:
        period = {"1h": "10d", "4h": "60d", "1d": "200d", "1wk": "2y"}.get(tf, "60d")
        df = yf.Ticker(yf_sym).history(period=period, interval=tf)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
        return df.tail(limit)
    except:
        return pd.DataFrame()


def detect_trend_structure(df, lookback=30):
    if len(df) < lookback:
        return "neutral"
    recent = df.tail(lookback)
    highs, lows = [], []
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
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "bullish"
    elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "bearish"
    return "neutral"


def detect_order_flow(df, direction, lookback=10):
    if len(df) < lookback + 1:
        return 0.0
    recent = df.tail(lookback)
    score = 0
    for i in range(1, len(recent)):
        if direction == "bullish":
            if recent["high"].iloc[i] > recent["high"].iloc[i-1]: score += 0.5
            if recent["low"].iloc[i]  > recent["low"].iloc[i-1]:  score += 0.5
        else:
            if recent["high"].iloc[i] < recent["high"].iloc[i-1]: score += 0.5
            if recent["low"].iloc[i]  < recent["low"].iloc[i-1]:  score += 0.5
    return round(score / (lookback - 1), 2)


def detect_liquidity_sweep(df, direction, lookback=50):
    if len(df) < lookback + 5:
        return False, 0.0
    search = df.iloc[-(lookback+5):-5]
    if direction == "bullish":
        ref_low = search["low"].min()
        for i in range(len(df)-5, len(df)):
            c = df.iloc[i]
            if c["low"] < ref_low and c["close"] > ref_low:
                return True, round(ref_low, 5)
    else:
        ref_high = search["high"].max()
        for i in range(len(df)-5, len(df)):
            c = df.iloc[i]
            if c["high"] > ref_high and c["close"] < ref_high:
                return True, round(ref_high, 5)
    return False, 0.0


def detect_dbos(df, direction):
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
            if len(segment) < 2 or len(segment) > 80: continue
            move_size = h2_val - segment["low"].min()
            if move_size <= 0: continue
            for j in range(h2_idx, min(h2_idx+8, len(df))):
                candle = df.iloc[j]
                if candle["close"] > h2_val:
                    return {"index": j, "price": h2_val,
                            "impulse_start": h1_idx,
                            "sweep_level": segment["low"].min(),
                            "h1_val": h1_val, "h2_val": h2_val}

    elif direction == "bearish" and len(l_list) >= 2:
        for i in range(len(l_list)-1, 0, -1):
            l2_idx, l2_val = l_list[i]
            l1_idx, l1_val = l_list[i-1]
            if l2_val >= l1_val: continue
            segment = df.iloc[l1_idx:l2_idx+1]
            if len(segment) < 2 or len(segment) > 80: continue
            move_size = segment["high"].max() - l2_val
            if move_size <= 0: continue
            for j in range(l2_idx, min(l2_idx+8, len(df))):
                candle = df.iloc[j]
                if candle["close"] < l2_val:
                    return {"index": j, "price": l2_val,
                            "impulse_start": l1_idx,
                            "sweep_level": segment["high"].max(),
                            "l1_val": l1_val, "l2_val": l2_val}
    return None


def find_idm(df, dbos_idx, direction):
    search_end = min(dbos_idx + 30, len(df))
    for i in range(dbos_idx + 1, search_end):
        c = df.iloc[i]
        candle_range = c["high"] - c["low"]
        if candle_range == 0: continue
        body = abs(c["close"] - c["open"])
        body_ratio = body / candle_range

        if direction == "bullish":
            lower_wick = min(c["open"], c["close"]) - c["low"]
            wick_ratio = lower_wick / candle_range
            is_pin_bar = wick_ratio > 0.40 and body_ratio < 0.55
            is_engulf  = c["close"] < c["open"] and body_ratio > 0.60
            if (is_pin_bar or is_engulf):
                if c["low"] < df["low"].iloc[max(0,i-5):i].min():
                    return {"index": i, "price": c["low"],
                            "wick_ratio": round(wick_ratio, 2),
                            "type": "pin_bar" if is_pin_bar else "engulf"}
        else:
            upper_wick = c["high"] - max(c["open"], c["close"])
            wick_ratio = upper_wick / candle_range
            is_pin_bar = wick_ratio > 0.40 and body_ratio < 0.55
            is_engulf  = c["close"] > c["open"] and body_ratio > 0.60
            if (is_pin_bar or is_engulf):
                if c["high"] > df["high"].iloc[max(0,i-5):i].max():
                    return {"index": i, "price": c["high"],
                            "wick_ratio": round(wick_ratio, 2),
                            "type": "pin_bar" if is_pin_bar else "engulf"}
    return None


def find_ob(df, idm_idx, direction):
    """
    منطق LuxAlgo Order Block:
    Bullish OB: شمعة أدنى low في المنطقة قبل الكسر
    Bearish OB: شمعة أعلى high في المنطقة قبل الكسر
    - OB zone (للرسم/الستوب) = كامل الشمعة high→low
    - entry = سقف/قاع الجسم فقط (ICT standard)
    """
    if idm_idx is None or idm_idx < 3:
        return None

    search_start = max(0, idm_idx - 30)
    segment = df.iloc[search_start:idm_idx]
    if len(segment) < 2:
        return None

    if direction == "bullish":
        local_pos  = int(segment["low"].values.argmin())
        ob_idx     = search_start + local_pos
        c          = df.iloc[ob_idx]
        candle_range = c["high"] - c["low"]
        if candle_range == 0: return None
        body_ratio = abs(c["close"] - c["open"]) / candle_range
        return {
            "top":         round(max(c["open"], c["close"]), 5),  # سقف الجسم = entry
            "bottom":      round(min(c["open"], c["close"]), 5),  # قاع الجسم
            "index":       ob_idx,
            "body_ratio":  round(body_ratio, 2),
            "candle_high": round(c["high"], 5),  # للرسم والستوب
            "candle_low":  round(c["low"],  5),
        }

    else:  # bearish
        local_pos  = int(segment["high"].values.argmax())
        ob_idx     = search_start + local_pos
        c          = df.iloc[ob_idx]
        candle_range = c["high"] - c["low"]
        if candle_range == 0: return None
        body_ratio = abs(c["close"] - c["open"]) / candle_range
        return {
            "top":         round(max(c["open"], c["close"]), 5),  # قمة الجسم
            "bottom":      round(min(c["open"], c["close"]), 5),  # قاع الجسم = entry
            "index":       ob_idx,
            "body_ratio":  round(body_ratio, 2),
            "candle_high": round(c["high"], 5),  # للرسم والستوب
            "candle_low":  round(c["low"],  5),
        }


def calc_sl_from_ob(ob, direction):
    ob_range  = ob["top"] - ob["bottom"]
    sl_buffer = ob_range * 0.20
    if direction == "bullish":
        return round(ob["candle_low"]  - sl_buffer, 5)
    else:
        return round(ob["candle_high"] + sl_buffer, 5)


def get_pdh_pdl(yf_sym):
    try:
        df = get_candles(yf_sym, "1d", 5)
        if len(df) < 2: return None, None
        prev = df.iloc[-2]
        return round(prev["high"], 5), round(prev["low"], 5)
    except:
        return None, None


def get_lwh_lwl(yf_sym):
    try:
        df = get_candles(yf_sym, "1wk", 5)
        if len(df) < 2: return None, None
        prev = df.iloc[-2]
        return round(prev["high"], 5), round(prev["low"], 5)
    except:
        return None, None


def check_bsl_ssl(df, direction, lookback=50):
    if len(df) < lookback: return False, 0.0
    current = df["close"].iloc[-1]
    recent  = df.tail(lookback)
    if direction == "bullish":
        bsl  = recent["high"].max()
        dist = (bsl - current) / current * 100
        return 0.3 < dist < 5.0, round(bsl, 5)
    else:
        ssl  = recent["low"].min()
        dist = (current - ssl) / current * 100
        return 0.3 < dist < 5.0, round(ssl, 5)


def calc_quality_new(liq_sweep, h4_of, daily_of, has_bsl, has_pdh_pdl,
                     has_lwh_lwl, idm_wick, ob_body, hard_news, monthly_match=True):
    if hard_news: return 0
    w = WEIGHTS_MEMORY
    score = 0
    if liq_sweep:        score += 30 * w.get("liq_sweep", 1.0)
    if h4_of >= 0.7:     score += 20 * w.get("h4_of", 1.0)
    elif h4_of >= 0.5:   score += 12 * w.get("h4_of", 1.0)
    if daily_of >= 0.6:  score += 15 * w.get("daily_of", 1.0)
    elif daily_of >= 0.4:score += 8  * w.get("daily_of", 1.0)
    if has_bsl:          score += 15 * w.get("bsl_ssl", 1.0)
    if has_pdh_pdl:      score += 10 * w.get("pdh_pdl", 1.0)
    if has_lwh_lwl:      score += 10 * w.get("lwh_lwl", 1.0)
    if idm_wick > 0.5:   score += 3  * w.get("idm_wick", 1.0)
    if ob_body > 0.6:    score += 3  * w.get("ob_body",  1.0)
    # الشهري مو شرط — لو يدعم يضيف نقاط بس، لو عكس ما يلغي شي
    if monthly_match:    score += 5
    return max(0, min(100, round(score)))


def get_risk_new(quality):
    if quality < 70:
        return 0, "جودة منخفضة ❌"
    max_dd    = float(ACCOUNT["max_drawdown"]   or 0)
    dd_used   = float(ACCOUNT["drawdown_used"]  or 0)
    dd_pct    = dd_used / max_dd * 100 if max_dd > 0 else 0
    if dd_pct >= 90:
        return 0, f"🚨 وصلت {dd_used:.1f}% من أصل {max_dd}% — الحساب في خطر!"
    if DAILY_RISK["loss_today"]:
        return 0, "🛑 خسرت صفقة اليوم — استريحي إلى الغد"
    if DAILY_RISK["trades_entered_today"] >= 2:
        return 0, "🛑 دخلتِ صفقتين اليوم — حد اليوم وصل"
    if quality >= 90:   base_risk = 1.0
    elif quality >= 80: base_risk = 0.75
    else:               base_risk = 0.5
    if dd_pct >= 80:
        base_risk   = 0.5
        label_extra = f"\n⚠️ دروداون كلي {dd_used:.1f}% — مخاطرة مخففة"
    else:
        label_extra = ""
    labels = {1.0: "ممتازة 🔥", 0.75: "قوية 💪", 0.5: "كويسة 👍"}
    return round(base_risk, 2), labels.get(base_risk, "") + label_extra


def is_trading_allowed():
    if DAILY_RISK.get("stopped_until_date"):
        today_str = datetime.now(RIYADH_TZ).strftime("%Y-%m-%d")
        if today_str < DAILY_RISK["stopped_until_date"]:
            return False, f"🛑 إيقاف حتى {DAILY_RISK['stopped_until_date']}"
    max_dd  = ACCOUNT["max_drawdown"]
    dd_used = ACCOUNT["drawdown_used"]
    if max_dd > 0 and dd_used / max_dd >= 0.90:
        return False, f"🚨 دروداون كلي {dd_used:.1f}% من {max_dd}% — وقف فوري!"
    if DAILY_RISK["loss_today"]:
        return False, "🛑 خسرت صفقة اليوم — الغد إن شاء الله"
    if DAILY_RISK["trades_entered_today"] >= 2:
        return False, "🛑 دخلتِ صفقتين اليوم — حد اليوم وصل"
    return True, ""


def analyze(sym_name, yf_sym, tf, news, debug=False):
    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 60:
        if debug: return f"{sym_name} {tf}: ❌ بيانات فاضية"
        return None
    if news.get("hard_block"):
        if debug: return f"{sym_name} {tf}: ❌ أخبار CPI/NFP/FOMC"
        return None
    trend = detect_trend_structure(df)
    if trend == "neutral":
        if debug: return f"{sym_name} {tf}: ❌ ترند محايد"
        return None
    df_h4    = get_candles(yf_sym, "4h", 50)
    h4_trend = detect_trend_structure(df_h4) if not df_h4.empty else "neutral"
    h4_of    = detect_order_flow(df_h4, trend) if not df_h4.empty else 0.0
    if h4_trend != "neutral" and h4_trend != trend:
        if debug: return f"{sym_name} {tf}: ❌ H4 عكس الاتجاه"
        return None
    if h4_of < 0.55:
        if debug: return f"{sym_name} {tf}: ❌ H4 order flow ضعيف ({h4_of})"
        return None
    df_d        = get_candles(yf_sym, "1d", 30)
    daily_trend = detect_trend_structure(df_d) if not df_d.empty else "neutral"
    daily_of    = detect_order_flow(df_d, trend) if not df_d.empty else 0.0
    if daily_trend != "neutral" and daily_trend != trend:
        if debug: return f"{sym_name} {tf}: ❌ Daily عكس الاتجاه"
        return None
    liq_sweep, liq_level = detect_liquidity_sweep(df, trend)
    dbos = detect_dbos(df, trend)
    if not dbos:
        if debug: return f"{sym_name} {tf}: ❌ ما في DBOS"
        return None
    idm = find_idm(df, dbos["index"], trend)
    if not idm:
        if debug: return f"{sym_name} {tf}: ❌ ما في IDM"
        return None
    ob = find_ob(df, idm["index"], trend)
    if not ob:
        if debug: return f"{sym_name} {tf}: ❌ ما في OB"
        return None
    ob_size = ob["top"] - ob["bottom"]
    min_ob = {"BTC": 300.0, "XAU": 3.0, "XAG": 0.05}.get(
        next((k for k in ["BTC","XAU","XAG"] if k in sym_name), ""), 0.0020)
    if ob_size < min_ob:
        if debug: return f"{sym_name} {tf}: ❌ OB صغير ({round(ob_size,4)})"
        return None
    current  = df["close"].iloc[-1]
    ob_range = ob["top"] - ob["bottom"]
    if trend == "bullish":
        if current < ob["bottom"] - ob_range * 1.5:
            if debug: return f"{sym_name} {tf}: ❌ السعر بعيد تحت OB"
            return None
        if current > ob["top"] + ob_range * 3:
            if debug: return f"{sym_name} {tf}: ⏳ السعر فوق OB بكثير"
            return None
    else:
        if current > ob["top"] + ob_range * 1.5:
            if debug: return f"{sym_name} {tf}: ❌ السعر بعيد فوق OB"
            return None
        if current < ob["bottom"] - ob_range * 3:
            if debug: return f"{sym_name} {tf}: ⏳ السعر تحت OB بكثير"
            return None
    in_ob   = ob["bottom"] <= current <= ob["top"]
    ob_age  = len(df) - ob["index"]
    idm_age = len(df) - idm["index"]
    if ob_age > 60:
        if debug: return f"{sym_name} {tf}: ❌ OB قديم ({ob_age} شمعة)"
        return None
    if idm_age > 50:
        if debug: return f"{sym_name} {tf}: ❌ IDM قديم ({idm_age} شمعة)"
        return None
    has_bsl, bsl_level = check_bsl_ssl(df, trend)
    pdh, pdl = get_pdh_pdl(yf_sym)
    has_pdh_pdl = False
    if pdh and pdl:
        if trend == "bullish" and pdh > current: has_pdh_pdl = True
        elif trend == "bearish" and pdl < current: has_pdh_pdl = True
    lwh, lwl = get_lwh_lwl(yf_sym)
    has_lwh_lwl = False
    if lwh and lwl:
        if trend == "bullish" and lwh > current: has_lwh_lwl = True
        elif trend == "bearish" and lwl < current: has_lwh_lwl = True
    df_w          = get_candles(yf_sym, "1wk", 20)
    weekly_trend  = detect_trend_structure(df_w) if not df_w.empty else "neutral"
    weekly_match  = weekly_trend == trend
    # HTF شهري — الصورة الكبيرة
    df_mo         = get_candles(yf_sym, "1mo", 12)
    monthly_trend = detect_trend_structure(df_mo) if not df_mo.empty else "neutral"
    monthly_match = monthly_trend == trend or monthly_trend == "neutral"
    quality = calc_quality_new(
        liq_sweep=liq_sweep, h4_of=h4_of, daily_of=daily_of,
        has_bsl=has_bsl, has_pdh_pdl=has_pdh_pdl, has_lwh_lwl=has_lwh_lwl,
        idm_wick=idm.get("wick_ratio", 0), ob_body=ob.get("body_ratio", 0),
        hard_news=news.get("hard_block", False),
        monthly_match=monthly_match,
    )
    if quality < 70:
        if debug: return f"{sym_name} {tf}: ❌ جودة {quality}% (أقل من 70)"
        return None
    sl = calc_sl_from_ob(ob, trend)
    if trend == "bullish":
        entry = round(ob["top"], 5)
        risk  = entry - sl
        if risk <= 0: return None
        tp1 = round(entry + risk * 2.0, 5)
        tp2 = round(entry + risk * 4.0, 5)
    else:
        entry = round(ob["bottom"], 5)
        risk  = sl - entry
        if risk <= 0: return None
        tp1 = round(entry - risk * 2.0, 5)
        tp2 = round(entry - risk * 4.0, 5)
    return {
        "symbol": sym_name, "tf": tf, "trend": trend,
        "current": current, "ob": ob, "in_ob": in_ob,
        "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
        "h4_of": h4_of, "daily_of": daily_of,
        "liq_sweep": liq_sweep, "liq_level": liq_level,
        "has_bsl": has_bsl, "bsl_level": bsl_level,
        "has_pdh_pdl": has_pdh_pdl, "pdh": pdh, "pdl": pdl,
        "has_lwh_lwl": has_lwh_lwl, "lwh": lwh, "lwl": lwl,
        "daily_trend": daily_trend, "weekly_match": weekly_match,
        "monthly_trend": monthly_trend, "monthly_match": monthly_match,
        "quality": quality, "news": news,
        "idm_type": idm.get("type", ""), "idm_wick": idm.get("wick_ratio", 0),
    }


# ============================================================
# ===== Morning Star =====
# ============================================================

def detect_morning_star(df, direction, lookback=30):
    """
    Morning Star Pattern مع سحب سيولة إلزامي قبل النمط
    c1 = شمعة عكسية كبيرة (جسم > 50%)
    c2 = شمعة صغيرة جداً (تردد)
    c3 = شمعة في الاتجاه قوية تغلق فوق/تحت منتصف c1
    يجب وجود سحب سيولة قبل c1
    """
    if len(df) < lookback + 6:
        return None

    search = df.tail(lookback + 6)

    for i in range(len(search) - 4, len(search) - 1):
        c1 = search.iloc[i - 2]
        c2 = search.iloc[i - 1]
        c3 = search.iloc[i]

        c1_range = c1["high"] - c1["low"]
        c3_range = c3["high"] - c3["low"]
        if c1_range == 0 or c3_range == 0:
            continue

        c1_body = abs(c1["close"] - c1["open"])
        c2_body = abs(c2["close"] - c2["open"])
        c3_body = abs(c3["close"] - c3["open"])
        c1_body_ratio = c1_body / c1_range
        c3_body_ratio = c3_body / c3_range

        if direction == "bullish":
            if not (c1["close"] < c1["open"] and c1_body_ratio > 0.50): continue
            if c1_body > 0 and c2_body / c1_body > 0.35: continue
            if c2["low"] < c1["low"] * 0.999: continue
            c1_mid = (c1["open"] + c1["close"]) / 2
            if not (c3["close"] > c3["open"] and c3["close"] > c1_mid and c3_body_ratio > 0.45): continue
            # سحب سيولة إلزامي قبل c1
            pre_c1   = search.iloc[max(0, i-12):i-2]
            if len(pre_c1) < 3: continue
            ref_low  = pre_c1["low"].min()
            has_sweep = any(
                pre_c1.iloc[j]["low"] < ref_low and pre_c1.iloc[j]["close"] > ref_low
                for j in range(len(pre_c1))
            )
            if not has_sweep: continue
            current = df["close"].iloc[-1]
            if current < c3["close"] * 0.997: continue
            return {
                "direction": "bullish",
                "entry":     round(c3["close"], 5),
                "sl":        round(c1["low"] - c1_range * 0.10, 5),
                "ob_top":    round(c1["open"], 5),
                "ob_bottom": round(c1["low"], 5),
                "liq_pool":  round(ref_low, 5),
                "current":   round(current, 5),
            }

        else:  # bearish
            if not (c1["close"] > c1["open"] and c1_body_ratio > 0.50): continue
            if c1_body > 0 and c2_body / c1_body > 0.35: continue
            if c2["high"] > c1["high"] * 1.001: continue
            c1_mid = (c1["open"] + c1["close"]) / 2
            if not (c3["close"] < c3["open"] and c3["close"] < c1_mid and c3_body_ratio > 0.45): continue
            pre_c1    = search.iloc[max(0, i-12):i-2]
            if len(pre_c1) < 3: continue
            ref_high  = pre_c1["high"].max()
            has_sweep = any(
                pre_c1.iloc[j]["high"] > ref_high and pre_c1.iloc[j]["close"] < ref_high
                for j in range(len(pre_c1))
            )
            if not has_sweep: continue
            current = df["close"].iloc[-1]
            if current > c3["close"] * 1.003: continue
            return {
                "direction": "bearish",
                "entry":     round(c3["close"], 5),
                "sl":        round(c1["high"] + c1_range * 0.10, 5),
                "ob_top":    round(c1["high"], 5),
                "ob_bottom": round(c1["open"], 5),
                "liq_pool":  round(ref_high, 5),
                "current":   round(current, 5),
            }
    return None


def analyze_morning_star(sym_name, yf_sym, tf, news, debug=False):
    if news.get("hard_block"):
        if debug: return f"{sym_name} {tf} 🌟: ❌ أخبار CPI/NFP/FOMC"
        return None
    df = get_candles(yf_sym, tf)
    if df.empty or len(df) < 60:
        if debug: return f"{sym_name} {tf} 🌟: ❌ بيانات فاضية"
        return None
    df_h4    = get_candles(yf_sym, "4h", 50)
    h4_trend = detect_trend_structure(df_h4) if not df_h4.empty else "neutral"
    if h4_trend == "neutral":
        if debug: return f"{sym_name} {tf} 🌟: ❌ H4 محايد"
        return None
    h4_of = detect_order_flow(df_h4, h4_trend) if not df_h4.empty else 0.0
    ms = detect_morning_star(df, h4_trend)
    if not ms:
        if debug: return f"{sym_name} {tf} 🌟: ❌ ما في نمط Morning Star"
        return None
    df_d        = get_candles(yf_sym, "1d", 30)
    daily_of    = detect_order_flow(df_d, h4_trend) if not df_d.empty else 0.0
    daily_trend = detect_trend_structure(df_d)       if not df_d.empty else "neutral"
    daily_match = daily_trend == h4_trend
    has_bsl, bsl_level = check_bsl_ssl(df, h4_trend)

    quality = 40  # base: نمط اكتمل
    quality += 20  # سحب سيولة — شرط النمط دايماً موجود
    if h4_of >= 0.7:   quality += 15
    elif h4_of >= 0.5: quality += 8
    if daily_match:    quality += 10
    if has_bsl:        quality += 15
    quality = max(0, min(100, quality))

    if quality < 70:
        if debug: return f"{sym_name} {tf} 🌟: ❌ جودة {quality}%"
        return None

    entry = ms["entry"]
    sl    = ms["sl"]
    risk  = abs(entry - sl)
    if risk <= 0: return None

    if h4_trend == "bullish":
        tp1 = round(entry + risk * 2.0, 5)
        tp2 = round(entry + risk * 4.0, 5)
    else:
        tp1 = round(entry - risk * 2.0, 5)
        tp2 = round(entry - risk * 4.0, 5)

    return {
        "symbol": sym_name, "tf": tf, "trend": h4_trend,
        "strategy": "morning_star",
        "current": ms["current"], "entry": entry, "sl": sl,
        "tp1": tp1, "tp2": tp2,
        "ob": {"top": ms["ob_top"], "bottom": ms["ob_bottom"],
               "candle_high": ms["ob_top"], "candle_low": ms["ob_bottom"]},
        "in_ob":       ms["ob_bottom"] <= ms["current"] <= ms["ob_top"],
        "liq_pool":    ms["liq_pool"],
        "has_bsl":     has_bsl, "bsl_level": bsl_level,
        "h4_of":       h4_of, "liq_sweep": True,
        "daily_match": daily_match, "daily_of": daily_of,
        "has_pdh_pdl": False, "has_lwh_lwl": False, "weekly_match": False,
        "quality":     quality, "news": news,
        "idm_type": "", "idm_wick": 0, "daily_trend": daily_trend,
    }


# ============================================================
# ===== رسائل السيتاب =====
# ============================================================

def morning_star_msg(a):
    direction   = "شراء 📈" if a["trend"] == "bullish" else "بيع 📉"
    risk, label = get_risk_new(a["quality"])
    risk_txt    = f"❌ ما ندخل — {label}" if risk == 0 else f"💰 مخاطرة: {risk}% — {label}"
    tv          = TRADINGVIEW_LINKS.get(a["symbol"], "https://www.tradingview.com")
    quality_bar = "█" * (a["quality"] // 20) + "░" * (5 - a["quality"] // 20)
    extras = []
    if a.get("liq_sweep"):           extras.append("✅ سحب سيولة قبل النمط (+20)")
    if a.get("h4_of", 0) >= 0.7:    extras.append("✅ H4 order flow قوي (+15)")
    elif a.get("h4_of", 0) >= 0.5:  extras.append("✅ H4 order flow متوسط (+8)")
    if a.get("daily_match"):         extras.append("✅ Daily ترند يدعم (+10)")
    if a.get("has_bsl"):             extras.append(f"✅ BSL/SSL عند {a.get('bsl_level','')} (+15)")
    news_txt = ""
    if a["news"]["has_news"]:
        news_txt = "⚠️ أخبار مهمة قريبة!\n"
        for ev in a["news"]["events"]:
            news_txt += f"  • {ev['title']} بعد {ev['hours']}س\n"
    sep  = "─────────────────"
    msg  = f"🌟 Morning Star | {direction} | {a['symbol']} | {a['tf']}\n{sep}\n"
    if extras: msg += "\n".join(extras) + "\n"
    if a.get("liq_pool"): msg += f"🎯 Liq Pool عند: {a['liq_pool']}\n"
    msg += news_txt
    msg += f"{sep}\n⚡ نمط اكتمل — ادخلي الحين!\n"
    msg += f"📌 دخول Market عند: {a['entry']}\n"
    msg += f"🛑 ستوب: {a['sl']}\n"
    msg += f"✅ هدف 1: {a['tp1']}  (1:2)\n🚀 هدف 2: {a['tp2']}  (1:4)\n"
    msg += f"السعر الحالي: {round(a['current'], 4)}\n"
    msg += f"منطقة OB: {round(a['ob']['bottom'],4)} — {round(a['ob']['top'],4)}\n"
    msg += f"{sep}\nجودة: {a['quality']}/100  {quality_bar}\n{risk_txt}\n📈 {tv}\nالقرار إلك 💪"
    return msg


def setup_msg(a):
    direction   = "شراء 📈" if a["trend"] == "bullish" else "بيع 📉"
    arrow       = "🟢" if a["trend"] == "bullish" else "🔴"
    risk, label = get_risk_new(a["quality"])
    risk_txt    = f"❌ ما ندخل — {label}" if risk == 0 else f"💰 مخاطرة: {risk}% — {label}"
    tv          = TRADINGVIEW_LINKS.get(a["symbol"], "https://www.tradingview.com")
    quality_bar = "█" * (a["quality"] // 20) + "░" * (5 - a["quality"] // 20)
    extras = []
    if a.get("liq_sweep"):    extras.append("✅ سحب سيولة قبل DBOS")
    if a.get("has_bsl"):      extras.append(f"💧 BSL عند {a.get('bsl_level','')}")
    if a.get("has_pdh_pdl"):  extras.append("📅 PDH/PDL في الاتجاه")
    if a.get("has_lwh_lwl"):  extras.append("📆 LWH/LWL في الاتجاه")
    if a.get("weekly_match"): extras.append("✅ أسبوعي يدعم")
    if a.get("monthly_match"):extras.append("🌙 شهري يدعم")
    news_txt = ""
    if a["news"]["has_news"]:
        news_txt = "⚠️ أخبار مهمة قريبة!\n"
        for ev in a["news"]["events"]:
            news_txt += f"  • {ev['title']} بعد {ev['hours']}س\n"
    action_header = "⚡ وصل الـ OB — ادخلي الحين!" if a["in_ob"] else "⏳ ما وصل بعد — حطي ليمت أوردر"
    order_type    = "دخول فوري (Market)" if a["in_ob"] else f"ليمت أوردر عند: {a['entry']}"
    sep  = "─────────────────"
    msg  = f"{arrow} {direction} | {a['symbol']} | {a['tf']}\n{sep}\n"
    if extras: msg += "\n".join(extras) + "\n"
    msg += news_txt
    msg += f"{sep}\n{action_header}\n📌 {order_type}\n"
    msg += f"🛑 ستوب:  {a['sl']}  (تحت/فوق شمعة OB)\n"
    msg += f"✅ هدف 1: {a['tp1']}  (1:2)\n🚀 هدف 2: {a['tp2']}  (1:4)\n"
    msg += f"السعر الحالي: {round(a['current'], 4)}\n"
    msg += f"منطقة OB:    {round(a['ob']['bottom'],4)} — {round(a['ob']['top'],4)}\n"
    msg += f"{sep}\nجودة: {a['quality']}/100  {quality_bar}\n{risk_txt}\n📈 {tv}\nالقرار إلك 💪"
    return msg


# ============================================================
# ===== الأوزان =====
# ============================================================

def update_weights_win(analysis):
    weights = load_weights()
    boost = 0.20
    if analysis.get("liq_sweep"):        weights["liq_sweep"] = min(3.0, weights["liq_sweep"] + boost)
    if analysis.get("h4_of",0) >= 0.7:  weights["h4_of"]     = min(3.0, weights["h4_of"]    + boost)
    if analysis.get("daily_of",0)>=0.6: weights["daily_of"]  = min(3.0, weights["daily_of"] + boost)
    if analysis.get("has_bsl"):          weights["bsl_ssl"]   = min(3.0, weights["bsl_ssl"]  + boost)
    if analysis.get("has_pdh_pdl"):      weights["pdh_pdl"]   = min(3.0, weights["pdh_pdl"]  + boost)
    if analysis.get("has_lwh_lwl"):      weights["lwh_lwl"]   = min(3.0, weights["lwh_lwl"]  + boost)
    save_weights(weights)

def update_weights_loss(analysis):
    weights = load_weights()
    penalty = 0.10
    if analysis.get("liq_sweep"): weights["liq_sweep"] = max(0.1, weights["liq_sweep"] - penalty)
    if analysis.get("has_bsl"):   weights["bsl_ssl"]   = max(0.1, weights["bsl_ssl"]  - penalty)
    save_weights(weights)


# ============================================================
# ===== جورنال وأزرار =====
# ============================================================

async def send_setup_with_buttons(bot, a, custom_msg=None):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    global TRADE_COUNTER
    TRADE_COUNTER[0] += 1
    trade_id = str(TRADE_COUNTER[0])
    JOURNAL[trade_id] = {
        "symbol": a["symbol"], "tf": a["tf"], "direction": a["trend"],
        "entry": a["entry"], "sl": a["sl"], "tp1": a["tp1"], "tp2": a["tp2"],
        "yf_sym": SYMBOLS.get(a["symbol"], ""),
        "risk": 0, "status": "pending", "result_r": None,
        "timestamp": datetime.now(RIYADH_TZ).strftime("%Y-%m-%d %H:%M"),
        "quality": a.get("quality", 0), "h4_of": a.get("h4_of", 0),
        "liq_sweep": a.get("liq_sweep", False), "has_bsl": a.get("has_bsl", False),
        "has_pdh_pdl": a.get("has_pdh_pdl", False), "has_lwh_lwl": a.get("has_lwh_lwl", False),
        "analysis": a,
    }
    journal_add(trade_id, JOURNAL[trade_id])
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ دخلت",    callback_data=f"entered_{trade_id}"),
        InlineKeyboardButton("❌ ما دخلت", callback_data=f"skipped_{trade_id}"),
    ]])
    msg_text = custom_msg if custom_msg else (
        morning_star_msg(a) if a.get("strategy") == "morning_star" else setup_msg(a)
    )
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
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("0.5%",  callback_data=f"risk_{trade_id}_0.5"),
            InlineKeyboardButton("0.75%", callback_data=f"risk_{trade_id}_0.75"),
            InlineKeyboardButton("1%",    callback_data=f"risk_{trade_id}_1.0"),
        ]])
        await query.edit_message_reply_markup(reply_markup=keyboard)
        await context.bot.send_message(chat_id=CHAT_ID, text="✅ دخلتِ الصفقة! كم المخاطرة؟")

    elif data.startswith("risk_"):
        parts = data.split("_")
        trade_id, risk = parts[1], float(parts[2])
        if trade_id in JOURNAL:
            JOURNAL[trade_id]["risk"]   = risk
            JOURNAL[trade_id]["status"] = "active"
            ACCOUNT["trades_week"]  += 1
            ACCOUNT["trades_today"] += 1
            DAILY_RISK["trades_entered_today"] += 1
            journal_set_status(trade_id, "active", risk)
            save_data()
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(
                chat_id=CHAT_ID,
                text=f"📌 مسجلة! {JOURNAL[trade_id]['symbol']} | مخاطرة: {risk}%\n"
                     "أراقبها وأخبرك لما تصل الهدف أو الستوب 👀"
            )

    elif data.startswith("skipped_"):
        trade_id = data.split("_")[1]
        if trade_id in JOURNAL:
            JOURNAL[trade_id]["status"] = "skipped"
            journal_set_status(trade_id, "skipped")
        await query.edit_message_reply_markup(reply_markup=None)

    elif data.startswith("result_"):
        parts = data.split("_")
        trade_id, result = parts[1], parts[2]
        if trade_id in JOURNAL:
            t = JOURNAL[trade_id]
            if result == "tp1":
                t["result_r"] = 2.0; t["status"] = "closed"
                update_weights_win(t.get("analysis", {}))
                journal_set_result(trade_id, 2.0)
                msg = f"✅ هدف 1 وصل! +2R على {t['symbol']} 🎯"
            elif result == "tp2":
                t["result_r"] = 4.0; t["status"] = "closed"
                update_weights_win(t.get("analysis", {}))
                journal_set_result(trade_id, 4.0)
                msg = f"🚀 هدف 2 وصل! +4R على {t['symbol']} 🔥"
            else:
                t["result_r"] = -1.0; t["status"] = "closed"
                DAILY_RISK["loss_today"] = True
                update_weights_loss(t.get("analysis", {}))
                journal_set_result(trade_id, -1.0)
                save_data()
                msg = f"🔴 ستوب على {t['symbol']} | -1R\n🛑 وقف التداول لباقي اليوم — الغد إن شاء الله 💪"
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(chat_id=CHAT_ID, text=msg)


# ============================================================
# ===== مراقبة الصفقات =====
# ============================================================

# تتبع التنبيهات عشان ما يكرر نفس الرسالة
OB_ALERTS_SENT   = set()   # trade_id اللي تم تنبيه OB قربها
TP1_TRAIL_SENT   = set()   # trade_id اللي تم إرسال تذكير Trailing

async def monitor_trades(bot):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    # ===== 1. مراقبة الصفقات المفتوحة =====
    active = {k: v for k, v in JOURNAL.items() if v["status"] == "active"}
    for trade_id, t in active.items():
        try:
            yf_sym = t["yf_sym"]
            if not yf_sym: continue
            df = get_candles(yf_sym, "1h", 5)
            if df.empty: continue
            current   = df["close"].iloc[-1]
            direction = t["direction"]

            hit_tp2 = (direction=="bullish" and current>=t["tp2"]) or (direction=="bearish" and current<=t["tp2"])
            hit_tp1 = (direction=="bullish" and current>=t["tp1"]) or (direction=="bearish" and current<=t["tp1"])
            hit_sl  = (direction=="bullish" and current<=t["sl"])  or (direction=="bearish" and current>=t["sl"])

            if hit_tp2:
                kb = InlineKeyboardMarkup([[InlineKeyboardButton("✅ أكدي TP2", callback_data=f"result_{trade_id}_tp2")]])
                await bot.send_message(chat_id=CHAT_ID, text=f"🚀 يبدو وصل هدف 2 على {t['symbol']}!", reply_markup=kb)

            elif hit_tp1:
                # Trailing Stop تذكير — مرة وحدة فقط
                if trade_id not in TP1_TRAIL_SENT:
                    TP1_TRAIL_SENT.add(trade_id)
                    trail_sl = round(t["entry"], 5)
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text=(
                            f"✅ {t['symbol']} وصل هدف 1! (+2R) 🎯\n"
                            f"─────────────────\n"
                            f"💡 Trailing Stop: حركي الستوب لنقطة الدخول\n"
                            f"📌 الستوب الجديد: {trail_sl}\n"
                            f"🚀 خلي الصفقة تشتغل للهدف 2\n"
                            f"─────────────────\n"
                            f"وين تغلقين؟"
                        )
                    )
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("✅ TP1", callback_data=f"result_{trade_id}_tp1"),
                    InlineKeyboardButton("🚀 TP2", callback_data=f"result_{trade_id}_tp2"),
                ]])
                await bot.send_message(chat_id=CHAT_ID, text=f"وين أغلقتِ {t['symbol']}؟", reply_markup=kb)

            elif hit_sl:
                kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔴 أكدي الستوب", callback_data=f"result_{trade_id}_sl")]])
                await bot.send_message(chat_id=CHAT_ID, text=f"⚠️ يبدو لمس الستوب على {t['symbol']}!", reply_markup=kb)

        except Exception as e:
            logger.error(f"خطأ مراقبة {trade_id}: {e}")

    # ===== 2. تتبع السعر من OB للصفقات الـ pending =====
    pending = {k: v for k, v in JOURNAL.items() if v["status"] == "pending"}
    for trade_id, t in pending.items():
        try:
            if trade_id in OB_ALERTS_SENT: continue
            yf_sym = t["yf_sym"]
            if not yf_sym: continue
            df = get_candles(yf_sym, "1h", 3)
            if df.empty: continue
            current   = df["close"].iloc[-1]
            direction = t["direction"]
            entry     = t["entry"]
            ob_top    = t.get("analysis", {}).get("ob", {}).get("top", entry)
            ob_bottom = t.get("analysis", {}).get("ob", {}).get("bottom", entry)
            ob_range  = ob_top - ob_bottom if ob_top and ob_bottom else 0

            if ob_range <= 0: continue

            # السعر اقترب من OB بـ 20% من حجمه
            proximity = ob_range * 0.20
            near_ob = (
                (direction == "bullish" and ob_bottom - proximity <= current <= ob_top + proximity) or
                (direction == "bearish" and ob_bottom - proximity <= current <= ob_top + proximity)
            )

            if near_ob:
                OB_ALERTS_SENT.add(trade_id)
                arrow = "📈" if direction == "bullish" else "📉"
                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=(
                        f"⚡ السعر اقترب من OB! {arrow}\n"
                        f"─────────────────\n"
                        f"📊 {t['symbol']} {t['tf']}\n"
                        f"💰 السعر الحالي: {round(current, 4)}\n"
                        f"🎯 منطقة OB: {round(ob_bottom,4)} — {round(ob_top,4)}\n"
                        f"📌 دخولك عند: {entry}\n"
                        f"─────────────────\n"
                        f"استعدي! 👀"
                    )
                )
        except Exception as e:
            logger.error(f"خطأ تتبع OB {trade_id}: {e}")


# ============================================================
# ===== Scan =====
# ============================================================

async def scan_markets(bot):
    allowed, reason = is_trading_allowed()
    if not allowed:
        return False, reason

    news      = check_news()
    found     = []
    now_ts    = datetime.now(RIYADH_TZ)
    today_str = now_ts.strftime("%Y-%m-%d")

    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            key = f"{name}_{tf}"
            if SENT_SETUPS.get(key) == today_str: continue
            try:
                r = analyze(name, yf_sym, tf, news)
                if r:
                    r["_sent_key"] = key
                    found.append(r)
            except Exception as e:
                logger.error(f"خطأ {name} {tf}: {e}")

    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            key = f"ms_{name}_{tf}"
            if SENT_SETUPS.get(key) == today_str: continue
            try:
                r = analyze_morning_star(name, yf_sym, tf, news)
                if r:
                    r["_sent_key"] = key
                    found.append(r)
            except Exception as e:
                logger.error(f"خطأ MS {name} {tf}: {e}")

    if found:
        found.sort(key=lambda x: x["quality"], reverse=True)
        for s in found:
            await send_setup_with_buttons(bot, s)
            SENT_SETUPS[s["_sent_key"]] = today_str
            await asyncio.sleep(2)
        return True, ""
    return False, ""


async def _background_scan(bot):
    try:
        await scan_markets(bot)
    except Exception as e:
        logger.error(f"خطأ سكان خلفية: {e}")


async def _market_status_report(bot):
    """تقرير السوق كل ساعتين — وش يشوف البوت فعلاً"""
    try:
        now  = datetime.now(RIYADH_TZ)
        news = check_news()

        if news.get("hard_block"):
            await bot.send_message(
                chat_id=CHAT_ID,
                text=f"📊 تقرير السوق | {now.strftime('%H:%M')}\n"
                     f"─────────────────\n"
                     f"🚫 أخبار CPI/NFP/FOMC — البوت واقف حتى تنتهي"
            )
            return

        STATUS_ICONS = {
            "محايد":        "⚫",
            "DBOS":         "🟡",
            "IDM":          "🟠",
            "OB":           "🔵",
            "جودة":         "🟢",
            "order flow":   "🔴",
            "عكس":          "🔴",
            "بيانات":       "⚪",
        }

        lines   = []
        found_setup = False

        for name, yf_sym in SYMBOLS.items():
            # نجرب 4h أول ثم 1h ونأخذ أفضل نتيجة
            best_line  = None
            best_stage = 0  # كلما أعلى كلما وصل أبعد في التحليل

            for tf in ["4h", "1h"]:
                r = analyze(name, yf_sym, tf, news, debug=True)
                if isinstance(r, dict):
                    # سيتاب كامل!
                    best_line  = f"🟢 {name} {tf}: سيتاب! جودة {r['quality']}% 🔥"
                    best_stage = 99
                    found_setup = True
                    break
                elif isinstance(r, str):
                    # نحدد المرحلة اللي وصلها
                    stage = (6 if "جودة" in r
                             else 5 if "OB" in r and "ما في" not in r
                             else 4 if "IDM" in r and "ما في" not in r
                             else 3 if "DBOS" in r and "ما في" not in r
                             else 2 if "order flow" in r or "عكس" in r
                             else 1)
                    if stage > best_stage:
                        best_stage = stage
                        # أيقونة بناء على المرحلة
                        icon = ("🔵" if stage >= 4
                                else "🟡" if stage == 3
                                else "🔴" if stage == 2
                                else "⚫")
                        # نظّف النص
                        clean = r.split(": ", 1)[-1] if ": " in r else r
                        best_line = f"{icon} {name}: {clean}"

            if best_line:
                lines.append(best_line)

        sep = "─────────────────"
        msg = f"📊 تقرير السوق | {now.strftime('%H:%M')} الرياض\n{sep}\n"
        msg += "\n".join(lines) + f"\n{sep}\n"

        if found_setup:
            msg += "⚡ فيه فرص — شوفي الإشارات فوق!"
        else:
            near  = sum(1 for l in lines if "🔵" in l or "🟡" in l)
            quiet = sum(1 for l in lines if "⚫" in l)
            if near:
                msg += f"⏳ {near} زوج وصل مراحل متقدمة — قريب!"
            else:
                msg += f"😴 السوق هادي ({quiet} زوج محايد)\nما في سيتاب بشروطنا الحين"

        if news["has_news"]:
            msg += "\n⚠️ " + " | ".join(ev["title"] for ev in news["events"][:2])

        await bot.send_message(chat_id=CHAT_ID, text=msg)

    except Exception as e:
        logger.error(f"خطأ تقرير السوق: {e}")


# ============================================================
# ===== رسائل الحساب =====
# ============================================================

def personal_stats_msg():
    """إحصائيات شخصية مفصلة من الجورنال"""
    closed = [t for t in JOURNAL.values() if t["status"] == "closed" and t.get("result_r") is not None]
    if len(closed) < 3:
        return "ما في بيانات كافية بعد 📊\nادخلي على الأقل 3 صفقات مغلقة وأعطيك إحصائياتك 💪"

    wins     = [t for t in closed if t["result_r"] > 0]
    losses   = [t for t in closed if t["result_r"] < 0]
    total_r  = round(sum(t["result_r"] for t in closed), 1)
    win_rate = round(len(wins)/len(closed)*100)
    avg_win  = round(sum(t["result_r"] for t in wins)/len(wins), 2)   if wins   else 0
    avg_loss = round(sum(t["result_r"] for t in losses)/len(losses), 2) if losses else 0

    # أفضل/أسوأ زوج
    by_symbol = {}
    for t in closed:
        by_symbol.setdefault(t["symbol"], []).append(t["result_r"])
    sym_stats = {s: round(sum(v),1) for s,v in by_symbol.items()}
    best_sym  = max(sym_stats, key=sym_stats.get)
    worst_sym = min(sym_stats, key=sym_stats.get)

    # أفضل/أسوأ يوم
    days_ar = {0:"الاثنين",1:"الثلاثاء",2:"الأربعاء",3:"الخميس",4:"الجمعة",5:"السبت",6:"الأحد"}
    by_day  = {}
    for t in closed:
        try:
            day = days_ar[datetime.strptime(t["timestamp"][:10], "%Y-%m-%d").weekday()]
            by_day.setdefault(day, []).append(t["result_r"])
        except: pass
    day_stats = {d: round(sum(v),1) for d,v in by_day.items()}
    best_day  = max(day_stats, key=day_stats.get) if day_stats else "-"
    worst_day = min(day_stats, key=day_stats.get) if day_stats else "-"

    # أفضل تايم فريم
    by_tf   = {}
    for t in closed:
        by_tf.setdefault(t.get("tf","?"), []).append(t["result_r"])
    tf_stats = {tf: round(sum(v)/len(v),2) for tf,v in by_tf.items()}
    best_tf  = max(tf_stats, key=tf_stats.get) if tf_stats else "-"

    # متوسط جودة رابحة vs خاسرة
    avg_q_win  = round(sum(t.get("quality",0) for t in wins)/len(wins))   if wins   else 0
    avg_q_loss = round(sum(t.get("quality",0) for t in losses)/len(losses)) if losses else 0

    sep = "─────────────────"
    msg  = f"📈 إحصائياتك الشخصية\n{sep}\n"
    msg += f"إجمالي: {len(closed)} | ✅{len(wins)} ربح | 🔴{len(losses)} خسارة\n"
    msg += f"نسبة الفوز: {win_rate}%\n"
    msg += f"مجموع R: {'+' if total_r>=0 else ''}{total_r}R\n"
    msg += f"متوسط ربح: +{avg_win}R | متوسط خسارة: {avg_loss}R\n"
    msg += f"{sep}\n"
    msg += f"🏆 أفضل زوج:  {best_sym}  ({sym_stats[best_sym]:+}R)\n"
    msg += f"💀 أسوأ زوج:  {worst_sym} ({sym_stats[worst_sym]:+}R)\n"
    msg += f"📅 أفضل يوم:  {best_day}  ({day_stats.get(best_day,0):+}R)\n"
    msg += f"😓 أصعب يوم: {worst_day} ({day_stats.get(worst_day,0):+}R)\n"
    msg += f"⏱ أفضل TF:   {best_tf}  (متوسط {tf_stats.get(best_tf,0):+}R)\n"
    msg += f"{sep}\n"
    msg += f"متوسط جودة الرابحة: {avg_q_win}%\n"
    msg += f"متوسط جودة الخاسرة: {avg_q_loss}%\n"
    if avg_q_win > avg_q_loss + 5:
        msg += "✅ الجودة العالية تفرق — واصلي الانتظار 💪\n"
    msg += sep
    return msg


def calc_auto_drawdown():
    original = ACCOUNT.get("balance", 0)
    current  = ACCOUNT.get("current_balance", 0)
    if original <= 0: return
    diff = original - current
    ACCOUNT["drawdown_used"] = round(max(0, diff / original * 100), 2)
    ACCOUNT["pnl_percent"]   = round((current - original) / original * 100, 2)


def daily_advice_msg():
    dd              = ACCOUNT["drawdown_used"]
    max_dd          = ACCOUNT["max_drawdown"]
    remaining_max   = max_dd - dd
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    pnl    = ACCOUNT["pnl_percent"]
    trades = ACCOUNT["trades_week"]
    phase_label = {"challenge":"Challenge 🔴","verification":"Verification 🟡","funded":"Funded 🟢"}.get(ACCOUNT["phase"],"")
    pnl_txt = (f"رابح {pnl}%، واصلي 🌟" if pnl > 3
               else f"رابح {pnl}%، شغل كويس 👍" if pnl > 0
               else "عند نقطة البداية 🎯" if pnl == 0
               else f"خسارة {abs(pnl)}%، حمي الحساب ❗")
    dd_pct = dd / max_dd * 100 if max_dd > 0 else 0
    dd_txt = (f"باقي {remaining_max:.1f}% الحمدلله ✅" if dd_pct < 50
              else f"باقي {remaining_max:.1f}% — تعاملي بحذر 🟡" if dd_pct < 80
              else f"باقي {remaining_max:.1f}% فقط! 🔴")
    msg  = f"صباح الخير ☀️\n─────────────────\n"
    msg += f"🏢 {ACCOUNT['firm_name']} | {phase_label}\n"
    msg += f"💰 ${ACCOUNT['current_balance']:,.0f}\n─────────────────\n"
    msg += f"الحساب: {pnl_txt}\nدروداون كلي: {dd_txt}\n"
    msg += f"دروداون يومي: باقي {remaining_daily:.1f}%\nصفقات الأسبوع: {trades}\n"
    msg += f"─────────────────\n{random.choice(DAILY_TIPS)}\nوفقك الله 🤍"
    return msg


def status_msg():
    now             = datetime.now(RIYADH_TZ)
    remaining_max   = ACCOUNT["max_drawdown"]   - ACCOUNT["drawdown_used"]
    remaining_daily = ACCOUNT["daily_drawdown"] - ACCOUNT["daily_used"]
    pnl  = ACCOUNT["pnl_percent"]
    icon = "🟢" if pnl >= 0 and remaining_max > 5 else "🟡" if remaining_max > 2 else "🔴"
    phase_label = {"challenge":"Challenge","verification":"Verification","funded":"Funded"}.get(ACCOUNT["phase"],"")
    msg  = f"{icon} حالة الحساب | {now.strftime('%H:%M')} الرياض\n─────────────────\n"
    msg += f"🏢 {ACCOUNT['firm_name']} | {phase_label}\n💰 ${ACCOUNT['current_balance']:,.0f}\n"
    msg += f"📊 PnL: {'+' if pnl>=0 else ''}{pnl}%\n"
    msg += f"📉 دروداون كلي: {ACCOUNT['drawdown_used']}% (باقي {remaining_max:.1f}%)\n"
    msg += f"📅 دروداون يومي: {ACCOUNT['daily_used']}% (باقي {remaining_daily:.1f}%)\n"
    msg += f"🔢 صفقات اليوم: {ACCOUNT['trades_today']} | الأسبوع: {ACCOUNT['trades_week']}"
    return msg


def challenge_progress_msg():
    phase         = ACCOUNT["phase"]
    pnl           = ACCOUNT["pnl_percent"]
    target        = PHASE_TARGETS.get(phase, {}).get("target", 0)
    remaining_max = ACCOUNT["max_drawdown"] - ACCOUNT["drawdown_used"]
    phase_label   = {"challenge":"Challenge","verification":"Verification","funded":"Funded"}.get(phase,"")
    if target:
        progress  = max(0, min(100, round(pnl / target * 100)))
        bar       = "█"*(progress//20) + "░"*(5-progress//20)
        target_txt= f"الهدف: {target}% | وصلت: {pnl}%\n{bar} {progress}%"
    else:
        target_txt= f"حساب ممول | ربح: {pnl}%"
    msg  = f"📊 {phase_label} Progress\n─────────────────\n{target_txt}\n"
    msg += f"دروداون باقي: {remaining_max:.1f}%\n"
    msg += ("✅ حققتِ الهدف! انتقلي للمرحلة التالية 🎉" if target and pnl >= target
            else "⚠️ دروداون ضيق، تعاملي بحذر" if remaining_max < 3
            else "واصلي 💪")
    return msg


def week_report_msg():
    """تقرير الأسبوع الحالي مرتب بالأيام"""
    days_ar = {0:"الاثنين",1:"الثلاثاء",2:"الأربعاء",3:"الخميس",4:"الجمعة",5:"السبت",6:"الأحد"}
    by_day  = {}
    for t in JOURNAL.values():
        ts = t.get("timestamp", "")
        try:
            day_name = days_ar.get(datetime.strptime(ts[:10], "%Y-%m-%d").weekday(), ts[:10])
        except:
            day_name = ts[:10] if ts else "غير محدد"
        by_day.setdefault(day_name, []).append(t)

    closed  = [t for t in JOURNAL.values() if t["status"] == "closed"]
    active  = [t for t in JOURNAL.values() if t["status"] == "active"]
    wins    = [t for t in closed if (t.get("result_r") or 0) > 0]
    losses  = [t for t in closed if (t.get("result_r") or 0) < 0]
    total_r = round(sum(t["result_r"] for t in closed if t.get("result_r")), 1) if closed else 0
    win_rate= round(len(wins)/len(closed)*100) if closed else 0

    msg = "📅 تقرير الأسبوع الحالي\n─────────────────\n"
    if not JOURNAL:
        msg += "ما في صفقات هالأسبوع بعد 📋\n"
    else:
        for day_name, trades in by_day.items():
            msg += f"\n{day_name}:\n"
            for t in trades:
                if t["status"] == "closed":
                    r    = t.get("result_r", 0)
                    icon = "✅" if r > 0 else "🔴"
                    msg += f"  {icon} {t['symbol']} {t['tf']} → {'+' if r>0 else ''}{r}R\n"
                elif t["status"] == "active":
                    msg += f"  ⏳ {t['symbol']} {t['tf']} — مفتوحة\n"
                else:
                    msg += f"  ⏭ {t['symbol']} {t['tf']} — تجاهلت\n"
        msg += f"\n─────────────────\n"
        msg += f"إجمالي: {len(closed)} | ✅{len(wins)} | 🔴{len(losses)}\n"
        if closed: msg += f"نسبة الفوز: {win_rate}% | مجموع: {'+' if total_r>=0 else ''}{total_r}R\n"
        if active: msg += f"⏳ مفتوحة: {len(active)}\n"

    dd_used   = ACCOUNT["drawdown_used"]
    remaining = ACCOUNT["max_drawdown"] - dd_used
    msg += f"📉 دروداون مستخدم: {dd_used}% | باقي: {remaining:.1f}%\n─────────────────\n"
    msg += ("الأسبوع بدأ، واصلي بالصبر 🎯" if not closed
            else "أسبوع ممتاز! فكري تأمني الأرباح 🌟" if total_r >= 6
            else "أسبوع كويس 💪 واصلي بنفس الانضباط" if total_r >= 3
            else "أسبوع متعادل، راجعي إيش تحسنيه 🧠" if total_r >= 0
            else "⚠️ خسارتين — خذي استراحة وراجعي الجورنال" if len(losses) >= 2
            else "أسبوع صعب، المهم حمايتِ الحساب 🛡️")
    return msg


def weekly_report_msg():
    """تقرير نهاية الأسبوع — يُرسل تلقائياً الجمعة ويصفّر الجورنال"""
    closed  = [t for t in JOURNAL.values() if t["status"] == "closed"]
    skipped = [t for t in JOURNAL.values() if t["status"] == "skipped"]
    active  = [t for t in JOURNAL.values() if t["status"] == "active"]
    if not closed and not active:
        return "ما في صفقات مسجلة هالأسبوع 📋\nبداية الأسبوع الجاي إن شاء الله 💪"
    wins    = [t for t in closed if (t.get("result_r") or 0) > 0]
    losses  = [t for t in closed if (t.get("result_r") or 0) < 0]
    win_rate= round(len(wins)/len(closed)*100) if closed else 0
    total_r = round(sum(t["result_r"] for t in closed if t.get("result_r")), 1)
    msg  = "📊 تقرير الأسبوع\n─────────────────\n"
    msg += f"إجمالي: {len(closed)} | ✅ {len(wins)} | 🔴 {len(losses)}\n"
    msg += f"📈 نسبة الفوز: {win_rate}%\n💰 مجموع R: {'+' if total_r>=0 else ''}{total_r}R\n"
    if skipped: msg += f"⏭ تجاهلت: {len(skipped)}\n"
    if active:  msg += f"⏳ مفتوحة: {len(active)}\n"
    msg += "─────────────────\n"
    for t in closed:
        r    = t.get("result_r") or 0
        icon = "✅" if r > 0 else "🔴"
        msg += f"{icon} {t['symbol']} {t['tf']} → {'+' if r>0 else ''}{r}R\n"
    msg += "─────────────────\n"
    msg += ("أسبوع ممتاز 🌟" if total_r >= 4 else "أسبوع كويس 💪" if total_r >= 0
            else "أسبوع صعب، راجعي الجورنال 🧠")
    stats_add_week(JOURNAL)
    JOURNAL.clear()
    return msg


# ============================================================
# ===== الأوامر =====
# ============================================================

async def start_cmd(update, context):
    if not ACCOUNT.get("setup_done") or not ACCOUNT.get("firm_name"):
        await update.message.reply_text("أهلاً! 👋 خليني أتعرف على حسابك أول\nاكتبي /setup لنبدأ")
    else:
        phase_label = {"challenge":"Challenge 🔴","verification":"Verification 🟡","funded":"Funded 🟢"}.get(ACCOUNT["phase"],"")
        await update.message.reply_text(
            f"أهلاً! 🌟\n─────────────────\n"
            f"🏢 {ACCOUNT['firm_name']} | {phase_label}\n💰 ${ACCOUNT['current_balance']:,.0f}\n"
            f"─────────────────\n"
            f"/scan     فحص فوري\n"
            f"/advice   نصايح اليوم\n"
            f"/status   حالة الحساب\n"
            f"/progress تقدم الـ Challenge\n"
            f"/week     تقرير الأسبوع\n"
            f"/stats    إحصائياتك الشخصية\n"
            f"/update   تحديث الحساب\n"
            f"/debug    تشخيص السوق"
        )

async def scan_cmd(update, context):
    msg = await update.message.reply_text(random.choice(WAITING_MSGS))
    asyncio.create_task(_do_scan_reply(context.bot, msg))

async def _do_scan_reply(bot, original_msg):
    try:
        found, reason = await scan_markets(bot)
        if found:
            await original_msg.edit_text("✅ الفحص انتهى — شوفي الإشارات فوق 👆")
        elif reason:
            await original_msg.edit_text(f"🛑 الفحص انتهى\n{reason}")
        else:
            await original_msg.edit_text(random.choice(NO_SETUP_MSGS))
    except Exception as e:
        await original_msg.edit_text(f"⚠️ خطأ في الفحص: {str(e)[:60]}")

async def stats_cmd(update, context):
    await update.message.reply_text(personal_stats_msg())

async def advice_cmd(update, context):
    await update.message.reply_text(daily_advice_msg())

async def status_cmd(update, context):
    await update.message.reply_text(status_msg())

async def progress_cmd(update, context):
    await update.message.reply_text(challenge_progress_msg())

async def week_cmd(update, context):
    await update.message.reply_text(week_report_msg())

async def journal_cmd(update, context):
    await update.message.reply_text(weekly_report_msg())

async def debug_cmd(update, context):
    news = check_news()
    msg  = "🔵 DBOS + IDM + OB:\n─────────────────\n"
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                r = analyze(name, yf_sym, tf, news, debug=True)
                if isinstance(r, str):   msg += f"{r}\n"
                elif isinstance(r, dict):msg += f"✅ {name} {tf}: جودة {r['quality']}%\n"
                else:                    msg += f"❌ {name} {tf}: ما في سيتاب\n"
            except Exception as e:
                msg += f"⚠️ {name} {tf}: {str(e)[:40]}\n"
    msg += "\n🌟 Morning Star:\n─────────────────\n"
    for name, yf_sym in SYMBOLS.items():
        for tf in ["4h", "1h"]:
            try:
                r = analyze_morning_star(name, yf_sym, tf, news, debug=True)
                if isinstance(r, str):   msg += f"{r}\n"
                elif isinstance(r, dict):msg += f"✅ {name} {tf} 🌟: جودة {r['quality']}%\n"
                else:                    msg += f"❌ {name} {tf} 🌟: ما في نمط\n"
            except Exception as e:
                msg += f"⚠️ {name} {tf} 🌟: {str(e)[:40]}\n"
    await update.message.reply_text(msg)


# ============================================================
# ===== الحلقة الرئيسية =====
# ============================================================

async def trading_loop(bot):
    load_data()
    calc_auto_drawdown()

    CMDS = (
        "─────────────────\n"
        "/scan     فحص فوري\n"
        "/advice   نصايح اليوم\n"
        "/status   حالة الحساب\n"
        "/week     تقرير الأسبوع\n"
        "/stats    إحصائياتك الشخصية\n"
        "/update   تحديث الحساب"
    )
    if ACCOUNT.get("setup_done") and ACCOUNT.get("firm_name"):
        phase_label   = {"challenge":"Challenge 🔴","verification":"Verification 🟡","funded":"Funded 🟢"}.get(ACCOUNT["phase"],"")
        dd_used       = ACCOUNT["drawdown_used"]
        remaining_max = ACCOUNT["max_drawdown"] - dd_used
        await bot.send_message(
            chat_id=CHAT_ID,
            text=(
                f"✅ بوتك اشتغل يا شذا ✅\n"
                f"─────────────────\n"
                f"🏢 {ACCOUNT['firm_name']} | {phase_label}\n"
                f"💰 ${ACCOUNT['current_balance']:,.0f} | PnL: {'+' if ACCOUNT['pnl_percent']>=0 else ''}{ACCOUNT['pnl_percent']}%\n"
                f"📉 دروداون مستخدم: {dd_used}% | باقي: {remaining_max:.1f}%\n"
                f"{CMDS}"
            )
        )
    else:
        await bot.send_message(
            chat_id=CHAT_ID,
            text=(
                f"✅ بوتك اشتغل يا شذا ✅\n"
                f"─────────────────\n"
                f"ما عندي بيانات حسابك بعد 📋\n"
                f"اكتبي /setup عشان نبدأ 👇\n"
                f"{CMDS}"
            )
        )

    last_advice_day = None
    last_warn_hour  = -1
    last_scan_slot  = -1
    last_report_day = None

    while True:
        try:
            now       = datetime.now(RIYADH_TZ)
            today     = now.date()
            today_str = today.strftime("%Y-%m-%d")

            # ===== صباح جديد =====
            if now.hour == 8 and now.minute < 5 and last_advice_day != today:
                ACCOUNT["daily_used"]    = 0.0
                ACCOUNT["trades_today"]  = 0
                DAILY_RISK["loss_today"] = False
                DAILY_RISK["trades_entered_today"] = 0
                save_data()
                await bot.send_message(chat_id=CHAT_ID, text=daily_advice_msg())
                await asyncio.sleep(1)
                await bot.send_message(chat_id=CHAT_ID, text=challenge_progress_msg())
                # لو في أخبار كبيرة اليوم — نرسل رسالة كوميدية
                morning_news = check_news()
                if morning_news.get("hard_block") and morning_news.get("events"):
                    await asyncio.sleep(2)
                    await bot.send_message(chat_id=CHAT_ID, text=news_comedy_msg(morning_news["events"]))
                last_advice_day = today

            # ===== تقرير الأسبوع — الجمعة من 8 مساءً =====
            if now.weekday() == 4 and now.hour >= 20 and last_report_day != today:
                await bot.send_message(chat_id=CHAT_ID, text=weekly_report_msg())
                last_report_day = today

            # ===== تحذير صفقتين — كل 4 ساعات =====
            if DAILY_RISK["trades_entered_today"] >= 2:
                if now.hour != last_warn_hour and now.hour % 4 == 0:
                    await bot.send_message(
                        chat_id=CHAT_ID,
                        text="⚠️ تنبيه: دخلتِ صفقتين اليوم\nانتبهي من الدروداون اليومي 🛡️\nما في إشارات جديدة حتى الغد"
                    )
                    last_warn_hour = now.hour

            # ===== تقرير السوق كل ساعتين =====
            if now.hour % 2 == 0 and now.minute < 2:
                if not hasattr(trading_loop, 'last_mkt_report') or trading_loop.last_mkt_report != now.hour:
                    asyncio.create_task(_market_status_report(bot))
                    trading_loop.last_mkt_report = now.hour

            # ===== سكان تلقائي كل 30 دقيقة =====
            current_slot = now.hour * 2 + (1 if now.minute >= 30 else 0)
            if current_slot != last_scan_slot:
                last_scan_slot = current_slot
                asyncio.create_task(_background_scan(bot))

            await monitor_trades(bot)
            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"خطأ في الحلقة: {e}")
            await asyncio.sleep(60)


# ============================================================
# ===== main =====
# ============================================================

async def main():
    setup_sheets()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    setup_conv = ConversationHandler(
        entry_points=[CommandHandler("setup", setup_start)],
        states={
            S_FIRM:     [MessageHandler(filters.TEXT & ~filters.COMMAND, got_firm)],
            S_PHASE:    [MessageHandler(filters.TEXT & ~filters.COMMAND, got_phase)],
            S_BALANCE:  [MessageHandler(filters.TEXT & ~filters.COMMAND, got_balance_setup)],
            S_MAX_DD:   [MessageHandler(filters.TEXT & ~filters.COMMAND, got_max_dd)],
            S_DAILY_DD: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_daily_dd)],
        },
        fallbacks=[CommandHandler("cancel", cancel_conv)],
    )

    update_conv = ConversationHandler(
        entry_points=[CommandHandler("update", update_start)],
        states={
            S_BALANCE:    [MessageHandler(filters.TEXT & ~filters.COMMAND, got_balance)],
            S_PNL:        [MessageHandler(filters.TEXT & ~filters.COMMAND, got_pnl)],
            S_DD_USED:    [MessageHandler(filters.TEXT & ~filters.COMMAND, got_dd_used)],
            S_DAILY_USED: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_daily_used)],
            S_TRADES_W:   [MessageHandler(filters.TEXT & ~filters.COMMAND, got_trades_w)],
            S_TRADES_D:   [MessageHandler(filters.TEXT & ~filters.COMMAND, got_trades_d)],
        },
        fallbacks=[CommandHandler("cancel", cancel_conv)],
    )

    app.add_handler(CommandHandler("start",    start_cmd))
    app.add_handler(CommandHandler("scan",     scan_cmd))
    app.add_handler(CommandHandler("advice",   advice_cmd))
    app.add_handler(CommandHandler("status",   status_cmd))
    app.add_handler(CommandHandler("progress", progress_cmd))
    app.add_handler(CommandHandler("week",     week_cmd))
    app.add_handler(CommandHandler("stats",    stats_cmd))
    app.add_handler(CommandHandler("journal",  journal_cmd))
    app.add_handler(CommandHandler("debug",    debug_cmd))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(setup_conv)
    app.add_handler(update_conv)

    bot = Bot(token=TELEGRAM_TOKEN)
    async with app:
        await app.start()
        await app.updater.start_polling()
        await trading_loop(bot)


if __name__ == "__main__":
    asyncio.run(main())
