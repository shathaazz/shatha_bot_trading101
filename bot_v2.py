
ICT/SMC - DBOS + IDM + OB
نسخة شخصية - باللهجة السعودية

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
from telegram.ext import Application, CommandHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
CHAT_ID = os.environ.get("CHAT_ID", "YOUR_CHAT_ID_HERE")
RIYADH_TZ = pytz.timezone(‘Asia/Riyadh’)

# ==================== إعدادات الحساب ====================

ACCOUNT = {
“balance”: 5000.0,
“max_drawdown”: 10.0,
“daily_drawdown”: 5.0,
“drawdown_used”: 0.0,
“daily_used”: 0.0,
“trades_week”: 0,
“pnl_percent”: 0.0,
}

SYMBOLS = {
“XAUUSD”: “GC=F”,
“XAGUSD”: “SI=F”,
“EURUSD”: “EURUSD=X”,
“GBPUSD”: “GBPUSD=X”,
“BTCUSD”: “BTC-USD”,
“USDCHF”: “USDCHF=X”,
“USDJPY”: “USDJPY=X”,
“AUDUSD”: “AUDUSD=X”,
}

HIGH_IMPACT_KEYWORDS = [
“Fed”, “Federal Reserve”, “FOMC”, “Interest Rate”,
“CPI”, “NFP”, “Non-Farm”, “GDP”, “Powell”, “ECB”, “BOE”, “BOJ”
]

# ==================== رسائل الانتظار ====================

WAITING_MSGS = [
“🔍 جالس أفحص الأسواق لك.. لحظة صبر يا بطلة”,
“👀 عيني على الشارت، لحظة وأخبرك”,
“⏳ البحث مستمر، السوق مو دايم يعطي فرص، بس أنا صاحي 💪”,
“🧐 فاحص كل زوج بعين.. لا شي يفوتني”,
]

NO_SETUP_MSGS = [
“🤷 ما لقيت سيتاب يستاهل الحين. دبر عمرك بشغلة ثانية وأنا أراقب لك 😄”,
“😴 السوق هادي الحين، ما في فرصة تستاهل. روحي اتقهوي وأنا هنا 👀”,
“🔎 فحصت كل شي، ما في سيتاب بشروطنا الحين. الصبر مفتاح، والفرص تجي 💙”,
“⏸️ السوق مو متحرك على شروطنا الحين. ما تدخلين بدون سيتاب صح، هذا اللي علمناه 😎”,
“🌙 هدوء في الأسواق الحين. استغلي الوقت تحللين أو تستريحين، وأنا أراقب 💙”,
]

STATUS_MSGS = [
“🔍 جالس أبحث لك عن سيتاب.. عيني على الشارت”,
“📊 أفحص الأزواج واحد واحد، لو في شي أنبهك فوراً”,
“👁️ صاحي ومراقب، لا تقلقين 💙”,
“⚡ شغّال بكامل طاقتي، ما شي يفوتني إن شاء الله”,
]

DAILY_TIPS = [
“ما في صفقة تستاهل تخلك تكسري خطتك. الخطة هي الملك 👑”,
“السوينق يحتاج صبر. الصفقة الصح تجيك، ما تروحين إليها 🎯”,
“الخسارة جزء من التداول. المهم إدارة المخاطرة مو الربح السريع 🧘”,
“أي ضغط داخل الصفقة؟ هذا إشارة توقفين مو تكملين ⛔”,
“الفرق بين المحترف والمبتدئ مو في الصفقات، في الانضباط 🏆”,
“اكتبي كل صفقة في الجورنال. اللي ما يوثق، ما يتعلم 📝”,
“لو حسيتِ بالثقل من السوق، خذي استراحة. الحساب أهم من الصفقة 💙”,
]

# ==================== الأخبار ====================

def check_news():
try:
r = requests.get(“https://nfs.faireconomy.media/ff_calendar_thisweek.json”, timeout=10)
if r.status_code != 200:
return {“has_news”: False, “events”: []}
now = datetime.utcnow()
upcoming = []
for ev in r.json():
try:
if ev.get(“impact”) != “High”:
continue
t = datetime.fromisoformat(ev.get(“date”,””).replace(“Z”,””))
diff = t - now
if timedelta(hours=-1) <= diff <= timedelta(hours=24):
title = ev.get(“title”,””)
if any(k.lower() in title.lower() for k in HIGH_IMPACT_KEYWORDS):
upcoming.append({
“title”: title,
“currency”: ev.get(“country”,””),
“hours”: round(diff.total_seconds()/3600, 1)
})
except:
continue
return {“has_news”: len(upcoming)>0, “events”: upcoming[:3]}
except:
return {“has_news”: False, “events”: []}

# ==================== البيانات والتحليل ====================

def get_candles(yf_sym, tf, limit=100):
try:
period = {“1h”:“7d”,“4h”:“60d”,“1d”:“180d”,“1wk”:“2y”}.get(tf,“60d”)
df = yf.Ticker(yf_sym).history(period=period, interval=tf)
df = df.rename(columns={‘Open’:‘open’,‘High’:‘high’,‘Low’:‘low’,‘Close’:‘close’})
return df.tail(limit)
except:
return pd.DataFrame()

def detect_trend(df):
if len(df) < 20:
return “neutral”
r = df.tail(20)
if r[‘high’].iloc[-1] > r[‘high’].iloc[0] and r[‘low’].iloc[-1] > r[‘low’].iloc[0]:
return “bullish”
if r[‘high’].iloc[-1] < r[‘high’].iloc[0] and r[‘low’].iloc[-1] < r[‘low’].iloc[0]:
return “bearish”
return “neutral”

def find_swings(df, lb=3):
highs, lows = [], []
for i in range(lb, len(df)-lb):
if df[‘high’].iloc[i] == df[‘high’].iloc[i-lb:i+lb+1].max():
highs.append((i, df[‘high’].iloc[i]))
if df[‘low’].iloc[i] == df[‘low’].iloc[i-lb:i+lb+1].

Shetradingg, [07/09/47 04:50 ص]
last_advice_day = today

        # رسالة الحالة كل 4 ساعات
        if now.hour % 4 == 0 and now.hour != last_status_hour and now.minute < 5:
            found = await scan_markets(bot)
            if not found:
                await bot.send_message(chat_id=CHAT_ID,
                    text=random.choice(NO_SETUP_MSGS))
            last_status_hour = now.hour
            scan_count += 1
        else:
            # فحص عادي كل ساعة بدون رسالة انتظار
            await scan_markets(bot)

        await asyncio.sleep(3600)

    except Exception as e:
        logger.error(f"خطأ: {e}")
        await asyncio.sleep(60)

# ==================== أوامر التيليغرام ====================

async def start_cmd(update, context):
await update.message.reply_text(
“🚀 أهلاً شذا!\n”
“أنا بوتك للتداول، أراقب الأسواق ٢٤/٧ 💙\n\n”
“الأوامر:\n”
“/scan - فحص فوري للأسواق\n”
“/advice - نصايح اليوم\n”
“/status - وش أسوي الحين\n”
“/update - تحديث وضع حسابك\n\n”
“ابدئي بـ /update عشان أعرف وضع حسابك 💪”
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

async def update_cmd(update, context):
“””
تحديث الحساب
مثال: /update pnl=+3.5 dd=2.5 daily=1.0 trades=2
“””
try:
args = “ “.join(context.args)
updated = []
    if "pnl=" in args:
        val = float(args.split("pnl=")[1].split()[0].replace("+",""))
        ACCOUNT['pnl_percent'] = val
        updated.append(f"PnL: {'+' if val>=0 else ''}{val}%")

    if "dd=" in args:
        val = float(args.split("dd=")[1].split()[0])
        ACCOUNT['drawdown_used'] = val
        updated.append(f"دروداون: {val}%")

    if "daily=" in args:
        val = float(args.split("daily=")[1].split()[0])
        ACCOUNT['daily_used'] = val
        updated.append(f"ديلي: {val}%")

    if "trades=" in args:
        val = int(args.split("trades=")[1].split()[0])
        ACCOUNT['trades_week'] = val
        updated.append(f"صفقات: {val}")

    if updated:
        await update.message.reply_text(
            f"✅ تم التحديث!\n" + "\n".join(updated) +
            "\n\nحسابك محفوظ عندي 💙"
        )
    else:
        await update.message.reply_text(
            "الاستخدام:\n"
            "/update pnl=+3.5 dd=2.5 daily=1.0 trades=2\n\n"
            "pnl = نسبة الربح أو الخسارة\n"
            "dd = الدروداون المستخدم\n"
            "daily = الديلي المستخدم\n"
            "trades = عدد الصفقات هالأسبوع\n\n"
            "مثال لو رابح 3.5% وعندك 2.5% دروداون:\n"
            "/update pnl=+3.5 dd=2.5 daily=0.5 trades=1"
        )
except Exception as e:
    await update.message.reply_text(
        "❌ في خطأ في البيانات\n"
        "مثال صح: /update pnl=+3.5 dd=2.5 daily=1.0 trades=2"
    )

async def main():
app = Application.builder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler(“start”, start_cmd))
app.add_handler(CommandHandler(“scan”, scan_cmd))
app.add_handler(CommandHandler(“advice”, advice_cmd))
app.add_handler(CommandHandler(“status”, status_cmd))
app.add_handler(CommandHandler(“update”, update_cmd))
bot = Bot(token=TELEGRAM_TOKEN)
async with app:
    await app.start()
    await app.updater.start_polling()
    await trading_loop(bot)

if name == “**main**”:
asyncio.run(main())
