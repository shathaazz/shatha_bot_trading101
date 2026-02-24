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
RIYADH_TZ = pytz.timezone(“Asia/Riyadh”)

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

WAITING_MSGS = [
“جالس افحص الاسواق لك، لحظة صبر يا بطلة”,
“عيني على الشارت، لحظة وأخبرك”,
“البحث مستمر، السوق مو دايم يعطي فرص، بس أنا صاحي”,
“فاحص كل زوج بعين، لا شي يفوتني”,
]

NO_SETUP_MSGS = [
“ما لقيت سيتاب يستاهل الحين. دبر عمرك بشغلة ثانية وأنا أراقب لك”,
“السوق هادي الحين، ما في فرصة تستاهل. روحي اتقهوي وأنا هنا”,
“فحصت كل شي، ما في سيتاب بشروطنا الحين. الصبر مفتاح، والفرص تجي”,
“السوق مو متحرك على شروطنا الحين. ما تدخلين بدون سيتاب صح”,
“هدوء في الاسواق الحين. استغلي الوقت تحللين أو تستريحين، وأنا أراقب”,
]

STATUS_MSGS = [
“جالس ابحث لك عن سيتاب، عيني على الشارت”,
“أفحص الازواج واحد واحد، لو في شي أنبهك فوراً”,
“صاحي ومراقب، لا تقلقين”,
“شغال بكامل طاقتي، ما شي يفوتني إن شاء الله”,
]

DAILY_TIPS = [
“ما في صفقة تستاهل تخلك تكسري خطتك. الخطة هي الملك”,
“السوينق يحتاج صبر. الصفقة الصح تجيك، ما تروحين إليها”,
“الخسارة جزء من التداول. المهم إدارة المخاطرة مو الربح السريع”,
“أي ضغط داخل الصفقة؟ هذا إشارة توقفين مو تكملين”,
“الفرق بين المحترف والمبتدئ مو في الصفقات، في الانضباط”,
“اكتبي كل صفقة في الجورنال. اللي ما يوثق، ما يتعلم”,
“لو حسيتِ بالثقل من السوق، خذي استراحة. الحساب أهم من الصفقة”,
]

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
t = datetime.fromisoformat(ev.get(“date”, “”).replace(“Z”, “”))
diff = t - now
if timedelta(hours=-1) <= diff <= timedelta(hours=24):
title = ev.get(“title”, “”)
if any(k.lower() in title.lower() for k in HIGH_IMPACT_KEYWORDS):
upcoming.append({
“title”: title,
“currency”: ev.get(“country”, “”),
“hours”: round(diff.total_seconds() / 3600, 1)
})
except:
continue
return {“has_news”: len(upcoming) > 0, “events”: upcoming[:3]}
except:
return {“has_news”: False, “events”: []}

def get_candles(yf_sym, tf, limit=100):
try:
period = {“1h”: “7d”, “4h”: “60d”, “1d”: “180d”, “1wk”: “2y”}.get(tf, “60d”)
df = yf.Ticker(yf_sym).history(period=period, interval=tf)
df = df.rename(columns={“Open”: “open”, “High”: “high”, “Low”: “low”, “Close”: “close”})
return df.tail(limit)
except:
return pd.DataFrame()

def detect_trend(df):
if len(df) < 20:
return “neutral”
r = df.tail(20)
if r[“high”].iloc[-1] > r[“high”].iloc[0] and r[“low”].iloc[-1] > r[“low”].iloc[0]:
return “bullish”
if r[“high”].iloc[-1] < r[“high”].iloc[0] and r[“low”].iloc[-1] < r[“low”].iloc[0]:
return “bearish”
return “neutral”

def find_swings(df, lb=3):
highs, lows = [], []
for i in range(lb, len(df) - lb):
if df[“high”].iloc[i] == df[“high”].iloc[i - lb:i + lb + 1].max():
highs.append((i, df[“high”].iloc[i]))
if df[“low”].iloc[i] == df[“low”].iloc[i - lb:i + lb + 1].min():
lows.append((i, df[“low”].iloc[i]))
return highs, lows

def detect_dbos(df, highs, lows, direction):
if direction == “bullish” and len(highs) >= 2:
for i in range(len(highs) - 1, 0, -1):
if highs[i][1] > highs[i - 1][1]:
for j in range(highs[i - 1][0], len(df)):
if df[“close”].iloc[j] > highs[i - 1][1]:
return {“index”: j, “price”: highs[i - 1][1]}


elif direction == “bearish” and len(lows) >= 2:
for i in range(len(lows) - 1, 0, -1):
if lows[i][1] < lows[i - 1][1]:
for j in range(lows[i - 1][0], len(df)):
if df[“close”].iloc[j] < lows[i - 1][1]:
return {“index”: j, “price”: lows[i - 1][1]}
return None

def find_idm(df, idx, direction):
for i in range(idx + 1, min(idx + 25, len(df))):
if direction == “bullish”:
if df[“close”].iloc[i] < df[“open”].iloc[i] and df[“low”].iloc[i] < df[“low”].iloc[i - 1]:
return {“index”: i, “price”: df[“low”].iloc[i]}
else:
if df[“close”].iloc[i] > df[“open”].iloc[i] and df[“high”].iloc[i] > df[“high”].iloc[i - 1]:
return {“index”: i, “price”: df[“high”].iloc[i]}
return None

def find_ob(df, idx, direction):
if not idx or idx < 2:
return None
for i in range(idx, max(idx - 15, 0), -1):
c = df.iloc[i]
if direction == “bullish” and c[“close”] < c[“open”]:
return {“top”: c[“open”], “bottom”: c[“close”]}
elif direction == “bearish” and c[“close”] > c[“open”]:
return {“top”: c[“close”], “bottom”: c[“open”]}
return None

def check_sweep(df, direction):
if len(df) < 15:
return False
rh = df[“high”].tail(15).iloc[:-2].max()
rl = df[“low”].tail(15).iloc[:-2].min()
last = df.iloc[-2]
if direction == “bullish”:
return last[“low”] < rl and df[“close”].iloc[-1] > rl
return last[“high”] > rh and df[“close”].iloc[-1] < rh

def calc_quality(dbos, idm, ob, sweep, daily_match, has_news):
score = 0
if dbos:
score += 25
if idm:
score += 25
if ob:
score += 25
if sweep:
score += 15
if daily_match:
score += 10
if has_news:
score -= 15
return max(0, min(100, score))

def get_risk(quality, account):
dd = account[“drawdown_used”]
remaining = account[“max_drawdown”] - dd
if remaining <= 2:
return 0, “الدروداون ضيق، استريحي”
if quality >= 90:
risk, label = 2.0, “ممتازة”
elif quality >= 75:
risk, label = 1.5, “قوية”
elif quality >= 60:
risk, label = 1.0, “كويسة”
else:
return 0, “ضعيفة”
if dd >= 6:
risk = min(risk, 0.5)
return risk, label

def analyze(sym_name, yf_sym, tf, news):
df = get_candles(yf_sym, tf)
if df.empty or len(df) < 30:
return None
trend = detect_trend(df)
if trend == “neutral”:
return None
highs, lows = find_swings(df)
dbos = detect_dbos(df, highs, lows, trend)
if not dbos:
return None
idm = find_idm(df, dbos[“index”], trend)
if not idm:
return None
ob = find_ob(df, idm[“index”], trend)
if not ob:
return None
current = df["close"].iloc[-1]
ob_range = ob["top"] - ob["bottom"]
in_ob = (ob["bottom"] - ob_range * 0.3) <= current <= (ob["top"] + ob_range * 0.3)
sweep = check_sweep(df, trend)

df_d = get_candles(yf_sym, "1d", 30)
daily_match = detect_trend(df_d) == trend if not df_d.empty else False

quality = calc_quality(dbos, idm, ob, sweep, daily_match, news["has_news"])
if quality < 60:
    return None

return {
    "symbol": sym_name, "tf": tf, "trend": trend,
    "current": current, "ob_top": ob["top"], "ob_bottom": ob["bottom"],
    "in_ob": in_ob, "sweep": sweep, "daily_match": daily_match,
    "quality": quality, "news": news,
}

def setup_msg(a):
arrow = “up” if a[“trend”] == “bullish” else “down”
direction = “شراء” if a[“trend”] == “bullish” else “بيع”
risk, label = get_risk(a[“quality”], ACCOUNT)
news_txt = ""
if a["news"]["has_news"]:
    news_txt = "\nتنبيه: في أخبار مهمة قريبة!\n"
    for ev in a["news"]["events"]:
        news_txt += "- " + ev["title"] + " بعد " + str(ev["hours"]) + " ساعة\n"
    news_txt += "خذي بالك وخففي المخاطرة\n"

extras = []
if a["sweep"]:
    extras.append("سحب سيولة قبل الحركة")
if a["daily_match"]:
    extras.append("اليومي يدعم النظرة")
extras_txt = "\n".join(extras)

zone_txt = "السعر داخل الـ OB الحين! لا تفوتينها" if a["in_ob"] else \
    "انتظري السعر يوصل للمنطقة: " + str(round(a["ob_bottom"], 4)) + " - " + str(round(a["ob_top"], 4))

quality_bar = "G" * (a["quality"] // 20) + "o" * (5 - a["quality"] // 20)

if risk == 0:
    risk_txt = "ما ندخل - " + label
else:
    risk_txt = "المخاطرة المقترحة: " + str(risk) + "% - " + label

msg = "سيتاب " + direction + " | " + a["symbol"] + "\n"
msg += "فريم: " + a["tf"] + "\n"
msg += "---\n"
msg += "DBOS - كسر هيكل مزدوج\n"
msg += "IDM - أول بول باك\n"

Shetradingg, [07/09/47 08:34 م]
msg += "OB - أوردر بلوك جاهز\n"
msg += extras_txt + "\n"
msg += news_txt
msg += "السعر الحالي: " + str(round(a["current"], 4)) + "\n"
msg += zone_txt + "\n"
msg += "الجودة: " + str(a["quality"]) + "/100\n"
msg += quality_bar + "\n"
msg += risk_txt + "\n"
msg += "القرار النهائي إلك شذا"
return msg

def daily_advice_msg():
dd = ACCOUNT[“drawdown_used”]
trades = ACCOUNT[“trades_week”]
balance = ACCOUNT[“balance”]
pnl = ACCOUNT[“pnl_percent”]
remaining = ACCOUNT[“max_drawdown”] - dd
if pnl > 0:
    pnl_txt = "الحساب رابح " + str(pnl) + "% واصلي بنفس المنهج"
elif pnl == 0:
    pnl_txt = "الحساب عند نقطة البداية، ركزي على الجودة"
elif pnl >= -5:
    pnl_txt = "الحساب خاسر " + str(abs(pnl)) + "% خففي المخاطرة"
else:
    pnl_txt = "الحساب خاسر " + str(abs(pnl)) + "% الاولوية حماية الحساب"

if dd == 0:
    dd_txt = "الحساب طازج ما استخدمتِ شي"
elif remaining >= 7:
    dd_txt = "استخدمتِ " + str(dd) + "% باقي " + str(remaining) + "% الحمدلله"
elif remaining >= 4:
    dd_txt = "باقي " + str(remaining) + "% دروداون، تعاملي بحذر"
else:
    dd_txt = "باقي " + str(remaining) + "% بس! الحساب يحتاج عناية قصوى"

if trades == 0:
    trades_txt = "ما دخلتِ صفقات، الصبر ذهب انتظري السيتاب الصح"
elif trades <= 2:
    trades_txt = "دخلتِ " + str(trades) + " صفقة، ممتاز"
else:
    trades_txt = str(trades) + " صفقات الاسبوع، شوي كثير للسوينق"

msg = "نصايح اليوم من بوتك\n---\n"
msg += "1 - وضع الحساب:\n" + pnl_txt + "\n\n"
msg += "2 - الدروداون:\n" + dd_txt + "\n\n"
msg += "3 - الصفقات:\n" + trades_txt + "\n\n"
msg += "4 - نصيحة:\n" + random.choice(DAILY_TIPS) + "\n\n"
msg += "وفقك الله شذا"
return msg

def status_msg():
now = datetime.now(RIYADH_TZ)
pnl = ACCOUNT[“pnl_percent”]
msg = random.choice(STATUS_MSGS) + “\n”
msg += “الوقت: “ + now.strftime(”%H:%M”) + “ الرياض\n”
msg += “الحساب: “ + str(pnl) + “%\n”
msg += “دروداون: “ + str(ACCOUNT[“drawdown_used”]) + “%\n”
msg += “صفقات الاسبوع: “ + str(ACCOUNT[“trades_week”])
return msg

async def scan_markets(bot):
news = check_news()
found = []
for name, yf_sym in SYMBOLS.items():
for tf in [“4h”, “1h”]:
try:
r = analyze(name, yf_sym, tf, news)
if r:
found.append(r)
except Exception as e:
logger.error(“خطأ “ + name + “ “ + tf + “: “ + str(e))
if found:
    found.sort(key=lambda x: x["quality"], reverse=True)
    for s in found:
        await bot.send_message(chat_id=CHAT_ID, text=setup_msg(s))
        await asyncio.sleep(2)
    return True
return False

async def trading_loop(bot):
await bot.send_message(
chat_id=CHAT_ID,
text=“بوتك شغال يا شذا!\nيفحص كل ساعة وينبهك بأي سيتاب\nالنصايح اليومية الساعة 8 صباحاً\n/scan فحص فوري\n/advice نصايح\n/status الحالة\n/update تحديث الحساب”
)
last_advice_day = None
last_status_hour = -1

while True:
    try:
        now = datetime.now(RIYADH_TZ)
        today = now.date()

        if now.hour == 8 and now.minute < 5 and last_advice_day != today:
            await bot.send_message(chat_id=CHAT_ID, text=daily_advice_msg())
            last_advice_day = today

        if now.hour % 4 == 0 and now.hour != last_status_hour and now.minute < 5:
            found = await scan_markets(bot)
            if not found:
                await bot.send_message(chat_id=CHAT_ID, text=random.choice(NO_SETUP_MSGS))
            last_status_hour = now.hour
        else:
            await scan_markets(bot)

        await asyncio.sleep(3600)

    except Exception as e:
        logger.error("خطأ: " + str(e))
        await asyncio.sleep(60)

async def start_cmd(update, context):
await update.message.reply_text(
“أهلاً شذا!\nأنا بوتك للتداول، أراقب الاسواق 24/7\n\n/scan فحص فوري\n/advice نصايح اليوم\n/status الحالة\n/update تحديث الحساب”
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
try:
args = “ “.


join(context.args)
updated = []
    if "pnl=" in args:
        val = float(args.split("pnl=")[1].split()[0].replace("+", ""))
        ACCOUNT["pnl_percent"] = val
        updated.append("PnL: " + str(val) + "%")

    if "dd=" in args:
        val = float(args.split("dd=")[1].split()[0])
        ACCOUNT["drawdown_used"] = val
        updated.append("دروداون: " + str(val) + "%")

    if "daily=" in args:
        val = float(args.split("daily=")[1].split()[0])
        ACCOUNT["daily_used"] = val
        updated.append("ديلي: " + str(val) + "%")

    if "trades=" in args:
        val = int(args.split("trades=")[1].split()[0])
        ACCOUNT["trades_week"] = val
        updated.append("صفقات: " + str(val))

    if updated:
        await update.message.reply_text("تم التحديث!\n" + "\n".join(updated))
    else:
        await update.message.reply_text(
            "الاستخدام:\n/update pnl=+3.5 dd=2.5 daily=1.0 trades=2"
        )
except:
    await update.message.reply_text("خطأ في البيانات، مثال: /update pnl=+3.5 dd=2.5 daily=1.0 trades=2")

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
