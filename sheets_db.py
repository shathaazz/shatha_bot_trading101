"""
sheets_db.py  ←  ضعيه في نفس مجلد bot.py
ذاكرة البوت الدائمة عبر Google Sheets
"""

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1RoSDo0yWZMiFog0UBbTeOQyi9l7iMZ2kbH7nNLjRCQo")
import tempfile

_creds_raw = os.environ.get("GOOGLE_CREDENTIALS", "")
if _creds_raw:
    _tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    _tmp.write(_creds_raw)
    _tmp.close()
    CREDENTIALS_FILE = _tmp.name
else:
    CREDENTIALS_FILE = "credentials.json"

_service = None


# ─────────────────────────────────────────
# الاتصال
# ─────────────────────────────────────────

def _get_service():
    global _service
    if _service:
        return _service
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        _service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        logger.info("✅ Google Sheets متصل")
        return _service
    except Exception as e:
        logger.error(f"❌ فشل الاتصال بـ Sheets: {e}")
        return None


def _write(range_name, values):
    try:
        svc = _get_service()
        if not svc:
            return False
        svc.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption="USER_ENTERED",
            body={"values": values}
        ).execute()
        return True
    except Exception as e:
        logger.error(f"Sheets write error [{range_name}]: {e}")
        return False


def _append(range_name, values):
    try:
        svc = _get_service()
        if not svc:
            return False
        svc.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": values}
        ).execute()
        return True
    except Exception as e:
        logger.error(f"Sheets append error [{range_name}]: {e}")
        return False


def _read(range_name):
    try:
        svc = _get_service()
        if not svc:
            return []
        res = svc.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=range_name
        ).execute()
        return res.get("values", [])
    except Exception as e:
        logger.error(f"Sheets read error [{range_name}]: {e}")
        return []


def _find_row(sheet, col, value):
    """يرجع رقم الصف (1-based) اللي فيه value في العمود col"""
    data = _read(f"{sheet}!A:Z")
    for i, row in enumerate(data):
        if len(row) > col and str(row[col]) == str(value):
            return i + 1
    return None


# ─────────────────────────────────────────
# إعداد الشيت (شغّليه مرة واحدة)
# ─────────────────────────────────────────

def setup_sheets():
    """ينشئ كل الـ Tabs والعناوين - مرة واحدة عند أول تشغيل"""
    svc = _get_service()
    if not svc:
        return False

    # إنشاء Tabs لو ما موجودة
    tabs = ["Journal", "Account", "Weights", "Stats"]
    try:
        meta = svc.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        existing = [s["properties"]["title"] for s in meta["sheets"]]
        reqs = []
        for t in tabs:
            if t not in existing:
                reqs.append({"addSheet": {"properties": {"title": t}}})
        if reqs:
            svc.spreadsheets().batchUpdate(
                spreadsheetId=SPREADSHEET_ID,
                body={"requests": reqs}
            ).execute()
    except Exception as e:
        logger.error(f"خطأ إنشاء Tabs: {e}")

    # عناوين Journal
    _write("Journal!A1:P1", [[
        "ID", "التاريخ", "الزوج", "الفريم", "الاتجاه",
        "الدخول", "الستوب", "هدف1", "هدف2",
        "المخاطرة%", "الحالة", "النتيجة R",
        "الجودة%", "H4 Flow", "يومي؟", "أسبوعي؟"
    ]])

    # عناوين Account
    _write("Account!A1:B1", [["المؤشر", "القيمة"]])
    _write("Account!A2:A12", [
        ["اسم الشركة"], ["المرحلة"], ["الرصيد الأصلي"],
        ["الرصيد الحالي"], ["PnL%"], ["دروداون كلي%"],
        ["دروداون يومي%"], ["صفقات الأسبوع"], ["صفقات اليوم"],
        ["آخر تحديث"], ["وقف التداول؟"]
    ])

    # عناوين Weights
    _write("Weights!A1:B1", [["الوزن", "القيمة"]])

    # عناوين Stats
    _write("Stats!A1:F1", [[
        "الأسبوع", "إجمالي", "رابحة", "خاسرة", "نسبة الفوز%", "مجموع R"
    ]])

    logger.info("✅ تم إعداد الشيت بنجاح")
    return True


# ─────────────────────────────────────────
# Journal
# ─────────────────────────────────────────

def journal_add(trade_id, trade):
    """يضيف صفقة جديدة"""
    _append("Journal!A:P", [[
        trade_id,
        trade.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M")),
        trade.get("symbol", ""),
        trade.get("tf", ""),
        "شراء 📈" if trade.get("direction") == "bullish" else "بيع 📉",
        trade.get("entry", ""),
        trade.get("sl", ""),
        trade.get("tp1", ""),
        trade.get("tp2", ""),
        trade.get("risk", ""),
        "⏳ انتظار",
        "",
        trade.get("quality", ""),
        trade.get("h4_of", ""),
        "✅" if trade.get("daily_match") else "❌",
        "✅" if trade.get("weekly_match") else "❌",
    ]])


def journal_set_status(trade_id, status, risk=None):
    """يحدث حالة الصفقة"""
    row = _find_row("Journal", 0, trade_id)
    if not row:
        return
    icons = {
        "active":  "✅ نشطة",
        "pending": "⏳ انتظار",
        "skipped": "❌ تجاهلت",
        "closed":  "🔒 مغلقة",
    }
    _write(f"Journal!K{row}", [[icons.get(status, status)]])
    if risk is not None:
        _write(f"Journal!J{row}", [[risk]])


def journal_set_result(trade_id, result_r):
    """يحدث نتيجة الصفقة"""
    row = _find_row("Journal", 0, trade_id)
    if not row:
        return
    status = "✅ رابحة" if result_r > 0 else "🔴 خاسرة"
    _write(f"Journal!K{row}", [[status]])
    _write(f"Journal!L{row}", [[result_r]])


def journal_load():
    """يحمل الجورنال كاملاً عند تشغيل البوت"""
    data = _read("Journal!A2:P")
    journal = {}
    dir_map = {"شراء 📈": "bullish", "بيع 📉": "bearish"}
    status_map = {
        "✅ نشطة": "active", "⏳ انتظار": "pending",
        "❌ تجاهلت": "skipped", "🔒 مغلقة": "closed",
        "✅ رابحة": "closed", "🔴 خاسرة": "closed",
    }
    for row in data:
        if not row or not row[0]:
            continue
        tid = str(row[0])

        def safe_float(idx, default=0.0):
            try:
                return float(row[idx]) if len(row) > idx and row[idx] else default
            except:
                return default

        def safe_int(idx, default=0):
            try:
                return int(row[idx]) if len(row) > idx and row[idx] else default
            except:
                return default

        journal[tid] = {
            "timestamp": row[1]  if len(row) > 1  else "",
            "symbol":    row[2]  if len(row) > 2  else "",
            "tf":        row[3]  if len(row) > 3  else "",
            "direction": dir_map.get(row[4], "bullish") if len(row) > 4 else "bullish",
            "entry":     safe_float(5),
            "sl":        safe_float(6),
            "tp1":       safe_float(7),
            "tp2":       safe_float(8),
            "risk":      safe_float(9),
            "status":    status_map.get(row[10], "pending") if len(row) > 10 else "pending",
            "result_r":  safe_float(11, None),
            "quality":   safe_int(12),
            "yf_sym":    "",  # نحسبه من SYMBOLS
        }
    logger.info(f"📋 Journal: حمّلت {len(journal)} صفقة")
    return journal


# ─────────────────────────────────────────
# Account
# ─────────────────────────────────────────

def account_save(account, daily_risk):
    _write("Account!B2:B12", [
        [account.get("firm_name", "")],
        [account.get("phase", "")],
        [account.get("balance", 0)],
        [account.get("current_balance", 0)],
        [account.get("pnl_percent", 0)],
        [account.get("drawdown_used", 0)],
        [account.get("daily_used", 0)],
        [account.get("trades_week", 0)],
        [account.get("trades_today", 0)],
        [datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["🛑 نعم" if daily_risk.get("trading_stopped") else "✅ لا"],
    ])


def account_load():
    data = _read("Account!A2:B12")
    keys = {
        "اسم الشركة": "firm_name", "المرحلة": "phase",
        "الرصيد الأصلي": "balance", "الرصيد الحالي": "current_balance",
        "PnL%": "pnl_percent", "دروداون كلي%": "drawdown_used",
        "دروداون يومي%": "daily_used", "صفقات الأسبوع": "trades_week",
        "صفقات اليوم": "trades_today",
    }
    result = {}
    for row in data:
        if len(row) < 2:
            continue
        key = keys.get(row[0])
        if not key:
            continue
        val = row[1]
        if key in ("balance", "current_balance", "pnl_percent",
                   "drawdown_used", "daily_used"):
            try:
                val = float(val)
            except:
                pass
        elif key in ("trades_week", "trades_today"):
            try:
                val = int(val)
            except:
                pass
        result[key] = val
    if result:
        logger.info("💰 Account: حمّلت من الشيت")
    return result


# ─────────────────────────────────────────
# Weights
# ─────────────────────────────────────────

def weights_save(weights):
    rows = [[k, round(v, 4)] for k, v in weights.items()]
    _write("Weights!A2:B100", rows)


def weights_load():
    data = _read("Weights!A2:B100")
    w = {}
    for row in data:
        if len(row) >= 2 and row[0]:
            try:
                w[row[0]] = float(row[1])
            except:
                pass
    if w:
        logger.info(f"⚖️ Weights: حمّلت {len(w)} وزن")
    return w


# ─────────────────────────────────────────
# Stats
# ─────────────────────────────────────────

def stats_add_week(journal):
    closed = [t for t in journal.values() if t.get("status") == "closed"]
    if not closed:
        return
    wins = [t for t in closed if (t.get("result_r") or 0) > 0]
    total_r = round(sum(t.get("result_r") or 0 for t in closed), 2)
    win_rate = round(len(wins) / len(closed) * 100) if closed else 0
    _append("Stats!A:F", [[
        datetime.now().strftime("%Y-W%W"),
        len(closed), len(wins), len(closed) - len(wins),
        win_rate, total_r
    ]])
