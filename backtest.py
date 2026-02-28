"""
===================================================
  BACKTEST — استراتيجيتين
  Strategy 1: DBOS + IDM + OB
  Strategy 2: Liquidity Sweep (Standalone)
  TP1 = 2R  |  TP2 = 4R
  آخر سنة — 8 أزواج
===================================================
تشغيل:
    pip install yfinance pandas numpy
    python backtest.py
النتائج تُحفظ في: backtest_results.json
===================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# ===== الإعدادات =====
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

TP1_R = 2.0
TP2_R = 4.0
SL_R  = 1.0

TARGET_TRADES = 300   # هدف 300 صفقة لكل استراتيجية
MIN_QUALITY   = 70    # حد الجودة الأدنى


# ============================================================
# ===== جلب البيانات =====
# ============================================================

def get_candles(yf_sym, tf, period="1y"):
    try:
        df = yf.Ticker(yf_sym).history(period=period, interval=tf)
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
        df = df[["open","high","low","close"]].dropna()
        return df
    except Exception as e:
        print(f"  ⚠️ خطأ جلب {yf_sym} {tf}: {e}")
        return pd.DataFrame()


# ============================================================
# ===== دوال التحليل (من bot.py) =====
# ============================================================

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
            if h2_val - segment["low"].min() <= 0: continue
            for j in range(h2_idx, min(h2_idx+8, len(df))):
                if df.iloc[j]["close"] > h2_val:
                    return {"index": j, "price": h2_val,
                            "h1_val": h1_val, "h2_val": h2_val,
                            "sweep_level": segment["low"].min()}
    elif direction == "bearish" and len(l_list) >= 2:
        for i in range(len(l_list)-1, 0, -1):
            l2_idx, l2_val = l_list[i]
            l1_idx, l1_val = l_list[i-1]
            if l2_val >= l1_val: continue
            segment = df.iloc[l1_idx:l2_idx+1]
            if len(segment) < 2 or len(segment) > 80: continue
            if segment["high"].max() - l2_val <= 0: continue
            for j in range(l2_idx, min(l2_idx+8, len(df))):
                if df.iloc[j]["close"] < l2_val:
                    return {"index": j, "price": l2_val,
                            "l1_val": l1_val, "l2_val": l2_val,
                            "sweep_level": segment["high"].max()}
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
            is_pin = wick_ratio > 0.40 and body_ratio < 0.55
            is_eng = c["close"] < c["open"] and body_ratio > 0.60
            if (is_pin or is_eng) and c["low"] < df["low"].iloc[max(0,i-5):i].min():
                return {"index": i, "price": c["low"], "wick_ratio": round(wick_ratio,2)}
        else:
            upper_wick = c["high"] - max(c["open"], c["close"])
            wick_ratio = upper_wick / candle_range
            is_pin = wick_ratio > 0.40 and body_ratio < 0.55
            is_eng = c["close"] > c["open"] and body_ratio > 0.60
            if (is_pin or is_eng) and c["high"] > df["high"].iloc[max(0,i-5):i].max():
                return {"index": i, "price": c["high"], "wick_ratio": round(wick_ratio,2)}
    return None


def find_ob(df, idm_idx, direction):
    if idm_idx is None or idm_idx < 3:
        return None
    search_start = max(0, idm_idx - 20)
    if direction == "bullish":
        for i in range(idm_idx - 1, search_start, -1):
            c = df.iloc[i]
            rng = c["high"] - c["low"]
            if rng == 0: continue
            body = abs(c["close"] - c["open"])
            body_ratio = body / rng
            if c["close"] < c["open"] and body_ratio > 0.45 and i+1 < len(df):
                nx = df.iloc[i+1]
                nx_body = abs(nx["close"] - nx["open"])
                nx_rng  = nx["high"] - nx["low"]
                if nx["close"] > nx["open"] and nx_rng > 0 and nx_body/nx_rng > 0.50:
                    return {"top": c["open"], "bottom": c["close"], "index": i,
                            "candle_high": c["high"], "candle_low": c["low"],
                            "body_ratio": round(body_ratio,2)}
    else:
        for i in range(idm_idx - 1, search_start, -1):
            c = df.iloc[i]
            rng = c["high"] - c["low"]
            if rng == 0: continue
            body = abs(c["close"] - c["open"])
            body_ratio = body / rng
            if c["close"] > c["open"] and body_ratio > 0.45 and i+1 < len(df):
                nx = df.iloc[i+1]
                nx_body = abs(nx["close"] - nx["open"])
                nx_rng  = nx["high"] - nx["low"]
                if nx["close"] < nx["open"] and nx_rng > 0 and nx_body/nx_rng > 0.50:
                    return {"top": c["close"], "bottom": c["open"], "index": i,
                            "candle_high": c["high"], "candle_low": c["low"],
                            "body_ratio": round(body_ratio,2)}
    return None


def calc_sl_from_ob(ob, direction):
    ob_range  = ob["top"] - ob["bottom"]
    sl_buffer = ob_range * 0.20
    if direction == "bullish":
        return round(ob["candle_low"] - sl_buffer, 5)
    else:
        return round(ob["candle_high"] + sl_buffer, 5)


def check_bsl_ssl(df, direction, lookback=50):
    if len(df) < lookback:
        return False
    current = df["close"].iloc[-1]
    recent  = df.tail(lookback)
    if direction == "bullish":
        bsl  = recent["high"].max()
        dist = (bsl - current) / current * 100
        return 0.3 < dist < 5.0
    else:
        ssl  = recent["low"].min()
        dist = (current - ssl) / current * 100
        return 0.3 < dist < 5.0


# ============================================================
# ===== محاكاة الدخول والخروج =====
# ============================================================

def simulate_trade(df, entry_bar_idx, entry, sl, tp1, tp2, direction):
    """
    تجري الشمعات بعد الإشارة وتشوف أيهم يتحقق أول: TP1 أو TP2 أو SL
    تعيد: نتيجة الصفقة كـ R
    """
    for i in range(entry_bar_idx + 1, len(df)):
        c = df.iloc[i]
        if direction == "bullish":
            # هل لمس TP2 أول
            if c["high"] >= tp2:
                return 4.0, "TP2", i
            # هل لمس TP1 أول
            if c["high"] >= tp1:
                return 2.0, "TP1", i
            # هل لمس SL
            if c["low"] <= sl:
                return -1.0, "SL", i
        else:
            if c["low"] <= tp2:
                return 4.0, "TP2", i
            if c["low"] <= tp1:
                return 2.0, "TP1", i
            if c["high"] >= sl:
                return -1.0, "SL", i
    # ما انتهت الصفقة قبل انتهاء البيانات
    return None, "OPEN", len(df)-1


# ============================================================
# ===== الاستراتيجية 1: DBOS + IDM + OB =====
# ============================================================

def backtest_strategy1(sym_name, yf_sym, tf, df_h4, df_d):
    """DBOS + IDM + OB"""
    trades = []
    df = get_candles(yf_sym, tf, period="1y")
    if df.empty or len(df) < 100:
        return trades

    window = 80  # نافذة التحليل

    for bar in range(window, len(df) - 10):
        window_df = df.iloc[bar - window: bar + 1].copy().reset_index(drop=True)

        trend = detect_trend_structure(window_df)
        if trend == "neutral":
            continue

        # H4 order flow
        h4_of = detect_order_flow(df_h4, trend) if not df_h4.empty else 0.0
        if h4_of < 0.55:
            continue

        # DBOS
        dbos = detect_dbos(window_df, trend)
        if not dbos:
            continue

        # IDM
        idm = find_idm(window_df, dbos["index"], trend)
        if not idm:
            continue

        # OB
        ob = find_ob(window_df, idm["index"], trend)
        if not ob:
            continue

        # حجم OB
        ob_size = ob["top"] - ob["bottom"]
        min_ob = {"BTC": 300.0, "XAU": 3.0, "XAG": 0.05}.get(
            next((k for k in ["BTC","XAU","XAG"] if k in sym_name), ""), 0.0020)
        if ob_size < min_ob:
            continue

        # OB عمر
        if (len(window_df) - ob["index"]) > 60:
            continue

        current = window_df["close"].iloc[-1]
        in_ob   = ob["bottom"] <= current <= ob["top"]

        # حساب الجودة
        liq_sweep, _ = detect_liquidity_sweep(window_df, trend)
        has_bsl = check_bsl_ssl(window_df, trend)
        daily_of = detect_order_flow(df_d, trend) if not df_d.empty else 0.0

        score = 0
        if liq_sweep:    score += 30
        if h4_of >= 0.7: score += 20
        elif h4_of >= 0.5: score += 12
        if daily_of >= 0.6: score += 15
        elif daily_of >= 0.4: score += 8
        if has_bsl:      score += 15
        quality = min(100, score)

        if quality < MIN_QUALITY:
            continue

        # دخول وخروج
        sl = calc_sl_from_ob(ob, trend)
        if trend == "bullish":
            entry = round(ob["top"], 5)
            risk  = entry - sl
        else:
            entry = round(ob["bottom"], 5)
            risk  = sl - entry

        if risk <= 0:
            continue

        tp1 = round(entry + (TP1_R * risk) * (1 if trend=="bullish" else -1), 5)
        tp2 = round(entry + (TP2_R * risk) * (1 if trend=="bullish" else -1), 5)

        # محاكاة الصفقة على الشمعات التالية
        result_r, result_label, exit_bar = simulate_trade(
            df, bar, entry, sl, tp1, tp2, trend
        )

        if result_r is None:  # مفتوحة
            continue

        trades.append({
            "symbol":    sym_name,
            "tf":        tf,
            "direction": trend,
            "entry":     entry,
            "sl":        round(sl, 5),
            "tp1":       round(tp1, 5),
            "tp2":       round(tp2, 5),
            "result_r":  result_r,
            "result":    result_label,
            "quality":   quality,
            "liq_sweep": liq_sweep,
            "has_bsl":   has_bsl,
            "h4_of":     h4_of,
            "bar_entry": bar,
            "bar_exit":  exit_bar,
            "date":      str(df.index[bar]) if hasattr(df.index[bar], '__str__') else str(bar),
            "strategy":  "DBOS_IDM_OB",
        })

        # تجنب التداخل — ننتقل للشمعة بعد الخروج
        bar = exit_bar

    return trades


# ============================================================
# ===== الاستراتيجية 2: Liquidity Sweep وحدها =====
# ============================================================

def backtest_strategy2(sym_name, yf_sym, tf, df_h4, df_d):
    """
    Liquidity Sweep Standalone:
    1. سحب سيولة واضح (ذيل يكسر قاع/قمة ويغلق بالداخل)
    2. الشمعة التالية تأكد الاتجاه (شمعة كبيرة في اتجاه السحب)
    3. الدخول عند إغلاق الشمعة التأكيدية
    4. SL تحت/فوق أدنى/أعلى الذيل
    """
    trades = []
    df = get_candles(yf_sym, tf, period="1y")
    if df.empty or len(df) < 60:
        return trades

    for bar in range(50, len(df) - 10):
        window_df = df.iloc[bar - 50: bar + 1].copy().reset_index(drop=True)

        trend = detect_trend_structure(window_df)
        if trend == "neutral":
            continue

        # سحب السيولة
        liq, liq_level = detect_liquidity_sweep(window_df, trend)
        if not liq:
            continue

        # الشمعة الأخيرة = شمعة السحب
        sweep_candle = window_df.iloc[-1]
        sweep_range  = sweep_candle["high"] - sweep_candle["low"]
        if sweep_range == 0:
            continue

        sweep_body = abs(sweep_candle["close"] - sweep_candle["open"])
        body_ratio = sweep_body / sweep_range

        # لازم يكون فيها ذيل واضح (Rejection)
        if direction == "bullish":
            pass

        # H4 order flow يدعم
        h4_of = detect_order_flow(df_h4, trend) if not df_h4.empty else 0.0
        if h4_of < 0.50:
            continue

        # جودة
        has_bsl  = check_bsl_ssl(window_df, trend)
        daily_of = detect_order_flow(df_d, trend) if not df_d.empty else 0.0

        score = 30  # sweep نفسه = 30
        if h4_of >= 0.7: score += 20
        elif h4_of >= 0.5: score += 12
        if daily_of >= 0.6: score += 15
        elif daily_of >= 0.4: score += 8
        if has_bsl: score += 15
        quality = min(100, score)

        if quality < MIN_QUALITY:
            continue

        # الدخول
        current = sweep_candle["close"]
        if trend == "bullish":
            entry = round(current, 5)
            sl    = round(sweep_candle["low"] * 0.9995, 5)   # تحت أدنى الذيل
        else:
            entry = round(current, 5)
            sl    = round(sweep_candle["high"] * 1.0005, 5)  # فوق أعلى الذيل

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        tp1 = round(entry + (TP1_R * risk) * (1 if trend=="bullish" else -1), 5)
        tp2 = round(entry + (TP2_R * risk) * (1 if trend=="bullish" else -1), 5)

        result_r, result_label, exit_bar = simulate_trade(
            df, bar, entry, sl, tp1, tp2, trend
        )

        if result_r is None:
            continue

        trades.append({
            "symbol":    sym_name,
            "tf":        tf,
            "direction": trend,
            "entry":     entry,
            "sl":        round(sl, 5),
            "tp1":       round(tp1, 5),
            "tp2":       round(tp2, 5),
            "result_r":  result_r,
            "result":    result_label,
            "quality":   quality,
            "liq_sweep": True,
            "has_bsl":   has_bsl,
            "h4_of":     h4_of,
            "bar_entry": bar,
            "bar_exit":  exit_bar,
            "date":      str(df.index[bar]) if hasattr(df.index[bar], '__str__') else str(bar),
            "strategy":  "LIQ_SWEEP",
        })

        bar = exit_bar

    return trades


# ============================================================
# ===== تشغيل الباك تيست =====
# ============================================================

def run_backtest():
    all_s1, all_s2 = [], []

    for sym_name, yf_sym in SYMBOLS.items():
        print(f"\n📊 {sym_name}...")

        # نجلب H4 و Daily مرة واحدة لكل زوج
        df_h4 = get_candles(yf_sym, "4h", period="1y")
        df_d  = get_candles(yf_sym, "1d", period="1y")

        for tf in ["4h", "1h"]:
            print(f"  ⏱  {tf} — Strategy 1 (DBOS+IDM+OB)...")
            s1 = backtest_strategy1(sym_name, yf_sym, tf, df_h4, df_d)
            all_s1.extend(s1)
            print(f"       → {len(s1)} صفقة")

            print(f"  ⏱  {tf} — Strategy 2 (Liq Sweep)...")
            s2 = backtest_strategy2(sym_name, yf_sym, tf, df_h4, df_d)
            all_s2.extend(s2)
            print(f"       → {len(s2)} صفقة")

    return all_s1, all_s2


def calc_stats(trades, name):
    if not trades:
        return {"name": name, "total": 0}

    wins   = [t for t in trades if t["result_r"] > 0]
    losses = [t for t in trades if t["result_r"] < 0]
    tp2s   = [t for t in trades if t["result"] == "TP2"]
    tp1s   = [t for t in trades if t["result"] == "TP1"]
    total_r = round(sum(t["result_r"] for t in trades), 2)
    win_rate = round(len(wins)/len(trades)*100, 1) if trades else 0

    profit_factor = round(
        sum(t["result_r"] for t in wins) /
        abs(sum(t["result_r"] for t in losses))
        if losses else 999, 2
    )

    # أطول سلسلة خسارة
    max_loss = cur_loss = 0
    for t in trades:
        if t["result_r"] < 0:
            cur_loss += 1
            max_loss = max(max_loss, cur_loss)
        else:
            cur_loss = 0

    # توزيع الأزواج
    pairs = {}
    for t in trades:
        k = t["symbol"]
        if k not in pairs:
            pairs[k] = {"total":0,"wins":0}
        pairs[k]["total"] += 1
        if t["result_r"] > 0:
            pairs[k]["wins"] += 1

    pair_stats = {k: {
        "total": v["total"],
        "win_rate": round(v["wins"]/v["total"]*100,1)
    } for k,v in pairs.items()}

    # منحنى الرصيد (بحساب $10,000 ومخاطرة 0.75%)
    balance = 10000.0
    curve   = [balance]
    for t in trades:
        balance += balance * (t["result_r"] * 0.75 / 100)
        curve.append(round(balance, 2))

    return {
        "name":            name,
        "total":           len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "tp1_count":       len(tp1s),
        "tp2_count":       len(tp2s),
        "win_rate":        win_rate,
        "total_r":         total_r,
        "profit_factor":   profit_factor,
        "max_loss_streak": max_loss,
        "avg_quality":     round(sum(t["quality"] for t in trades)/len(trades), 1),
        "final_balance":   round(curve[-1], 2),
        "profit_pct":      round((curve[-1]-10000)/100, 2),
        "balance_curve":   curve,
        "pair_stats":      pair_stats,
        "trades":          trades,
    }


def save_results(s1_stats, s2_stats):
    results = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "settings": {
            "tp1_r": TP1_R,
            "tp2_r": TP2_R,
            "min_quality": MIN_QUALITY,
            "period": "1 year",
            "symbols": list(SYMBOLS.keys()),
        },
        "strategy1": s1_stats,
        "strategy2": s2_stats,
    }

    # احفظ بدون balance_curve و trades في JSON الرئيسي (ثقيلة)
    results_summary = {
        "generated_at": results["generated_at"],
        "settings":     results["settings"],
        "strategy1":    {k:v for k,v in s1_stats.items() if k not in ["trades","balance_curve"]},
        "strategy2":    {k:v for k,v in s2_stats.items() if k not in ["trades","balance_curve"]},
        "s1_trades":    s1_stats.get("trades", []),
        "s2_trades":    s2_stats.get("trades", []),
        "s1_curve":     s1_stats.get("balance_curve", []),
        "s2_curve":     s2_stats.get("balance_curve", []),
    }

    with open("backtest_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n✅ تم الحفظ في: backtest_results.json")
    return results_summary


def print_summary(stats):
    print(f"\n{'='*45}")
    print(f"  {stats['name']}")
    print(f"{'='*45}")
    print(f"  إجمالي الصفقات : {stats['total']}")
    print(f"  رابحة          : {stats['wins']}  ({stats['win_rate']}%)")
    print(f"  خاسرة          : {stats['losses']}")
    print(f"  TP1 وصل        : {stats.get('tp1_count',0)}")
    print(f"  TP2 وصل        : {stats.get('tp2_count',0)}")
    print(f"  إجمالي R       : {stats['total_r']}R")
    print(f"  Profit Factor  : {stats['profit_factor']}")
    print(f"  أطول سلسلة خسارة: {stats['max_loss_streak']}")
    print(f"  متوسط الجودة   : {stats.get('avg_quality',0)}/100")
    print(f"  الرصيد النهائي : ${stats['final_balance']:,.2f}")
    print(f"  ربح الحساب     : +{stats['profit_pct']}%")
    print(f"\n  أداء الأزواج:")
    for pair, ps in sorted(stats.get("pair_stats",{}).items(),
                           key=lambda x: x[1]["win_rate"], reverse=True):
        bar = "█" * int(ps["win_rate"]//10) + "░" * (10 - int(ps["win_rate"]//10))
        print(f"    {pair:8s} {bar} {ps['win_rate']}%  ({ps['total']} صفقة)")


# ============================================================
# ===== الرئيسي =====
# ============================================================

if __name__ == "__main__":
    print("🚀 بدأ الباك تيست...")
    print(f"   TP1={TP1_R}R  |  TP2={TP2_R}R  |  حد الجودة={MIN_QUALITY}")
    print(f"   الأزواج: {', '.join(SYMBOLS.keys())}")
    print(f"   الفترة: آخر سنة\n")

    s1_trades, s2_trades = run_backtest()

    print(f"\n📈 إجمالي إشارات Strategy 1: {len(s1_trades)}")
    print(f"📈 إجمالي إشارات Strategy 2: {len(s2_trades)}")

    # خذ أعلى 300 جودة من كل استراتيجية
    s1_top = sorted(s1_trades, key=lambda x: x["quality"], reverse=True)[:TARGET_TRADES]
    s2_top = sorted(s2_trades, key=lambda x: x["quality"], reverse=True)[:TARGET_TRADES]

    s1_stats = calc_stats(s1_top, "Strategy 1 — DBOS + IDM + OB")
    s2_stats = calc_stats(s2_top, "Strategy 2 — Liquidity Sweep")

    print_summary(s1_stats)
    print_summary(s2_stats)

    results = save_results(s1_stats, s2_stats)

    print("\n" + "="*45)
    print("  ✅ الباك تيست انتهى!")
    print("  📁 النتائج: backtest_results.json")
    print("  🌐 افتحي backtest_web.html وأضيفي النتائج")
    print("="*45)
