"""
===================================================
Backtest - DBOS + Morning Star | آخر سنتين
===================================================
شغّليه: python backtest.py
النتائج: backtest_results.txt + backtest_trades.json
===================================================
"""

import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta

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

TIMEFRAMES = ["1h", "4h"]
LOOKBACK_DAYS = 730

def get_candles(yf_sym, tf):
    try:
        end = datetime.now()
        start = end - timedelta(days=LOOKBACK_DAYS)
        df = yf.download(yf_sym, start=start, end=end, interval="1h", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df[["open","high","low","close"]].dropna()
        if tf == "4h":
            df = df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        return df
    except Exception as e:
        print(f"  خطا {yf_sym}: {e}")
        return pd.DataFrame()

def detect_trend_structure(df, lookback=30):
    if len(df) < lookback:
        return "neutral"
    sub = df.tail(lookback)
    lb = 3
    highs, lows = [], []
    for i in range(lb, len(sub)-lb):
        if sub["high"].iloc[i] == sub["high"].iloc[i-lb:i+lb+1].max():
            highs.append(sub["high"].iloc[i])
        if sub["low"].iloc[i] == sub["low"].iloc[i-lb:i+lb+1].min():
            lows.append(sub["low"].iloc[i])
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]: return "bullish"
        if highs[-1] < highs[-2] and lows[-1] < lows[-2]: return "bearish"
    return "neutral"

def detect_dbos(df, direction):
    lb = 5
    h_list, l_list = [], []
    for i in range(lb, len(df)-lb):
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
                body_top = max(df["open"].iloc[j], df["close"].iloc[j])
                if body_top > h2_val:
                    return {"index": j, "price": h2_val, "impulse_start": h1_idx}

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
                body_bottom = min(df["open"].iloc[j], df["close"].iloc[j])
                if body_bottom < l2_val:
                    return {"index": j, "price": l2_val, "impulse_start": l1_idx}
    return None

def find_idm(df, dbos_idx, direction):
    for i in range(dbos_idx+1, min(dbos_idx+25, len(df))):
        c = df.iloc[i]
        r = c["high"] - c["low"]
        if r == 0: continue
        body = abs(c["close"] - c["open"])
        if direction == "bullish":
            lower_wick = min(c["open"], c["close"]) - c["low"]
            if lower_wick/r > 0.35 and c["low"] < df["low"].iloc[max(0,i-3):i].min():
                return {"index": i, "price": c["low"]}
        else:
            upper_wick = c["high"] - max(c["open"], c["close"])
            if upper_wick/r > 0.35 and c["high"] > df["high"].iloc[max(0,i-3):i].max():
                return {"index": i, "price": c["high"]}
    return None

def find_ob(df, idm_idx, direction):
    if not idm_idx or idm_idx < 2: return None
    for rng in [7, 13]:
        for i in range(idm_idx-1, max(idm_idx-rng, 0), -1):
            c = df.iloc[i]
            r = c["high"] - c["low"]
            if r == 0: continue
            body = abs(c["close"] - c["open"])
            ratio = 0.45 if rng == 7 else 0.40
            if direction == "bullish" and c["close"] < c["open"] and body/r >= ratio:
                if i+1 < len(df) and df["close"].iloc[i+1] > df["open"].iloc[i+1]:
                    return {"top": c["open"], "bottom": c["close"], "index": i}
            if direction == "bearish" and c["close"] > c["open"] and body/r >= ratio:
                if i+1 < len(df) and df["close"].iloc[i+1] < df["open"].iloc[i+1]:
                    return {"top": c["close"], "bottom": c["open"], "index": i}
    return None

def detect_morning_star_bt(df, direction):
    for i in range(max(3, len(df)-30), len(df)-2):
        c1, c2, c3 = df.iloc[i], df.iloc[i+1], df.iloc[i+2]
        r1 = c1["high"] - c1["low"]
        if r1 == 0: continue
        body1 = abs(c1["close"] - c1["open"])
        body2 = abs(c2["close"] - c2["open"])
        if direction == "bullish":
            if c1["close"] >= c1["open"]: continue
            if body1/r1 < 0.50: continue
            prev_low = df["low"].iloc[max(0,i-10):i].min()
            if c1["low"] >= prev_low: continue
            if body2 > body1*0.30: continue
            if c3["close"] <= c3["open"]: continue
            if c3["close"] <= (c1["open"]+c1["close"])/2: continue
            return {"c3_idx": i+2, "entry": round(c3["close"],5),
                    "sl": round(c1["low"]-(c1["high"]-c1["low"])*0.15, 5)}
        else:
            if c1["close"] <= c1["open"]: continue
            if body1/r1 < 0.50: continue
            prev_high = df["high"].iloc[max(0,i-10):i].max()
            if c1["high"] <= prev_high: continue
            if body2 > body1*0.30: continue
            if c3["close"] >= c3["open"]: continue
            if c3["close"] >= (c1["open"]+c1["close"])/2: continue
            return {"c3_idx": i+2, "entry": round(c3["close"],5),
                    "sl": round(c1["high"]+(c1["high"]-c1["low"])*0.15, 5)}
    return None

def simulate_trade(df_full, entry_idx, entry, sl, tp1, tp2, direction):
    for i in range(entry_idx+1, min(entry_idx+100, len(df_full))):
        h = df_full["high"].iloc[i]
        l = df_full["low"].iloc[i]
        if direction == "bullish":
            if l <= sl: return "SL", -1.0, i-entry_idx
            if h >= tp2: return "TP2", 4.0, i-entry_idx
            if h >= tp1: return "TP1", 2.0, i-entry_idx
        else:
            if h >= sl: return "SL", -1.0, i-entry_idx
            if l <= tp2: return "TP2", 4.0, i-entry_idx
            if l <= tp1: return "TP1", 2.0, i-entry_idx
    return "OPEN", 0, 100

def backtest_dbos(sym_name, yf_sym, tf):
    df_full = get_candles(yf_sym, tf)
    if df_full.empty or len(df_full) < 100: return []
    trades = []
    last_bar = 0
    for start in range(80, len(df_full)-10):
        if start - last_bar < 20: continue
        df = df_full.iloc[:start]
        trend = detect_trend_structure(df)
        if trend == "neutral": continue
        dbos = detect_dbos(df, trend)
        if not dbos: continue
        idm = find_idm(df, dbos["index"], trend)
        if not idm: continue
        ob = find_ob(df, idm["index"], trend)
        if not ob: continue
        ob_size = ob["top"] - ob["bottom"]
        if ob_size <= 0: continue
        min_risk = ob_size * 1.5
        if trend == "bullish":
            entry = ob["top"]
            sl = ob["bottom"] - ob_size*0.15
            if (entry-sl) < min_risk: sl = entry - min_risk
            risk = entry - sl
            tp1 = entry + risk*2
            tp2 = entry + risk*4
        else:
            entry = ob["bottom"]
            sl = ob["top"] + ob_size*0.15
            if (sl-entry) < min_risk: sl = entry + min_risk
            risk = sl - entry
            tp1 = entry - risk*2
            tp2 = entry - risk*4
        result, rr, bars = simulate_trade(df_full, start, entry, sl, tp1, tp2, trend)
        last_bar = start
        trades.append({
            "strategy":"DBOS","symbol":sym_name,"tf":tf,
            "date":str(df_full.index[start])[:10],
            "direction":trend,"entry":round(entry,5),"sl":round(sl,5),
            "tp1":round(tp1,5),"tp2":round(tp2,5),
            "result":result,"rr":rr,"bars":bars,
            "conditions":"DBOS+IDM+OB"
        })
    return trades

def backtest_ms(sym_name, yf_sym, tf):
    df_full = get_candles(yf_sym, tf)
    if df_full.empty or len(df_full) < 60: return []
    trades = []
    last_bar = 0
    # H4 للفلترة
    df_h4 = df_full.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    for start in range(50, len(df_full)-10):
        if start - last_bar < 10: continue
        df = df_full.iloc[:start]
        # ترند H4
        h4_end = df_h4[df_h4.index <= df_full.index[start]]
        h4_trend = detect_trend_structure(h4_end)
        if h4_trend == "neutral": continue
        pattern = detect_morning_star_bt(df, h4_trend)
        if not pattern: continue
        if pattern["c3_idx"] < len(df)-6: continue
        entry = pattern["entry"]
        sl    = pattern["sl"]
        risk  = abs(entry-sl)
        if risk <= 0: continue
        if h4_trend == "bullish":
            tp1 = entry+risk*2; tp2 = entry+risk*4
        else:
            tp1 = entry-risk*2; tp2 = entry-risk*4
        result, rr, bars = simulate_trade(df_full, start, entry, sl, tp1, tp2, h4_trend)
        last_bar = start
        trades.append({
            "strategy":"Morning Star","symbol":sym_name,"tf":tf,
            "date":str(df_full.index[start])[:10],
            "direction":h4_trend,"entry":entry,"sl":sl,
            "tp1":round(tp1,5),"tp2":round(tp2,5),
            "result":result,"rr":rr,"bars":bars,
            "conditions":"C1-bearish+C2-small+C3-bullish"
        })
    return trades

def report(trades, name):
    if not trades: return f"\n{name}: لا صفقات\n"
    total = len(trades)
    tp2c = sum(1 for t in trades if t["result"]=="TP2")
    tp1c = sum(1 for t in trades if t["result"]=="TP1")
    slc  = sum(1 for t in trades if t["result"]=="SL")
    opn  = sum(1 for t in trades if t["result"]=="OPEN")
    wins = tp1c+tp2c
    wr   = wins/(wins+slc)*100 if (wins+slc)>0 else 0
    avg_rr = sum(t["rr"] for t in trades if t["result"]!="OPEN")/max(len([t for t in trades if t["result"]!="OPEN"]),1)

    by_sym = {}
    for t in trades:
        s = t["symbol"]
        if s not in by_sym: by_sym[s]={"w":0,"l":0}
        if t["result"] in ["TP1","TP2"]: by_sym[s]["w"]+=1
        elif t["result"]=="SL": by_sym[s]["l"]+=1

    sym_lines = "\n".join([f"  {s}: {v['w']}W/{v['l']}L = {v['w']/max(v['w']+v['l'],1)*100:.0f}%" for s,v in sorted(by_sym.items(), key=lambda x: x[1]['w']/(x[1]['w']+x[1]['l']+0.001), reverse=True)])

    return f"""
{'='*50}
{name}
{'='*50}
اجمالي الصفقات : {total}
TP2 (4R)       : {tp2c} ({tp2c/total*100:.1f}%)
TP1 (2R)       : {tp1c} ({tp1c/total*100:.1f}%)
SL  (-1R)      : {slc}  ({slc/total*100:.1f}%)
مفتوحة         : {opn}
Win Rate       : {wr:.1f}%
متوسط RR       : {avg_rr:.2f}R
{'─'*30}
النتائج حسب الزوج:
{sym_lines}
"""

def detail(trades, n=50):
    lines = []
    for t in trades[:n]:
        icon = "✅" if t["result"] in ["TP1","TP2"] else "❌" if t["result"]=="SL" else "⏳"
        lines.append(f"{icon} {t['date']} | {t['symbol']} {t['tf']} | {t['direction'][:4]} | دخول:{t['entry']} ستوب:{t['sl']} | {t['result']}({t['rr']:+.1f}R) | {t['bars']}شمعة | {t['conditions']}")
    return "\n".join(lines)

if __name__ == "__main__":
    print("بدء الباك تيست - سنتين")
    print("="*50)

    all_dbos, all_ms = [], []

    for sym, yf_sym in SYMBOLS.items():
        for tf in TIMEFRAMES:
            print(f"{sym} {tf}...", end=" ", flush=True)
            d = backtest_dbos(sym, yf_sym, tf)
            all_dbos.extend(d)
            m = backtest_ms(sym, yf_sym, tf)
            all_ms.extend(m)
            print(f"DBOS:{len(d)} MS:{len(m)}")

    r1 = report(all_dbos, "DBOS Strategy")
    r2 = report(all_ms,   "Morning Star")

    full = f"""BACKTEST - آخر سنتين
{datetime.now().strftime('%Y-%m-%d %H:%M')}
{r1}
{r2}
{'='*50}
تفاصيل DBOS (اول 50):
{'='*50}
{detail(all_dbos)}

{'='*50}
تفاصيل Morning Star (اول 50):
{'='*50}
{detail(all_ms)}
"""

    with open("backtest_results.txt","w",encoding="utf-8") as f:
        f.write(full)
    with open("backtest_trades.json","w",encoding="utf-8") as f:
        json.dump(all_dbos+all_ms, f, ensure_ascii=False, indent=2)

    print(full)
    print(f"\nتم الحفظ: backtest_results.txt")
    print(f"DBOS:{len(all_dbos)} | Morning Star:{len(all_ms)}")
