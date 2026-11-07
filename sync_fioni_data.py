#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
菲欧尼之心 · 云端数据全量同步引擎 (v4.0 含 SPX / VOO / NQMAIN 2612 期货主连)
"""
import os
import json
import base64
import math
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf

# 敏感标的（SPLV）Base64 动态脱敏，源码不留明文
_FACTOR_Y_REAL = base64.b64decode("U1BMVg==").decode("utf-8")

SECTOR_CONFIG = {
    "👑 置顶3X多空": ['SOXL', 'SOXS', 'TQQQ', 'SQQQ'],
    "🌐 置顶大盘与期货": ['SOXX', 'QQQ', 'SPY', '^GSPC', 'VOO', 'NQZ26.CME'],
    "⚡ 核心高弹性池": ['SPY', 'VOO', '^GSPC', 'NQZ26.CME', 'MRVL', 'LITE', 'BE', 'SOXL', 'TQQQ', 'MRNA', 'PLTR', 'MSFT', 'NOW', 'COHR', 'AXTI', 'CRCL', 'SPCX', 'NBIS'],
    "💾 芯片与存储硬件": ['SOXX', 'NVDA', 'AVGO', 'TSM', 'ASML', 'AMD', 'ARM', 'INTC', 'MU', 'STX', 'WDC', 'SNDK'],
    "☁️ 科技巨头与加密": ['QQQ', 'BTC-USD', 'MSTR', 'AAPL', 'GOOG', 'AMZN', 'META', 'TSLA', 'ORCL', 'DELL', 'CRWV', 'ALAB'],
    "🛡️ 因子Y与宏观防御": [_FACTOR_Y_REAL, 'KO', 'SCHD', 'GLD', 'GDX', 'VST', 'NLR']
}

ALL_TICKERS = list(dict.fromkeys([s for group in SECTOR_CONFIG.values() for s in group]))

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain.iloc[-1] / (avg_loss.iloc[-1] + 1e-9)
    return float(100.0 - (100.0 / (1.0 + rs)))

def calculate_fioni_levels(df_1h: pd.DataFrame, sym: str) -> dict:
    sub = df_1h.tail(60).copy()
    spot = float(sub['close'].iloc[-1])
    rsi = compute_rsi(sub['close'], 14)

    recent20 = sub.tail(20)
    mean20 = float(recent20['close'].mean())
    std20 = float(recent20['close'].std())
    if math.isnan(std20) or std20 <= 0:
        std20 = spot * 0.015
    boll_lower = mean20 - 2.0 * std20
    boll_upper = mean20 + 2.0 * std20
    swing_low = float(recent20['low'].min())
    swing_high = float(recent20['high'].max())

    low_min = float(sub['low'].min())
    high_max = float(sub['high'].max())
    bins = 24
    edges = np.linspace(low_min, high_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vols = np.zeros(bins, dtype=float)

    typ = (sub['high'].values + sub['low'].values + sub['close'].values) / 3.0
    v_arr = sub['volume'].values
    idx = np.clip(np.digitize(typ, edges) - 1, 0, bins - 1)
    for i, v in zip(idx, v_arr):
        vols[i] += float(v) if v > 0 else 1.0  # 兼容 SPX 指数无成交量时按等权价格分布计算

    poc_idx = int(np.argmax(vols))
    tot_vol = vols.sum() if vols.sum() > 0 else 1.0
    cum_vol = vols[poc_idx]
    l, r = poc_idx, poc_idx
    while cum_vol < 0.70 * tot_vol and (l > 0 or r < bins - 1):
        vl = vols[l - 1] if l > 0 else -1
        vr = vols[r + 1] if r < bins - 1 else -1
        if vl >= vr:
            l -= 1
            cum_vol += vols[l]
        else:
            r += 1
            cum_vol += vols[r]

    val_price = float(centers[l])
    vah_price = float(centers[r])

    sup = max(swing_low, min(val_price, boll_lower))
    res = min(swing_high, max(vah_price, boll_upper))
    if sup >= spot:
        sup = spot * 0.982
    if res <= spot:
        res = spot * 1.018

    dense = min(spot * 0.99, 0.45 * sup + 0.35 * val_price + 0.20 * swing_low)
    if dense >= sup:
        dense = sup * 0.988

    tag = "安全持股"
    if rsi <= 35:
        tag = "🔥双周期见底"
    elif rsi <= 41:
        tag = "⚡4H超卖"
    elif rsi >= 68:
        tag = "🚨双周期触顶"
    elif rsi >= 62:
        tag = "📡4H阻力"

    return {
        "spot": round(spot, 2),
        "dense": round(dense, 2),
        "sup": round(sup, 2),
        "res": round(res, 2),
        "rsi": round(rsi, 1),
        "tag": tag
    }

def fetch_single_ticker(sym: str):
    # 若为纳指100期货主连2612，优先抓取 NQZ26.CME，若数据源需要连续合约名则自动回退 NQ=F
    candidates = ['NQZ26.CME', 'NQ=F'] if sym == 'NQZ26.CME' else [sym]
    for query_sym in candidates:
        try:
            tk = yf.Ticker(query_sym)
            df_1h = tk.history(period="1mo", interval="1h")
            if df_1h is not None and not df_1h.empty and len(df_1h) >= 15:
                df_1h.columns = [str(c).lower() for c in df_1h.columns]
                levels = calculate_fioni_levels(df_1h, sym)
                return sym, levels
        except Exception:
            continue
    return None

def main():
    bj_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[*] 启动全市场标的行情与量化矩阵同步 (北京时间: {bj_time})...")

    quotes = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_single_ticker, sym): sym for sym in ALL_TICKERS}
        for f in as_completed(futures):
            res = f.result()
            if res:
                sym, data = res
                quotes[sym] = data

    print(f"[+] 成功抓取并解算 {len(quotes)}/{len(ALL_TICKERS)} 支标的真实数据")

    export_payload = {
        "updated_at": bj_time,
        "quotes": quotes
    }

    with open("fioni_data.json", "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    print("[✓] 成功写入 fioni_data.json！")

if __name__ == "__main__":
    main()
