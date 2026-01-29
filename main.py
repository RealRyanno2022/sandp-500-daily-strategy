import os
import math
import asyncio
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple

import httpx

# =========================
# Fixed trading constraints
# =========================
TRADE_SYMBOL = "SPX500"
ACCOUNT_EQUITY = 6000.0
RISK_PCT = 0.02
TARGET_CASH_UTIL = 0.95
MIN_CASH_UTIL = 0.90

# =========================
# Trend / Thesis config
# =========================
TREND_LOOKBACK = 20

IMPULSE_EFFICIENCY = 0.70
GRIND_EFFICIENCY = 0.40
CHOP_EFFICIENCY = 0.25

IMPULSE_NORM_SLOPE = 0.03
GRIND_NORM_SLOPE = 0.015
FLAT_NORM_SLOPE = 0.01

# =========================
# Broker / contract config
# =========================
PRICE_SCALE = 10.0
CONTRACT_MULTIPLIER = 0.1
CONTRACT_STEP = 0.1
TICK_SIZE_PROXY = 0.10

HISTORICAL_PROXY_CANDIDATES = ["SPX500", "SPX", "SPY"]


# ---------------------------
# MassiveStocksProvider
# ---------------------------
class MassiveStocksProvider:
    def __init__(self):
        self.api_key = os.getenv("MASSIVE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing MASSIVE_API_KEY in environment")
        self.base = "https://api.massive.com"

    async def get_daily_data(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        adjusted: bool = True,
    ) -> List[Dict]:

        if not start:
            start = date.today() - timedelta(days=365 * 10)
        if not end:
            end = date.today()

        url = (
            f"{self.base}/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
            f"?adjusted={'true' if adjusted else 'false'}"
            f"&sort=asc&limit=50000&apiKey={self.api_key}"
        )

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)

        if r.status_code != 200:
            raise RuntimeError(r.text)

        results = r.json().get("results") or []
        if not results:
            raise RuntimeError(f"No data for {symbol}")

        return [
            {
                "date": row["t"],
                "open": row["o"],
                "high": row["h"],
                "low": row["l"],
                "close": row["c"],
                "volume": row.get("v"),
            }
            for row in results
        ]


# ---------------------------
# ATR(14)
# ---------------------------
def compute_atr14(daily: List[Dict]) -> float:
    window = sorted(daily, key=lambda x: x["date"])[-15:]
    trs = []
    prev_close = window[0]["close"]

    for bar in window[1:]:
        tr = max(
            bar["high"] - bar["low"],
            abs(bar["high"] - prev_close),
            abs(bar["low"] - prev_close),
        )
        trs.append(tr)
        prev_close = bar["close"]

    return sum(trs) / len(trs)


# ---------------------------
# Helpers
# ---------------------------
def floor_to_step(x: float, step: float) -> float:
    return math.floor(x / step) * step


def floor_to_tick(x: float, tick: float) -> float:
    return math.floor(x / tick) * tick


# ---------------------------
# Position sizing
# ---------------------------
def size_position_no_margin(
    equity: float,
    entry_price: float,
    tick_size: float,
    contract_step: float,
    risk_pct: float,
    target_cash_util: float,
    contract_multiplier: float,
) -> Tuple[float, float, float]:

    max_risk = equity * risk_pct
    max_notional = equity * target_cash_util

    cash_units = floor_to_step(
        max_notional / (entry_price * contract_multiplier),
        contract_step,
    )
    risk_units = floor_to_step(
        max_risk / (tick_size * contract_multiplier),
        contract_step,
    )

    units = max(contract_step, min(cash_units, risk_units))

    stop_distance = floor_to_tick(
        max_risk / (units * contract_multiplier),
        tick_size,
    )

    cash_util = (units * entry_price * contract_multiplier) / equity
    return units, stop_distance, cash_util


# ---------------------------
# Trend & Thesis classifier
# ---------------------------
def classify_trend_and_thesis(
    daily: List[Dict],
    atr14: float,
) -> Dict[str, str]:

    closes = [b["close"] for b in daily[-TREND_LOOKBACK:]]
    returns = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    x = list(range(len(closes)))
    mx = sum(x) / len(x)
    my = sum(closes) / len(closes)

    slope = sum((xi - mx) * (yi - my) for xi, yi in zip(x, closes)) / (
        sum((xi - mx) ** 2 for xi in x) or 1
    )

    efficiency = abs(closes[-1] - closes[0]) / (sum(abs(r) for r in returns) + 1e-9)
    norm_slope = slope / atr14 if atr14 else 0.0

    if efficiency > IMPULSE_EFFICIENCY and abs(norm_slope) > IMPULSE_NORM_SLOPE:
        trend = "Impulse"
    elif efficiency > GRIND_EFFICIENCY and abs(norm_slope) > GRIND_NORM_SLOPE:
        trend = "Grind"
    elif efficiency < CHOP_EFFICIENCY and abs(norm_slope) < FLAT_NORM_SLOPE:
        trend = "Chop"
    elif abs(norm_slope) < FLAT_NORM_SLOPE:
        trend = "Flat"
    else:
        trend = "Transition"

    if trend == "Impulse":
        thesis = "Strong buy" if norm_slope > 0 else "Strong sell"
    elif trend == "Grind":
        thesis = "Conditional buy" if norm_slope > 0 else "Conditional sell"
    elif trend == "Transition":
        thesis = "Risky buy" if norm_slope > 0 else "Risky sell"
    else:
        thesis = "No trade"

    return {"trend": trend, "thesis": thesis}


# ---------------------------
# Signal generation
# ---------------------------
async def generate_spx500_signal() -> Dict:
    provider = MassiveStocksProvider()

    for proxy in HISTORICAL_PROXY_CANDIDATES:
        try:
            daily = await provider.get_daily_data(proxy)
            used_proxy = proxy
            break
        except Exception:
            continue
    else:
        raise RuntimeError("No valid historical proxy")

    atr_proxy = compute_atr14(daily)
    entry_proxy = daily[-1]["close"]

    atr14 = atr_proxy * PRICE_SCALE
    entry = entry_proxy * PRICE_SCALE
    tick_size = TICK_SIZE_PROXY * PRICE_SCALE

    units, stop_cap, cash_util = size_position_no_margin(
        ACCOUNT_EQUITY,
        entry,
        tick_size,
        CONTRACT_STEP,
        RISK_PCT,
        TARGET_CASH_UTIL,
        CONTRACT_MULTIPLIER,
    )

    stop_distance = min(stop_cap, floor_to_tick(1.5 * atr14, tick_size))
    tp_distance = floor_to_tick(2 * stop_distance, tick_size)

    trend_info = classify_trend_and_thesis(daily, atr_proxy)

    return {
        "symbol": TRADE_SYMBOL,
        "proxy": used_proxy,
        "equity": ACCOUNT_EQUITY,
        "entry": round(entry, 2),
        "atr14": round(atr14, 4),
        "tick_size": tick_size,
        "trend": trend_info["trend"],
        "thesis": trend_info["thesis"],
        "stop": f"{entry - stop_distance:.2f} (Short {entry + tp_distance:.2f})",
        "tp": f"{entry + tp_distance:.2f} (Short {entry - stop_distance:.2f})",
        "units": round(units / 100, 4),
        "notional": round(units * entry * CONTRACT_MULTIPLIER, 2),
        "cash_util": round(cash_util, 4),
        "risk": round(units * stop_distance * CONTRACT_MULTIPLIER, 2),
    }


def print_trade_plan(s: Dict) -> None:
    print("\n=== SPX500 TRADE PLAN (CASH ONLY, NO MARGIN) ===")
    print(f"Symbol:                {s['symbol']}  (proxy: {s['proxy']})")
    print(f"Account equity:        ${s['equity']:,.2f}")
    print(f"Entry (last close):    {s['entry']:.2f}")
    print(f"ATR(14):               {s['atr14']:.4f}")
    print(f"Tick size:             {s['tick_size']:.2f}")
    print()
    print(f"Trend:                 {s['trend']}")
    print(f"Trade thesis:          {s['thesis']}")
    print()
    print("Levels (long-first, short-mirror in brackets):")
    print(f"  Stop loss:           {s['stop']}")
    print(f"  Take profit:         {s['tp']}")
    print("==============================================\n")


async def main() -> None:
    signal = await generate_spx500_signal()
    print_trade_plan(signal)


if __name__ == "__main__":
    asyncio.run(main())
