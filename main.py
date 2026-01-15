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
ACCOUNT_EQUITY = 6000.0          # hard cap (USD)
RISK_PCT = 0.02                  # <= 2% risk per trade
TARGET_CASH_UTIL = 0.95          # aim to deploy ~95% of cash (>=90% whenever feasible)
MIN_CASH_UTIL = 0.90

# Broker / contract conventions (per your note):
# - SPX500 is quoted with 10x "price scale" versus the proxy (decimal moved right 1 place).
# - Units are also moved right 1 place (x10 units).
# To keep notional/risk consistent in USD while scaling price and units, we price contracts with a multiplier.
PRICE_SCALE = 10.0               # e.g., proxy 677.06 -> SPX500 6770.60
CONTRACT_MULTIPLIER = 0.1       # notional = units * price * multiplier

# If your broker allows smaller/larger increments, change CONTRACT_STEP accordingly.
CONTRACT_STEP = 0.1

# Tick size on the PROXY scale. It will be scaled by PRICE_SCALE internally.
# Example: 0.10 proxy ticks -> 1.0 ticks on SPX500 quoted scale after x10.
TICK_SIZE_PROXY = 0.10


# Massive API symbols to try for historical daily OHLC.
# We use a proxy for ATR and last close, then scale to SPX500 quoted level.
HISTORICAL_PROXY_CANDIDATES = [
    "SPX500",   # try direct, if Massive supports it
    "SPX",      # if Massive supports S&P 500 index
    "SPY",      # ETF fallback
]


# ---------------------------
# MassiveStocksProvider (async)
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
        """
        Return a list of dicts with open/high/low/close/volume/date using Massive v2 aggs.
        """
        if not start:
            start = date.today() - timedelta(days=365 * 10)
        if not end:
            end = date.today()

        s = start.isoformat()
        e = end.isoformat()

        url = (
            f"{self.base}/v2/aggs/ticker/{symbol}/range/1/day/{s}/{e}"
            f"?adjusted={'true' if adjusted else 'false'}"
            f"&sort=asc&limit=50000&apiKey={self.api_key}"
        )

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)

        if r.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Massive aggs error {r.status_code}: {r.text}",
                request=r.request,
                response=r,
            )

        data = r.json()
        results = data.get("results") or []

        # IMPORTANT: empty results must be treated as a failure so we can try fallbacks.
        if len(results) == 0:
            raise RuntimeError(
                f"No daily OHLC results for ticker '{symbol}'. "
                f"Check Massive symbol support / permissions. Response keys: {list(data.keys())}"
            )

        out: List[Dict] = []
        for row in results:
            out.append(
                {
                    "date": row["t"],  # ms timestamp
                    "open": row["o"],
                    "high": row["h"],
                    "low": row["l"],
                    "close": row["c"],
                    "volume": row.get("v"),
                }
            )
        return out


# ---------------------------
# ATR(14) - simple Wilder TR average over last 14 TRs
# ---------------------------
def compute_atr14(daily: List[Dict]) -> float:
    if len(daily) < 15:
        raise ValueError("Not enough daily history to compute ATR(14). Need at least 15 bars.")

    daily_sorted = sorted(daily, key=lambda x: x["date"])
    window = daily_sorted[-15:]  # 15 bars gives 14 TRs

    trs: List[float] = []
    prev_close = window[0]["close"]

    for bar in window[1:]:
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)
        prev_close = close

    return sum(trs) / len(trs)


# ---------------------------
# Rounding helpers
# ---------------------------
def floor_to_step(x: float, step: float) -> float:
    return math.floor(x / step) * step


def floor_to_tick(x: float, tick: float) -> float:
    return math.floor(x / tick) * tick


# ---------------------------
# Core sizing logic (NO MARGIN + <=2% risk + aggressive cash usage)
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
    """
    Returns:
      (units, stop_distance, cash_utilisation)

    Guarantees:
      - notional = units * entry_price * contract_multiplier <= equity * target_cash_util (=> no margin)
      - risk = units * stop_distance * contract_multiplier <= equity * risk_pct
      - targets ~target_cash_util cash usage where feasible, without margin
    """
    max_risk = equity * risk_pct

    # 1) Aggressive cash-first sizing (no margin)
    max_notional = equity * target_cash_util
    cash_units = floor_to_step(max_notional / (entry_price * contract_multiplier), contract_step)
    if cash_units < contract_step:
        cash_units = contract_step

    # 2) Risk floor: even minimum stop (1 tick) must not violate risk
    min_stop = tick_size
    risk_cap_units = floor_to_step(max_risk / (min_stop * contract_multiplier), contract_step)
    if risk_cap_units < contract_step:
        risk_cap_units = contract_step

    units = min(cash_units, risk_cap_units)

    # 3) Choose stop distance so risk is <= cap (snapped DOWN to ticks)
    raw_stop = max_risk / (units * contract_multiplier)
    stop_distance = floor_to_tick(raw_stop, tick_size)
    if stop_distance < tick_size:
        stop_distance = tick_size  # safe because units respects risk at 1 tick

    cash_util = (units * entry_price * contract_multiplier) / equity
    return units, stop_distance, cash_util


# ---------------------------
# Signal generation (SPX500 only)
# ---------------------------
async def generate_spx500_signal(
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 3.0,
) -> Dict:
    provider = MassiveStocksProvider()

    # Resolve best available historical proxy through Massive
    daily: Optional[List[Dict]] = None
    used_proxy: Optional[str] = None
    last_err: Optional[Exception] = None

    for proxy in HISTORICAL_PROXY_CANDIDATES:
        try:
            daily = await provider.get_daily_data(symbol=proxy)
            used_proxy = proxy
            break
        except Exception as e:
            last_err = e
            continue

    if daily is None:
        raise RuntimeError(f"Unable to fetch historical data from Massive. Last error: {last_err}")

    if len(daily) < 20:
        raise RuntimeError("Not enough history to compute ATR-based levels.")

    # Proxy values
    atr14_proxy = compute_atr14(daily)
    entry_proxy = float(daily[-1]["close"])

    # Scale to SPX500 quoted level (decimal moved right by 1 place)
    atr14 = atr14_proxy * PRICE_SCALE
    entry = entry_proxy * PRICE_SCALE
    tick_size = TICK_SIZE_PROXY * PRICE_SCALE

    # 1) Aggressive size that NEVER uses margin and NEVER risks >2%
    units, stop_dist_cap, cash_util = size_position_no_margin(
        equity=ACCOUNT_EQUITY,
        entry_price=entry,
        tick_size=tick_size,
        contract_step=CONTRACT_STEP,
        risk_pct=RISK_PCT,
        target_cash_util=TARGET_CASH_UTIL,
        contract_multiplier=CONTRACT_MULTIPLIER,
    )

    min_cash_target_met = cash_util >= MIN_CASH_UTIL

    # 2) ATR-aware stop (but never wider than the risk-cap stop)
    atr_stop = floor_to_tick(sl_atr_mult * atr14, tick_size)
    if atr_stop < tick_size:
        atr_stop = tick_size

    stop_distance = min(stop_dist_cap, atr_stop)  # never increases risk beyond cap

    # Keep classic RR by scaling TP off chosen stop.
    rr_ratio = (tp_atr_mult / sl_atr_mult) if sl_atr_mult > 0 else 2.0
    tp_distance = floor_to_tick(stop_distance * rr_ratio, tick_size)
    if tp_distance < tick_size:
        tp_distance = tick_size

    # 3) Prices (LONG-only level values + (Short X) mirror for downstream reversals)
    stop_price = entry - stop_distance
    take_profit_price = entry + tp_distance

    # 4) Hard checks (include multiplier)
    max_risk = ACCOUNT_EQUITY * RISK_PCT
    actual_risk = units * stop_distance * CONTRACT_MULTIPLIER
    notional = units * entry * CONTRACT_MULTIPLIER

    if notional > ACCOUNT_EQUITY + 1e-9:
        raise RuntimeError("Invariant failed: margin would be required (notional > equity).")

    if actual_risk > max_risk + 1e-9:
        raise RuntimeError("Invariant failed: risk > 2% cap.")

    # 5) Output signal payload
    signal = {
        "symbol": TRADE_SYMBOL,
        "historical_proxy_used": used_proxy,

        "equity": ACCOUNT_EQUITY,
        "entry_price": round(entry, 2),

        # Units are on the x10 scale because entry is x10 and multiplier is 0.01
        "units": round(units / 100.0, 4),

        # Notional/risk are computed in USD with multiplier
        "notional": round(notional, 2),
        "cash_utilisation": round(cash_util, 4),
        "cash_utilisation_meets_90pct": bool(min_cash_target_met),

        "atr14": round(atr14, 4),
        "tick_size": tick_size,

        # LONG-only level values + bracketed short mirror (exactly as requested)
        "stop_loss": f"{stop_price:.2f} (Short {take_profit_price:.2f})",
        "take_profit": f"{take_profit_price:.2f} (Short {stop_price:.2f})",

        # For auditing
        "stop_distance": round(stop_distance, 4),
        "take_profit_distance": round(tp_distance, 4),
        "max_risk": round(max_risk, 2),
        "actual_risk": round(actual_risk, 2),
        "risk_pct": RISK_PCT,

        # Contract conventions (for downstream auditing/execution)
        "price_scale": PRICE_SCALE,
        "contract_multiplier": CONTRACT_MULTIPLIER,
    }

    return signal


def print_trade_plan(signal: Dict) -> None:
    print("\n=== SPX500 TRADE PLAN (CASH ONLY, NO MARGIN) ===")
    print(f"Symbol:                {signal['symbol']}  (proxy: {signal['historical_proxy_used']})")
    print(f"Account equity:        ${signal['equity']:,.2f}")
    print(f"Entry (last close):    {signal['entry_price']:.2f}")
    print(f"ATR(14):               {signal['atr14']:.4f}")
    print(f"Tick size:             {signal['tick_size']:.2f}")

    print("Levels (long-first, short-mirror in brackets):")
    print(f"  Stop loss:           {signal['stop_loss']}")
    print(f"  Take profit:         {signal['take_profit']}")
    print()

    print("Position sizing (cash-only):")
    print(f"  Units:               {signal['units']}")
    print(f"  Notional:            ${signal['notional']:,.2f}")
    print(
        f"  Cash utilisation:    {signal['cash_utilisation']*100:.2f}% "
        f"(>=90%: {signal['cash_utilisation_meets_90pct']})"
    )
    print()

    print("Risk controls:")
    print(f"  Max risk (2%):        ${signal['max_risk']:,.2f}")
    print(f"  Actual risk @ SL:     ${signal['actual_risk']:,.2f}")
    print(f"  Stop distance:        {signal['stop_distance']:.2f}")
    print(f"  Take profit distance: {signal['take_profit_distance']:.2f}")
    print("==============================================\n")


async def main() -> None:
    signal = await generate_spx500_signal()
    print_trade_plan(signal)

    # Optional: print raw dict for logging/automation
    # print(signal)


if __name__ == "__main__":
    asyncio.run(main())
