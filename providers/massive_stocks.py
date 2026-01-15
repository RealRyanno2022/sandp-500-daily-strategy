import os
import math
from datetime import date, timedelta
from typing import List, Dict, Optional

import httpx
import asyncio


# ---------------------------
# MassiveStocksProvider (async)
# ---------------------------
class MassiveStocksProvider:
    def __init__(self):
        self.api_key = os.getenv("MASSIVE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing MASSIVE_API_KEY in .env or environment")
        self.base = "https://api.massive.com"

    async def get_daily_data(
        self,
        symbol: str = "SPY",
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

        out: List[Dict] = []
        for row in data.get("results", []):
            out.append(
                {
                    "date": row["t"],    # ms timestamp
                    "open": row["o"],
                    "high": row["h"],
                    "low": row["l"],
                    "close": row["c"],
                    "volume": row.get("v"),
                    "vw": row.get("vw"),
                    "count": row.get("n"),
                }
            )
        return out


# ---------------------------
# Utility: ATR(14)
# ---------------------------
def compute_atr14(daily: List[Dict]) -> float:
    """
    Compute ATR(14) using Wilder TR.
    Expects daily sorted ascending by 'date'.
    """
    if len(daily) < 15:
        raise ValueError("Not enough daily history to compute ATR(14). Need at least 15 bars.")

    daily_sorted = sorted(daily, key=lambda x: x["date"])
    window = daily_sorted[-15:]

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

    atr14 = sum(trs) / len(trs)
    return atr14


# ---------------------------
# CFD signal generation (async)
# ---------------------------
async def generate_cfd_signal(
    symbol: str,
    equity: float = 6000.0,
    risk_pct: float = 0.02,      # 2%
    tick_size: float = 0.01,     # default US stock tick
    sl_atr_mult: float = 4.0,    # wide SL
    tp_atr_mult: float = 8.0,    # wide TP
) -> None:
    """
    Generate a CFD trade plan:
      - Max risk: risk_pct * equity (default 2% of 6000 = 120)
      - Stop distance: sl_atr_mult * ATR(14)
      - Take profit:  tp_atr_mult * ATR(14)
      - Output SL/TP in ticks + prices, position size, $/tick, actual $ risk.

    Replace this function body with your own CFD logic if you prefer.
    """
    provider = MassiveStocksProvider()
    daily = await provider.get_daily_data(symbol=symbol)

    if len(daily) < 20:
        raise SystemExit(f"Not enough history for {symbol} to compute ATR-based levels.")

    atr14 = compute_atr14(daily)
    last_close = daily[-1]["close"]

    # Max risk
    risk_amount = equity * risk_pct  # e.g. 6000 * 0.02 = 120

    # Wide stop / target based on ATR
    sl_distance_dollars = sl_atr_mult * atr14
    tp_distance_dollars = tp_atr_mult * atr14

    # Convert to ticks
    sl_ticks = max(1, int(round(sl_distance_dollars / tick_size)))
    tp_ticks = max(1, int(round(tp_distance_dollars / tick_size)))

    # Position sizing: risk_amount ≈ position_size * sl_ticks * tick_size
    denom = sl_ticks * tick_size
    if denom <= 0:
        raise SystemExit("Invalid configuration: stop distance or tick size is zero.")

    position_size = int(risk_amount / denom)
    if position_size < 1:
        position_size = 1  # at least 1 unit

    # Actual risk & per-tick value
    actual_risk = position_size * sl_ticks * tick_size
    per_tick_dollars = position_size * tick_size

    # Price levels (assuming long)
    sl_price = last_close - sl_ticks * tick_size
    tp_price = last_close + tp_ticks * tick_size

    print("\n=== CFD TRADE PLAN ===")
    print(f"Symbol:            {symbol}")
    print(f"Portfolio equity:  ${equity:,.2f}")
    print(f"Max risk (2%):     ${risk_amount:,.2f}")
    print(f"Last close:        ${last_close:,.2f}")
    print(f"ATR(14):           ${atr14:,.2f}")
    print(f"Tick size:         ${tick_size:.4f}")
    print()
    print("Stop loss:")
    print(f"  Distance:        {sl_ticks} ticks (~${sl_ticks * tick_size:,.2f})")
    print(f"  Stop price:      ${sl_price:,.2f}")
    print()
    print("Take profit:")
    print(f"  Distance:        {tp_ticks} ticks (~${tp_ticks * tick_size:,.2f})")
    print(f"  Target price:    ${tp_price:,.2f}")
    print()
    print(f"Position size:     {position_size} CFDs (shares/units)")
    print(f"Per tick P&L:      ${per_tick_dollars:,.2f} per tick")
    print(f"Actual risk @ SL:  ${actual_risk:,.2f}")
    print("======================\n")


# ---------------------------
# Futures flow placeholder
# ---------------------------
async def run_futures_signal(symbol: str) -> None:
    """
    Placeholder for your existing futures signal logic.
    Replace this with your real futures branch.
    """
    print("\nRunning existing FUTURES signal logic...")
    print(f"(Ticker: {symbol})")
    # TODO: import and call your actual futures logic here.
    print("TODO: wire in your futures signal generator here.\n")


# ---------------------------
# CLI helpers
# ---------------------------
def ask_ticker_and_is_cfd() -> dict:
    """
    Ask for a ticker (3–5 letters, uppercased) and whether this is a CFD trade.
    Always runs both prompts.
    Returns a dict: {"ticker": <str>, "is_cfd": <bool>}
    """
    while True:
        # Ticker prompt + validation
        ticker = input("Enter ticker (3–5 letters, e.g. AVGO): ").strip().upper()
        if len(ticker) < 3 or len(ticker) > 5:
            print("Ticker must be 3–5 characters.\n")
            continue
        if not ticker.isalpha():
            print("Ticker must contain only letters.\n")
            continue

        # CFD prompt – accepts capital or lowercase Y/N
        while True:
            cfd_answer = input("Is this a CFD trade? (Y/N): ").strip().lower()
            if cfd_answer in ("y", "n"):
                cfd_boolean = (cfd_answer == "y")
                return {"ticker": ticker, "is_cfd": cfd_boolean}
            print("Please answer with Y or N.\n")
e

    




# ---------------------------
# Main async entry
# ---------------------------
async def main():
    # Ask both questions together
    info = ask_ticker_and_is_cfd()
    ticker = info["ticker"]
    is_cfd = info["is_cfd"]

    if not is_cfd:
        # FUTURES PATH (existing behaviour)
        await run_futures_signal(ticker)
        return

    # CFD PATH – call your CFD logic here
    equity_str = input("Portfolio equity (default 6000): ").strip()
    equity = float(equity_str) if equity_str else 6000.0

    tick_str = input("Tick size (default 0.01): ").strip()
    tick_size = float(tick_str) if tick_str else 0.01

    print("\nGenerating CFD signal...")
    await generate_cfd_signal(
        symbol=ticker,
        equity=equity,
        risk_pct=0.02,
        tick_size=tick_size,
        sl_atr_mult=4.0,
        tp_atr_mult=8.0,
    )




if __name__ == "__main__":
    asyncio.run(main())
