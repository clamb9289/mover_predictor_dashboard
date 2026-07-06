"""
Alpaca Trader
==============
Reads today's predictions from the database and executes trades via
the Alpaca API — no interactive login, no 2FA device approval required.

Run modes:
  python trader.py buy     — place opening orders at 10:00am ET
  python trader.py monitor — check positions, apply stop/floor (every 15 min)
  python trader.py sell    — close all remaining positions at 3:55pm ET
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, date

import pytz
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrderByIdRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tracker.db")
CANDIDATE_POOL_PATH = os.path.join(BASE_DIR, "candidate_pool.json")
TRADE_LOG = os.path.join(BASE_DIR, "trade_log.json")
TAX_LOG = os.path.join(BASE_DIR, "tax_liability.json")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
PERF_PATH = os.path.join(BASE_DIR, "performance.json")

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

STOP_LOSS_PCT    = -0.05   # Hard stop: sell if down 5% from buy price

# Staircase trailing stop — floor steps up as stock climbs
# Each tuple is (trigger_pct, floor_pct)
# e.g. once up 5%, floor locks at 5% — if it falls back below 5% we sell
TRAIL_STEPS = [
    (0.03, 0.03),   # Hit +3%  → floor at +3%
    (0.05, 0.05),   # Hit +5%  → floor steps up to +5%
    (0.07, 0.07),   # Hit +7%  → floor steps up to +7%
    (0.10, 0.10),   # Hit +10% → floor steps up to +10%
    (0.15, 0.15),   # Hit +15% → floor steps up to +15%
]
# Beyond the last step, trail dynamically at this % below peak
DYNAMIC_TRAIL_BELOW_PEAK = 0.02  # sell if drops 2% from all-time high
TAX_RATE          = 0.30

# Opening-move check — run right before buying, using intraday data now
# available since predictor.py's picks were based on yesterday's close.
# If a pick has already dropped hard since today's open and hasn't shown
# signs of stabilizing, skip buying it rather than chasing a same-day
# breakdown (e.g. TSLA's Q2-delivery "sell the news" reversal on 7/2).
OPENING_DROP_THRESHOLD = -0.03   # flag if down 3%+ from today's open at buy time
MIN_BOUNCE_TO_OVERRIDE = 0.01    # but allow the buy if it's already bounced 1%+ off the intraday low (sign of stabilizing)
ET = pytz.timezone("America/New_York")

# Alpaca credentials from GitHub Secrets
ALPACA_API_KEY    = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

# --------------------------------------------------------------------------
# MARKET HOURS
# --------------------------------------------------------------------------

MARKET_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-04-02",
    "2027-05-31", "2027-06-19", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
}


def is_market_open():
    now_et = datetime.now(ET)
    today_str = now_et.strftime("%Y-%m-%d")
    if now_et.weekday() >= 5:
        return False, f"Market closed — weekend ({now_et.strftime('%A')})"
    if today_str in MARKET_HOLIDAYS:
        return False, f"Market closed — holiday ({today_str})"
    market_open  = now_et.replace(hour=10, minute=0, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if now_et < market_open:
        return False, "Market not yet open (opens 9:30am ET)"
    if now_et >= market_close:
        return False, "Market already closed (4:00pm ET)"
    return True, f"Market open ({now_et.strftime('%I:%M%p ET')})"

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def is_trading_enabled():
    config = load_config()
    enabled = config.get("trading_enabled", False)
    if not enabled:
        print("Trading is DISABLED in config.json. Set trading_enabled to true to activate.")
    return enabled


def is_paper_mode():
    """Returns True if we should use Alpaca paper trading endpoints."""
    config = load_config()
    return config.get("alpaca_paper", True)


def check_risk_limits():
    config = load_config()
    starting_capital = float(config.get("starting_capital", 300.0))
    max_daily_loss_pct = float(config.get("max_daily_loss_pct", 0.05))
    max_drawdown_pct   = float(config.get("max_total_drawdown_pct", 0.15))

    today = date.today().strftime("%Y-%m-%d")
    trade_log = load_trade_log()
    tax_log   = load_tax_log()

    daily_pnl = 0.0
    if today in trade_log:
        for pos in trade_log[today].get("positions", {}).values():
            daily_pnl += pos.get("pnl", 0.0)

    if starting_capital > 0 and (daily_pnl / starting_capital) <= -max_daily_loss_pct:
        return False, (f"Daily loss limit hit: {daily_pnl/starting_capital:.1%} "
                       f"(limit -{max_daily_loss_pct:.0%}). Pausing for the day.")

    net_pnl = tax_log.get("net_pnl", 0.0)
    current_value = starting_capital + net_pnl
    drawdown_pct  = (current_value - starting_capital) / starting_capital if starting_capital else 0

    if drawdown_pct <= -max_drawdown_pct:
        reason = (f"TOTAL DRAWDOWN LIMIT HIT: portfolio at ${current_value:.2f} "
                  f"({drawdown_pct:.1%} from starting ${starting_capital:.2f}). "
                  f"Bot shutting itself down.")
        config["trading_enabled"] = False
        config["auto_disabled_reason"] = reason
        config["auto_disabled_date"] = today
        save_config(config)
        print(f"\n{'!'*60}\n{reason}\n{'!'*60}\n")
        return False, reason

    print(f"Risk check OK — daily P&L: ${daily_pnl:.2f}, "
          f"portfolio: ${current_value:.2f} ({drawdown_pct:+.1%} from start)")
    return True, None

# --------------------------------------------------------------------------
# ALPACA CLIENT
# --------------------------------------------------------------------------

def get_alpaca_client():
    """Returns an authenticated Alpaca TradingClient.
    paper=True  → paper trading endpoints (safe, no real money)
    paper=False → live trading endpoints
    """
    paper = is_paper_mode()
    client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=paper,
    )
    mode_label = "PAPER" if paper else "LIVE"
    print(f"Alpaca client initialized ({mode_label} mode)")
    return client

# --------------------------------------------------------------------------
# PRICE + ACCOUNT
# --------------------------------------------------------------------------

def get_current_price(ticker):
    """Uses yfinance for price quotes — no Alpaca market data subscription needed."""
    try:
        price = float(yf.Ticker(ticker).fast_info["lastPrice"])
        return price if price > 0 else None
    except Exception as e:
        print(f"  Price fetch failed for {ticker}: {e}")
        return None


def get_safe_buying_power(client):
    """Returns buying power capped at tracked portfolio value to prevent margin use."""
    config = load_config()
    starting_capital = float(config.get("starting_capital", 300.0))
    tax_log = load_tax_log()
    net_pnl = tax_log.get("net_pnl", 0.0)
    portfolio_value = starting_capital + net_pnl

    account = client.get_account()
    actual_bp = float(account.buying_power)

    safe_bp = min(actual_bp, portfolio_value)
    print(f"Buying power — Alpaca: ${actual_bp:.2f}, "
          f"Tracked portfolio: ${portfolio_value:.2f}, "
          f"Using (lower of two): ${safe_bp:.2f}")
    return safe_bp

def get_trail_floor(pct_change, peak_pct):
    """
    Returns (floor_pct, floor_label) given current % change and peak % reached.
    Implements the staircase trailing stop — floor steps up as stock climbs,
    then trails dynamically 2% below peak once past the last step.
    """
    # Find the highest step the peak has crossed
    active_floor = None
    active_label = None
    for trigger, floor in TRAIL_STEPS:
        if peak_pct >= trigger:
            active_floor = floor
            active_label = f"+{floor:.0%} floor (step)"

    if active_floor is None:
        return None, None  # No floor yet — below first trigger

    # If peak is beyond the last defined step, use dynamic trail
    last_trigger = TRAIL_STEPS[-1][0]
    if peak_pct > last_trigger:
        dynamic_floor = peak_pct - DYNAMIC_TRAIL_BELOW_PEAK
        active_floor = dynamic_floor
        active_label = f"+{dynamic_floor:.1%} floor (trailing {DYNAMIC_TRAIL_BELOW_PEAK:.0%} below peak +{peak_pct:.1%})"

    return active_floor, active_label


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def load_trade_log():
    if os.path.exists(TRADE_LOG):
        with open(TRADE_LOG) as f:
            return json.load(f)
    return {}


def save_trade_log(log):
    with open(TRADE_LOG, "w") as f:
        json.dump(log, f, indent=2)


def load_tax_log():
    if os.path.exists(TAX_LOG):
        with open(TAX_LOG) as f:
            return json.load(f)
    return {
        "total_gains": 0.0, "total_losses": 0.0, "net_pnl": 0.0,
        "tax_liability": 0.0, "set_aside": 0.0, "trades": []
    }


def save_tax_log(log):
    with open(TAX_LOG, "w") as f:
        json.dump(log, f, indent=2)


def get_today_predictions():
    conn = sqlite3.connect(DB_PATH)
    today = date.today().strftime("%Y-%m-%d")
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, rank FROM predictions WHERE predict_date=? ORDER BY rank",
        (today,)
    )
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]

# --------------------------------------------------------------------------
# ORDER FILL RECONCILIATION
# --------------------------------------------------------------------------

def get_order_fill_details(client, order_id, max_wait_seconds=30, poll_interval=3):
    """
    Polls Alpaca for the actual fill price after order submission.
    Returns dict with state, filled_qty, average_price, or None on timeout.
    """
    waited = 0
    while waited < max_wait_seconds:
        try:
            order = client.get_order_by_id(order_id)
            state = str(order.status)
            if state in ("filled", "rejected", "cancelled", "expired"):
                avg_price = float(order.filled_avg_price) if order.filled_avg_price else None
                filled_qty = float(order.filled_qty) if order.filled_qty else 0.0
                return {
                    "state": state,
                    "filled_quantity": filled_qty,
                    "average_price": avg_price,
                }
        except Exception as e:
            print(f"    (order status check failed: {e})")
        time.sleep(poll_interval)
        waited += poll_interval

    print(f"    WARNING: order {order_id} did not reach final state within {max_wait_seconds}s")
    return None

# --------------------------------------------------------------------------
# BUY
# --------------------------------------------------------------------------

def reconcile_unconfirmed_orders(client):
    """Positions logged as 'unconfirmed' (buy) or 'sell_unconfirmed' (sell) at
    order time may have actually filled on Alpaca shortly after our
    confirmation polling gave up. Re-check those orders by their saved
    order_id before doing anything else, so a real position never gets
    silently abandoned just because the initial confirmation timed out.

    Checks EVERY day in trade_log.json, not just today — a stuck
    unconfirmed/sell_unconfirmed status from days ago will otherwise never
    get revisited, since nothing else ever looks back at prior days."""
    log = load_trade_log()
    any_changed = False

    for day_str, day in log.items():
        for ticker, pos in day.get("positions", {}).items():
            if pos.get("status") not in ("unconfirmed", "sell_unconfirmed"):
                continue
            order_id = pos.get("order_id")
            if not order_id:
                continue

            print(f"  [{day_str}] {ticker}: re-checking previously unconfirmed order {order_id}...")
            fill = get_order_fill_details(client, order_id, max_wait_seconds=15, poll_interval=3)

            if not fill or fill["state"] != "filled" or not fill["average_price"]:
                if pos["status"] == "unconfirmed":
                    # Order status polling didn't reach a terminal state — this can
                    # happen with notional (dollar-based) orders on Alpaca even long
                    # after the trade actually filled. Fall back to the Positions
                    # endpoint, which reflects real holdings regardless of what the
                    # Orders endpoint reports.
                    print(f"    order status inconclusive (state={fill['state'] if fill else 'unknown'}) — checking actual position instead...")
                    try:
                        real_pos = client.get_open_position(ticker)
                        real_price = float(real_pos.avg_entry_price)
                        real_shares = float(real_pos.qty)
                        fill = {"state": "filled", "average_price": real_price, "filled_quantity": real_shares}
                        print(f"    FOUND real position: {real_shares:.6f} shares @ avg ${real_price:.2f}")
                    except Exception as e:
                        print(f"    no open position found either ({e}) — will retry next run")
                        continue
                else:
                    # sell_unconfirmed: if the position no longer exists on Alpaca
                    # at all, that confirms the sell genuinely went through — we
                    # just can't get an exact fill price from a position that's
                    # gone. sell_position() already stored a reasonable fallback
                    # sell_price (and already called record_pnl() once at the
                    # original attempt), so just finalize the status here rather
                    # than recomputing or re-recording anything.
                    print(f"    order status inconclusive (state={fill['state'] if fill else 'unknown'}) — checking if position still exists...")
                    try:
                        client.get_open_position(ticker)
                        print(f"    still shows as an open position — sell hasn't gone through yet, will retry next run")
                        continue
                    except Exception:
                        pos["status"] = "closed"
                        print(f"    CONFIRMED gone from account — sell succeeded (using previously recorded "
                              f"fallback sell price of ${pos.get('sell_price', 0):.2f})")
                        any_changed = True
                        continue

            real_price = fill["average_price"]
            real_shares = fill["filled_quantity"]

            if pos["status"] == "unconfirmed":
                pos["status"] = "open"
                pos["buy_price"] = real_price
                pos["shares"] = real_shares
                pos["cost"] = real_price * real_shares
                pos.pop("reason", None)
                print(f"    CONFIRMED: filled {real_shares:.6f} shares @ ${real_price:.2f}")
            else:  # sell_unconfirmed, but order-status polling itself resolved this time
                # Refine the sell_price to the real confirmed fill, but do NOT call
                # record_pnl() again — it was already recorded once at the original
                # sell attempt using the fallback price, and calling it again here
                # would double-count the gain/loss in tax_liability.json.
                pos["status"] = "closed"
                pos["sell_price"] = real_price
                pos["pnl"] = (real_price - pos["buy_price"]) * pos["shares"]
                pos.pop("reason", None)
                print(f"    SELL CONFIRMED: filled @ ${real_price:.2f} (refined from fallback estimate)")

            any_changed = True

        # A day may have been incorrectly marked 'closed' earlier because every
        # position looked non-open at the time. If any position in THIS day is
        # now confirmed genuinely open, re-open that day so monitoring resumes.
        if any(p.get("status") == "open" for p in day.get("positions", {}).values()):
            day["status"] = "open"

    if any_changed:
        save_trade_log(log)
        print("  Reconciliation complete — trade_log.json updated.")


def check_opening_move(ticker):
    """Check price action since today's open, using intraday 5-min bars.
    Returns (skip: bool, reason: str). skip=True means the stock has
    already dropped hard since the open and hasn't shown a meaningful
    bounce off its intraday low — i.e. still breaking down, not just
    normal early-session noise."""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval="5m")
        if hist.empty or len(hist) < 2:
            return False, "Insufficient intraday data — proceeding without opening-move check."

        open_price = float(hist["Open"].iloc[0])
        current_price = float(hist["Close"].iloc[-1])
        low_so_far = float(hist["Low"].min())

        if not open_price or not low_so_far:
            return False, "Invalid intraday data — proceeding without opening-move check."

        pct_from_open = (current_price - open_price) / open_price
        pct_off_low = (current_price - low_so_far) / low_so_far

        if pct_from_open <= OPENING_DROP_THRESHOLD:
            if pct_off_low >= MIN_BOUNCE_TO_OVERRIDE:
                return False, (f"down {pct_from_open:+.1%} from today's open, but bounced "
                               f"{pct_off_low:+.1%} off the intraday low — proceeding.")
            else:
                return True, (f"down {pct_from_open:+.1%} from today's open with no real "
                               f"bounce off the low ({pct_off_low:+.1%}) — skipping.")
        return False, f"{pct_from_open:+.1%} from today's open — within normal range."
    except Exception as e:
        return False, f"opening-move check failed ({e}) — proceeding without it."


def get_candidate_pool():
    """Returns today's full ranked candidate list (top-3 + bench runners-up)
    for the opening-move fallback in buy_positions(). Falls back to the
    plain top-3 from get_today_predictions() if the pool file is missing,
    unreadable, or stale (predictor.py didn't run today with the pool
    feature, or wrote it for a different date)."""
    today = date.today().strftime("%Y-%m-%d")
    try:
        with open(CANDIDATE_POOL_PATH) as f:
            pool = json.load(f)
        if pool.get("date") != today:
            print(f"  candidate_pool.json is stale (date={pool.get('date')}, expected {today}) — falling back to top-3 only.")
            return get_today_predictions()
        return [c["ticker"] for c in sorted(pool["candidates"], key=lambda c: c["rank"])]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"  candidate_pool.json unavailable ({e}) — falling back to top-3 only.")
        return get_today_predictions()


def buy_positions(client):
    today = date.today().strftime("%Y-%m-%d")

    existing_log = load_trade_log()
    if today in existing_log:
        print(f"Buy already executed for {today} — skipping to avoid overwriting real trade data.")
        return

    candidate_pool = get_candidate_pool()

    if not candidate_pool:
        print("No predictions found for today — skipping buy.")
        return

    NUM_POSITIONS = 3  # target number of positions to hold, regardless of how many candidates get filtered out

    buying_power = get_safe_buying_power(client)
    if buying_power < 10:
        print(f"SKIPPED: Insufficient buying power (${buying_power:.2f}).")
        return

    per_stock = buying_power / NUM_POSITIONS
    print(f"Splitting ${buying_power:.2f} across {NUM_POSITIONS} stocks — ${per_stock:.2f} each")
    print(f"Candidate pool ({len(candidate_pool)} ranked): {', '.join(candidate_pool)}")

    log = load_trade_log()
    log[today] = {"positions": {}, "status": "open", "date": today}

    bought_count = 0
    for rank, ticker in enumerate(candidate_pool, start=1):
        if bought_count >= NUM_POSITIONS:
            break

        pre_trade_quote = get_current_price(ticker)

        skip, opening_reason = check_opening_move(ticker)
        print(f"  #{rank} {ticker}: {opening_reason}")
        if skip:
            log[today]["positions"][ticker] = {
                "status": "skipped",
                "reason": f"Opening-move filter: {opening_reason}",
                "pre_trade_quote": pre_trade_quote
            }
            continue

        try:
            # Use notional (dollar amount) instead of shares — cleaner for
            # fixed-dollar position sizing and avoids fractional share math
            order_request = MarketOrderRequest(
                symbol=ticker,
                notional=round(per_stock, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(order_request)
        except Exception as e:
            print(f"  {ticker}: ORDER FAILED TO SUBMIT — {e}")
            log[today]["positions"][ticker] = {
                "status": "error", "reason": str(e),
                "pre_trade_quote": pre_trade_quote
            }
            continue

        order_id = str(order.id)
        print(f"  {ticker}: order submitted (id={order_id}), confirming fill...")
        fill = get_order_fill_details(client, order_id, max_wait_seconds=45)

        if fill is None:
            log[today]["positions"][ticker] = {
                "status": "unconfirmed",
                "reason": "Order submitted but fill not confirmed within timeout.",
                "notional": per_stock,
                "buy_price": pre_trade_quote,
                "shares": round(per_stock / pre_trade_quote, 6) if pre_trade_quote else None,
                "order_id": order_id,
                "pre_trade_quote": pre_trade_quote,
            }
            bought_count += 1
            continue

        if fill["state"] != "filled" or not fill["average_price"]:
            print(f"  {ticker}: ORDER {fill['state'].upper()} — not filled.")
            log[today]["positions"][ticker] = {
                "status": fill["state"],
                "reason": f"Order ended in state '{fill['state']}'.",
                "order_id": order_id,
                "pre_trade_quote": pre_trade_quote,
            }
            continue

        real_price = fill["average_price"]
        real_shares = fill["filled_quantity"]
        real_cost = real_price * real_shares
        slippage_pct = round((real_price - pre_trade_quote) / pre_trade_quote * 100, 3) if pre_trade_quote else None

        print(f"  {ticker}: FILLED {real_shares:.6f} shares @ ${real_price:.2f} = ${real_cost:.2f}" +
              (f" (slippage {slippage_pct:+.3f}%)" if slippage_pct is not None else ""))

        log[today]["positions"][ticker] = {
            "shares": real_shares,
            "buy_price": real_price,
            "cost": real_cost,
            "status": "open",
            "order_id": order_id,
            "pre_trade_quote": pre_trade_quote,
            "slippage_pct": slippage_pct,
        }
        bought_count += 1

    if bought_count < NUM_POSITIONS:
        print(f"WARNING: only filled {bought_count}/{NUM_POSITIONS} positions — "
              f"all {len(candidate_pool)} candidates in today's pool were filtered out, "
              f"errored, or exhausted before reaching the target.")

    save_trade_log(log)
    sync_real_trades_to_performance()
    print("Buy orders placed and reconciled.")

# --------------------------------------------------------------------------
# UPDATE PERFORMANCE.JSON WITH LIVE PRICES
# --------------------------------------------------------------------------

def sync_real_trades_to_performance(target_date=None):
    """Mirror actual trade_log.json fill data (real entry/exit prices, real
    exit reasons) into performance.json, so the dashboard reflects what was
    actually bought/sold on Alpaca instead of the 9:30am prediction-time
    price snapshot or the independent yfinance-based paper simulation.
    Defaults to today, but accepts an explicit date (e.g. when closing out
    a prior day's positions that were found via close_all_positions()'s
    fallback)."""
    if not os.path.exists(PERF_PATH):
        return

    with open(PERF_PATH) as f:
        perf = json.load(f)

    today = target_date or date.today().strftime("%Y-%m-%d")
    today_entry = next((d for d in perf.get("days", []) if d["date"] == today), None)
    if not today_entry:
        return

    trade_log = load_trade_log()
    today_trades = trade_log.get(today, {}).get("positions", {})
    if not today_trades:
        return

    updated = False
    now_et = datetime.now(ET).strftime("%H:%M ET")

    for pred in today_entry.get("predictions", []):
        ticker = pred["ticker"]
        pos = today_trades.get(ticker)
        if not pos:
            continue

        if pos.get("status") == "skipped":
            if pred.get("skipped_reason") != pos.get("reason"):
                pred["skipped_reason"] = pos.get("reason")
                updated = True
            continue

        if pos.get("status") in ("error", "unconfirmed", "sell_error", "sell_unconfirmed"):
            continue

        real_buy_price = pos.get("buy_price")
        if real_buy_price and pred.get("entry_price") != real_buy_price:
            if "prediction_price" not in pred:
                pred["prediction_price"] = pred.get("entry_price")  # keep original 9:30am snapshot for reference
            pred["entry_price"] = real_buy_price
            updated = True

        # "sell_unconfirmed" already carries a real sell_price (fallback to the
        # pre-trade quote when Alpaca's fill confirmation times out — see
        # sell_position()), so it's just as displayable as a fully "closed"
        # position. Only "sell_error" (order submission itself failed) has no
        # real sell data and correctly stays excluded.
        if pos.get("status") in ("closed", "sell_unconfirmed") and pos.get("sell_price"):
            entry = pred["entry_price"]
            exit_pct = round((pos["sell_price"] - entry) / entry * 100, 2) if entry else None
            if pred.get("paper_exit_price") != round(pos["sell_price"], 2):
                pred["paper_exit_price"] = round(pos["sell_price"], 2)
                pred["paper_exit_pct"] = exit_pct
                pred["paper_exit_reason"] = pos.get("close_reason", "Closed")
                pred["paper_exit_time"] = now_et
                updated = True
        elif pos.get("status") == "open" and pred.get("paper_exit_price") is not None:
            # This position is genuinely still open on Alpaca, but performance.json
            # has a leftover phantom exit written by the old unconditional
            # simulation (pre-fix). Clear it so the dashboard stops showing a
            # fake "Exited" badge for a position that never actually sold.
            pred["paper_exit_price"] = None
            pred["paper_exit_pct"] = None
            pred["paper_exit_reason"] = None
            pred["paper_exit_time"] = None
            pred.pop("paper_trail_floor_active", None)
            pred.pop("paper_current_floor_pct", None)
            pred.pop("paper_peak_pct", None)
            updated = True

    # Any ticker actually bought (open/closed) that ISN'T one of the original
    # top-3 predictions is a substitute pulled from the candidate bench after
    # an opening-move skip. Without this, a substitute buy would be silently
    # invisible in performance.json even though it's a real position.
    known_tickers = {p["ticker"] for p in today_entry.get("predictions", [])}
    substitute_tickers = [t for t, pos in today_trades.items()
                           if t not in known_tickers and pos.get("status") in ("open", "closed")]

    pool_scores = {}
    if substitute_tickers:
        try:
            with open(CANDIDATE_POOL_PATH) as f:
                pool = json.load(f)
            if pool.get("date") == today:
                pool_scores = {c["ticker"]: c for c in pool.get("candidates", [])}
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

    for ticker in substitute_tickers:
        pos = today_trades[ticker]
        pool_info = pool_scores.get(ticker, {})
        today_entry.setdefault("predictions", []).append({
            "ticker": ticker,
            "rank": pool_info.get("rank"),
            "score": pool_info.get("composite_score", 0),
            "entry_price": pos.get("buy_price"),
            "substitute_pick": True,
            "substitute_note": "Bought from the candidate bench after an original pick was skipped by the opening-move filter.",
            "paper_exit_price": round(pos["sell_price"], 2) if pos.get("status") in ("closed", "sell_unconfirmed") and pos.get("sell_price") else None,
            "paper_exit_pct": (round((pos["sell_price"] - pos["buy_price"]) / pos["buy_price"] * 100, 2)
                                if pos.get("status") in ("closed", "sell_unconfirmed") and pos.get("sell_price") and pos.get("buy_price") else None),
            "paper_exit_reason": pos.get("close_reason") if pos.get("status") in ("closed", "sell_unconfirmed") else None,
        })
        updated = True

    if updated:
        today_entry["prices_last_updated"] = now_et
        perf["days"] = [d for d in perf["days"] if d["date"] != today]
        perf["days"].append(today_entry)
        perf["days"].sort(key=lambda x: x["date"])
        with open(PERF_PATH, "w") as f:
            json.dump(perf, f, indent=2)
        print(f"performance.json reconciled with real trade fills at {now_et}")


def update_performance_prices():
    """Fetch current prices via yfinance and update performance.json.
    Only simulates paper exits (win-floor/stop-loss) when trading is
    DISABLED — i.e. there are no real orders to reconcile against. When
    real trading is active, sync_real_trades_to_performance() is the
    source of truth for entry/exit prices instead."""
    if not os.path.exists(PERF_PATH):
        return

    with open(PERF_PATH) as f:
        perf = json.load(f)

    today = date.today().strftime("%Y-%m-%d")
    today_entry = next((d for d in perf.get("days", []) if d["date"] == today), None)
    if not today_entry:
        return

    updated = False
    now_et = datetime.now(ET).strftime("%H:%M ET")
    market_open, _ = is_market_open()
    near_close = datetime.now(ET).hour == 15 and datetime.now(ET).minute >= 50

    for pred in today_entry.get("predictions", []):
        ticker = pred["ticker"]
        price = get_current_price(ticker)
        if not price:
            continue

        entry = pred.get("entry_price")
        pred["live_price"] = round(price, 2)
        pct = round((price - entry) / entry * 100, 2) if entry else None
        pred["live_pct"] = pct
        pred["price_updated"] = now_et
        updated = True

        # Simulated paper exit — ONLY runs when trading is disabled (no real
        # orders exist to reconcile against). When real trading is active,
        # sync_real_trades_to_performance() populates these same fields from
        # actual Alpaca fills instead, so this block is skipped entirely.
        if not is_trading_enabled() and pred.get("paper_exit_price") is None and entry and pct is not None:
            pct_decimal = pct / 100
            peak_pct = max(pred.get("paper_peak_pct", 0.0), pct_decimal)
            pred["paper_peak_pct"] = round(peak_pct, 4)

            floor_pct, floor_label = get_trail_floor(pct_decimal, peak_pct)

            exit_reason = None
            if floor_pct is not None and pct_decimal < floor_pct:
                exit_reason = f"Trail stop: {floor_label} ({pct:+.1f}%)"
            elif pct_decimal <= STOP_LOSS_PCT:
                exit_reason = f"Stop loss ({pct:+.1f}%)"
            elif near_close or not market_open:
                exit_reason = "End of day close"

            if floor_pct is not None:
                pred["paper_trail_floor_active"] = True
                pred["paper_current_floor_pct"] = round(floor_pct, 4)

            if exit_reason:
                pred["paper_exit_price"] = round(price, 2)
                pred["paper_exit_pct"] = pct
                pred["paper_exit_reason"] = exit_reason
                pred["paper_exit_time"] = now_et
                print(f"  {ticker}: PAPER EXIT @ ${price:.2f} ({pct:+.2f}%) — {exit_reason}")

        print(f"  {ticker}: ${price:.2f}" +
              (f" ({pct:+.2f}% vs entry ${entry:.2f})" if entry else ""))

    if updated:
        today_entry["prices_last_updated"] = now_et
        perf["days"] = [d for d in perf["days"] if d["date"] != today]
        perf["days"].append(today_entry)
        perf["days"].sort(key=lambda x: x["date"])
        with open(PERF_PATH, "w") as f:
            json.dump(perf, f, indent=2)
        print(f"performance.json updated at {now_et}")

# --------------------------------------------------------------------------
# MONITOR
# --------------------------------------------------------------------------

def monitor_positions(client):
    today = date.today().strftime("%Y-%m-%d")
    log = load_trade_log()

    if today not in log:
        print("No trades for today yet.")
        update_performance_prices()
        return

    day = log[today]
    if day["status"] == "closed":
        print("All positions already closed.")
        sync_real_trades_to_performance()
        update_performance_prices()
        return

    any_open = False
    for ticker, pos in day["positions"].items():
        if pos["status"] != "open":
            continue
        any_open = True
        current_price = get_current_price(ticker)
        if not current_price:
            continue

        buy_price  = pos["buy_price"]
        pct_change = (current_price - buy_price) / buy_price
        shares     = pos["shares"]
        print(f"  {ticker}: bought @ ${buy_price:.2f}, now ${current_price:.2f} ({pct_change:+.1%})")

        # Track peak % reached for this position
        peak_pct = max(pos.get("peak_pct", 0.0), pct_change)
        pos["peak_pct"] = round(peak_pct, 4)

        floor_pct, floor_label = get_trail_floor(pct_change, peak_pct)

        if floor_pct is not None:
            pos["trail_floor_active"] = True
            pos["current_floor_pct"] = round(floor_pct, 4)

            if pct_change < floor_pct:
                sell_position(client, ticker, shares, current_price, pos, day,
                              f"Trail stop: {floor_label} triggered ({pct_change:+.1%})")
            else:
                print(f"    ↑ Trail floor active — {floor_label}, currently {pct_change:+.1%}, peak {peak_pct:+.1%}")
        elif pct_change <= STOP_LOSS_PCT:
            sell_position(client, ticker, shares, current_price, pos, day,
                          f"Stop loss ({pct_change:+.1%})")

    if not any_open:
        day["status"] = "closed"

    save_trade_log(log)
    sync_real_trades_to_performance()
    update_performance_prices()

# --------------------------------------------------------------------------
# SELL SINGLE
# --------------------------------------------------------------------------

def sell_position(client, ticker, shares, current_price, pos, day, reason):
    try:
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_request)
    except Exception as e:
        print(f"  SELL FAILED TO SUBMIT for {ticker}: {e}")
        pos["status"] = "sell_error"
        pos["sell_error_reason"] = str(e)
        return

    order_id = str(order.id)
    print(f"  {ticker}: sell order submitted (id={order_id}), confirming fill...")
    fill = get_order_fill_details(client, order_id, max_wait_seconds=45)

    if fill is None or fill["state"] != "filled" or not fill["average_price"]:
        state = fill["state"] if fill else "unconfirmed"
        print(f"  SELL {state.upper()} for {ticker} — using pre-trade quote as fallback.")
        sell_price = current_price
        pos["status"] = "sell_unconfirmed"
        pos["sell_error_reason"] = f"Sell ended in state '{state}' — fallback to quote."
    else:
        sell_price = fill["average_price"]
        slippage_pct = round((sell_price - current_price) / current_price * 100, 3) if current_price else None
        print(f"  {ticker}: SELL FILLED @ ${sell_price:.2f}" +
              (f" (slippage {slippage_pct:+.3f}%)" if slippage_pct is not None else ""))
        pos["slippage_pct"] = slippage_pct
        pos["status"] = "closed"

    buy_price = pos["buy_price"]
    pnl = (sell_price - buy_price) * shares
    pos["sell_price"] = sell_price
    pos["pnl"] = pnl
    pos["close_reason"] = reason
    print(f"  SOLD {ticker}: {shares:.6f} @ ${sell_price:.2f} | P&L: ${pnl:+.2f} | {reason}")
    record_pnl(ticker, buy_price, sell_price, shares, pnl, reason)

# --------------------------------------------------------------------------
# CLOSE ALL
# --------------------------------------------------------------------------

def close_all_positions(client):
    today = date.today().strftime("%Y-%m-%d")
    log = load_trade_log()

    target_date = today if today in log else None
    if target_date is None:
        # Today has no entry (e.g. running manually on a non-trading day, or
        # a scheduled EOD close was missed entirely) — fall back to the most
        # recent day still marked "open" so real positions never get
        # silently stranded just because the calendar date has moved on.
        open_days = [d for d, day in log.items() if day.get("status") == "open"]
        if open_days:
            target_date = max(open_days)
            print(f"No trades for today — found still-open positions from {target_date}, closing those instead.")

    if target_date is None:
        print("No open trades found to close.")
        return

    day = log[target_date]
    for ticker, pos in day["positions"].items():
        if pos["status"] != "open":
            continue
        current_price = get_current_price(ticker) or pos["buy_price"]
        sell_position(client, ticker, pos["shares"], current_price, pos, day, "End of day close")

    day["status"] = "closed"
    save_trade_log(log)
    sync_real_trades_to_performance(target_date)
    print_daily_summary(target_date, day)

# --------------------------------------------------------------------------
# P&L + TAX
# --------------------------------------------------------------------------

def record_pnl(ticker, buy_price, sell_price, shares, pnl, reason):
    tax_log = load_tax_log()
    today = date.today().strftime("%Y-%m-%d")
    tax_log["trades"].append({
        "date": today, "ticker": ticker, "shares": shares,
        "buy_price": buy_price, "sell_price": sell_price,
        "pnl": pnl, "reason": reason, "type": "short_term"
    })
    if pnl > 0:
        tax_log["total_gains"]    += pnl
        tax_log["tax_liability"]  += pnl * TAX_RATE
        tax_log["set_aside"]      += pnl * TAX_RATE
    else:
        tax_log["total_losses"]   += abs(pnl)
        tax_log["tax_liability"]   = max(0, tax_log["tax_liability"] + pnl * TAX_RATE)
    tax_log["net_pnl"] = tax_log["total_gains"] - tax_log["total_losses"]
    save_tax_log(tax_log)


def print_daily_summary(today, day):
    print(f"\n{'='*55}\nDAILY TRADING SUMMARY — {today}\n{'='*55}")
    total_pnl = 0
    for ticker, pos in day["positions"].items():
        if "pnl" in pos:
            total_pnl += pos["pnl"]
            print(f"  {ticker}: ${pos['pnl']:+.2f} ({pos.get('close_reason', '')})")
        else:
            print(f"  {ticker}: {pos['status']}")
    print(f"\nTotal P&L today: ${total_pnl:+.2f}")
    tax_log = load_tax_log()
    config  = load_config()
    starting = float(config.get("starting_capital", 300.0))
    current  = starting + tax_log["net_pnl"]
    print(f"\nPortfolio:\n  Start: ${starting:.2f}  Current: ${current:.2f} ({(current-starting)/starting:+.1%})")
    print(f"  Net P&L: ${tax_log['net_pnl']:.2f}")
    print(f"\nTax tracker ({int(TAX_RATE*100)}% short-term):")
    print(f"  Gains: ${tax_log['total_gains']:.2f}  Losses: ${tax_log['total_losses']:.2f}")
    print(f"  Liability: ${tax_log['tax_liability']:.2f}  Set aside: ${tax_log['set_aside']:.2f}")
    print(f"{'='*55}\n")

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY env vars required.")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "monitor"

    if not is_trading_enabled():
        if mode == "monitor" and is_market_open()[0]:
            print("Paper/disabled mode — updating live prices only...")
            update_performance_prices()
        else:
            print("Exiting — trading is disabled.")
        sys.exit(0)

    if mode == "sell":
        # No day-of-week/holiday gate here on purpose. The scheduled cron
        # trigger already only fires on weekdays, so this only ever matters
        # for a manually-triggered sell — and Alpaca safely accepts and
        # queues orders submitted while the market's closed, executing them
        # at the next open (see Alpaca docs on order queuing). Blocking that
        # ourselves would only get in the way of legitimate manual cleanup
        # (e.g. closing out positions from a missed EOD close before the
        # next trading day).
        pass
    else:
        open_ok, open_reason = is_market_open()
        if not open_ok:
            print(f"Skipping — {open_reason}")
            sys.exit(0)

    ok, reason = check_risk_limits()
    if not ok:
        print(f"RISK LIMIT: {reason}")
        sys.exit(0)

    client = get_alpaca_client()
    reconcile_unconfirmed_orders(client)

    if mode == "buy":
        print("MODE: Buy opening positions")
        buy_positions(client)
    elif mode == "monitor":
        print("MODE: Monitor positions")
        monitor_positions(client)
    elif mode == "sell":
        print("MODE: Close all positions")
        close_all_positions(client)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
