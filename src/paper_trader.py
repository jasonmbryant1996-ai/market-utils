"""
paper_trader.py
================
Configurable paper-trading engine for the regime monitor.

Instead of treating every 5-minute Bear (or Bull) prediction as a separate
signal, this tracks ONE open paper position at a time and manages it across
multiple 5-minute checks, using:

  - Fixed initial stop-loss and target (R:R based)
  - Optional DYNAMIC TARGET: while the regime model keeps confirming the
    same direction, the target is pushed further away (more room to run)
  - Optional TRAILING STOP: once price has moved TRAIL_ACTIVATION_RR in
    your favor, the stop trails TRAIL_DISTANCE_RR behind the best price
    reached, protecting profit without capping the upside
  - Optional EMA EXIT: exits early if price crosses back through the EMA,
    a momentum-loss signal independent of the R-multiple math
  - A hard time-based exit as a safety net

Everything is driven by PAPER_TRADE_CONFIG below — including MODE, which
lets you switch between a Bear-only model, a Bull-only model, or a
combined Bull/Bear model later without touching any other code.
"""

from datetime import datetime, timezone
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — tune everything here
# ══════════════════════════════════════════════════════════════════════════════
PAPER_TRADE_CONFIG = {
    "ENABLED": True,

    # "bear_only"  -> only shorts on regime_pred == 1 (Bear)
    # "bull_only"  -> only longs  on regime_pred == 0 (Bull)
    # "bull_bear"  -> longs on Bull, shorts on Bear (needs a bull/bear model)
    "MODE": "bear_only",

    "REGIME_CONFIDENCE_THRESHOLD": 0.45,   # minimum confidence to open a trade

    # ── Position sizing & fixed risk definition ──────────────────────────────
    "STARTING_EQUITY"     : 100.0,
    "RISK_PER_TRADE_PCT"  : 1.0,     # % of current paper equity risked per trade
    "STOP_LOSS_PCT"       : 0.010,   # initial stop distance = 1R (e.g. 1.0%)
    "INITIAL_TARGET_RR"   : 2.5,     # initial target = STOP_LOSS_PCT * this
    "FEE_RATE"            : 0.0005,  # per-side fee (matches your backtest config)
    "MAX_HOLD_HOURS"      : 24,      # hard time-exit safety net

    # ── Dynamic target: extend target further while signal persists ─────────
    "ENABLE_DYNAMIC_TARGET"    : True,
    "DYNAMIC_TARGET_EXTEND_RR" : 0.5,   # each confirming bar pushes target this
                                        # many R further from the CURRENT price
                                        # (never moves the target backwards)

    # ── Trailing stop: activate after price moves in your favor ──────────────
    "ENABLE_TRAILING_STOP" : True,
    "TRAIL_ACTIVATION_RR"  : 1.0,   # activate once price has moved this many R
                                    # in your favor from entry
    "TRAIL_DISTANCE_RR"    : 1.0,   # trail this many R behind the best price
                                    # reached since entry (only ever tightens)

    # ── EMA exit: close early if price crosses back through the EMA ──────────
    "ENABLE_EMA_EXIT" : True,
    "EMA_PERIOD"      : 21,
}


# ══════════════════════════════════════════════════════════════════════════════
# Direction mapping — makes MODE swappable without touching logic below
# ══════════════════════════════════════════════════════════════════════════════

def get_desired_direction(pred: int, conf: float, config: dict) -> str | None:
    """
    Maps a (pred, conf) pair to a desired trade direction ('long'/'short'),
    or None if no trade is currently warranted. pred: 0=Bull, 1=Bear, 2=Neutral.
    """
    if conf < config["REGIME_CONFIDENCE_THRESHOLD"]:
        return None

    mode = config["MODE"]
    if mode == "bear_only":
        return "short" if pred == 1 else None
    if mode == "bull_only":
        return "long" if pred == 0 else None
    if mode == "bull_bear":
        if pred == 0:
            return "long"
        if pred == 1:
            return "short"
        return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Compute EMA from raw OHLCV
# ══════════════════════════════════════════════════════════════════════════════

def compute_ema(df_raw: pd.DataFrame, period: int) -> float:
    return float(df_raw["close"].ewm(span=period, adjust=False).mean().iloc[-1])


# ══════════════════════════════════════════════════════════════════════════════
# Open a new paper trade
# ══════════════════════════════════════════════════════════════════════════════

def open_trade(direction: str, price: float, equity: float, config: dict) -> dict:
    stop_pct = config["STOP_LOSS_PCT"]
    target_rr = config["INITIAL_TARGET_RR"]

    if direction == "short":
        stop   = price * (1 + stop_pct)
        target = price * (1 - stop_pct * target_rr)
    else:  # long
        stop   = price * (1 - stop_pct)
        target = price * (1 + stop_pct * target_rr)

    risk_amount   = equity * (config["RISK_PER_TRADE_PCT"] / 100.0)
    position_size = risk_amount / stop_pct   # $ notional, matches backtest sizing

    return {
        "open"                 : True,
        "direction"             : direction,
        "entry_price"           : price,
        "entry_time"            : datetime.now(timezone.utc).isoformat(),
        "initial_stop"          : stop,
        "current_stop"          : stop,
        "initial_target"        : target,
        "current_target"        : target,
        "risk_amount"           : risk_amount,
        "stop_pct"              : stop_pct,
        "position_size"         : position_size,
        "peak_favorable_price"  : price,
        "trail_active"          : False,
        "consecutive_same_signal": 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Manage an open paper trade — one call per 5-minute check
# ══════════════════════════════════════════════════════════════════════════════

def manage_trade(
    trade: dict,
    price: float,
    ema_value: float,
    pred: int,
    conf: float,
    config: dict,
) -> tuple[dict, bool, str | None, float | None]:
    """
    Updates the trade in place (trailing stop, dynamic target, signal streak),
    checks exit conditions, and returns:
        (trade, closed: bool, close_reason: str|None, net_pnl: float|None)
    """
    direction = trade["direction"]
    stop_pct  = trade["stop_pct"]
    entry     = trade["entry_price"]

    # ── Track best (most favorable) price reached since entry ────────────────
    if direction == "short":
        if price < trade["peak_favorable_price"]:
            trade["peak_favorable_price"] = price
        favorable_move_pct = (entry - trade["peak_favorable_price"]) / entry
    else:
        if price > trade["peak_favorable_price"]:
            trade["peak_favorable_price"] = price
        favorable_move_pct = (trade["peak_favorable_price"] - entry) / entry

    r_multiple = favorable_move_pct / stop_pct

    # ── Trailing stop: activate once R-multiple threshold is reached ─────────
    if config["ENABLE_TRAILING_STOP"] and r_multiple >= config["TRAIL_ACTIVATION_RR"]:
        trade["trail_active"] = True
        trail_distance_pct = stop_pct * config["TRAIL_DISTANCE_RR"]
        if direction == "short":
            candidate_stop = trade["peak_favorable_price"] * (1 + trail_distance_pct)
            trade["current_stop"] = min(trade["current_stop"], candidate_stop)  # only tighten
        else:
            candidate_stop = trade["peak_favorable_price"] * (1 - trail_distance_pct)
            trade["current_stop"] = max(trade["current_stop"], candidate_stop)

    # ── Dynamic target: extend further while the model keeps confirming ──────
    desired_dir = get_desired_direction(pred, conf, config)
    if desired_dir == direction:
        trade["consecutive_same_signal"] += 1
        if config["ENABLE_DYNAMIC_TARGET"]:
            extend_pct = stop_pct * config["DYNAMIC_TARGET_EXTEND_RR"]
            if direction == "short":
                candidate_target = price * (1 - extend_pct)
                trade["current_target"] = min(trade["current_target"], candidate_target)
            else:
                candidate_target = price * (1 + extend_pct)
                trade["current_target"] = max(trade["current_target"], candidate_target)
    else:
        trade["consecutive_same_signal"] = 0

    # ── Check exit conditions (priority: stop -> target -> EMA -> time) ──────
    closed, reason = False, None

    if direction == "short":
        if price >= trade["current_stop"]:
            closed, reason = True, "stop"
        elif price <= trade["current_target"]:
            closed, reason = True, "target"
        elif config["ENABLE_EMA_EXIT"] and price >= ema_value:
            closed, reason = True, "ema_exit"
    else:
        if price <= trade["current_stop"]:
            closed, reason = True, "stop"
        elif price >= trade["current_target"]:
            closed, reason = True, "target"
        elif config["ENABLE_EMA_EXIT"] and price <= ema_value:
            closed, reason = True, "ema_exit"

    if not closed:
        entry_time  = datetime.fromisoformat(trade["entry_time"])
        hours_open  = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
        if hours_open >= config["MAX_HOLD_HOURS"]:
            closed, reason = True, "time_exit"

    # ── Compute P&L if closed ─────────────────────────────────────────────────
    net_pnl = None
    if closed:
        exit_price = price
        if reason == "stop":
            exit_price = trade["current_stop"]
        elif reason == "target":
            exit_price = trade["current_target"]

        coins = trade["position_size"] / entry
        if direction == "short":
            gross = (entry - exit_price) * coins
        else:
            gross = (exit_price - entry) * coins

        fee = config["FEE_RATE"] * trade["position_size"] + config["FEE_RATE"] * (coins * exit_price)
        net_pnl = gross - fee

        trade["exit_price"] = exit_price
        trade["exit_time"]  = datetime.now(timezone.utc).isoformat()
        trade["close_reason"] = reason
        trade["net_pnl"] = net_pnl
        trade["open"] = False

    return trade, closed, reason, net_pnl


# ══════════════════════════════════════════════════════════════════════════════
# Formatting helpers for Telegram messages
# ══════════════════════════════════════════════════════════════════════════════

def format_open_message(trade: dict, config: dict, equity: float) -> str:
    d = trade["direction"].upper()
    emoji = "🔴" if trade["direction"] == "short" else "🟢"
    return (
        f"{emoji} *PAPER TRADE OPENED — {d}*\n\n"
        f"Entry     : `${trade['entry_price']:,.2f}`\n"
        f"Stop      : `${trade['current_stop']:,.2f}`  "
        f"({config['STOP_LOSS_PCT']:.1%})\n"
        f"Target    : `${trade['current_target']:,.2f}`  "
        f"({config['INITIAL_TARGET_RR']:.1f}R)\n"
        f"Size      : `${trade['position_size']:,.2f}`\n"
        f"Risk      : `${trade['risk_amount']:,.2f}`\n"
        f"Equity    : `${equity:,.2f}`\n"
    )


def format_close_message(trade: dict, equity: float) -> str:
    d = trade["direction"].upper()
    pnl = trade["net_pnl"]
    emoji = "✅" if pnl > 0 else "❌"
    reason_labels = {
        "stop": "Stop hit", "target": "Target hit",
        "ema_exit": "EMA exit", "time_exit": "Time exit",
    }
    reason = reason_labels.get(trade["close_reason"], trade["close_reason"])
    return (
        f"{emoji} *PAPER TRADE CLOSED — {d}*\n\n"
        f"Reason    : {reason}\n"
        f"Entry     : `${trade['entry_price']:,.2f}`\n"
        f"Exit      : `${trade['exit_price']:,.2f}`\n"
        f"Net P&L   : `${pnl:+,.2f}`\n"
        f"New Equity: `${equity:,.2f}`\n"
        f"Held      : {trade['consecutive_same_signal']} confirming bars\n"
    )
