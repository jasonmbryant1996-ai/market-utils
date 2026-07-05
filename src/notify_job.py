"""
notify_job.py
==============
Sends a Telegram message when the monitor job starts or ends — separate
from the per-iteration trade-open/trade-close messages sent by
live_monitor.py. Gives you visibility into whether the loop is actually
running, how long it lasted, and what the paper-trading state looked like
at each boundary.

Usage (called from the workflow):
    python src/notify_job.py start
    python src/notify_job.py end
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import requests

STATE_PATH = Path(__file__).parent.parent / "state" / "last_signal.json"


def send_telegram(text: str) -> None:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram secrets not set — skipping job notification")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as exc:
        print(f"Telegram send failed: {exc}")


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def fmt_trade_status(state: dict) -> str:
    trade = state.get("paper_trade", {})
    if trade.get("open"):
        d = trade["direction"].upper()
        return (
            f"🔓 *Open {d} trade*\n"
            f"  Entry : `${trade['entry_price']:,.2f}`\n"
            f"  Stop  : `${trade['current_stop']:,.2f}`\n"
            f"  Target: `${trade['current_target']:,.2f}`\n"
            f"  Trail : {'ACTIVE' if trade.get('trail_active') else 'not yet'}\n"
            f"  Confirming bars: {trade.get('consecutive_same_signal', 0)}"
        )
    return "🔒 No open trade"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("start", "end"):
        print("Usage: python notify_job.py [start|end]")
        sys.exit(1)

    mode        = sys.argv[1]
    state       = load_state()
    equity      = state.get("equity", 100.0)
    trades_ever = len(state.get("trade_log", []))
    run_id      = os.environ.get("GITHUB_RUN_ID", "?")
    run_number  = os.environ.get("GITHUB_RUN_NUMBER", "?")
    now_str     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if mode == "start":
        msg = (
            f"🟢 *Monitor job started*\n\n"
            f"Time        : `{now_str} UTC`\n"
            f"Run         : `#{run_number}` (id `{run_id}`)\n"
            f"Equity      : `${equity:,.2f}`\n"
            f"Trades ever : `{trades_ever}`\n\n"
            f"{fmt_trade_status(state)}"
        )

    else:  # end
        iterations   = os.environ.get("ITERATIONS", "?")
        start_epoch  = os.environ.get("JOB_START_EPOCH")
        if start_epoch:
            duration_min = (time.time() - float(start_epoch)) / 60
            duration_str = f"{duration_min:.0f} min"
        else:
            duration_str = "?"

        msg = (
            f"🔴 *Monitor job ending*\n\n"
            f"Time        : `{now_str} UTC`\n"
            f"Run         : `#{run_number}` (id `{run_id}`)\n"
            f"Duration    : `{duration_str}`\n"
            f"Iterations  : `{iterations}`\n"
            f"Equity      : `${equity:,.2f}`\n"
            f"Trades ever : `{trades_ever}`\n\n"
            f"{fmt_trade_status(state)}\n\n"
            f"_Next scheduled restart in ~5h, or trigger manually._"
        )

    send_telegram(msg)


if __name__ == "__main__":
    main()
