"""
telegram_commands.py
=====================
Minimal Telegram "ask on demand" support for the regime monitor.

How it works
------------
Telegram's Bot API has no push mechanism usable from a GitHub Actions
runner (no public HTTPS endpoint to receive webhooks), so this uses
long-polling instead: each iteration of the monitor loop (every ~5 min,
matching the existing `sleep 300` in regime_monitor.yml) calls
`poll_and_reply()` once. It fetches any *new* messages you've sent the
bot since the last check, and if any of them are a recognized command,
replies with the most recently computed prediction/state for both models.

This means: send `/status` in Telegram, and you'll get a reply within
one loop iteration (~5 minutes worst case) — not instantly. If you need
sub-minute responses you'd need a real webhook receiver (e.g. a small
always-on server or a serverless function), which is a different
architecture than "GitHub Actions runs a loop".

Commands recognized
--------------------
/status or /predict  → replies with both models' latest pred/conf/equity/
                        open-trade snapshot (pulled from state, not a
                        fresh inference call — inference already runs
                        every iteration regardless of whether you ask).
"""

import os
import requests

API_BASE = "https://api.telegram.org/bot{token}/{method}"

KNOWN_COMMANDS = {"/status", "/predict", "/pred"}


def _get(method: str, params: dict) -> dict:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {}
    try:
        r = requests.get(API_BASE.format(token=token, method=method), params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"Telegram {method} failed: {exc}")
        return {}


def _send(text: str) -> None:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        requests.post(
            API_BASE.format(token=token, method="sendMessage"),
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as exc:
        print(f"Telegram sendMessage failed: {exc}")


def format_status_reply(state: dict) -> str:
    """Builds a combined status message from both models' last-known state."""
    lines = ["📡 *On-demand status — both models*", ""]
    models = state.get("models", {})

    if not models:
        return "📡 No model state recorded yet — wait for the next iteration."

    for m in models.values():
        label = m.get("model_label") or m.get("key", "model")
        trade = m.get("paper_trade", {}) or {}
        pred_names = {0: "Bull", 1: "Bear", 2: "Neutral"}
        pred_name  = pred_names.get(m.get("pred"), "?")

        if trade.get("open"):
            trade_str = (
                f"open {trade['direction'].upper()} @ "
                f"${trade['entry_price']:,.2f} "
                f"(stop ${trade['current_stop']:,.2f}, "
                f"target ${trade['current_target']:,.2f})"
            )
        else:
            trade_str = "flat"

        lines.append(
            f"*{label}*\n"
            f"  Pred    : {pred_name}  (`{m.get('conf', 0.0):.1%}` confidence)\n"
            f"  Equity  : `${m.get('equity', 100.0):,.2f}`\n"
            f"  Trade   : {trade_str}\n"
            f"  As of   : `{m.get('ts', '?')} UTC`\n"
        )

    return "\n".join(lines)


def poll_and_reply(state: dict) -> dict:
    """
    Checks for new Telegram messages since the last stored offset, and
    replies with a combined status if a recognized command is found.
    Returns the (possibly updated) state dict — call this AFTER you've
    saved this iteration's fresh predictions into `state`, so the reply
    reflects current numbers.

    Mutates/returns state["telegram_offset"].
    """
    offset  = state.get("telegram_offset", 0)
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    resp = _get("getUpdates", {"offset": offset, "timeout": 0})
    updates = resp.get("result", [])
    if not updates:
        return state

    new_offset = offset
    replied = False

    for upd in updates:
        new_offset = max(new_offset, upd.get("update_id", 0) + 1)
        msg = upd.get("message") or upd.get("channel_post") or {}
        if not msg:
            continue

        # Only respond to messages from your configured chat — ignore anyone else.
        msg_chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id and msg_chat_id != str(chat_id):
            continue

        text = (msg.get("text") or "").strip().lower()
        if text in KNOWN_COMMANDS and not replied:
            _send(format_status_reply(state))
            replied = True   # avoid spamming multiple replies in one batch

    state["telegram_offset"] = new_offset
    return state