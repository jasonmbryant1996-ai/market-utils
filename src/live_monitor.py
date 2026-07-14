"""
live_monitor.py
===============
Fetches the latest BTC data ONCE, computes the shared 35-feature matrix
ONCE, then runs inference for every model in MODEL_REGISTRY against it
(each with its own scaler and its own independent paper-trade ledger).

Designed to run every 5 minutes via GitHub Actions (see regime_monitor.yml).

State is persisted between runs in state/last_signal.json, keyed per model
under state["models"][<model_key>], plus a shared state["telegram_offset"]
used by telegram_commands.py for the on-demand /status command.

Exit codes:
  0 — ran successfully
  1 — error (data fetch failed, model load failed, etc.)
"""

import os
import sys
import json
import pickle
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import requests

ROOT       = Path(__file__).parent.parent
MODEL_ROOT = ROOT / "model"
STATE_PATH = ROOT / "state" / "last_signal.json"

sys.path.insert(0, str(Path(__file__).parent))
from model_arch import RegimeTransformer
from features  import (
    FEATURE_COLUMNS, fetch_btc_bars, fetch_macro, build_features,
    fetch_current_price,
)
import paper_trader as pt
import telegram_commands as tc

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK     = 240          # must match training, shared by both models
N_BARS       = 8700         # ~30 days of 5-min bars
NUM_CLASSES  = 3
REGIME_NAMES = {0: "Bull Trend", 1: "Bear Trend", 2: "Neutral"}
REGIME_EMOJIS = {0: "🟢", 1: "🔴", 2: "⚪"}

CONF_CHANGE_DELTA = 0.05    # re-notify if confidence shifts by this much

MODEL_PARAMS = dict(
    input_dim  = 35,
    d_model    = 128,
    nhead      = 4,
    num_layers = 3,
    dim_ffn    = 512,
    dropout    = 0.2,
    num_classes= NUM_CLASSES,
)

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry is one deployed model. `model_dir` must match the directory name
# used when uploading via upload_model.py --model-name <model_dir> (see the
# updated upload_model.yml / upload_model.py). Both currently share the same
# target/stop (2.5% / 1.0%) per your training logs — only MODE and hold time
# differ. Adjust conf_threshold per model if your EV tables suggest otherwise.
MODEL_REGISTRY = [
    {
        "key"           : "bear_6h",
        "model_dir"     : "bear_6h",
        "label"         : "🐻 BEAR-6H",
        "mode"          : "bear_only",
        "conf_threshold": 0.45,
        "max_hold_hours": 6,
        "stop_loss_pct" : 0.010,
        "target_rr"     : 2.5,
    },
    {
        "key"           : "bull_48h",
        "model_dir"     : "bull_48h",
        "label"         : "🐂 BULL-48H",
        "mode"          : "bull_bear",
        "conf_threshold": 0.75,
        "max_hold_hours": 48,
        "stop_loss_pct" : 0.010,
        "target_rr"     : 2.5,
    },
]

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.warning("Telegram env vars not set — skipping notification")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as exc:
        log.error(f"Telegram send failed: {exc}")
        return False


# ── State management ──────────────────────────────────────────────────────────

def _default_model_state(reg: dict) -> dict:
    return {
        "key"          : reg["key"],
        "model_label"  : reg["label"],
        "pred"         : 2,
        "conf"         : 0.0,
        "notified_conf": 0.0,
        "ts"           : "",
        "equity"       : pt.PAPER_TRADE_CONFIG["STARTING_EQUITY"],
        "paper_trade"  : {"open": False},
        "trade_log"    : [],
    }


def load_state() -> dict:
    """Loads state and migrates the old single-model flat format if found."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
        except Exception:
            state = {}
    else:
        state = {}

    if "models" not in state:
        migrated = {"telegram_offset": 0, "models": {}}
        # If an old flat-format state exists, keep it as the bear_6h model's
        # history rather than discarding your existing equity/trade_log.
        if "pred" in state:
            old = dict(state)
            old["key"] = "bear_6h"
            old["model_label"] = "🐻 BEAR-6H"
            migrated["models"]["bear_6h"] = old
        state = migrated

    state.setdefault("telegram_offset", 0)
    state.setdefault("models", {})
    for reg in MODEL_REGISTRY:
        state["models"].setdefault(reg["key"], _default_model_state(reg))

    return state


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def git_commit_state() -> None:
    """Commits state/last_signal.json so paper-trade state survives restarts."""
    try:
        subprocess.run(["git", "-C", str(ROOT), "config", "user.email",
                         "regime-bot@users.noreply.github.com"],
                        check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "config", "user.name",
                         "regime-monitor-bot"],
                        check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "add", "state/last_signal.json"],
                        check=True, capture_output=True, text=True)

        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"])
        if diff.returncode == 0:
            return

        subprocess.run(["git", "-C", str(ROOT), "commit", "-m",
                         "Update paper trade state"],
                        check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(ROOT), "push"],
                        check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        log.warning(f"Git commit/push failed (non-fatal): {exc.stderr}")


# ── Model loading / inference ─────────────────────────────────────────────────

def load_model_and_scaler(model_dir: str) -> tuple:
    pth = MODEL_ROOT / model_dir / "best_regime_transformer.pth"
    pkl = MODEL_ROOT / model_dir / "scaler_X.pkl"
    if not pth.exists() or not pkl.exists():
        raise FileNotFoundError(f"Missing model files under model/{model_dir}/")

    device = torch.device("cpu")
    ckpt   = torch.load(pth, map_location=device, weights_only=True)
    model  = RegimeTransformer(**MODEL_PARAMS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(pkl, "rb") as f:
        scaler = pickle.load(f)

    log.info(f"[{model_dir}] loaded (epoch {ckpt.get('epoch', 0) + 1}, val_f1={ckpt.get('val_f1', 0):.4f})")
    return model, scaler


@torch.no_grad()
def predict(model, scaler, df_features: pd.DataFrame) -> tuple[int, float, list[float]]:
    X_all    = df_features[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = scaler.transform(X_all)

    if len(X_scaled) < LOOKBACK:
        raise ValueError(f"Not enough bars after feature warmup: got {len(X_scaled)}, need {LOOKBACK}")
    seq = X_scaled[-LOOKBACK:]
    t   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    logits = model(t)
    probs  = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_class = int(np.argmax(probs))
    confidence = float(max(probs))
    return pred_class, confidence, probs


# ── Notification logic (per-model, direction-specific) ────────────────────────

def should_notify(reg: dict, pred: int, conf: float, model_state: dict) -> tuple[bool, str]:
    """
    Each model only ever cares about ONE direction (bear_6h watches for
    Bear, bull_48h watches for Bull) — simpler than the old combined logic.
    """
    watch_class = 1 if reg["mode"] == "bear_only" else 0
    threshold   = reg["conf_threshold"]
    last_pred   = model_state.get("pred", 2)
    last_conf   = model_state.get("notified_conf", 0.0)

    if pred == watch_class and conf >= threshold and last_pred != watch_class:
        return True, "NEW_SIGNAL"
    if pred != watch_class and last_pred == watch_class and last_conf >= threshold:
        return True, "SIGNAL_ENDED"
    if pred == watch_class and conf >= threshold and last_pred == watch_class:
        if abs(conf - last_conf) >= CONF_CHANGE_DELTA:
            return True, "SIGNAL_UPDATE"
    return False, "NO_CHANGE"


def build_message(reg: dict, pred: int, conf: float, probs: list, reason: str,
                   bar_time: str, btc_price: float) -> str:
    emoji = REGIME_EMOJIS[pred]
    name  = REGIME_NAMES[pred]
    label = reg["label"]

    headers = {
        "NEW_SIGNAL"   : f"🚨 *NEW SIGNAL — {label}*",
        "SIGNAL_ENDED" : f"✅ *SIGNAL ENDED — {label}*",
        "SIGNAL_UPDATE": f"📊 *UPDATE — {label}*",
    }
    header = headers.get(reason, f"📊 *REGIME UPDATE — {label}*")

    filled = int(conf * 10)
    bar    = "█" * filled + "░" * (10 - filled)

    lines = [
        header, "",
        f"{emoji} *{name}*  —  `{conf:.1%}` confidence",
        f"`[{bar}]`", "",
        f"🟢 Bull  : `{probs[0]:.1%}`",
        f"🔴 Bear  : `{probs[1]:.1%}`",
        f"⚪ Neutral: `{probs[2]:.1%}`", "",
        f"₿ BTC Price : `${btc_price:,.0f}`",
        f"🕐 Bar time : `{bar_time} UTC`", "",
    ]

    if reason == "NEW_SIGNAL":
        direction = "SHORT" if reg["mode"] == "bear_only" else "LONG"
        lines += [
            "*📋 Paper Trade Setup*",
            f"  Direction : {direction}",
            f"  Stop      : {reg['stop_loss_pct']:.1%}",
            f"  Target    : {reg['stop_loss_pct'] * reg['target_rr']:.1%}",
            f"  Hold max  : {reg['max_hold_hours']}h",
        ]
    elif reason == "SIGNAL_ENDED":
        lines += ["_Monitor for next signal._"]

    return "\n".join(lines)


# ── Per-model run ──────────────────────────────────────────────────────────────

def run_model(reg: dict, df_features: pd.DataFrame, df_raw: pd.DataFrame,
              live_price: float, btc_price: float, bar_time: str,
              state: dict) -> None:
    model_state = state["models"][reg["key"]]

    try:
        model, scaler = load_model_and_scaler(reg["model_dir"])
    except FileNotFoundError as exc:
        log.error(f"[{reg['key']}] {exc}")
        return

    try:
        pred, conf, probs = predict(model, scaler, df_features)
    except Exception as exc:
        log.error(f"[{reg['key']}] inference failed: {exc}")
        return

    log.info(f"[{reg['key']}] {REGIME_EMOJIS[pred]} {REGIME_NAMES[pred]}  conf={conf:.1%}")

    do_notify, reason = should_notify(reg, pred, conf, model_state)
    if do_notify:
        send_telegram(build_message(reg, pred, conf, probs, reason, bar_time, btc_price))
        model_state["notified_conf"] = conf

    # ── Paper trade lifecycle (independent ledger per model) ─────────────────
    config = dict(pt.PAPER_TRADE_CONFIG)
    config.update({
        "MODE"                        : reg["mode"],
        "REGIME_CONFIDENCE_THRESHOLD" : reg["conf_threshold"],
        "MAX_HOLD_HOURS"              : reg["max_hold_hours"],
        "STOP_LOSS_PCT"               : reg["stop_loss_pct"],
        "INITIAL_TARGET_RR"           : reg["target_rr"],
        "MODEL_KEY"                   : reg["key"],
        "MODEL_LABEL"                 : reg["label"],
    })

    equity    = model_state.get("equity", config["STARTING_EQUITY"])
    trade     = model_state.get("paper_trade", {"open": False})
    trade_log = model_state.get("trade_log", [])
    ema_value = pt.compute_ema(df_raw, pt.PAPER_TRADE_CONFIG["EMA_PERIOD"])

    if config["ENABLED"]:
        if trade.get("open"):
            trade, closed, close_reason, net_pnl = pt.manage_trade(
                trade, live_price, ema_value, pred, conf, config
            )
            if closed:
                equity += net_pnl
                trade_log.append(dict(trade))
                log.info(f"[{reg['key']}] trade CLOSED ({close_reason})  net_pnl=${net_pnl:+.2f}  equity=${equity:.2f}")
                send_telegram(pt.format_close_message(trade, equity))
        else:
            desired_dir = pt.get_desired_direction(pred, conf, config)
            if desired_dir is not None:
                trade = pt.open_trade(desired_dir, live_price, equity, config)
                log.info(f"[{reg['key']}] trade OPENED  dir={desired_dir}  entry=${live_price:,.2f}")
                send_telegram(pt.format_open_message(trade, config, equity))

    state["models"][reg["key"]] = {
        "key"          : reg["key"],
        "model_label"  : reg["label"],
        "pred"         : pred,
        "conf"         : conf,
        "notified_conf": model_state.get("notified_conf", 0.0),
        "ts"           : bar_time,
        "equity"       : equity,
        "paper_trade"  : trade,
        "trade_log"    : trade_log,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("BTC Regime Monitor — starting run (%d models)", len(MODEL_REGISTRY))

    log.info("Fetching BTC bars from Binance...")
    try:
        df_raw = fetch_btc_bars(n=N_BARS)
    except Exception as exc:
        log.error(f"Data fetch failed: {exc}")
        sys.exit(1)

    log.info("Fetching macro proxies (SPX, DXY, ETH/BTC)...")
    start_dt = df_raw.index[0].strftime("%Y-%m-%d")
    end_dt   = (df_raw.index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        spx_map, dxy_map, eth_btc_map = fetch_macro(start_dt, end_dt)
    except Exception as exc:
        log.warning(f"Macro fetch failed ({exc}) — continuing with zeros")
        spx_map = dxy_map = eth_btc_map = pd.Series(dtype=float)

    log.info("Computing shared feature matrix (once for both models)...")
    try:
        df_features = build_features(df_raw, spx_map, dxy_map, eth_btc_map)
    except Exception as exc:
        log.error(f"Feature computation failed: {exc}")
        sys.exit(1)

    bar_open_time  = df_features.index[-1]
    bar_close_time = bar_open_time + pd.Timedelta(minutes=5)
    bar_time  = bar_close_time.strftime("%Y-%m-%d %H:%M")
    btc_price = float(df_raw["close"].iloc[-1])

    try:
        live_price = fetch_current_price()
    except Exception as exc:
        log.warning(f"Live price fetch failed ({exc}) — using last closed bar")
        live_price = btc_price

    state = load_state()

    for reg in MODEL_REGISTRY:
        run_model(reg, df_features, df_raw, live_price, btc_price, bar_time, state)

    # ── On-demand Telegram status command (poll AFTER this run's numbers saved) ─
    state = tc.poll_and_reply(state)

    save_state(state)
    git_commit_state()
    log.info("Run complete")


if __name__ == "__main__":
    main()