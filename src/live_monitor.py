"""
live_monitor.py
===============
Fetches the latest BTC data, runs regime inference, and sends a Telegram
notification when conditions change meaningfully.

Designed to run every 5 minutes via GitHub Actions.

State is persisted between runs via a JSON file (state/last_signal.json)
which is saved and restored using the GitHub Actions cache action.

Exit codes:
  0 — ran successfully
  1 — error (data fetch failed, model load failed, etc.)
"""

import os
import sys
import json
import math
import pickle
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
MODEL_PATH  = ROOT / "model" / "best_regime_transformer.pth"
SCALER_PATH = ROOT / "model" / "scaler_X.pkl"
STATE_PATH  = ROOT / "state"  / "last_signal.json"

sys.path.insert(0, str(Path(__file__).parent))
from model_arch import RegimeTransformer
from features  import FEATURE_COLUMNS, fetch_btc_bars, fetch_macro, build_features

# ── Config ────────────────────────────────────────────────────────────────────
LOOKBACK          = 240          # must match training
N_BARS            = 8700         # ~30 days of 5-min bars
NUM_CLASSES       = 3
REGIME_NAMES      = {0: "Bull Trend", 1: "Bear Trend", 2: "Neutral"}
REGIME_EMOJIS     = {0: "🟢", 1: "🔴", 2: "⚪"}

# Notification thresholds
BEAR_THRESHOLD    = 0.45         # minimum confidence to alert on Bear
BULL_THRESHOLD    = 0.45         # minimum confidence to alert on Bull (set high to mute)
CONF_CHANGE_DELTA = 0.05         # re-notify if confidence shifts by this much

# Model architecture — must match exactly what was used during training
MODEL_PARAMS = dict(
    input_dim  = 35,
    d_model    = 128,
    nhead      = 4,
    num_layers = 3,
    dim_ffn    = 512,
    dropout    = 0.2,
    num_classes= NUM_CLASSES,
)

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
        log.info("Telegram message sent")
        return True
    except Exception as exc:
        log.error(f"Telegram send failed: {exc}")
        return False


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load the last known signal state from disk."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"pred": 2, "conf": 0.0, "notified_conf": 0.0, "ts": ""}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ── Model inference ───────────────────────────────────────────────────────────

def load_model() -> tuple:
    """Load model and scaler. Returns (model, scaler)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            "Add best_regime_transformer.pth to the model/ directory."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}\n"
            "Add scaler_X.pkl to the model/ directory."
        )

    device = torch.device("cpu")   # GitHub Actions has no GPU

    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = RegimeTransformer(**MODEL_PARAMS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    log.info(
        f"Model loaded  (epoch {ckpt.get('epoch', 0) + 1}, "
        f"val_f1={ckpt.get('val_f1', 0):.4f})"
    )
    return model, scaler


@torch.no_grad()
def predict(model, scaler, df_features: object) -> tuple[int, float, list[float]]:
    """
    Run inference on the last bar of df_features.

    Returns
    -------
    pred_class : int   — 0=Bull, 1=Bear, 2=Neutral
    confidence : float — max softmax probability
    probs      : list  — [p_bull, p_bear, p_neutral]
    """
    X_all    = df_features[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = scaler.transform(X_all)

    # Last LOOKBACK bars
    if len(X_scaled) < LOOKBACK:
        raise ValueError(
            f"Not enough bars after feature warmup: got {len(X_scaled)}, need {LOOKBACK}"
        )
    seq = X_scaled[-LOOKBACK:]                      # (lookback, features)
    t   = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, lookback, features)

    logits = model(t)
    probs  = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_class = int(np.argmax(probs))
    confidence = float(max(probs))
    return pred_class, confidence, probs


# ── Notification logic ────────────────────────────────────────────────────────

def should_notify(pred: int, conf: float, last: dict) -> tuple[bool, str]:
    """
    Decide whether to send a Telegram message and what type.

    Returns (send: bool, reason: str)
    """
    last_pred = last.get("pred", 2)
    last_conf = last.get("notified_conf", 0.0)

    # ── New Bear signal (model just switched to Bear above threshold) ─────────
    if pred == 1 and conf >= BEAR_THRESHOLD and last_pred != 1:
        return True, "NEW_BEAR"

    # ── Bear signal ended ─────────────────────────────────────────────────────
    if pred != 1 and last_pred == 1 and last.get("notified_conf", 0) >= BEAR_THRESHOLD:
        return True, "BEAR_ENDED"

    # ── Bear signal still active but confidence shifted significantly ─────────
    if pred == 1 and conf >= BEAR_THRESHOLD and last_pred == 1:
        if abs(conf - last_conf) >= CONF_CHANGE_DELTA:
            return True, "BEAR_UPDATE"

    # ── New Bull signal (optional — muted by default via high threshold) ──────
    if pred == 0 and conf >= BULL_THRESHOLD and last_pred != 0:
        return True, "NEW_BULL"

    return False, "NO_CHANGE"


def build_message(
    pred: int,
    conf: float,
    probs: list,
    reason: str,
    bar_time: str,
    btc_price: float,
) -> str:
    emoji  = REGIME_EMOJIS[pred]
    name   = REGIME_NAMES[pred]

    headers = {
        "NEW_BEAR"   : "🚨 *NEW BEAR SIGNAL*",
        "BEAR_ENDED" : "✅ *BEAR SIGNAL ENDED*",
        "BEAR_UPDATE": "📊 *BEAR UPDATE*",
        "NEW_BULL"   : "🚀 *NEW BULL SIGNAL*",
    }
    header = headers.get(reason, "📊 *REGIME UPDATE*")

    # Confidence bar (10 chars)
    filled = int(conf * 10)
    bar    = "█" * filled + "░" * (10 - filled)

    lines = [
        header,
        "",
        f"{emoji} *{name}*  —  `{conf:.1%}` confidence",
        f"`[{bar}]`",
        "",
        f"🟢 Bull  : `{probs[0]:.1%}`",
        f"🔴 Bear  : `{probs[1]:.1%}`",
        f"⚪ Neutral: `{probs[2]:.1%}`",
        "",
        f"₿ BTC Price : `${btc_price:,.0f}`",
        f"🕐 Bar time : `{bar_time} UTC`",
        "",
    ]

    if reason == "NEW_BEAR" and conf >= 0.45:
        lines += [
            "*📋 Paper Trade Setup*",
            f"  Direction : SHORT",
            f"  Stop      : +1.0% above entry",
            f"  Target    : -2.5% below entry",
            f"  Hold max  : 6 hours (72 bars)",
            "",
            "_Track this in your paper trade log._",
        ]
    elif reason == "BEAR_ENDED":
        lines += ["_Monitor for next signal._"]
    elif reason == "NEW_BULL" and conf >= 0.45:
        lines += [
            "*📋 Paper Trade Setup*",
            f"  Direction : LONG",
            f"  Stop      : -1.0% below entry",
            f"  Target    : +2.5% above entry",
            f"  Hold max  : 6 hours (72 bars)",
            "",
            "_Track this in your paper trade log._",
        ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("BTC Regime Monitor — starting run")
    log.info(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model, scaler = load_model()
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ── Fetch data ────────────────────────────────────────────────────────────
    log.info("Fetching BTC bars from Binance...")
    try:
        df_raw = fetch_btc_bars(n=N_BARS)
    except Exception as exc:
        log.error(f"Data fetch failed: {exc}")
        sys.exit(1)

    # ── Macro proxies ─────────────────────────────────────────────────────────
    log.info("Fetching macro proxies (SPX, DXY, ETH/BTC)...")
    start_dt = df_raw.index[0].strftime("%Y-%m-%d")
    import pandas as pd
    end_dt = (df_raw.index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    try:
        spx_map, dxy_map, eth_btc_map = fetch_macro(start_dt, end_dt)
    except Exception as exc:
        log.warning(f"Macro fetch partial failure: {exc} — continuing with zeros")
        import pandas as pd
        spx_map = dxy_map = eth_btc_map = pd.Series(dtype=float)

    # ── Feature engineering ───────────────────────────────────────────────────
    log.info("Computing features...")
    try:
        df_features = build_features(df_raw, spx_map, dxy_map, eth_btc_map)
    except Exception as exc:
        log.error(f"Feature computation failed: {exc}")
        sys.exit(1)

    log.info(f"Feature matrix shape: {df_features.shape}")

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        pred, conf, probs = predict(model, scaler, df_features)
    except Exception as exc:
        log.error(f"Inference failed: {exc}")
        sys.exit(1)

    bar_time  = df_features.index[-1].strftime("%Y-%m-%d %H:%M")
    btc_price = float(df_raw["close"].iloc[-1])

    log.info(f"Prediction : {REGIME_EMOJIS[pred]} {REGIME_NAMES[pred]}")
    log.info(f"Confidence : {conf:.1%}")
    log.info(f"Probs      : Bull={probs[0]:.1%}  Bear={probs[1]:.1%}  Neutral={probs[2]:.1%}")
    log.info(f"BTC price  : ${btc_price:,.0f}")

    # Always print a one-liner for GitHub Actions log readability
    print(
        f"REGIME | {REGIME_NAMES[pred]} {conf:.1%} | "
        f"Bull={probs[0]:.1%} Bear={probs[1]:.1%} Neutral={probs[2]:.1%} | "
        f"BTC=${btc_price:,.0f} | {bar_time} UTC"
    )

    # ── Notification decision ─────────────────────────────────────────────────
    last_state = load_state()
    send, reason = should_notify(pred, conf, last_state)

    if send:
        msg = build_message(pred, conf, probs, reason, bar_time, btc_price)
        log.info(f"Sending Telegram notification: {reason}")
        send_telegram(msg)
        notified_conf = conf if pred in (0, 1) else 0.0
    else:
        log.info(f"No notification needed ({reason})")
        notified_conf = last_state.get("notified_conf", 0.0)

    # ── Persist state ─────────────────────────────────────────────────────────
    save_state({
        "pred"         : pred,
        "conf"         : conf,
        "notified_conf": notified_conf,
        "ts"           : bar_time,
        "btc_price"    : btc_price,
        "probs"        : probs,
    })

    log.info("Run complete")


if __name__ == "__main__":
    main()
