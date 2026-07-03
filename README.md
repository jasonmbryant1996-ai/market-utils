# BTC Regime Monitor

Runs every 5 minutes via GitHub Actions. Fetches live BTC data, runs
the trained RegimeTransformer, and sends a Telegram message when a
Bear (or Bull) signal fires above your confidence threshold.

---

## What You Get

- **Telegram alert** when the model predicts Bear ≥ 45% confidence
- **Paper trade setup** included in the message (direction, stop, target)
- **Exit alert** when the Bear signal ends
- All signals logged in the GitHub Actions run history

---

## Setup Guide — Every Step

### STEP 1 — Create the Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name, e.g. `BTC Regime Bot`
4. Choose a username, e.g. `btc_regime_42_bot` (must end in `bot`)
5. BotFather replies with your **Bot Token** — looks like:
   ```
   7412365890:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   Save this. You will need it in Step 4.

6. Now get your **Chat ID**:
   - Send any message to your new bot (e.g. "hello")
   - Open this URL in your browser (replace YOUR_TOKEN):
     ```
     https://api.telegram.org/botYOUR_TOKEN/getUpdates
     ```
   - Find `"chat":{"id":` in the response — the number after it is your Chat ID
   - Example: `"chat":{"id":123456789,...}`
   - Save this number.

---

### STEP 2 — Create the GitHub Repository

1. Go to **github.com** → click **+** (top right) → **New repository**
2. Name it: `market_utils`
3. Set visibility to **Public** ← important (GitHub Actions is free for public repos)
4. Do NOT initialise with README (you'll push the files yourself)
5. Click **Create repository**
6. Copy the repository URL shown — looks like:
   ```
   https://github.com/jasonmbryant1996-ai/market_utils.git
   ```

---

### STEP 3 — Add the Code Files

On your computer, open a terminal and run:

```bash
# Clone the empty repo
git clone https://github.com/jasonmbryant1996-ai/market_utils.git
cd market_utils

# Create the folder structure
mkdir -p .github/workflows src model state
```

Now copy all the files from this project into the cloned folder,
keeping the same structure:

```
market_utils/
├── .github/
│   └── workflows/
│       └── regime_monitor.yml
├── model/
│   └── README.md           ← add your .pth and .pkl here next
├── src/
│   ├── model_arch.py
│   ├── features.py
│   └── live_monitor.py
├── state/
│   └── README.md
├── requirements.txt
└── README.md               ← this file
```

---

### STEP 4 — Add Your Model Files

Download these two files from Kaggle:

| File | Kaggle path |
|------|-------------|
| `best_regime_transformer.pth` | `/kaggle/working/best_regime_transformer.pth` |
| `scaler_X.pkl` | `/kaggle/working/scaler_X.pkl` |

**How to download from Kaggle:**
- In your notebook, run: `print(os.listdir('/kaggle/working/'))`
- In the Kaggle sidebar: Files → working → click the filename → Download

Place both files in the `model/` folder:

```
model/
├── README.md
├── best_regime_transformer.pth   ← add this
└── scaler_X.pkl                  ← add this
```

---

### STEP 5 — Verify MODEL_PARAMS Matches Your Checkpoint

Open `src/live_monitor.py` and find `MODEL_PARAMS`:

```python
MODEL_PARAMS = dict(
    input_dim  = 35,
    d_model    = 128,
    nhead      = 4,
    num_layers = 3,
    dim_ffn    = 512,
    dropout    = 0.2,
    num_classes= 3,
)
```

This must exactly match the architecture you trained with.
If you trained with different values, update them here.

---

### STEP 6 — Push Everything to GitHub

```bash
cd market_utils

git add .
git commit -m "Initial deploy: regime monitor + model files"
git push origin main
```

If git push asks for credentials:
- Username: your GitHub username
- Password: use a Personal Access Token (not your account password)
  - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate new token → check `repo` scope → copy the token
  - Use it as the password

---

### STEP 7 — Add Secrets to GitHub

This is where you add your Telegram bot token and chat ID so GitHub
Actions can send messages without exposing them in the code.

1. Go to your repo on GitHub
2. Click **Settings** (top menu of the repo, not your account)
3. Left sidebar → **Secrets and variables** → **Actions**
4. Click **New repository secret** — add these two:

| Name | Value |
|------|-------|
| `TELEGRAM_BOT_TOKEN` | `7412365890:AAHxxx...` (your bot token from Step 1) |
| `TELEGRAM_CHAT_ID` | `123456789` (your chat ID from Step 1) |

---

### STEP 8 — Trigger a Test Run

1. Go to your repo on GitHub
2. Click **Actions** tab (top menu)
3. Click **BTC Regime Monitor** in the left sidebar
4. Click **Run workflow** → **Run workflow** (green button)
5. Wait ~3 minutes for it to complete
6. Click the run to see logs — you should see lines like:
   ```
   REGIME | Bear Trend 48.3% | Bull=22.1% Bear=48.3% Neutral=29.6% | BTC=$67,420 | 2025-08-15 14:35 UTC
   ```
7. If the model predicted Bear ≥ 45%, you should receive a Telegram message

**If the run fails**, click the failed step to see the error message.
Common issues are listed in the Troubleshooting section below.

---

### STEP 9 — Confirm Automatic Scheduling

After the first successful manual run, the workflow will run automatically
every 5 minutes. You can verify by:

1. Waiting 5-10 minutes
2. Going to Actions tab → BTC Regime Monitor
3. You should see new runs appearing automatically

You will only receive Telegram messages when:
- A new Bear signal fires (model switches to Bear ≥ 45%)
- The Bear signal ends
- The Bear confidence changes by more than 5%

---

## Paper Trading Log

Every time you receive a Bear entry alert, log it:

| # | Date/Time UTC | BTC Entry | Stop (+1%) | Target (-2.5%) | Conf | Result | Notes |
|---|--------------|-----------|------------|-----------------|------|--------|-------|
| 1 | | | | | | | |

Check the outcome after max 6 hours (72 bars). Log win/loss.
After 20-30 trades, compare your live win rate to the backtest (54-58%).

---

## Changing Notification Settings

All thresholds are in `src/live_monitor.py`:

```python
BEAR_THRESHOLD    = 0.45    # lower = more alerts, higher = fewer but more selective
BULL_THRESHOLD    = 0.45    # set to 0.99 to effectively mute Bull alerts
CONF_CHANGE_DELTA = 0.05    # how much confidence must shift to re-notify on same signal
```

After changing, commit and push:
```bash
git add src/live_monitor.py
git commit -m "Adjust notification thresholds"
git push
```

---

## Troubleshooting

**"Model not found" error**
→ You forgot to add `.pth` and `.pkl` files to `model/`

**"Not enough bars after feature warmup" error**
→ Binance returned fewer bars than expected. Wait 5 minutes and retry.

**No Telegram message received**
→ Check secrets are set correctly (Step 7)
→ Check Actions logs for "Telegram send failed" line
→ Test your bot manually: send it a message, confirm it responds

**"state_dict mismatch" error**
→ MODEL_PARAMS in live_monitor.py doesn't match your checkpoint
→ Update the params to match exactly what you trained with

**GitHub Actions not running every 5 minutes**
→ This is normal — GitHub has up to ~1 minute of jitter
→ Also, GitHub may pause scheduled workflows if the repo has no activity
   for 60 days. To prevent this, push a commit or trigger manually

**High number of Telegram messages**
→ Raise `BEAR_THRESHOLD` from 0.45 to 0.50 or 0.55
→ Or raise `CONF_CHANGE_DELTA` from 0.05 to 0.10

---

## File Structure

```
market_utils/
├── .github/
│   └── workflows/
│       └── regime_monitor.yml   GitHub Actions schedule + steps
├── model/
│   ├── README.md
│   ├── best_regime_transformer.pth  (you add this — not in repo template)
│   └── scaler_X.pkl                 (you add this — not in repo template)
├── src/
│   ├── model_arch.py       RegimeTransformer class (must match training)
│   ├── features.py         35-feature live engineering pipeline
│   └── live_monitor.py     Main script: fetch → infer → notify
├── state/
│   └── README.md           last_signal.json lives here (managed by CI)
├── requirements.txt        CPU-only PyTorch + dependencies
└── README.md               This file
```
